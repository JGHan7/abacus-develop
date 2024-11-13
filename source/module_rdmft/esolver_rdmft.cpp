//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft.h"
#include "module_rdmft/optimizer/optimizer_tools.h" // temporary
#include <cmath> // temporary

namespace rdmft
{


template <typename TK, typename TR>
ESolver_RDMFT<TK, TR>::ESolver_RDMFT()
{
    this->classname = "ESolver_RDMFT";
}


template <typename TK, typename TR>
ESolver_RDMFT<TK, TR>::~ESolver_RDMFT()
{
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::before_all_runners(const Input_para& inp, UnitCell& ucell)
{
    rdmft_solver.before_all_runners(inp, ucell);
    this->maxniter = rdmft_solver.maxniter;
    this->maxniter_occ_num = this->maxniter;
    this->maxniter_orb = this->maxniter;

    this->iter_diag_ethr = 1e-8;
    this->lambda_thr = 1e-4;
    this->occ_num_thr = 1e-3; // how much is proper?

    this->dft_optimize = true;

    // initialize rdmft
    // if( rdmft_optimize_type == "iterDiag" )
    if( 1 )
    {
        rdmft_solver.init(ucell, PARAM.inp.dft_functional, PARAM.inp.rdmft_power_alpha, true);
    }
    else
    {
        rdmft_solver.init(ucell, PARAM.inp.dft_functional, PARAM.inp.rdmft_power_alpha);
    }

    this->iter_diag_rdmft.init(rdmft_solver.nk_total, rdmft_solver.para_Eij, *(rdmft_solver.ParaV));

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::runner(const int istep, UnitCell& ucell)
{
    // rdmft_solver.before_scf(istep);
    rdmft_solver.update_ion(istep, ucell);

    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        // the command to stop the runner is in the exx_iter_finish() function of Exx_LRI_interface.hpp
        rdmft_solver.runner(istep, ucell);
    }
    else
    {
        rdmft_solver.maxniter = 1;
        rdmft_solver.runner(istep, ucell);
    }
    this->rdmft_solver.inital_wfc_occNum();

    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    // this->rdmft_solver.cal_Hk_Hpsi(); // has done in update_elec()
    this->rdmft_solver.cal_Energy();

    /****** get start guess natural orbitals and occ_number ******/

    // TODO: optimize occ_number
    // this->rdmft_solver.update_elec(occ_num_temp); // could be in optimize occ_number as opti_occNum(rdmft_solver)
    
    // to get start guess natural orbitals
    this->iter_diag_rdmft.get_start_guess(rdmft_solver);

    // TODO: optimize occ_number // to get start guess_occNum
    // this->rdmft_solver.update_elec(occ_num_temp);

    /****** get start guess natural orbitals and occ_number ******/

    if(dft_optimize) this->update_occ_num_dft(this->rdmft_solver);

    for(int iter_occ_num=1; iter_occ_num <= this->maxniter_occ_num; ++iter_occ_num)
    {
        
        this->iter_diag_rdmft.before_inner_loop();
        for(int iter_orb=1; iter_orb <= this->maxniter_orb; ++iter_orb)
        {
            double diff_etotal = this->iter_diag_rdmft.optimize_orb(this->rdmft_solver);

            if(dft_optimize) this->update_occ_num_dft(this->rdmft_solver);

            std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
            std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal << "\ndiff_E: " << diff_etotal << "\n******\n" << std::endl << std::defaultfloat;

            if( std::abs(diff_etotal) < iter_diag_ethr  && iter_orb > 5) break; // reference: relative error < 1e-7
        }

        if(dft_optimize) break;
        break;

        // TODO: optimize occ_number
        // this->rdmft_solver.update_elec(occ_num_temp);

    }

    std::cout << "\n******\n" << "maxniter of rdmft is: " << this->maxniter << "\n******\n" << std::endl;
    std::cout << "\n\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n" << std::endl;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver_in)
{
    std::vector< std::vector<double> > ekb_rdmft(rdmft_solver_in.nk_total, std::vector<double>(PARAM.inp.nbands));

    for(int ik=0; ik<ekb_rdmft.size(); ++ik)
    {
        std::vector<TK> Hij_rdmft = rdmft_solver_in.Hij_no_exx[ik];
        if(GlobalC::exx_info.info_global.cal_exx)
        {
            for(int iloc=0; iloc<Hij_rdmft.size(); ++iloc) Hij_rdmft[iloc] += rdmft_solver_in.Hij_exx[ik][iloc];
        }

        std::vector<TK> temp_egivector(rdmft_solver_in.para_Eij.get_local_size());
        rdmft::pdiag_scalapack( &(rdmft_solver_in.para_Eij), PARAM.inp.nbands, Hij_rdmft.data(), ekb_rdmft[ik].data(), temp_egivector.data(), false );

        for(int ib=0; ib<ekb_rdmft[ik].size(); ++ib) rdmft_solver_in.pelec->ekb(ik, ib) = ekb_rdmft[ik][ib];
    }

    rdmft_solver_in.pelec->calEBand();
    rdmft_solver_in.pelec->calculate_weights();
    ModuleBase::matrix occ_number_ks = (rdmft_solver_in.pelec->wg);
    for(int ik=0; ik < occ_number_ks.nr; ++ik)
    {
        for(int inb=0; inb < occ_number_ks.nc; ++inb)
        {
            occ_number_ks(ik, inb) /= rdmft_solver_in.kv.wk[ik];
        }
    }
    rdmft_solver_in.update_elec(&occ_number_ks);

}















template <typename TK, typename TR>
double ESolver_RDMFT<TK, TR>::cal_energy()
{
    return 0.0;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::cal_force(ModuleBase::matrix& force)
{
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::cal_stress(ModuleBase::matrix& stress)
{
    ;
}



template class ESolver_RDMFT<double, double>;
template class ESolver_RDMFT<std::complex<double>, double>;
template class ESolver_RDMFT<std::complex<double>, std::complex<double>>;

}

