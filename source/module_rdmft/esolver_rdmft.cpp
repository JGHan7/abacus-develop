//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft.h"
#include "module_rdmft/rdmft_tools.h"
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
void ESolver_RDMFT<TK, TR>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    rdmft_solver.before_all_runners(ucell, inp);
    this->maxniter = inp.scf_nmax;
    this->maxniter_occ_num = this->maxniter;
    this->maxniter_orb = this->maxniter;

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

    this->iter_diag_orb.init(rdmft_solver.nk_total, rdmft_solver.para_Eij, *(rdmft_solver.ParaV));
    this->ls_opti_occ_num.init(&this->rdmft_solver);
    // this->ebi.init(rdmft_solver.nk_total);
    // this->bfgs_rdmft.init(rdmft_solver.nk_total, PARAM.inp.nbands);

    // convergence parameters
    this->dft_optimize = true;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->iter_diag_ethr = 1e-8;
    this->lambda_thr = 1e-4;
    this->occ_num_thr = 1e-3; // how much is proper?

    this->iter_diag_orb.scale_zeta = 0.01;
    this->ls_opti_occ_num.ebi.solve_mu_thr = 1e-10;
    this->ls_opti_occ_num.ebi.random_inital = true;
    this->ls_opti_occ_num.ls_c1 = 0.0001;
    this->ls_opti_occ_num.ls_c2 = 0.999;
    this->ls_opti_occ_num.ls_condition = "swolfe";
    this->ls_opti_occ_num.max_step_size = 1000;

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    // rdmft_solver.before_scf(istep);
    rdmft_solver.update_ion(istep, ucell);

    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        // the command to stop the runner is in the exx_iter_finish() function of Exx_LRI_interface.hpp
        rdmft_solver.runner(ucell, istep);
    }
    else
    {
        rdmft_solver.modify_scf_nmax(1);
        rdmft_solver.runner(ucell, istep);
    }
    this->rdmft_solver.inital_wfc_occNum();

    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    this->rdmft_solver.cal_Energy();

    this->get_start_guess();

    for(int iter_occ_num=1; iter_occ_num <= this->maxniter_occ_num; ++iter_occ_num)
    {
        int small_diffE = 0;
        this->iter_diag_orb.before_opti();
        for(int iter_orb=1; iter_orb <= this->maxniter_orb; ++iter_orb)
        {
            // optimize natural orbitals
            double diff_etotal = this->iter_diag_orb.optimize_orb(this->rdmft_solver);

            // delete or save?
            if(this->dft_optimize) { this->update_occ_num_dft(this->rdmft_solver); }

            std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
            std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal << "\ndiff_E: " << diff_etotal << "\n******\n" << std::endl << std::defaultfloat;

            if( std::abs(diff_etotal) < iter_diag_ethr ) ++small_diffE;
            if( small_diffE > 3 && iter_orb > 5) break; // reference: relative error < 1e-7
            // if( iter_orb > 200 ) this->iter_diag_orb.scale_zeta *= 0.1; // test 
        }

        // optimize natural occupation numbers
        this->opti_occ_num(this->dft_optimize);

        // add something, to determine whether the optimization of the occ_number has converged
        if(dft_optimize) break;

        std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num << std::endl << std::fixed << std::setprecision(5);
        rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number");
        std::cout << std::endl << std::defaultfloat;

        // TODO: optimize occ_number
        // this->rdmft_solver.update_elec(occ_num_temp);

    }

    rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number");

    std::cout << "\n******\n" << "maxniter of rdmft is: " << this->maxniter << "\n******\n" << std::endl;
    std::cout << "\n\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n" << std::endl;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::opti_occ_num(bool dft_type, bool first_time)
{
    if(dft_type)
    {
        this->update_occ_num_dft(this->rdmft_solver);
    }
    else
    {
        this->ls_opti_occ_num.do_line_search(first_time);
    }
    // if(first_time)
    // {
    //     if( this->ebi.random_inital == true )
    //     {
    //         this->ebi.get_inital_guess();
    //         ModuleBase::matrix occ_num = this->ebi.get_occ_number();
    //         this->rdmft_solver.update_elec( &occ_num );
    //         // this->rdmft_solver.cal_E_grad_occ_num();
    //     }
    //     else
    //     {
    //         this->ebi.get_inital_guess( &this->rdmft_solver.occ_number );
    //     }
    //     this->rdmft_solver.cal_E_grad_occ_num();
    //     // transfer this->rdmft_solver.occNum_wfcHamiltWfc -> std::vector
    //     // this->ebi.get_dE_dx(std::vector, this->bfgs_rdmft.dE_dx1);
    // }
    // else
    // {

    // }
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::get_start_guess()
{
    // get start guess occ_number and optimize once
    this->opti_occ_num(this->dft_optimize, true);
    
    // get start guess natural orbitals
    this->iter_diag_orb.get_start_guess(rdmft_solver);

    // optimize occ_number
    this->opti_occ_num(this->dft_optimize);
}







template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver_in)
{
    elecstate::ElecState* pelec_ = rdmft_solver.get_pelec();
    K_Vectors& kv_ = rdmft_solver.get_kv();
 
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

        for(int ib=0; ib<ekb_rdmft[ik].size(); ++ib) pelec_->ekb(ik, ib) = ekb_rdmft[ik][ib];
    }

    pelec_->calEBand();
    pelec_->calculate_weights();
    ModuleBase::matrix occ_number_ks = (pelec_->wg);
    for(int ik=0; ik < occ_number_ks.nr; ++ik)
    {
        for(int inb=0; inb < occ_number_ks.nc; ++inb)
        {
            occ_number_ks(ik, inb) /= kv_.wk[ik];
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
void ESolver_RDMFT<TK, TR>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::cal_stress(UnitCell& ucell, ModuleBase::matrix& force)
{
    ;
}



template class ESolver_RDMFT<double, double>;
template class ESolver_RDMFT<std::complex<double>, double>;
template class ESolver_RDMFT<std::complex<double>, std::complex<double>>;

}

