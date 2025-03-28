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
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_all_runners(ucell, inp);
    this->maxniter_occ_num = PARAM.inp.maxniter_occ_num;
    this->maxniter_orb = PARAM.inp.scf_nmax;

    // initialize rdmft
    if( PARAM.inp.rdmft_orb_opti == "iter_diag" || PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        rdmft_solver.init(this->GG,
                            this->GK,
                            this->pv,
                            ucell,
                            this->kv,
                            *(this->pelec),
                            this->orb_,
                            this->two_center_bundle_,
                            PARAM.inp.dft_functional,
                            PARAM.inp.rdmft_power_alpha,
                            true);
    }
    else
    {
        rdmft_solver.init(this->GG,
                            this->GK,
                            this->pv,
                            ucell,
                            this->kv,
                            *(this->pelec),
                            this->orb_,
                            this->two_center_bundle_,
                            PARAM.inp.dft_functional,
                            PARAM.inp.rdmft_power_alpha,
                            false);
    }

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
    {
        this->iter_diag_orb.init(rdmft_solver.nk_total, this->kv.get_nkstot_full(), rdmft_solver.para_Eij, this->pv, &this->rdmft_solver);
        this->ls_opti_occ_num.init(this->kv, &this->rdmft_solver);

        this->iter_diag_orb.init_orb_by_lambda = PARAM.inp.init_orb_by_lambda;
        this->iter_diag_orb.scale_F = PARAM.inp.scale_fock;
        this->iter_diag_orb.scale_zeta = PARAM.inp.scale_zeta;
        // this->iter_diag_orb.if_rotate_Fock = PARAM.inp.rotate_fock;
        // this->iter_diag_orb.if_rotate_Fock = false;

        this->ls_opti_occ_num.ebi.random_inital = PARAM.inp.random_occ_num;
        this->ls_opti_occ_num.ebi.solve_mu_thr = PARAM.inp.solve_mu_thr;
        this->ls_opti_occ_num.ebi.tot_nelec_thr = PARAM.inp.tot_nelec_thr;
        this->ls_opti_occ_num.ls_wolfe_c1 = PARAM.inp.ls_wolfe_c1;
        this->ls_opti_occ_num.ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2;
        this->ls_opti_occ_num.ls_armijo_c1 = PARAM.inp.ls_armijo_c1;
        this->ls_opti_occ_num.ls_armijo_c2 = PARAM.inp.ls_armijo_c2;
        this->ls_opti_occ_num.ls_condition = PARAM.inp.ls_condition;
        this->ls_opti_occ_num.max_step_size = PARAM.inp.max_step_size;
        this->ls_opti_occ_num.min_step_size = PARAM.inp.min_step_size;
    }
    else if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        this->idmft.init(rdmft_solver.nk_total, this->kv, rdmft_solver.para_Eij, this->pv, &this->rdmft_solver);
    }

    // this->ebi.init(rdmft_solver.nk_total);
    // this->bfgs_rdmft.init(rdmft_solver.nk_total, PARAM.inp.nbands);

    // // convergence parameters
    // this->dft_optimize = true;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // this->conver_initial_value = true; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // this->iter_diag_ethr = 1e-8;
    // this->occ_num_thr = 1e-5; // how much is proper?
    // this->lambda_thr = 1e-4;

    // this->iter_diag_orb.scale_zeta = 0.01;

    // this->ls_opti_occ_num.ebi.random_inital = true;
    // this->ls_opti_occ_num.ebi.solve_mu_thr = 1e-10;
    // this->ls_opti_occ_num.ebi.tot_nelec_thr = 1e-10;
    // this->ls_opti_occ_num.ls_wolfe_c1 = 0.0001;
    // this->ls_opti_occ_num.ls_wolfe_c2 = 0.999;
    // this->ls_opti_occ_num.ls_armijo_c1 = 0.0001;
    // this->ls_opti_occ_num.ls_armijo_c2 = 0.9;
    // this->ls_opti_occ_num.ls_condition = "swolfe";
    // this->ls_opti_occ_num.max_step_size = 1000.0;
    // this->ls_opti_occ_num.min_step_size = 1e-10;

    // convergence parameters
    this->dft_optimize = PARAM.inp.dft_opti;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->conver_initial_value = PARAM.inp.conv_inital_value; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    this->iter_diag_ethr = PARAM.inp.iter_diag_ethr;
    this->occ_num_thr = PARAM.inp.occ_num_thr; // how much is proper?
    this->lambda_thr = PARAM.inp.lambda_thr;

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);
    rdmft_solver.update_ion(ucell, *(this->pw_rho), this->ppcell.vloc, this->sf.strucFac);

    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        // the command to stop the runner is in the exx_iter_finish() function of Exx_LRI_interface.hpp
        ModuleESolver::ESolver_KS<TK>::runner(ucell, istep);
    }
    else
    {
        this->maxniter = 1;
        ModuleESolver::ESolver_KS<TK>::runner(ucell, istep);
    }
    this->rdmft_solver.inital_wfc_occNum(this->pelec->wg, this->psi);


    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    this->rdmft_solver.cal_Energy();

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
    {
        this->get_start_guess();

        double diff_etotal = 1.0;
        double diff_occ_num_max = 1.0;

        for(int iter_occ_num=1; iter_occ_num <= this->maxniter_occ_num; ++iter_occ_num)
        {
            int small_diffE = 0;
            this->iter_diag_orb.before_opti();

            double temp_diag_ethr = iter_diag_ethr;
            // if( !this->conver_initial_value && !this->dft_optimize )
            // {
            //     if( diff_occ_num_max > 50 * this->occ_num_thr )
            //     {
            //         temp_diag_ethr *= 100; // 1000 ?
            //     }
            // }

            for(int iter_orb=1; iter_orb <= this->maxniter_orb; ++iter_orb)
            {
                // delete or save?
                if(this->dft_optimize)
                {
                    this->update_occ_num_dft(this->rdmft_solver);
                    // GlobalV::ofs_running << "\n******\nniter_occ_number of rdmft: " << iter_occ_num << std::endl << std::fixed << std::setprecision(7);
                    // rdmft::global_printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
                    // GlobalV::ofs_running << std::endl << std::defaultfloat;

                    // std::cout << "\n******\nupdate_occ_num_dft: " << std::endl << std::fixed << std::setprecision(7);
                    // rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
                    // std::cout << std::endl << std::defaultfloat;
                }

                // optimize natural orbitals
                diff_etotal = this->iter_diag_orb.optimize_orb(this->rdmft_solver);

                std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
                std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal << "\ndiff_E: " << diff_etotal << "\n******" << std::endl << std::defaultfloat;

                // if( std::abs(diff_etotal) < iter_diag_ethr )
                if( std::abs(diff_etotal) < temp_diag_ethr )
                {
                    ++small_diffE;
                }
                else
                {
                    small_diffE = 0;
                }
                // if( small_diffE >= 2 && iter_orb >= 3) break; // reference: relative error < 1e-7
                if( small_diffE >= 1 && iter_orb >= 3) break; // reference: relative error < 1e-7
                // if( iter_orb > 200 ) this->iter_diag_orb.scale_zeta *= 0.1; // test 
            }

            // 
            if(dft_optimize) break;

            // optimize natural occupation numbers
            diff_occ_num_max = this->opti_occ_num(this->dft_optimize);

            std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num  << "\ndiff_occ_num_max(*num_symm_k): " << diff_occ_num_max << std::endl << std::fixed << std::setprecision(7);
            rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
            
            // double sys_nelec_now = 0.0;
            // for(int i=0; i<rdmft_solver.occ_number.nr*rdmft_solver.occ_number.nc; ++i)
            // {
            //     sys_nelec_now += rdmft_solver.occ_number.c[i];
            // }
            // std::cout << "\n system nelec now: " <<  sys_nelec_now  << "\ndiff_occ_num_max: " << diff_occ_num_max << std::endl << std::defaultfloat;
            // std::vector<double> sys_nelec_spin =  this->ls_opti_occ_num.ebi.get_nelec_spin();
            // if( std::abs( sys_nelec_now - sys_nelec_spin[0] ) > 1e-10 )
            // {
            //     rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, this->ls_opti_occ_num.get_var_x()->data(), "now var_x", 10);
            // }

            if( diff_occ_num_max < this->occ_num_thr )
            {
                double max_off_diag_F = this->iter_diag_orb.check_hermi_lambda();
                if( max_off_diag_F < this->lambda_thr )
                {
                    break;
                }
                else
                {
                    std::cout << "\n******\nmax_off_diag_F > lambda_thr: " << max_off_diag_F << "\n******\n" << std::endl;
                }
            }


            // TODO: optimize occ_number
            // this->rdmft_solver.update_elec(occ_num_temp);

        }

    }
    else if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        // this->idmft.solve_zero_occ_num();
        for(int iter_orb=1; iter_orb <= this->maxniter_orb; ++iter_orb)
        {
            double diff_etotal = this->idmft.optimize();

            std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
            std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
                        << "\n\nE(TV + Hartree + XC) by RDMFT:   " << this->rdmft_solver.E_RDMFT[3]
                        << "\nEcum_entropy: " << this->rdmft_solver.Ecum_entropy
                        << "\nidmft_entropy: " << this->rdmft_solver.idmft_entropy
                        << "\n\ndiff_E: " << diff_etotal
                        << "\ndiff_DM_max: " << this->idmft.get_diff_DM_max()
                        << "\n******" << std::endl << std::defaultfloat;

            // this->idmft.opti_occ_num();

            // temporary
            this->print_info_idmft();

            if( std::abs(diff_etotal) < iter_diag_ethr && this->idmft.get_diff_DM_max() < PARAM.inp.scf_thr  )
            {
                break;
            }
        }
    }

    this->print_info_idmft();

    // std::cout << std::scientific << std::setprecision(1) << std::endl;
    // rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
    // std::cout << std::defaultfloat;

    std::cout << "\n******\n" << "maxniter of NOs in rdmft is: " << this->maxniter_orb << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n\n\n" << std::endl;
}


template <typename TK, typename TR>
double ESolver_RDMFT<TK, TR>::opti_occ_num(bool dft_type, bool first_time)
{
    double diff_occ_num_max = 0.0;
    if(dft_type)
    {
        diff_occ_num_max = this->update_occ_num_dft(this->rdmft_solver);
    }
    else
    {
        diff_occ_num_max = this->ls_opti_occ_num.do_line_search(first_time);
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
    return diff_occ_num_max;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::get_start_guess()
{
    // get start guess occ_number and optimize once
    this->opti_occ_num(this->dft_optimize, true);
    
    std::cout << "\n******\n" << "get inital value in occ_num !!!!!!" << "\n******\n" << std::endl;

    // get start guess natural orbitals
    this->iter_diag_orb.get_start_guess(rdmft_solver, this->conver_initial_value);

    std::cout << "\n******\n" << "get inital value in orbitals !!!!!!" << "\n******\n" << std::endl;

    // optimize occ_number
    this->opti_occ_num(this->dft_optimize);

    std::cout << "\n******\n" << "after ESolver_RDMF::get_start_guess() !!!!!!" << "\n******\n" << std::endl;
}







template <typename TK, typename TR>
double ESolver_RDMFT<TK, TR>::update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver_in)
{
    elecstate::ElecState* pelec_ = this->pelec;
    K_Vectors& kv_ = this->kv;
 
    std::vector< std::vector<double> > ekb_rdmft(rdmft_solver_in.nk_total, std::vector<double>(PARAM.inp.nbands, 0.0));

    for(int ik=0; ik<ekb_rdmft.size(); ++ik)
    {
        std::vector<TK> Hij_rdmft = rdmft_solver_in.Hij_no_exx[ik];  // should times wk?
        if(GlobalC::exx_info.info_global.cal_exx)
        {
            for(int iloc=0; iloc<Hij_rdmft.size(); ++iloc) Hij_rdmft[iloc] += rdmft_solver_in.Hij_exx[ik][iloc];
        }

        std::vector<TK> temp_egivector(rdmft_solver_in.para_Eij.get_local_size(), 0.0);
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

    return 1.0;

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::print_info_idmft()
{
    if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        for(int ik=0; ik < rdmft_solver.nk_total; ++ik)
        {
            std::cout << "\n\nik: " << ik << std::endl; // << std::fixed << std::setprecision(10);
            std::cout << "---------------------------------------\nnbands      " << "occ_number      " << "energy level(Rydberg)" << std::endl; 
            for(int ib=0; ib < rdmft_solver.nbands_total; ++ib)
            {
                std::cout << ib << "           " << rdmft_solver.occ_number(ik, ib) << "        " << this->idmft.get_energy_level()[ik][ib] << std::endl;
            }
            std::cout << "---------------------------------------\n" << std::endl;
        }
    }
    else
    {
        ;
    }
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

