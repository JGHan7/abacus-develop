//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/optimizer_tools.h" // temporary
#include <cmath> // temporary
#include "module_elecstate/elecstate_tools.h" // temporary
#include "module_rdmft/optimizer/bfgs_method.h" // temporary
#include "module_rdmft/optimizer/cg_method.h" // temporary

#include <torch/torch.h>


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
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_all_runners(ucell, inp);

    // initialize rdmft
    if( PARAM.inp.rdmft_orb_opti == "iter_diag" || PARAM.inp.rdmft_orb_opti == "idmft" || 
            PARAM.inp.rdmft_orb_opti == "adam" || PARAM.inp.rdmft_orb_opti == "ft_rdmft" ||
            PARAM.inp.rdmft_orb_opti == "bfgs" || PARAM.inp.rdmft_orb_opti == "cg" || 
            PARAM.inp.opti_by_torch )
    {
        rdmft_solver.init(this->GG,
                            this->GK,
                            this->pv,
                            ucell,
                            this->gd,
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
                            this->gd,
                            this->kv,
                            *(this->pelec),
                            this->orb_,
                            this->two_center_bundle_,
                            PARAM.inp.dft_functional,
                            PARAM.inp.rdmft_power_alpha,
                            false);
    }

    // this->nk_total = this->rdmft_solver.nk_total;
    // this->para_Fij = &this->rdmft_solver.para_Eij;

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" || PARAM.inp.rdmft_orb_opti == "adam" )
    {
        this->iter_diag_orb.init(rdmft_solver.nk_total, this->kv.get_nkstot_full(), rdmft_solver.para_Eij, this->pv, &this->rdmft_solver);
        this->ls_opti_occ_num.init(this->kv, &this->rdmft_solver);

        // this->iter_diag_orb.init_orb_by_lambda = PARAM.inp.init_orb_by_lambda;
        // this->iter_diag_orb.scale_F = PARAM.inp.scale_fock;
        // this->iter_diag_orb.scale_zeta = PARAM.inp.scale_zeta;
        // this->iter_diag_orb.if_rotate_Fock = PARAM.inp.rotate_fock;
        // this->iter_diag_orb.if_rotate_Fock = false;

        // this->ls_opti_occ_num.ebi.random_inital = PARAM.inp.random_occ_num;
        // this->ls_opti_occ_num.ebi.solve_mu_thr = PARAM.inp.solve_mu_thr;
        // this->ls_opti_occ_num.ebi.tot_nelec_thr = PARAM.inp.tot_nelec_thr;
        // this->ls_opti_occ_num.ls_wolfe_c1 = PARAM.inp.ls_wolfe_c1;
        // this->ls_opti_occ_num.ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2;
        // this->ls_opti_occ_num.ls_armijo_c1 = PARAM.inp.ls_armijo_c1;
        // this->ls_opti_occ_num.ls_armijo_c2 = PARAM.inp.ls_armijo_c2;
        // this->ls_opti_occ_num.ls_condition = PARAM.inp.ls_condition;
        // this->ls_opti_occ_num.max_step_size = PARAM.inp.max_step_size;
        // this->ls_opti_occ_num.min_step_size = PARAM.inp.min_step_size;
    }
    else if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        this->idmft.init(rdmft_solver.nk_total, this->kv, rdmft_solver.para_Eij, this->pv, &this->rdmft_solver);
    }
    else if( PARAM.inp.rdmft_orb_opti == "ft_rdmft" )
    {
        this->ft_rdmft.init(rdmft_solver.nk_total, this->kv, rdmft_solver.para_Eij, this->pv, &this->rdmft_solver, this->pelec->ekb);
    }
    else if( PARAM.inp.rdmft_orb_opti == "bfgs" || PARAM.inp.rdmft_orb_opti == "cg" )
    {
        this->ls_opti_occ_num.init(this->kv, &this->rdmft_solver);
        this->ls_opti_orb.init(&this->rdmft_solver);
    }

    this->DM.resize(rdmft_solver.nk_total, std::vector<TK>(this->pv.nloc, 0.0));

    // convergence parameters
    this->dft_optimize = PARAM.inp.dft_opti;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->conver_initial_value = PARAM.inp.conv_inital_value; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    this->iter_diag_ethr = PARAM.inp.iter_diag_ethr;
    this->occ_num_thr = PARAM.inp.occ_num_thr; // how much is proper?
    this->lambda_thr = PARAM.inp.lambda_thr;

    if( PARAM.inp.rdmft_orb_opti == "test" )
    {
        if( PARAM.inp.occ_num_opti == "cg" )
        {
            // this->x_optimizer = std::make_unique< rdmft::CG_method<double> >();
            this->x_optimizer = std::make_unique< rdmft::CG_method<std::complex<double>> >();
            // this->ls.get_options().ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2_cg;
        }
        else
        {
            // this->x_optimizer = std::make_unique< rdmft::BFGS_method<double> >();
            this->x_optimizer = std::make_unique< rdmft::BFGS_method<std::complex<double>> >();
        }
        this->x_optimizer->init(this->dim_x);
        this->data_x.resize(this->dim_x, 0.0);
        this->ls_data_x.resize(this->dim_x, 0.0);
        this->df_dx.resize(this->dim_x, 0.0);
        this->pk.resize(this->dim_x, 0.0);
    }

    // if( PARAM.inp.opti_by_torch )
    // {
    //     this->var_x.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    //     this->dE_dx.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    //     this->R_vec_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    //     this->dE_dR_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    //     this->occ_number_new.create(this->rdmft_solver.nk_total, PARAM.inp.nbands);
    //     this->wfc_new.resize(this->rdmft_solver.nk_total, this->pv.ncol_bands, this->pv.nrow);
    //     this->wfc_new.zero_out();

    //     this->ebi_torch.init(this->rdmft_solver.nk_total, this->kv.get_nkstot_full(), this->kv.wk);
    // }

    // // test torch
    // auto torchTest = torch::rand({4, 4});
    // std::cout << "torchTest:\n" << torchTest << std::endl;

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);
    rdmft_solver.update_ion(ucell, *(this->pw_rho), this->locpp.vloc, this->sf.strucFac);

    this->get_start_guess(ucell, istep);

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
    {
        double diff_etotal = 1.0;
        double diff_occ_num_max = 1.0;

        for(int iter_occ_num=1; iter_occ_num <= PARAM.inp.maxniter_occ_num; ++iter_occ_num)
        {
            int small_diffE = 0;
            this->iter_diag_orb.restart_opti(this->p_hamilt);

            double temp_diag_ethr = iter_diag_ethr;
            // if( !this->conver_initial_value && !this->dft_optimize )
            // {
            //     if( diff_occ_num_max > 50 * this->occ_num_thr )
            //     {
            //         temp_diag_ethr *= 100; // 1000 ?
            //     }
            // }

            for(int iter_orb=1; iter_orb <= PARAM.inp.maxniter_orb; ++iter_orb)
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
                std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
                            << "\n\ndiff_E: " << diff_etotal
                            << "\ndiff_DM_max: " << this->iter_diag_orb.get_diff_DM_max()
                            << "\n******" << std::endl << std::defaultfloat;

                // if( std::abs(diff_etotal) < iter_diag_ethr )
                if( std::abs(diff_etotal) < temp_diag_ethr )
                {
                    ++small_diffE;
                }
                else
                {
                    small_diffE = 0;
                }

                // // if( small_diffE >= 2 && iter_orb >= 3) break; // reference: relative error < 1e-7
                // if( small_diffE >= 1 && iter_orb >= 3) break; // reference: relative error < 1e-7
                // // if( iter_orb > 200 ) this->iter_diag_orb.scale_zeta *= 0.1; // test 

                // if( std::abs(diff_etotal) < iter_diag_ethr && this->iter_diag_orb.get_diff_DM_max() < PARAM.inp.scf_thr )
                if( this->iter_diag_orb.get_diff_DM_max() < PARAM.inp.scf_thr && iter_orb >= 3 )
                {
                    break;
                }

            }

            // 
            if(dft_optimize) break;

            // optimize natural occupation numbers
            diff_occ_num_max = this->opti_occ_num(this->dft_optimize);

            std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num  << "\ndiff_occ_num_max(*num_symm_k): " << diff_occ_num_max << std::endl << std::fixed << std::setprecision(7);
            rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
            
            // temporary
            this->print_info();

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
    else if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        int init_maxniter = 10;
        if(dft_optimize)
        {
            init_maxniter = PARAM.inp.maxniter_orb;
        }
        double diff_E = 1.0;
        double final_diff_E = 1.0;
        double diff_occ_num_max = 1.0;
        double Etotal = this->rdmft_solver.Etotal;

        int tot_orb_iter = 0;
        int tot_occ_num_iter = 0;
        int tot_exteral_iter = 0;

        for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
        {
            tot_exteral_iter = iter; // temp
            bool orb_conv = false;
            bool occ_num_conv = false;

            init_maxniter = std::min(init_maxniter, PARAM.inp.maxniter_orb);
            this->iter_diag_orb.restart_opti(this->p_hamilt);
            for(int iter_orb=1; iter_orb <= init_maxniter; ++iter_orb)
            {
                // // temp test
                // if( iter > 3 )
                // {
                //     break;
                // }

                // delete or save?
                if(this->dft_optimize)
                {
                    this->update_occ_num_dft(this->rdmft_solver);
                }

                // optimize natural orbitals
                diff_E = this->iter_diag_orb.optimize_orb(this->rdmft_solver);

                std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
                std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
                            << "\n\ndiff_E: " << diff_E
                            << "\ndiff_DM_max: " << this->iter_diag_orb.get_diff_DM_max()
                            << "\n******" << std::endl << std::defaultfloat;

                if( this->iter_diag_orb.get_diff_DM_max() < PARAM.inp.scf_thr && iter_orb >= 3 )
                {
                    tot_orb_iter += iter_orb;
                    orb_conv = true;
                    break;
                }
            }
            if( !orb_conv )
            {
                tot_orb_iter += init_maxniter;
            }
            
            final_diff_E = this->rdmft_solver.Etotal - Etotal;
            if( final_diff_E > 0 )
            {
                this->iter_diag_orb.modify_learn_rate();
                init_maxniter += 10;
            }
            Etotal = this->rdmft_solver.Etotal;

            if( dft_optimize )
            {
                break;
            }

            if(!this->dft_optimize) { this->ls_opti_occ_num.restart_opti(); }
            for(int iter_occ_num=1; iter_occ_num <= PARAM.inp.maxniter_occ_num; ++iter_occ_num)
            {
                // optimize natural occupation numbers
                diff_occ_num_max = this->opti_occ_num(this->dft_optimize);
                std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num  << "\ndiff_occ_num_max(*num_symm_k): " << diff_occ_num_max << std::endl << std::fixed << std::setprecision(7);
                rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);

                if( diff_occ_num_max < this->occ_num_thr ) // || (!this->dft_optimize && this->ls_opti_occ_num.diff_rate_max < 0.01)
                {
                    tot_occ_num_iter += iter_occ_num;
                    occ_num_conv = true;
                    break;
                }
            }
            if( !occ_num_conv  )
            {
                tot_occ_num_iter += PARAM.inp.maxniter_occ_num;
            }
            
            final_diff_E = this->rdmft_solver.Etotal - Etotal;
            if( final_diff_E > -1e-6 )
            {
                init_maxniter += 10;
            }
            Etotal = this->rdmft_solver.Etotal;

            double max_off_diag_F = this->iter_diag_orb.check_hermi_lambda();
            if( max_off_diag_F < this->lambda_thr )
            {
                std::cout << "\n******\n\nConvergence!\n\nmax_off_diag_F < lambda_thr: " << max_off_diag_F << "\n******\n" << std::endl;
                break;
            }
            else if( orb_conv && occ_num_conv )
            {
                std::cout << "\n******\n\nNOs(DM) and ONs Converge respectively!\n\nmax_off_diag_F is " << max_off_diag_F << "\n******\n" << std::endl;
                break;
            }
            else
            {
                std::cout << "\n******\nstill optimize NOs and ONs, because max_off_diag_F > lambda_thr: " << max_off_diag_F << "\n******\n" << std::endl;
            }
        }

        this->print_info();

        // // temp
        // // rearrange the ONs of each k point from large to small
        // std::cout << "\n******\nrearrange the ONs of each k point from large to small\n******\n" << std::endl;
        // ModuleBase::matrix temp_num = this->rdmft_solver.occ_number;
        // for(int ik=0; ik < rdmft_solver.nk_total; ++ik)
        // {
        //     int num_bands = this->rdmft_solver.nbands_total;
        //     std::sort(&temp_num(ik, 0), &temp_num(ik, 0) + num_bands, std::greater<>());

        //     std::cout << "\n\nik: " << ik << std::endl; // << std::fixed << std::setprecision(10);
        //     std::cout << "---------------------------------------\nnbands      " << "occ_number      " << std::endl; 
        //     for(int ib=0; ib < num_bands; ++ib)
        //     {
        //         std::cout << ib << "           " << temp_num(ik, ib) << "        " << std::endl;
        //     }
        //     std::cout << "---------------------------------------\n" << std::endl;
        // }

        // std::cout << "\n******\n" << std::fixed << std::setprecision(10);
        // std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
        //             << "\n\nE(TV + Hartree + XC) by RDMFT:   " << this->rdmft_solver.E_RDMFT[3]
        //             // << "\n\ndiff_E: " << diff_etotal
        //             // << "\ndiff_DM_max: " << this->idmft.get_diff_DM_max()
        //             << "\n******" << std::endl << std::defaultfloat;

        if( !PARAM.inp.rdmft_couple_opti )
        {
            std::cout << "\n******\nexternal iter = " << tot_exteral_iter 
                        << "\ntotal orbital iter = " << tot_orb_iter 
                        << "\ntotal occ_num iter = " << tot_occ_num_iter
                        << "\n******\n" << std::endl;
        }

    }
    else if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        // this->idmft.solve_zero_occ_num();
        for(int iter_orb=1; iter_orb <= PARAM.inp.maxniter_orb; ++iter_orb)
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
            this->print_info();

            // if( std::abs(diff_etotal) < iter_diag_ethr && this->idmft.get_diff_DM_max() < PARAM.inp.scf_thr )
            if( this->idmft.get_diff_DM_max() < PARAM.inp.scf_thr )
            {
                break;
            }
        }
    }
    else if( PARAM.inp.rdmft_orb_opti == "ft_rdmft" )
    {
        for(int iter_kappa=1; iter_kappa <= PARAM.inp.maxniter_occ_num; ++iter_kappa)
        {
            // this->ft_rdmft.solve_zero_occ_num();
            for(int iter_orb=1; iter_orb <= PARAM.inp.scf_nmax; ++iter_orb)
            {
                double diff_etotal = this->ft_rdmft.optimize();

                std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
                std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
                            << "\n\nE(TV + Hartree + XC) by RDMFT:   " << this->rdmft_solver.E_RDMFT[3]
                            << "\n\ndiff_E: " << diff_etotal
                            << "\ndiff_DM_max: " << this->ft_rdmft.get_diff_DM_max()
                            << "\n******" << std::endl << std::defaultfloat;

                // this->ft_rdmft.opti_occ_num();

                // temporary
                this->print_info();

                rdmft::printMatrix_pointer(this->rdmft_solver.nk_total, this->rdmft_solver.nbands_total, this->ft_rdmft.occ_number[0].data(), "occ_number in idmft", 10);

                // if( std::abs(diff_etotal) < iter_diag_ethr && this->ft_rdmft.get_diff_DM_max() < PARAM.inp.scf_thr )
                if( this->ft_rdmft.get_diff_DM_max() < PARAM.inp.scf_thr )
                {
                    break;
                }
            }

            std::cout << "\n" << "iter_kappa: " << iter_kappa << "\n" << std::endl;
            this->ft_rdmft.optimize_kappa();
            // if( std::abs(this->ft_rdmft.dE_dk_sum) < 1e-7 || std::abs(this->ft_rdmft.kappa) < 1e-5 ) { break; }
            if( this->ft_rdmft.max_diff_kappa < 1e-6 ) {break;}
        }

    }
    else if( PARAM.inp.rdmft_orb_opti == "test" )
    {   
        bool first_time = true;
        double f_value = 0.0;
        for(int i=0; i<=PARAM.inp.scf_nmax; ++i)
        {
            f_value = this->fx(this->data_x);
            this->df_dx = this->grad_f(this->data_x);
            double norm_grad_x = this->norm_grad(this->df_dx);

            std::cout << "\n\n******\niter: " << i << "\nf(x):  " << f_value << "\nnorm_grad: " << norm_grad_x << std::endl;
            rdmft::printMatrix_pointer(1, this->data_x.size(), this->data_x.data(), "x");
            rdmft::printMatrix_pointer(1, this->data_x.size(), this->df_dx.data(), "df_dx");


            if( norm_grad_x < 1e-8 ) { break; }
            
            this->x_optimizer->get_pk(this->df_dx, this->data_x, this->pk, first_time);
            double step_size = 0.0;
            if(PARAM.inp.ls_condition != "test")
            {
                std::complex<double> dphi_0_ = 0.0;
                double max_pk_elem = 0.0;
                for(int j=0; j<this->dim_x; ++j)
                {
                    dphi_0_ += this->df_dx[j] * std::conj(this->pk[j]);
                    max_pk_elem = std::max( max_pk_elem, std::abs(this->pk[j]) );
                }
                double dphi_0 = std::real(dphi_0_);

                // std::vector<std::complex<double>> ls_data_x(this->dim_x, 0.0);

                auto update_x = [this](const double trial_alpha)
                {
                    for(int j=0; j<this->dim_x; ++j)
                    {
                        this->ls_data_x[j] = this->data_x[j] + trial_alpha * this->pk[j];
                    }
                };

                auto phi = [this, update_x](const double trial_alpha)
                {
                    update_x(trial_alpha);
                    double trial_phi = this->fx(this->ls_data_x);
                    return trial_phi;
                };

                auto dphi = [this]()
                {
                    this->df_dx = this->grad_f(this->ls_data_x);
                    std::complex<double> dphi_ = 0.0;
                    for(int j=0; j<this->dim_x; ++j)
                    {
                        dphi_ += this->df_dx[j] * std::conj(this->pk[j]);
                    }
                    double trial_dphi = std::real(dphi_);
                    return trial_dphi;
                };

                step_size = this->ls.do_line_search(phi, dphi, f_value, dphi_0, max_pk_elem, 1.0);
                std::cout << "\nstep_size in test: " << step_size << "\nphi_0: " << f_value << "\ndphi_0: " << dphi_0 << "\n" << std::endl;
            }


            rdmft::printMatrix_pointer(1, this->data_x.size(), this->pk.data(), "pk");
            std::cout << "\n******\n\n" << std::endl;

            for(int j=0; j<this->data_x.size(); ++j)
            {
                if(PARAM.inp.ls_condition == "test")
                {
                    this->data_x[j] += PARAM.inp.ls_fixed_step * this->pk[j];
                }
                else
                {
                     this->data_x[j] += step_size * this->pk[j];
                }
            }

            first_time = false;
        }
    }
    else if( PARAM.inp.rdmft_orb_opti == "bfgs" || PARAM.inp.rdmft_orb_opti == "cg" )
    {
        
        double Etotal_old = this->rdmft_solver.Etotal;
        int E1_rise = 0;
        int E2_rise = 0;
        for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
        {
            // record the occ_num obtained from the previous optimization
            ModuleBase::matrix occ_num_old = this->ls_opti_occ_num.get_occ_num();

            // optimize occ_num
            double diff_occ_num_max = this->opti_occ_num(this->dft_optimize);
            double E_new1 = this->rdmft_solver.Etotal;

            // optimize NOs using the gradient before optimizing ONs (the gradient of the previous step)
            if( !this->dft_optimize )
            {
                this->rdmft_solver.update_elec( &occ_num_old, nullptr );
            }
            double E_new2 = this->ls_opti_orb.do_line_search();

            // "optimize occ_num"
            if( !this->dft_optimize )
            {
                ModuleBase::matrix occ_num_new = this->ls_opti_occ_num.get_occ_num();
                this->rdmft_solver.update_elec( &occ_num_new, nullptr );
            }

            double diff_E1 = E_new1 - Etotal_old;
            double diff_E2 = E_new2 - Etotal_old;
            double diff_E_all =  this->rdmft_solver.cal_Energy() - Etotal_old;
            Etotal_old = this->rdmft_solver.Etotal;

            if( diff_E1 > 0 )
            {
                ++E1_rise;
                if( E1_rise >= 3 )
                {
                    // restart of turn-off precond ?
                    this->ls_opti_occ_num.restart_opti();
                    E1_rise = 0;
                }
            }
            if( diff_E2 > 0 )
            {
                ++E2_rise;
                if( E2_rise >= 3 )
                {
                    this->ls_opti_orb.restart_opti();
                    E2_rise = 0;
                }
            }

            std::cout << "\n******\nniter of rdmft: " << iter 
                        << std::fixed << std::setprecision(15);
            std::cout << "\n\nEtotal_rdmft by opti ONs: " << E_new1
                        << "\ndiff_E: " << diff_E1 
                        << "\ndiff_occ_num_max: " << diff_occ_num_max
                        << "\n\nEtotal_rdmft by opti NOs: " << E_new2
                        << "\ndiff_E: " << diff_E2
                        << "\n\ndiff_E_all: " << diff_E_all
                        // << "\ndiff_DM_max: " << this->diff_DM_max
                        // << "\n\nmax_off_diag_F: " << this->max_off_diag_Fock
                        // << "\n\ndiff_wfc_norm: " << this->diff_wfc_norm
                        // << "\n\ndiff_wfc_max: " << this->diff_wfc_max
                        << std::endl;

            if( iter%10 == 1 )
            {
                rdmft::printMatrix_pointer(this->rdmft_solver.nk_total, PARAM.inp.nbands, this->rdmft_solver.occ_number.c, "occ_number", 10);
            }
            std::cout << "******" << std::endl << std::defaultfloat;

            if( diff_occ_num_max < this->occ_num_thr && std::abs(diff_E_all) < 1e-7 && iter > 10 )
            {
                break;
            }

        }

        this->print_info();

        std::cout << "\n******\nNumber of NOs restart_opti = " << this->ls_opti_orb.num_restart 
                    << "\nNumber of ONs restart_opti = " << this->ls_opti_occ_num.num_restart 
                    // << "\n\nNumber of line search(NOs) = " << this->ls_opti_orb.get_num_bfgs_restart()
                    // << "\nNumber of line search(ONs) = " << this->ls_opti_occ_num.get_num_bfgs_restart()
                    << "\n\nNumber of BFGS(NOs) restart_skip = " << this->ls_opti_orb.get_num_bfgs_restart()
                    << "\nNumber of BFGS(ONs) restart_skip = " << this->ls_opti_occ_num.get_num_bfgs_restart()
                    << "\n******\n" << std::endl;

    }

    // this->print_info();

    // std::cout << std::scientific << std::setprecision(1) << std::endl;
    // rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
    // std::cout << std::defaultfloat;

    std::cout << "\n******\n" << "maxniter of NOs in rdmft is: " << PARAM.inp.maxniter_orb << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n\n\n" << std::endl;


    // // just test
    // TK* pwfc_dft = &( this->psi->operator()(0, 0, 0) );
    // TK* pwfc_rdmft = &(this->rdmft_solver.wfc(0, 0, 0));
    // for(int i=0; i<this->psi->size(); ++i) { pwfc_dft[i] = pwfc_rdmft[i]; }
    // this->pelec->wg = this->rdmft_solver.wg;

    // this->psi_tmp = new psi::Psi<TK>;
    // this->psi_tmp->resize(this->rdmft_solver.nk_total, this->pv.ncol_bands, this->pv.nrow);
    // *(this->psi_tmp) = this->rdmft_solver.wfc;
    // this->wg_tmp = this->rdmft_solver.wg;

    // ModuleESolver::ESolver_KS<TK>::runner(ucell, istep);

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
void ESolver_RDMFT<TK, TR>::get_start_guess(UnitCell& ucell, const int istep)
{
    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        // ensure that exx calculation can be started
        // a better way is to force the start of exx even if pbe does not converge when esolver_type == "rdmft"?
        this->maxniter = std::max(200, PARAM.inp.scf_nmax);
        // the command to stop the runner is in the exx_iter_finish() function of Exx_LRI_interface.hpp
        ModuleESolver::ESolver_KS<TK>::runner(ucell, istep);

        if( this->niter == this->maxniter )
        {
            std::cout << "\n******\n" << "dft-PBE calculation does not converge and cannot provide initial values with exx calculation" << "\n******\n" << std::endl;
            assert(0);
        }
    }
    else
    {
        this->maxniter = 1;
        ModuleESolver::ESolver_KS<TK>::runner(ucell, istep);
    }
    this->rdmft_solver.inital_wfc_occNum(this->pelec->wg, this->psi);

    // // test
    // GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    // std::cout << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    // this->rdmft_solver.cal_Energy();

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
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
    else if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        // get start guess occ_number and optimize once
        this->opti_occ_num(this->dft_optimize, true);
        
        std::cout << "\n******\n" << "get inital value in occ_num !!!!!!" << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.rdmft_orb_opti == "idmft" )
    {
        if( PARAM.inp.mixing_rdmft )
        {
            rdmft::cal_special_DM(&this->pv, this->rdmft_solver.wg, this->rdmft_solver.wfc, this->DM);
            this->idmft.restart_opti(this->DM, this->p_hamilt);
        }
    }
    else if( PARAM.inp.rdmft_orb_opti == "ft_rdmft" )
    {
        if( PARAM.inp.mixing_rdmft )
        {
            rdmft::cal_special_DM(&this->pv, this->rdmft_solver.wg, this->rdmft_solver.wfc, this->DM);
            this->ft_rdmft.restart_opti(this->DM, this->p_hamilt);
        }
    }
    else if( PARAM.inp.rdmft_orb_opti == "bfgs" || PARAM.inp.rdmft_orb_opti == "cg" )
    {
        // get start guess occ_number and optimize once
        this->opti_occ_num(this->dft_optimize, true);
        
        std::cout << "\n******\n" << "get inital value in occ_num !!!!!!" << "\n******\n" << std::endl;

        // optimize occ_number
        this->opti_occ_num(this->dft_optimize);

        this->ls_opti_orb.get_start_guess();
    }


}







template <typename TK, typename TR>
double ESolver_RDMFT<TK, TR>::update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver_in)
{
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

        for(int ib=0; ib<ekb_rdmft[ik].size(); ++ib) this->pelec->ekb(ik, ib) = ekb_rdmft[ik][ib];
    }

    elecstate::calEBand(this->pelec->ekb,this->pelec->wg,this->pelec->f_en);
    elecstate::calculate_weights(this->pelec->ekb,
                                    this->pelec->wg,
                                    this->pelec->klist,
                                    this->pelec->eferm,
                                    this->pelec->f_en,
                                    this->pelec->nelec_spin,
                                    this->pelec->skip_weights);

    ModuleBase::matrix occ_number_ks = (this->pelec->wg);
    for(int ik=0; ik < occ_number_ks.nr; ++ik)
    {
        for(int inb=0; inb < occ_number_ks.nc; ++inb)
        {
            occ_number_ks(ik, inb) /= kv_.wk[ik];
        }
    }
    rdmft_solver_in.update_elec(&occ_number_ks);

    return 0.0;

}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::print_info()
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
    else if( PARAM.inp.rdmft_orb_opti == "ft_rdmft" )
    {
        std::cout << "\n" << "kappa = " << this->ft_rdmft.kappa << std::endl;
        for(int ik=0; ik < rdmft_solver.nk_total; ++ik)
        {
            std::cout << "\n\nik: " << ik << std::endl; // << std::fixed << std::setprecision(10);
            std::cout << "---------------------------------------\nnbands      " << "occ_number      " << "energy level(Rydberg)" << std::endl; 
            for(int ib=0; ib < rdmft_solver.nbands_total; ++ib)
            {
                std::cout << ib << "           " << rdmft_solver.occ_number(ik, ib) << "        " << this->ft_rdmft.get_energy_level()[ik][ib] << std::endl;
            }
            std::cout << "---------------------------------------\n" << std::endl;
        }
    }
    else
    {
        for(int ik=0; ik < rdmft_solver.nk_total; ++ik)
        {
            std::cout << "\n\nik: " << ik << std::endl; // << std::fixed << std::setprecision(10);
            std::cout << "---------------------------------------\nnbands      " << "occ_number      " << std::endl; 
            for(int ib=0; ib < rdmft_solver.nbands_total; ++ib)
            {
                std::cout << ib << "           " << rdmft_solver.occ_number(ik, ib) << "        " << std::endl;
            }
            std::cout << "---------------------------------------\n" << std::endl;
        }
    }

    // temp
    // rearrange the ONs of each k point from large to small
    std::cout << "\n******\nrearrange the ONs of each k point from large to small\n******\n" << std::endl;
    ModuleBase::matrix temp_num = this->rdmft_solver.occ_number;
    for(int ik=0; ik < rdmft_solver.nk_total; ++ik)
    {
        int num_bands = this->rdmft_solver.nbands_total;
        std::sort(&temp_num(ik, 0), &temp_num(ik, 0) + num_bands, std::greater<>());

        std::cout << "\n\nik: " << ik << std::endl; // << std::fixed << std::setprecision(10);
        std::cout << "---------------------------------------\nnbands      " << "occ_number      " << std::endl; 
        for(int ib=0; ib < num_bands; ++ib)
        {
            std::cout << ib << "           " << temp_num(ik, ib) << "        " << std::endl;
        }
        std::cout << "---------------------------------------\n" << std::endl;
    }

    std::cout << "\n******\n" << std::fixed << std::setprecision(10);
    std::cout << "Etotal_rdmft: " << this->rdmft_solver.Etotal
                << "\n\nE(TV + Hartree + XC) by RDMFT:   " << this->rdmft_solver.E_RDMFT[3]
                // << "\n\ndiff_E: " << diff_etotal
                // << "\ndiff_DM_max: " << this->idmft.get_diff_DM_max()
                << "\n******\n" << std::endl << std::defaultfloat;

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

