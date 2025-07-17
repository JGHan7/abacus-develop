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
            PARAM.inp.rdmft_orb_opti == "bfgs" || PARAM.inp.opti_by_torch )
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

    this->DM.resize(rdmft_solver.nk_total, std::vector<TK>(this->pv.nloc, 0.0));

    // convergence parameters
    this->dft_optimize = PARAM.inp.dft_opti;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->conver_initial_value = PARAM.inp.conv_inital_value; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    this->iter_diag_ethr = PARAM.inp.iter_diag_ethr;
    this->occ_num_thr = PARAM.inp.occ_num_thr; // how much is proper?
    this->lambda_thr = PARAM.inp.lambda_thr;

    if( PARAM.inp.rdmft_orb_opti == "test" )
    {
        this->data_x.resize(this->dim_x, 0.0);
        this->df_dx.resize(this->dim_x, 0.0);
        this->bfgs_opti_x.init(1, this->dim_x);
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

    {
        if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
        {
            double diff_etotal = 1.0;
            double diff_occ_num_max = 1.0;

            for(int iter_occ_num=1; iter_occ_num <= PARAM.inp.maxniter_occ_num; ++iter_occ_num)
            {
                int small_diffE = 0;
                this->iter_diag_orb.before_opti(this->p_hamilt);

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

            for(int iter=1; iter<PARAM.inp.scf_nmax; ++iter)
            {
                tot_exteral_iter = iter; // temp
                bool orb_conv = false;
                bool occ_num_conv = false;

                init_maxniter = std::min(init_maxniter, PARAM.inp.maxniter_orb);
                this->iter_diag_orb.before_opti(this->p_hamilt);
                for(int iter_orb=1; iter_orb <= init_maxniter; ++iter_orb)
                {
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

                if(!this->dft_optimize) { this->ls_opti_occ_num.before_opti(); }
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
                tot_occ_num_iter += PARAM.inp.maxniter_occ_num;

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
                        << "\n******" << std::endl << std::defaultfloat;

            std::cout << "\n******\nexternal iter = " << tot_exteral_iter 
                        << "\ntotal orbital iter = " << tot_orb_iter 
                        << "\ntotal occ_num iter = " << tot_occ_num_iter
                        << "\n******\n" << std::endl;

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
            for(int i=0; i<PARAM.inp.scf_nmax; ++i)
            {
                f_value = this->fx(this->data_x);
                this->df_dx = this->grad_f(this->data_x);
                double norm_grad_x = this->norm_grad(this->df_dx);

                std::cout << "\n\n******\niter: " << i << "\nf(x):  " << f_value << "\nnorm_grad: " << norm_grad_x << std::endl;
                rdmft::printMatrix_pointer(1, this->data_x.size(), this->data_x.data(), "x");
                rdmft::printMatrix_pointer(1, this->data_x.size(), this->df_dx.data(), "df_dx");


                if( norm_grad_x < 1e-8 ) { break; }
                
                this->bfgs_opti_x.get_pk(this->df_dx, this->data_x, this->pk, first_time);
                rdmft::printMatrix_pointer(1, this->data_x.size(), this->pk.data(), "pk");
                std::cout << "\n******\n\n" << std::endl;

                for(int j=0; j<this->data_x.size(); ++j)
                {
                    this->data_x[j] += PARAM.inp.ls_fixed_step * this->pk[j];
                }

                first_time = false;
            }
        }
    }

    // this->print_info();

    // std::cout << std::scientific << std::setprecision(1) << std::endl;
    // rdmft::printMatrix_pointer(rdmft_solver.nk_total, rdmft_solver.nbands_total, rdmft_solver.occ_number.c, "occ_number", 10);
    // std::cout << std::defaultfloat;

    std::cout << "\n******\n" << "maxniter of NOs in rdmft is: " << PARAM.inp.maxniter_orb << "\n******\n" << std::endl;
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

    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    this->rdmft_solver.cal_Energy();


    // if( PARAM.inp.opti_by_torch )
    // {
    //     // std::fill(this->R_vec_global.begin(), this->R_vec_global.end(), 0.0);

    //     std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    //     if(PARAM.inp.random_occ_num)
    //     {
    //         this->ebi_torch.get_inital_guess(this->var_x);
    //     }
    //     else
    //     {
    //         this->ebi_torch.get_inital_guess(this->var_x, &this->rdmft_solver.occ_number);
    //     }

    //     torch::Tensor var_x_tensor;
    //     rdmft::vector2tensor(this->var_x, var_x_tensor, { static_cast<int>(this->var_x.size()) });

    //     this->cal_Etotal(&var_x_tensor, nullptr, 1, 0);

    //     // this->occ_number = this->param_occ_num->get_occ_number();
    //     // this->rdmft_solver->update_elec( &(this->occ_number) );
    //     // this->phi_0 = this->rdmft_solver->cal_Energy();
    // }
    // else
    {
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
                this->idmft.before_opti(this->DM, this->p_hamilt);
            }
        }
        else if( PARAM.inp.rdmft_orb_opti == "ft_rdmft" )
        {
            if( PARAM.inp.mixing_rdmft )
            {
                rdmft::cal_special_DM(&this->pv, this->rdmft_solver.wg, this->rdmft_solver.wfc, this->DM);
                this->ft_rdmft.before_opti(this->DM, this->p_hamilt);
            }
        }
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

    return 1.0;

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
}




// template <typename TK, typename TR>
// double ESolver_RDMFT<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
//                                             const std::vector<torch::Tensor>* R_tensor,
//                                             bool cal_by_occ_num,
//                                             bool cal_by_orb)
// {
//     // std::vector< std::vector<TK> > R_vec_global(this->nk_total);

//     if( var_x_tensor != nullptr )
//     {
//         this->occ_number_new.zero_out(); 
//         // convert data
//         rdmft::tensor2vector(*var_x_tensor, this->var_x);
//         this->ebi_torch.update_x_occ_num(this->var_x);

//         // get new ONs
//         this->occ_number_new = this->ebi_torch.get_occ_number();
//     }

//     if( R_tensor != nullptr )
//     {
//         this->wfc_new.zero_out();
//         std::vector<TK> R_vec_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
//         std::vector<TK> exp_R_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

//         for(int ik=0; ik<this->nk_total;++ik)
//         {
//             // convert data formats
//             rdmft::tensor2vector( (*R_tensor)[ik], this->R_vec_global[ik]);
//             rdmft::distribute_vec(this->para_Fij, this->R_vec_global[ik], R_vec_local);

//             // get exp_R i.e. rotation_mat
//             rdmft::exp_skew_hermi_mat(this->para_Fij, R_vec_local, exp_R_local);

//             // get C_t+1 = C_t * exp(R)
//             rdmft::pTgemm_scalapack( this->para_Fij, &(this->rdmft_solver.wfc(ik, 0, 0)), exp_R_local.data(),
//                             &(this->wfc_new(ik, 0, 0)), PARAM.inp.nbands, PARAM.inp.nbands, PARAM.inp.nbands, 'N', 'N' );
//         }
//     }

//     double Etotal = 0.0;
//     if( cal_by_occ_num && cal_by_orb )
//     {
//         this->rdmft_solver.update_elec( &(this->occ_number_new), &(this->wfc_new) );
//         this->has_cal_E_occ_num = true;
//         this->has_cal_E_wfc = true;
//     }
//     else if( cal_by_occ_num )
//     {
//         this->rdmft_solver.update_elec( &(this->occ_number_new) );
//         this->has_cal_E_occ_num = true;
//     }
//     else if( cal_by_orb )
//     {
//         this->rdmft_solver.update_elec( nullptr, &(this->wfc_new) );
//         this->has_cal_E_wfc = true;
//     }
//     else
//     {
//         return 0.0;
//     }

//     return Etotal;
// }


// template <typename TK, typename TR>
// void ESolver_RDMFT<TK, TR>::cal_dE_dx(torch::Tensor& dE_dx_tensor, const torch::Tensor* var_x_tensor)
// {
//     if( !this->has_cal_E_occ_num )
//     {
//         if( var_x_tensor != nullptr )
//         {
//             this->cal_Etotal(var_x_tensor, nullptr, 1, 0);
//         }
//         else
//         {
//             std::cout << "\n***\n" << "cal_Etotal() was not performed before cal_dE_dx(), and no new iteration point was provided" << "\n***\n" << std::endl;
//             assert(0);
//         }
//     }
//     // get dE_dx
//     this->rdmft_solver.cal_E_grad_occ_num();
//     std::vector<double> dE_docc_num = this->rdmft_solver.get_dE_docc_num();
//     this->ebi_torch.get_dE_dx(dE_docc_num, this->dE_dx);

//     // convert data formats
//     rdmft::vector2tensor(this->dE_dx, dE_dx_tensor, { this->nk_total * PARAM.inp.nbands });

//     this->has_cal_E_occ_num = false;
// }


// template <typename TK, typename TR>
// void ESolver_RDMFT<TK, TR>::cal_dE_dR(std::vector<torch::Tensor>& dE_dR_tensor, const std::vector<torch::Tensor>* R_tensor)
// {
//     if( !this->has_cal_E_wfc )
//     {
//         if( R_tensor != nullptr )
//         {
//             this->cal_Etotal(nullptr, R_tensor, 0, 1);
//         }
//         else
//         {
//             std::cout << "\n***\n" << "cal_Etotal() was not performed before cal_dE_dR(), and no new iteration point was provided" << "\n***\n" << std::endl;
//             assert(0);
//         }
//     }

//     // antisymm lambda to get dE_dR_local
//     std::vector<TK> dE_dR_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

//     for(int ik=0; ik<this->nk_total; ++ik)
//     {
//         // get dE_dR
//         this->rdmft_solver.cal_antisym_lambda(ik, dE_dR_local, 1.0); // re-derive the formula to determine the factor !!!!!!!!!!!!!!!!!!!!!!

//         // convert data formats
//         rdmft::collect_vec(this->para_Fij, dE_dR_local, dE_dR_global[ik]);
//         rdmft::vector2tensor(dE_dR_global[ik], dE_dR_tensor[ik], { PARAM.inp.nbands * PARAM.inp.nbands });
//     }

//     this->has_cal_E_occ_num = false;
// }






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

