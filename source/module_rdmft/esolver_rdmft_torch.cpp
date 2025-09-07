//==========================================================
// Author: Jingang Han
// DATE : 2025-07-16
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft_torch.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/optimizer_tools.h" // temporary
// #include <cmath> // temporary
// #include "module_elecstate/elecstate_tools.h" // temporary



namespace rdmft
{

template <typename TK, typename TR>
ESolver_RDMFT_Torch<TK, TR>::ESolver_RDMFT_Torch()
{
    this->classname = "ESolver_RDMFT_Torch";
}


template <typename TK, typename TR>
ESolver_RDMFT_Torch<TK, TR>::~ESolver_RDMFT_Torch()
{
    // delete this->var_x_optimizer;
    // delete this->R_optimizer;
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_all_runners(ucell, inp);
    // ModuleESolver::ESolver_RDMFT<TK, TR>::before_all_runners(ucell, inp);

    // test torch
    auto torchTest = torch::rand({4, 4});
    std::cout << "\ntorchTest in ESolver_RDMFT_Torch:\n" << torchTest << "\n" << std::endl;

    this->rdmft_solver.init(this->GG,
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

    this->ebi.init(this->rdmft_solver.nk_total, this->kv.get_nkstot_full(), this->kv.wk);

    // convergence parameters
    this->iter_diag_ethr = PARAM.inp.iter_diag_ethr;
    this->occ_num_thr = PARAM.inp.occ_num_thr;
    this->lambda_thr = PARAM.inp.lambda_thr;
    this->dft_optimize = PARAM.inp.dft_opti;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->conver_initial_value = PARAM.inp.conv_inital_value; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    this->nk_total = this->rdmft_solver.nk_total;
    this->para_Fij = &this->rdmft_solver.para_Eij;
    this->DM.resize(this->nk_total, std::vector<TK>(this->pv.nloc, 0.0));
    this->occ_number_new.create(this->rdmft_solver.nk_total, PARAM.inp.nbands, true);
    this->occ_number_old.create(this->rdmft_solver.nk_total, PARAM.inp.nbands, true);
    this->wfc_new.resize(this->rdmft_solver.nk_total, this->pv.ncol_bands, this->pv.nrow);
    this->wfc_new.zero_out();
    this->wfc_old = this->wfc_new;
    // this->wfc_record = this->wfc_new;

    this->cal_molecular = ( PARAM.inp.gamma_only || this->nk_total == 1 || ( this->nk_total == 2 && PARAM.inp.nspin == 2 ) );
    this->opti_deltaR = true;

    this->init_opti_param();

    this->set_opti_options();

    this->select_optimizer();
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::init_opti_param()
{
    // ONs
    this->var_x.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    this->dE_dx.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    rdmft::vector2tensor(this->var_x, this->var_x_tensor, { static_cast<int>(this->var_x.size()) });
    this->var_x_tensor.mutable_grad() = torch::zeros_like(this->var_x_tensor);

    // NOs
    this->R_vec_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    this->dE_dR_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    this->R_tensor.resize(this->nk_total);
    if( !PARAM.inp.gamma_only )
    {
        this->R_tensor_real.resize(this->nk_total);
        this->R_tensor_imag.resize(this->nk_total);
    }
    this->dE_dR_tensor.resize(this->nk_total);
    this->R_optimizer.resize(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        // std::fill(this->R_vec_global[ik].begin(), this->R_vec_global[ik].end(), 0.0);
        rdmft::vector2tensor( this->R_vec_global[ik], this->R_tensor[ik], { PARAM.inp.nbands * PARAM.inp.nbands } );
        
        // malloc grad
        if( PARAM.inp.gamma_only )
        {
            this->R_tensor[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
        }
        else
        {
            this->R_tensor_real[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
            this->R_tensor_imag[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
        }
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::set_opti_options()
{
    // for ONs
    if( PARAM.inp.occ_num_opti == "adam" )
    {
        this->x_need_ls = false;
        this->x_options_adam.lr(PARAM.inp.adam_lr_occ_num);
        // this->x_options_adam.betas(std::make_tuple(PARAM.inp.adam_beta1, PARAM.inp.adam_beta2));
        // this->x_options_adam.amsgrad(true);
    }
    else if( PARAM.inp.occ_num_opti == "lbfgs" )
    {
        this->x_need_ls = true;
        this->x_options_lbfgs.lr(1.0);
        this->x_options_lbfgs.max_iter(1);
        this->x_options_lbfgs.line_search_fn("strong_wolfe");
        // this->x_options_lbfgs.tolerance_grad(1e-8);
    }

    // for NOs
    if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        this->R_need_ls = false;
        this->R_options_adam.lr(PARAM.inp.adam_learn_rate);
        this->R_options_adam.betas(std::make_tuple(PARAM.inp.adam_beta1, PARAM.inp.adam_beta2));
        this->R_options_adam.amsgrad(true);
    }
    else if( PARAM.inp.rdmft_orb_opti == "lbfgs" )
    {
        this->R_need_ls = true;
        this->R_options_lbfgs.lr(1.0);
        this->R_options_lbfgs.max_iter(1);
        this->R_options_lbfgs.line_search_fn("strong_wolfe");
        // this->R_options_lbfgs.tolerance_grad(1e-8);
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);
    this->rdmft_solver.update_ion(ucell, *(this->pw_rho), this->locpp.vloc, this->sf.strucFac);

    this->get_start_guess(ucell, istep);

    // this->select_optimizer();

    this->do_optimize();

    this->print_info();

    // if( !PARAM.inp.rdmft_couple_opti )
    // {
    //     std::cout << "\n******\nexternal iter = " << tot_exteral_iter 
    //                 << "\ntotal orbital iter = " << tot_orb_iter 
    //                 << "\ntotal occ_num iter = " << tot_occ_num_iter
    //                 << "\n******\n" << std::endl;
    // }

}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::select_optimizer()
{
    // for ONs
    // before specifying the optimizer, param must be initialized to a certain extent
    this->var_x_tensor.set_requires_grad(true);
    std::vector<torch::Tensor> param_x = {this->var_x_tensor};
    if( PARAM.inp.occ_num_opti == "adam" )
    {
        this->var_x_optimizer = std::make_unique<torch::optim::Adam>(param_x, this->x_options_adam);
    }
    else if( PARAM.inp.occ_num_opti == "lbfgs" )
    {
        this->var_x_optimizer = std::make_unique<torch::optim::LBFGS>(param_x, this->x_options_lbfgs);
    }

    // for NOs
    // this->R_optimizer.resize(this->nk_total);
    std::vector< std::vector<torch::Tensor> > param_R(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        if( PARAM.inp.gamma_only ) // GlobalC::exx_info.info_ri.real_number
        {
            this->R_tensor[ik].set_requires_grad(true);
            param_R[ik] = { this->R_tensor[ik] };
        }
        else
        {
            // this->R_tensor_real[ik] = torch::empty(this->R_tensor[ik].sizes(), torch::dtype(torch::kDouble).requires_grad(true));
            // this->R_tensor_imag[ik] = torch::empty(this->R_tensor[ik].sizes(), torch::dtype(torch::kDouble).requires_grad(true));
            rdmft::split_complex_tensor(this->R_tensor[ik], this->R_tensor_real[ik], this->R_tensor_imag[ik]);

            this->R_tensor_real[ik].set_requires_grad(true);
            this->R_tensor_imag[ik].set_requires_grad(true);
            param_R[ik] = { this->R_tensor_real[ik], this->R_tensor_imag[ik] };
        }

        if( PARAM.inp.rdmft_orb_opti == "adam" )
        {
            this->R_optimizer[ik] = std::make_unique<torch::optim::Adam>(param_R[ik], this->R_options_adam);
        }
        else if( PARAM.inp.rdmft_orb_opti == "lbfgs" )
        {
            this->R_optimizer[ik] = std::make_unique<torch::optim::LBFGS>(param_R[ik], this->R_options_lbfgs);
        }
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::before_opti()
{
    this->var_x_optimizer->state().clear();
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        this->R_optimizer[ik]->state().clear();
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::do_optimize()
{
    if( PARAM.inp.rdmft_couple_opti )
    {
        this->couple_opti();
    }
    else
    {
        this->decouple_opti();
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::couple_opti()
{
    double Etotal_old = 0.0;
    // ModuleBase::matrix occ_num_old(this->nk_total, PARAM.inp.nbands);
    double diff_E1 = 0.0;
    double diff_E2 = 0.0;
    // double final_diff_E = 0.0;
    // double diff_occ_num_max = 0.0;

    if( PARAM.inp.occ_num_opti == "adam" && PARAM.inp.rdmft_orb_opti == "adam" ) // ( !this->x_need_ls && !this->R_need_ls )
    {
        for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
        {
            // get Etotal and update the gradient
            // after finding a suitable learning rate, x and R can be passed together when updating the energy to reduce the amount of calculation
            double E_new1 = this->trial_Ex_Egrad().item().toDouble();
            double E_new2 = 0.0;
            for(int ik=0; ik<this->nk_total; ++ik)
            {
                E_new2 = this->trial_ER_Egrad(ik).item().toDouble();
            }

            diff_E1 = E_new1 - Etotal_old;
            diff_E2 = E_new2 - E_new1;
            Etotal_old = E_new2;

            bool conv = this->converge();

            std::cout << "\n******\nniter of rdmft: " << iter 
                        << std::fixed << std::setprecision(10);
            std::cout << "\n\nEtotal_rdmft by opti ONs: " << E_new1
                        << "\ndiff_E: " << diff_E1 
                        << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
                        << "\n\nEtotal_rdmft by opti NOs: " << E_new2
                        << "\ndiff_E: " << diff_E2 
                        << "\ndiff_DM_max: " << this->diff_DM_max 
                        << "\n\nmax_off_diag_F: " << this->max_off_diag_Fock 
                        << std::endl;

            if( iter%10 == 1 )
            {
                rdmft::printMatrix_pointer(this->nk_total, this->rdmft_solver.nbands_total, this->rdmft_solver.occ_number.c, "occ_number", 10);
            }
            std::cout << "******" << std::endl << std::defaultfloat;

            if ( conv && std::abs(diff_E1 + diff_E2) < 1e-5 )
            {
                break;
            }

            // optimize (update) parameters x, R[ik]
            this->var_x_optimizer->step();
            for(int ik=0; ik<this->nk_total; ++ik)
            {
                this->R_optimizer[ik]->step();
            }

            if( this->opti_deltaR )
            {
                this->wfc_old = this->wfc_new; //this->rdmft_solver.wfc;
            }

        }

    }
    else if( PARAM.inp.occ_num_opti == "lbfgs" && PARAM.inp.rdmft_orb_opti == "lbfgs" )
    {
        for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
        {
            // record the x obtained from the previous optimization
            torch::Tensor x_old = this->var_x_tensor.clone();

            double E_new1 = this->optimize_x();

            // in each iteration, x and R are optimized "simultaneously", that is, the latest ONs(x) are not used when optimizing NOs(R)
            this->cal_Etotal( &x_old, nullptr, 1, 0 );

            double E_new2 = this->optimize_R();

            // "optimize x"
            E_new1 = this->cal_Etotal( &this->var_x_tensor, nullptr, 1, 0 );
            diff_E1 = E_new1 - E_new2;
            diff_E2 = E_new2 - Etotal_old;
            Etotal_old = E_new1;

            // diff_E1 = E_new1 - Etotal_old;
            // diff_E2 = E_new2 - E_new1;
            // Etotal_old = E_new2;

            bool conv = this->converge();

            std::cout << "\n******\nniter of rdmft: " << iter 
                        << std::fixed << std::setprecision(10);
            std::cout << "\n\nEtotal_rdmft by opti ONs: " << E_new1
                        << "\ndiff_E: " << diff_E1 
                        << "\ndiff_occ_num_max: " << this->diff_occ_num_max
                        << "\n\nEtotal_rdmft by opti NOs: " << E_new2
                        << "\ndiff_E: " << diff_E2
                        << "\ndiff_DM_max: " << this->diff_DM_max
                        << "\n\nmax_off_diag_F: " << this->max_off_diag_Fock
                        // << "\n\ndiff_wfc_norm: " << this->diff_wfc_norm
                        // << "\n\ndiff_wfc_max: " << this->diff_wfc_max
                        << std::endl;

            if( iter%10 == 1 )
            {
                rdmft::printMatrix_pointer(this->nk_total, this->rdmft_solver.nbands_total, this->rdmft_solver.occ_number.c, "occ_number", 10);
            }
            std::cout << "******" << std::endl << std::defaultfloat;

            if ( conv && std::abs(diff_E1 + diff_E2) < 1e-5 )
            {
                break;
            }
        }
    }
    else
    {
        std::cout << "\n******\n" << "coupled optimization for optimizers requiring line search is not yet implemented" << "\n******\n" << std::endl;
        assert(0);
    }

    // std::cout << "\n******\n" << "maxniter of NOs in rdmft is: " << PARAM.inp.maxniter_orb << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n\n\n" << std::endl;
 
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::decouple_opti()
{
    int init_maxniter = 10;
    if(this->dft_optimize)
    {
        init_maxniter = PARAM.inp.maxniter_orb;
    }
    // double final_diff_E = 1.0;
    double diff_occ_num_max = 0.0;
    double E_new = this->rdmft_solver.Etotal;
    double diff_E = E_new;
    double Etotal_old = 0.0;

    int tot_orb_iter = 0;
    int tot_occ_num_iter = 0;
    int tot_exteral_iter = 0;

    for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
    {
        tot_exteral_iter = iter; // temp
        bool orb_conv = false;
        bool occ_number_conv = false;

        // need more testing and thinking, has the landscape been changed ? ? ?
        this->before_opti();

        ModuleBase::matrix occ_num_old(this->nk_total, PARAM.inp.nbands);
        for(int iter_occ_num=1; iter_occ_num <= PARAM.inp.maxniter_occ_num; ++iter_occ_num)
        {
            if( this->dft_optimize )
            {
                break;
            }

            // temp
            torch::Tensor x_old = this->var_x_tensor.clone();

            // only replace the value, other properties remain unchanged
            // this->var_x_tensor.copy_(x_old);

            E_new = this->optimize_x();
            diff_E = E_new - Etotal_old;
            Etotal_old = E_new;

            occ_number_conv = this->occ_num_conv();

            std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num  << "\ndiff_occ_num_max: " << this->diff_occ_num_max << std::endl;
            std::cout << std::fixed << std::setprecision(10) 
                        << "Etotal_rdmft: " << E_new
                        << "\ndiff_E: " << diff_E;
                        // << "\ndiff_DM_max: " << this->iter_diag_orb.get_diff_DM_max()
            rdmft::printMatrix_pointer(this->nk_total, this->rdmft_solver.nbands_total, this->rdmft_solver.occ_number.c, "occ_number", 10);
            // std::cout << x_old << std::endl;
            std::cout << "\n******" << std::endl << std::defaultfloat;

            if( occ_number_conv )
            {
                tot_occ_num_iter += iter_occ_num;
                // occ_number_conv = true;
                break;
            }

            // prevent NOs in rdmft from being too different from NOs corresponding to var_x when convergence is not achieved
            if( iter_occ_num == PARAM.inp.maxniter_occ_num )
            {
                this->cal_Etotal(&this->var_x_tensor, nullptr, 1, 0);
            }
        }
        double diff_E_tot = diff_E;

        if( occ_number_conv == true )
        {
            init_maxniter += 10;
        }
        else
        {
            tot_occ_num_iter += PARAM.inp.maxniter_occ_num;
        }

        double E_temp = Etotal_old;
        init_maxniter = std::min(init_maxniter, PARAM.inp.maxniter_orb);
        for(int iter_orb=1; iter_orb <= init_maxniter; ++iter_orb)
        {
            if(this->dft_optimize)
            {
                this->update_occ_num_dft(this->rdmft_solver);
            }

            double E_new = this->optimize_R();
            diff_E = E_new - Etotal_old;
            Etotal_old = E_new;

            orb_conv = this->dm_conv();

            std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
            std::cout << "by Torch:\nEtotal_rdmft: " << E_new
                        << "\n\ndiff_E: " << diff_E
                        << "\ndiff_DM_max: " << this->diff_DM_max
                        << "\n******" << std::endl << std::defaultfloat;
            
            if( orb_conv ) // if( orb_conv || std::abs(diff_E) < 1e-8 )
            {
                tot_orb_iter += iter_orb;
                break;
            }
        }
        diff_E_tot += diff_E;

        if( !orb_conv )
        {
            tot_orb_iter += init_maxniter;
        }

        if( PARAM.inp.rdmft_orb_opti == "adam" && (E_new - E_temp > -1e-6) )
        {
            for(int ik=0; ik<this->nk_total; ++ik)
            {
                for (auto& group : R_optimizer[ik]->param_groups() )
                {
                    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
                    options.lr( options.lr() * PARAM.inp.adam_scaling_lr );
                }
            }
            init_maxniter += 10;
        }

        if( this->dft_optimize )
        {
            break;
        }

        if( this->converge( &occ_number_conv, &orb_conv ) && std::abs( diff_E_tot ) < 1e-5 )
        {
            break;
        }
    }

    std::cout << "\n******\nexternal iter = " << tot_exteral_iter 
                << "\ntotal orbital iter = " << tot_orb_iter 
                << "\ntotal occ_num iter = " << tot_occ_num_iter
                << "\n******\n" << std::endl;

}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::get_start_guess(UnitCell& ucell, const int istep)
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

    // must have
    this->wfc_new = *(this->psi);
    this->wfc_old = this->wfc_new;

    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    this->rdmft_solver.cal_Energy();

    // initialize R
    // for(int ik=0; ik<this->nk_total; ++ik)
    // {
    //     std::fill(this->R_vec_global[ik].begin(), this->R_vec_global[ik].end(), 0.0);
    //     rdmft::vector2tensor( this->R_vec_global[ik], this->R_tensor[ik], { PARAM.inp.nbands * PARAM.inp.nbands } );
        
    //     // malloc grad
    //     if( PARAM.inp.gamma_only )
    //     {
    //         this->R_tensor[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
    //     }
    //     else
    //     {
    //         this->R_tensor_real[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
    //         this->R_tensor_imag[ik].mutable_grad() = torch::zeros_like(this->R_tensor[ik]);
    //     }
    // }

    // initialize var_x
    std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    if(PARAM.inp.random_occ_num)
    {
        this->ebi.get_inital_guess(this->var_x);
    }
    else
    {
        this->ebi.get_inital_guess(this->var_x, &this->rdmft_solver.occ_number);
    }
    rdmft::vector2tensor(this->var_x, this->var_x_tensor, { static_cast<int>(this->var_x.size()) });
    // this->var_x_tensor.mutable_grad() = torch::zeros_like(this->var_x_tensor);

    this->cal_Etotal(&this->var_x_tensor, nullptr, 1, 0);
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch<TK, TR>::optimize_x()
{
    double Etotal = 0.0;
    if( this->x_need_ls )
    {
        Etotal = this->var_x_optimizer->step( [this]() { return this->trial_Ex_Egrad(true); } ).item().toDouble();

        // because the step size corresponding to the last call of the trial_E_Egrad() function by torch is not necessarily the final step size
        // torch's line search will continue to look for a more suitable step size after first finding one that satisfies the strong Wolfe condition.
        Etotal = this->trial_Ex_Egrad(false).item().toDouble();
        // Etotal = this->trial_Ex_Egrad(true).item().toDouble();
    }
    else
    {
        // the optimizer does not have a built-in (not needed) line search, so use this function to perform an update
        Etotal = this->trial_Ex_Egrad().item().toDouble();
        this->var_x_optimizer->step();
        // Etotal = this->trial_Ex_Egrad().item().toDouble();
    }

    return Etotal;
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch<TK, TR>::optimize_R()
{
    double Etotal = 0.0;

    if( this->R_need_ls )
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            Etotal = this->R_optimizer[ik]->step( [this, ik]() { return this->trial_ER_Egrad(ik); } ).item().toDouble();
        }

        // because the step size corresponding to the last call of the trial_E_Egrad() function by torch is not necessarily the final step size
        // torch's line search will continue to look for a more suitable step size after first finding one that satisfies the strong Wolfe condition.
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            Etotal = this->trial_ER_Egrad(ik, false).item().toDouble();
        }
    }
    else
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            Etotal = this->trial_ER_Egrad(ik).item().toDouble();
            this->R_optimizer[ik]->step();
            // Etotal = this->trial_ER_Egrad(ik).item().toDouble();
        }
    }

    if( this->opti_deltaR )
    {
        this->wfc_old = this->wfc_new; //this->rdmft_solver.wfc;
    }

    return Etotal;
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch<TK, TR>::trial_Ex_Egrad(bool cal_grad)
{
    // this->var_x_optimizer->zero_grad();

    double E = this->cal_Etotal( &this->var_x_tensor, nullptr, 1, 0 );

    if( cal_grad )
    {
        this->var_x_optimizer->zero_grad();
        this->cal_dE_dx( this->dE_dx_tensor );
        // this->var_x_optimizer->zero_grad();
        this->var_x_tensor.mutable_grad() = this->dE_dx_tensor;
    }

    return torch::tensor(E, torch::dtype(torch::kDouble).requires_grad(true));
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch<TK, TR>::trial_ER_Egrad(const int ik, bool cal_grad)
{


    if( !PARAM.inp.gamma_only )
    {
        rdmft::merge_complex_tensor(this->R_tensor_real[ik], this->R_tensor_imag[ik], this->R_tensor[ik]);
    }

    // in the future, the orbital contribution to energy in cal_Etotal() can be modified to distinguish k-points
    double E = this->cal_Etotal( nullptr, &this->R_tensor[ik], 0, 1, ik );

    if( this->opti_deltaR )
    {
        if( PARAM.inp.gamma_only )
        {
            this->R_tensor[ik].data().zero_();
        }
        else
        {
            this->R_tensor_real[ik].data().zero_();
            this->R_tensor_imag[ik].data().zero_();
        }
    }

    if( cal_grad )
    {
        this->R_optimizer[ik]->zero_grad();
        // in the future, it can be modified to distinguish k points
        this->cal_dE_dR( this->dE_dR_tensor[ik], ik );
        if( PARAM.inp.gamma_only )
        {
            this->R_tensor[ik].mutable_grad() = this->dE_dR_tensor[ik];
        }
        else
        {
            // optimizers such as Adam will continue to accumulate gradient errors,
            // causing the imaginary part that is theoretically 0 to gradually become non-zero ?
            if( this->cal_molecular )
            {
                this->R_tensor_real[ik].mutable_grad() =  1.0 * torch::real(this->dE_dR_tensor[ik]);
                this->R_tensor_imag[ik].mutable_grad() =  0.0 * torch::imag(this->dE_dR_tensor[ik]);
            }
            else
            {
                this->R_tensor_real[ik].mutable_grad() =  2.0 * torch::real(this->dE_dR_tensor[ik]);
                this->R_tensor_imag[ik].mutable_grad() =  -2.0 * torch::imag(this->dE_dR_tensor[ik]);
            }
            // std::cout << "dE_dR_tensor_real:\n" << this->R_tensor_real[ik].mutable_grad() << std::endl;
            // std::cout << "dE_dR_tensor_imag:\n" << this->R_tensor_imag[ik].mutable_grad() << std::endl;
        }
    }


    return torch::tensor(E, torch::dtype(torch::kDouble).requires_grad(true));
}



// template <typename TK, typename TR>
// double ESolver_RDMFT_Torch<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
//                                             const std::vector<torch::Tensor>* R_tensor,
//                                             bool cal_by_occ_num,
//                                             bool cal_by_orb)
template <typename TK, typename TR>
double ESolver_RDMFT_Torch<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
                                            const torch::Tensor* R_tensor_ik,
                                            bool cal_by_occ_num,
                                            bool cal_by_orb,
                                            const int ik)
{
    // std::vector< std::vector<TK> > R_vec_global(this->nk_total);

    if( var_x_tensor != nullptr )
    {
        this->occ_number_new.zero_out(); 
        // convert data
        rdmft::tensor2vector(*var_x_tensor, this->var_x);
        this->ebi.update_x_occ_num(this->var_x);

        // get new ONs
        this->occ_number_new = this->ebi.get_occ_number();
    }

    // if( R_tensor != nullptr )
    if( R_tensor_ik != nullptr )
    {
        // When wfc is optimized k points at a time, the following code cannot appear
        // that is, all historical data of wfc_new except the ik point must be retained
        // (currently only the ik point is optimized, and other points remain unchanged)
        // this->wfc_new.zero_out();
        
        std::vector<TK> R_vec_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
        std::vector<TK> exp_R_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

        // for(int ik=0; ik<this->nk_total;++ik)
        {
            // convert data formats
            // rdmft::tensor2vector( (*R_tensor)[ik], this->R_vec_global[ik]);
            rdmft::tensor2vector( *R_tensor_ik, this->R_vec_global[ik]);
            rdmft::distribute_vec(this->para_Fij, this->R_vec_global[ik], R_vec_local);

            // get exp_R i.e. rotation_mat
            rdmft::exp_skew_hermi_mat(this->para_Fij, R_vec_local, exp_R_local);

            // get C_t+1 = C_t * exp(R_t)
            rdmft::pTgemm_scalapack( this->para_Fij, &(this->wfc_old(ik, 0, 0)), exp_R_local.data(),
                            &(this->wfc_new(ik, 0, 0)), PARAM.inp.nbands, PARAM.inp.nbands, PARAM.inp.nbands, 'N', 'N' );
        }

        // std::cout << "R_tensor_ik:\n" << *R_tensor_ik << std::endl;
        // rdmft::printMatrix_pointer(PARAM.inp.nbands, PARAM.inp.nbands, exp_R_local.data(), "exp_R_loacl", 10);
    }

    double Etotal = 0.0;
    if( cal_by_occ_num && cal_by_orb )
    {
        this->rdmft_solver.update_elec( &(this->occ_number_new), &(this->wfc_new) );
        this->has_cal_E_occ_num = true;
        this->has_cal_E_wfc = true;
    }
    else if( cal_by_occ_num )
    {
        this->rdmft_solver.update_elec( &(this->occ_number_new) );
        this->has_cal_E_occ_num = true;
    }
    else if( cal_by_orb )
    {
        this->rdmft_solver.update_elec( nullptr, &(this->wfc_new) );
        this->has_cal_E_wfc = true;
    }
    else
    {
        return 0.0;
    }
    Etotal = this->rdmft_solver.cal_Energy();

    return Etotal;
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::cal_dE_dx(torch::Tensor& dE_dx_tensor, const torch::Tensor* var_x_tensor)
{
    if( !this->has_cal_E_occ_num )
    {
        if( var_x_tensor != nullptr )
        {
            this->cal_Etotal(var_x_tensor, nullptr, 1, 0);
        }
        else
        {
            std::cout << "\n***\n" << "cal_Etotal() was not performed before cal_dE_dx(), and no new iteration point was provided" << "\n***\n" << std::endl;
            assert(0);
        }
    }
    // get dE_dx
    this->rdmft_solver.cal_E_grad_occ_num();
    std::vector<double> dE_docc_num = this->rdmft_solver.get_dE_docc_num();
    this->ebi.get_dE_dx(dE_docc_num, this->dE_dx);

    // convert data formats
    rdmft::vector2tensor(this->dE_dx, dE_dx_tensor, { this->nk_total * PARAM.inp.nbands });

    this->has_cal_E_occ_num = false;
}


// template <typename TK, typename TR>
// void ESolver_RDMFT_Torch<TK, TR>::cal_dE_dR(std::vector<torch::Tensor>& dE_dR_tensor, const std::vector<torch::Tensor>* R_tensor)
template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik)
{
    if( !this->has_cal_E_wfc )
    {
        if( R_tensor_ik != nullptr )
        {
            this->cal_Etotal(nullptr, R_tensor_ik, 0, 1, ik);
        }
        else
        {
            std::cout << "\n***\n" << "cal_Etotal() was not performed before cal_dE_dR(), and no new iteration point was provided" << "\n***\n" << std::endl;
            assert(0);
        }
    }

    // antisymm lambda to get dE_dR_local
    std::vector<TK> dE_dR_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

    // for(int ik=0; ik<this->nk_total; ++ik)
    {
        // get dE_dR
        this->rdmft_solver.cal_antisym_lambda(ik, dE_dR_local, this->grad_factor); // re-derive the formula to determine the factor !!!!!!!!!!!!!!!!!!!!!!

        // convert data formats
        rdmft::collect_vec(this->para_Fij, dE_dR_local, dE_dR_global[ik]);
        // rdmft::vector2tensor(dE_dR_global[ik], dE_dR_tensor[ik], { PARAM.inp.nbands * PARAM.inp.nbands });
        rdmft::vector2tensor(dE_dR_global[ik], dE_dR_tensor_ik, { PARAM.inp.nbands * PARAM.inp.nbands });

        // test !!!!!!!!!!!!!!!!!!!!!!!!
        // dE_dR_tensor_ik= torch::zeros_like(dE_dR_tensor_ik);
    }

    this->has_cal_E_occ_num = false;
}


template <typename TK, typename TR>
bool ESolver_RDMFT_Torch<TK, TR>::converge(const bool* occ_num_conv_in, const bool* dm_conv_in)
{

    bool occ_num_conv = false;
    if(occ_num_conv_in == nullptr)
    {
        occ_num_conv = this->occ_num_conv();
    }
    else
    {
        occ_num_conv = *occ_num_conv_in;
    }

    bool dm_conv = false;
    if(dm_conv_in == nullptr)
    {
        dm_conv = this->dm_conv();
    }
    else
    {
        dm_conv = *dm_conv_in;
    }

    double max_off_diag_F = this->check_hermi_lambda();

    // bool orb_conv = this->orb_conv();

    this->max_off_diag_Fock = max_off_diag_F;

    // if( max_off_diag_F < this->lambda_thr )
    // {
    //     std::cout << "\n******\n\nConvergence!\n" 
    //                 << "\nmax_off_diag_F < lambda_thr: " << max_off_diag_F 
    //                 << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
    //                 << "\ndiff_DM_max: " << this->diff_DM_max 
    //                 << "\n******\n" << std::endl;
    //     return 1;
    // }
    // else if( dm_conv && occ_num_conv )
    // {
    //     std::cout << "\n******\n\nNOs(DM) and ONs Converge respectively!\n" 
    //                 << "\nmax_off_diag_F is " << max_off_diag_F 
    //                 << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
    //                 << "\ndiff_DM_max: " << this->diff_DM_max 
    //                 << "\n******\n" << std::endl;
    //     return 1;
    // }
    // else
    // {
    //     std::cout << "\n******\nstill optimize NOs and ONs, because max_off_diag_F > lambda_thr: " << max_off_diag_F << "\n******\n" << std::endl;
    // }

    if( max_off_diag_F < this->lambda_thr && dm_conv && occ_num_conv )
    {
        std::cout << "\n******\n\nConvergence with dm_conv!\n" 
                    << "\nmax_off_diag_F < lambda_thr: " << max_off_diag_F 
                    << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
                    << "\ndiff_DM_max: " << this->diff_DM_max 
                    // << "\n\ndiff_wfc_norm: " << this->diff_wfc_norm
                    << "\n******\n" << std::endl;
        return 1;
    }
    else if( !PARAM.inp.rdmft_dm_conv && max_off_diag_F < this->lambda_thr && occ_num_conv )
    {
        std::cout << "\n******\n\nConvergence with max_off_diag_F and Etotal!\n" 
                    << "\nmax_off_diag_F < lambda_thr: " << max_off_diag_F 
                    << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
                    << "\ndiff_DM_max: " << this->diff_DM_max 
                    // << "\n\ndiff_wfc_norm: " << this->diff_wfc_norm
                    << "\n******\n" << std::endl;
        return 1;
    }
    else
    {
        if( !PARAM.inp.rdmft_couple_opti )
        {
            std::cout << "\n******\n\nstill optimize NOs and ONs !!!!!!!!!!!!!!! \n" 
                        << "\nmax_off_diag_F is " << max_off_diag_F 
                        << "\ndiff_occ_num_max: " << this->diff_occ_num_max 
                        << "\ndiff_DM_max: " << this->diff_DM_max 
                        // << "\n\ndiff_wfc_norm: " << this->diff_wfc_norm
                        << "\n******\n" << std::endl;
        }
    }

    return 0;
}


// define it as follows or refer to the class iterDiag
template <typename TK, typename TR>
double ESolver_RDMFT_Torch<TK, TR>::check_hermi_lambda()
{
    double max_off_diag_Fock = 0.0;
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        for(int i=0; i<this->dE_dR_global[ik].size(); ++i)
        {
            double norm_Fij = std::abs( this->dE_dR_global[ik][i] );
            max_off_diag_Fock = std::max( max_off_diag_Fock, norm_Fij );
        }
    }

    return max_off_diag_Fock;
}


template <typename TK, typename TR>
bool ESolver_RDMFT_Torch<TK, TR>::dm_conv()
{
    // ModuleBase::matrix occ_num = this->ebi.get_occ_number();
    ModuleBase::matrix temp_wg(this->rdmft_solver.wg);
    std::vector< std::vector<TK> > DM_new(this->nk_total, std::vector<TK>(this->pv.nloc, 0.0));
    rdmft::cal_special_DM(this->para_Fij, temp_wg, this->wfc_new, DM_new);

    this->diff_DM_max = 0.0;
    for(int ik=0; ik<nk_total; ++ik)
    {
        for(int iloc=0; iloc<DM_new[ik].size(); ++iloc)
        {
            this->diff_DM_max = std::max( this->diff_DM_max, std::abs(this->DM[ik][iloc] - DM_new[ik][iloc]) );

            // update DM
            this->DM[ik][iloc] = DM_new[ik][iloc];
        }
    }
    rdmft::reduce_all_max(this->diff_DM_max);

    if( this->diff_DM_max < PARAM.inp.scf_thr )
    {
        return 1;
    }

    return 0;
}


// template <typename TK, typename TR>
// bool ESolver_RDMFT_Torch<TK, TR>::orb_conv()
// {
//     double sum_norm = 0.0;
//     this->diff_wfc_max = 0.0;

//     TK* pwfc_new = &( this->wfc_new(0, 0, 0) );
//     TK* pwfc_record = &( this->wfc_record(0, 0, 0) );
//     for(int i=0; i<this->wfc_new.size(); ++i)
//     {
//         sum_norm += std::norm( pwfc_new[i] - pwfc_record[i] );
//         this->diff_wfc_max = std::max( this->diff_wfc_max, std::abs(pwfc_new[i] - pwfc_record[i]) );
//         pwfc_record[i] = pwfc_new[i];
//     }
//     sum_norm = std::sqrt( sum_norm );
//     this->diff_wfc_norm = sum_norm;

//     // this->wfc_record = this->wfc_new;

//     if( sum_norm < PARAM.inp.scf_thr || this->diff_wfc_max < PARAM.inp.scf_thr )
//     {
//         return 1;
//     }

//     return 0;
// }


template <typename TK, typename TR>
bool ESolver_RDMFT_Torch<TK, TR>::occ_num_conv()
{
    // ModuleBase::matrix occ_num_new = this->ebi.get_occ_number();
    const int num = this->occ_number_new.nr * this->occ_number_new.nc;
    this->diff_occ_num_max = 0.0;
    for(int i=0; i<num; ++i)
    {
        this->diff_occ_num_max = std::max( this->diff_occ_num_max, std::abs(this->occ_number_new.c[i] - this->occ_number_old.c[i]) );
    }

    // update occ_num
    this->occ_number_old = this->occ_number_new;

    if( this->diff_occ_num_max < this->occ_num_thr )
    {
        return 1;
    }

    return 0;
}










template class ESolver_RDMFT_Torch<double, double>;
template class ESolver_RDMFT_Torch<std::complex<double>, double>;
template class ESolver_RDMFT_Torch<std::complex<double>, std::complex<double>>;

}

