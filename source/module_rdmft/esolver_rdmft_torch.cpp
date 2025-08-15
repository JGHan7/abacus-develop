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

    // temp
    if( PARAM.inp.rdmft_orb_opti == "iter_diag" || PARAM.inp.rdmft_orb_opti == "adam" )
    {
        this->iter_diag_orb.init(this->rdmft_solver.nk_total, this->kv.get_nkstot_full(), this->rdmft_solver.para_Eij, this->pv, &this->rdmft_solver);
        this->ls_opti_occ_num.init(this->kv, &this->rdmft_solver);
    }
    this->iter_diag_ethr = PARAM.inp.iter_diag_ethr;
    this->occ_num_thr = PARAM.inp.occ_num_thr;
    this->lambda_thr = PARAM.inp.lambda_thr;

    // convergence parameters
    this->dft_optimize = PARAM.inp.dft_opti;  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    this->conver_initial_value = PARAM.inp.conv_inital_value; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    this->nk_total = this->rdmft_solver.nk_total;
    this->para_Fij = &this->rdmft_solver.para_Eij;

    this->var_x.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    this->dE_dx.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    this->R_vec_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    this->dE_dR_global.resize( this->nk_total, std::vector<TK>(PARAM.inp.nbands * PARAM.inp.nbands, 0.0) );
    this->occ_number_new.create(this->rdmft_solver.nk_total, PARAM.inp.nbands);
    this->wfc_new.resize(this->rdmft_solver.nk_total, this->pv.ncol_bands, this->pv.nrow);
    this->wfc_new.zero_out();
    this->wfc_old = this->wfc_new;
    this->R_tensor.resize(this->nk_total);
    // if( !PARAM.inp.gamma_only )
    {
        this->R_tensor_real.resize(this->nk_total);
        this->R_tensor_imag.resize(this->nk_total);
    }
    this->dE_dR_tensor.resize(this->nk_total);
    this->R_optimizer.resize(this->nk_total);

    this->ebi.init(this->rdmft_solver.nk_total, this->kv.get_nkstot_full(), this->kv.wk);

    // test torch
    auto torchTest = torch::rand({4, 4});
    std::cout << "torchTest in ESolver_RDMFT_Torch:\n" << torchTest << std::endl;


    // for ONs
    if( PARAM.inp.occ_num_opti == "adam" )
    {
        this->x_need_ls = false;
        this->x_options_adam.lr(PARAM.inp.adam_learn_rate);
        // this->x_options_adam.betas(std::make_tuple(PARAM.inp.adam_beta1, PARAM.inp.adam_beta2));
        // this->x_options_adam.amsgrad(true);
    }
    else if( PARAM.inp.occ_num_opti == "lbfgs" )
    {
        this->x_need_ls = true;
        this->x_options_lbfgs.lr(1.0);
        this->x_options_lbfgs.max_iter(1);
        // this->R_options_lbfgs.tolerance_grad(1e-6);
        // this->R_options_lbfgs..line_search_fn("strong_wolfe");
    }

    // for NOs
    if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        this->R_need_ls = false;
        this->R_options_adam.lr(PARAM.inp.adam_learn_rate);
        this->R_options_adam.betas(std::make_tuple(PARAM.inp.adam_beta1, PARAM.inp.adam_beta2));
        this->R_options_adam.amsgrad(true);

        // test
        this->grad.resize(nk_total);
        this->moment_m.resize(nk_total);
        this->moment_v.resize(nk_total);
        this->vhat_max.resize(nk_total);
        for(int ik=0; ik<nk_total; ++ik)
        {
            this->grad[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->moment_m[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->moment_v[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->vhat_max[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        }

        this->m_hat.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->v_hat.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        
    }
    else if( PARAM.inp.rdmft_orb_opti == "lbfgs" )
    {
        this->R_need_ls = true;
        this->R_options_lbfgs.lr(1.0);
        this->R_options_lbfgs.max_iter(1);
        // this->R_options_lbfgs.tolerance_grad(1e-6);
        // this->R_options_lbfgs..line_search_fn("strong_wolfe");
    }





}



template <typename TK, typename TR>
void ESolver_RDMFT_Torch<TK, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);
    this->rdmft_solver.update_ion(ucell, *(this->pw_rho), this->locpp.vloc, this->sf.strucFac);

    this->get_start_guess(ucell, istep);

    // test torch
    auto torchTest = torch::rand({6, 6});
    std::cout << "torchTest:\n" << torchTest << std::endl;

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
    std::vector< std::vector<torch::Tensor> > param_R(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        if( PARAM.inp.gamma_only ) // GlobalC::exx_info.info_ri.real_number
        {
            param_R[ik] = { this->R_tensor[ik] };
        }
        else
        {
            // this->R_tensor_real[ik] = torch::empty(this->R_tensor[ik].sizes(), torch::dtype(torch::kDouble).requires_grad(true));
            // this->R_tensor_imag[ik] = torch::empty(this->R_tensor[ik].sizes(), torch::dtype(torch::kDouble).requires_grad(true));

            rdmft::split_complex_tensor(this->R_tensor[ik], this->R_tensor_real[ik], this->R_tensor_imag[ik]);
            param_R[ik] = { this->R_tensor_real[ik], this->R_tensor_imag[ik] };

            // test
            // param_R[ik] = { this->R_tensor_real[ik] };
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

    // // refactor together with the start_guess() function to reduce unnecessary calculations
    // this->trial_Ex_Egrad();
    // for(int ik=0; ik<this->nk_total; ++ik)
    // {
    //     this->trial_ER_Egrad(ik);
    // }



    this->trial_Ex_Egrad();
    int init_maxniter = 30;
    if(this->dft_optimize)
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

    init_maxniter = std::min(init_maxniter, PARAM.inp.maxniter_orb);
    double Etotal_old;

    // // test
    // ModuleBase::matrix occ_num_old;
    // // double Etotal_old;
    // for(int iter_occ_num=1; iter_occ_num <= PARAM.inp.maxniter_occ_num; ++iter_occ_num)
    // {
    //     // optimize natural occupation numbers
    //     // diff_occ_num_max = this->opti_occ_num(this->dft_optimize);
    //     occ_num_old = this->ebi.get_occ_number();
    //     Etotal_old = this->rdmft_solver.Etotal;

    //     double E_new = this->optimize_x();

    //     ModuleBase::matrix occ_num_new = this->ebi.get_occ_number();
    //     double diff_occ_num_max = 0.0;
    //     for(int i=0; i<occ_num_new.nr * occ_num_new.nc; ++i)
    //     {
    //         diff_occ_num_max = std::max( diff_occ_num_max, std::abs(occ_num_new.c[i] - occ_num_old.c[i]) );
    //     }

    //     std::cout << "\n******\nniter_occ_number of rdmft: " << iter_occ_num  << "\ndiff_occ_num_max(*num_symm_k): " << diff_occ_num_max << std::endl << std::fixed << std::setprecision(7);
    //     rdmft::printMatrix_pointer(this->nk_total, this->rdmft_solver.nbands_total, this->rdmft_solver.occ_number.c, "occ_number", 10);

    //     std::cout << "by Torch:\nEtotal_rdmft: " << this->rdmft_solver.Etotal
    //                 << "\n\ndiff_E: " << E_new - Etotal_old
    //                 << "\ndiff_DM_max: " << this->iter_diag_orb.get_diff_DM_max()
    //                 << "\n******" << std::endl << std::defaultfloat;

    //     if( diff_occ_num_max < this->occ_num_thr * 1e-2 ) // || (!this->dft_optimize && this->ls_opti_occ_num.diff_rate_max < 0.01)
    //     {
    //         tot_occ_num_iter += iter_occ_num;
    //         // occ_num_conv = true;
    //         break;
    //     }
    // }




    for(int ik=0; ik<this->nk_total; ++ik)
    {
        this->trial_ER_Egrad(ik).item().toDouble();
    }

    // test
    for(int iter_orb=1; iter_orb <= init_maxniter; ++iter_orb)
    {
        // delete or save?
        if(this->dft_optimize)
        {
            this->update_occ_num_dft(this->rdmft_solver);
        }
        Etotal_old = this->rdmft_solver.Etotal;

        double E_new = this->optimize_R();

        std::cout << "\n******\nniter_orb of rdmft: " << iter_orb << std::endl << std::fixed << std::setprecision(10);
        std::cout << "by Torch:\nEtotal_rdmft: " << E_new // this->rdmft_solver.Etotal
                    << "\n\ndiff_E: " << E_new - Etotal_old
                    // << "\ndiff_DM_max: " << this->iter_diag_orb.get_diff_DM_max()
                    << "\n******" << std::endl << std::defaultfloat;

        diff_E = E_new - Etotal_old;
        if( std::abs(diff_E) < 1e-8 ) { break; }

    }








    this->print_info();

    // temp
    // rearrange the ONs of each k point from large to small
    std::cout << "\n******\nrearrange the ONs of each k point from large to small\n******\n" << std::endl;
    ModuleBase::matrix temp_num = this->rdmft_solver.occ_number;
    for(int ik=0; ik < this->nk_total; ++ik)
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
    TK* pwfc_in = &( this->psi->operator()(0, 0, 0) );
    TK* pwfc = &this->wfc_new(0, 0, 0);
    for(int i=0; i<this->wfc_new.size(); ++i) { pwfc[i] = pwfc_in[i]; }

    // this->wfc_old = this->wfc_new;
    TK* pwfc_new = &this->wfc_new(0, 0, 0);
    TK* pwfc_old = &this->wfc_old(0, 0, 0);
    for(int i=0; i<this->wfc_new.size(); ++i) { pwfc_old[i] = pwfc_new[i]; }

    // test
    GlobalV::ofs_running << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    std::cout << "\n******\n" << "test: cal once rdmft after get inital values" << "\n******\n" << std::endl;
    this->rdmft_solver.cal_Energy();

    // initialize R
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        std::fill(this->R_vec_global[ik].begin(), this->R_vec_global[ik].end(), 0.0);
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
    this->var_x_tensor.mutable_grad() = torch::zeros_like(this->var_x_tensor);

    this->cal_Etotal(&var_x_tensor, nullptr, 1, 0);
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch<TK, TR>::optimize_x()
{
    double Etotal = 0.0;
    if( this->x_need_ls )
    {
        Etotal = this->var_x_optimizer->step( [this]() { return this->trial_Ex_Egrad(); } ).item().toDouble();
    }
    else
    {
        // the optimizer does not have a built-in (not needed) line search, so use this function to perform an update
        // Etotal = this->trial_Ex_Egrad().item().toDouble();
        this->var_x_optimizer->step();
        Etotal = this->trial_Ex_Egrad().item().toDouble();
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
    }
    else
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            // // Etotal = this->trial_ER_Egrad(ik).item().toDouble();
            // this->R_optimizer[ik]->step();
            // Etotal = this->trial_ER_Egrad(ik).item().toDouble();
        }

        std::vector<TK> R_vec_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
        std::vector<TK> exp_R_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

        std::cout << "\n******\ntest adam which writed by myself\n******" << std::endl;

        // test adam which writed by myself
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            std::fill(R_vec_local.begin(), R_vec_local.end(), 0.0);
            std::fill(exp_R_local.begin(), exp_R_local.end(), 0.0);

            std::fill(this->grad[ik].begin(), this->grad[ik].end(), 0.0);


            // this->rdmft_solver.cal_antisym_lambda(ik, this->grad[ik], 2.0);
            this->trial_ER_Egrad(ik).item().toDouble();
            rdmft::distribute_vec(this->para_Fij, this->dE_dR_global[ik], this->grad[ik]);



            for(int i=0; i<this->grad[ik].size(); ++i)
            {
                this->moment_m[ik][i] = PARAM.inp.adam_beta1 * this->moment_m[ik][i] + (1.0 - PARAM.inp.adam_beta1) * this->grad[ik][i];
                this->moment_v[ik][i] = PARAM.inp.adam_beta2 * this->moment_v[ik][i] + (1.0 - PARAM.inp.adam_beta2) * std::norm(this->grad[ik][i]);
                this->m_hat[i] = this->moment_m[ik][i] / (1.0 - PARAM.inp.adam_beta1);
                this->v_hat[i] = this->moment_v[ik][i] / (1.0 - PARAM.inp.adam_beta2);
                this->vhat_max[ik][i] = std::max(this->vhat_max[ik][i], this->v_hat[i]);
                // this->vhat_max[ik][i] = this->v_hat[i];
                R_vec_local[i] = - PARAM.inp.adam_learn_rate * this->m_hat[i] / std::sqrt( this->vhat_max[ik][i] + 1e-16 );
            }

            // rdmft::exp_skew_hermi_mat(this->para_Fij, R_vec_local, exp_R_local);

            // rdmft::pTgemm_scalapack( this->para_Fij, &(this->wfc_old(ik, 0, 0)), exp_R_local.data(),
            //                 &(this->wfc_new(ik, 0, 0)), PARAM.inp.nbands, PARAM.inp.nbands, PARAM.inp.nbands, 'N', 'N' );

            torch::Tensor test_R;
            rdmft::collect_vec(this->para_Fij, R_vec_local, this->R_vec_global[ik]);
            rdmft::vector2tensor(this->R_vec_global[ik], test_R, { PARAM.inp.nbands * PARAM.inp.nbands } );
            Etotal = this->cal_Etotal( nullptr, &test_R, 0, 1, ik );


        }

        this->rdmft_solver.update_elec( nullptr, &(this->wfc_new) );
        TK* pwfc_new = &this->wfc_new(0, 0, 0);
        TK* pwfc_old = &this->wfc_old(0, 0, 0);
        for(int i=0; i<this->wfc_new.size(); ++i) { pwfc_old[i] = pwfc_new[i]; }
        Etotal = this->rdmft_solver.cal_Energy();

    }

    this->wfc_old = this->wfc_new;

    return Etotal;
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch<TK, TR>::trial_Ex_Egrad()
{
    // this->var_x_tensor.grad().zero_();
    this->var_x_optimizer->zero_grad();

    double E = this->cal_Etotal( &this->var_x_tensor, nullptr, 1, 0 );

    this->cal_dE_dx( dE_dx_tensor );

    this->var_x_tensor.mutable_grad() = this->dE_dx_tensor;

    return torch::tensor(E, torch::dtype(torch::kDouble).requires_grad(true));
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch<TK, TR>::trial_ER_Egrad(const int ik)
{
    // if( PARAM.inp.gamma_only )
    // {
    //     // this->R_tensor[ik].grad().zero_();
    // }
    // else
    // {
    //     // this->R_tensor_real[ik].grad().zero_();
    //     // this->R_tensor_imag[ik].grad().zero_();
    // }
    this->R_optimizer[ik]->zero_grad();

    if( !PARAM.inp.gamma_only )
    {
        rdmft::merge_complex_tensor(this->R_tensor_real[ik], this->R_tensor_imag[ik], this->R_tensor[ik]);
    }

    // in the future, the orbital contribution to energy in cal_Etotal() can be modified to distinguish k-points
    double E = this->cal_Etotal( nullptr, &this->R_tensor[ik], 0, 1, ik );

    // in the future, it can be modified to distinguish k points
    this->cal_dE_dR( this->dE_dR_tensor[ik], ik );

    if( PARAM.inp.gamma_only )
    {
        this->R_tensor[ik].mutable_grad() = this->dE_dR_tensor[ik];
    }
    else
    {
        // if (!this->R_tensor_real[ik].grad().defined())
        //     this->R_tensor_real[ik].mutable_grad() = torch::zeros_like(this->R_tensor_real[ik]);
        // if (!this->R_tensor_imag[ik].grad().defined())
        //     this->R_tensor_imag[ik].mutable_grad() = torch::zeros_like(this->R_tensor_imag[ik]);
        // this->R_tensor_real[ik].mutable_grad().copy_( 2 * torch::real(this->dE_dR_tensor[ik]) );
        // this->R_tensor_imag[ik].mutable_grad().copy_( -2 * torch::imag(this->dE_dR_tensor[ik]) );

        this->R_tensor_real[ik].mutable_grad() =  2 * torch::real(this->dE_dR_tensor[ik]);
        this->R_tensor_imag[ik].mutable_grad() =  -2 * torch::imag(this->dE_dR_tensor[ik]);

        // std::cout << "dE_dR_tensor_real:\n" << this->R_tensor_real[ik].mutable_grad() << std::endl;
        // std::cout << "dE_dR_tensor_imag:\n" << this->R_tensor_imag[ik].mutable_grad() << std::endl;

    }

    return torch::tensor(E, torch::dtype(torch::kDouble).requires_grad(true));
    // return torch::tensor(E, torch::dtype(torch::kDouble));
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

            // get C_t+1 = C_t * exp(R)
            rdmft::pTgemm_scalapack( this->para_Fij, &(this->rdmft_solver.wfc(ik, 0, 0)), exp_R_local.data(),
                            &(this->wfc_new(ik, 0, 0)), PARAM.inp.nbands, PARAM.inp.nbands, PARAM.inp.nbands, 'N', 'N' );

            // rdmft::pTgemm_scalapack( this->para_Fij, &(this->wfc_old(ik, 0, 0)), exp_R_local.data(),
            //                 &(this->wfc_new(ik, 0, 0)), PARAM.inp.nbands, PARAM.inp.nbands, PARAM.inp.nbands, 'N', 'N' );
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
        this->rdmft_solver.cal_antisym_lambda(ik, dE_dR_local, 2.0); // re-derive the formula to determine the factor !!!!!!!!!!!!!!!!!!!!!!

        // convert data formats
        rdmft::collect_vec(this->para_Fij, dE_dR_local, dE_dR_global[ik]);
        // rdmft::vector2tensor(dE_dR_global[ik], dE_dR_tensor[ik], { PARAM.inp.nbands * PARAM.inp.nbands });
        rdmft::vector2tensor(dE_dR_global[ik], dE_dR_tensor_ik, { PARAM.inp.nbands * PARAM.inp.nbands });

        // test !!!!!!!!!!!!!!!!!!!!!!!!
        // dE_dR_tensor_ik= torch::zeros_like(dE_dR_tensor_ik);
    }

    this->has_cal_E_occ_num = false;
}












template class ESolver_RDMFT_Torch<double, double>;
template class ESolver_RDMFT_Torch<std::complex<double>, double>;
template class ESolver_RDMFT_Torch<std::complex<double>, std::complex<double>>;

}

