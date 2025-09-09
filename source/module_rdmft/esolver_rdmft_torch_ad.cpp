//==========================================================
// Author: Jingang Han
// DATE : 2025-09-07
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft_torch_ad.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/optimizer_tools.h" // temporary
// #include <cmath> // temporary
// #include "module_elecstate/elecstate_tools.h" // temporary



namespace rdmft
{


template <typename TK, typename TR>
ESolver_RDMFT_Torch_AD<TK, TR>::ESolver_RDMFT_Torch_AD()
{
    this->classname = "ESolver_RDMFT_Torch_AD";
}


template <typename TK, typename TR>
ESolver_RDMFT_Torch_AD<TK, TR>::~ESolver_RDMFT_Torch_AD()
{
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::init_opti_param()
{
    std::cout << "\n***\n" << "Enter RDMFT with automatic differentiation provided by libTorch" << "\n***\n" << std::endl;
    this->nbands64 = PARAM.inp.nbands;
    this->nbasis64 = this->pv.get_wfc_global_nbasis();

    // ONs
    this->var_x.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    this->dE_dx.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    rdmft::vector2tensor(this->var_x, this->var_x_tensor, { static_cast<int>(this->var_x.size()) });
    this->var_x_tensor.mutable_grad() = torch::zeros_like(this->var_x_tensor);

    // NOs
    this->opti_deltaR = false;
    this->R_optimizer.resize(this->nk_total);
    thetaR_real.resize(this->nk_total);
    thetaR_imag.resize(this->nk_total);
    this->R_tensor.resize(this->nk_total);

    wfc_new_tensor.resize(this->nk_total);
    wfc_0_tensor.resize(this->nk_total);
    dE_dwfc_tensor.resize(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        thetaR_real[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        thetaR_imag[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        this->R_tensor[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        
        wfc_new_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        wfc_0_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        dE_dwfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::select_optimizer()
{

    if( PARAM.inp.rdmft_one_opti )
    {
        std::vector<torch::Tensor> param = {this->var_x_tensor};
        param.insert( param.end(), this->thetaR_real.begin(), this->thetaR_real.end() );
        if( !PARAM.inp.gamma_only )
        {
            param.insert( param.end(), this->thetaR_imag.begin(), this->thetaR_imag.end() );
        }

        if( PARAM.inp.rdmft_orb_opti == "adam" )
        {
            this->one_optimizer = std::make_unique<torch::optim::Adam>(param, this->R_options_adam);
        }
        else if( PARAM.inp.rdmft_orb_opti == "lbfgs" )
        {
            this->one_optimizer = std::make_unique<torch::optim::LBFGS>(param, this->R_options_lbfgs);
        }
    }
    else
    {
        // for ONs
        // before specifying the optimizer, param must be initialized to a certain extent
        // this->var_x_tensor.set_requires_grad(true);
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
                // this->thetaR_real[ik].set_requires_grad(true);
                param_R[ik] = { this->thetaR_real[ik] };
            }
            else
            {
                // this->thetaR_real[ik].set_requires_grad(true);
                // this->thetaR_imag[ik].set_requires_grad(true);
                param_R[ik] = { this->thetaR_real[ik], this->thetaR_imag[ik] };
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

}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::get_start_guess(UnitCell& ucell, const int istep)
{
    rdmft::ESolver_RDMFT_Torch<TK, TR>::get_start_guess(ucell, istep);

    // get wfc0 by tensor type
    std::vector<std::vector<TK>> wfc_vec(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        rdmft::psi2vec(ik, this->pv, *(this->psi), wfc_vec[ik]);
        rdmft::vector2tensor( wfc_vec[ik], this->wfc_0_tensor[ik], {nbands64, nbasis64} );
    }
}


// template <typename TK, typename TR>
// void ESolver_RDMFT_Torch_AD<TK, TR>::couple_opti()
// {
    
// }


// template <typename TK, typename TR>
// void ESolver_RDMFT_Torch_AD<TK, TR>::decouple_opti()
// {
    
// }


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::one_opti()
{
    double Etotal_old = 0.0;
    double diff_E = 0.0;

    for(int iter=1; iter<=PARAM.inp.scf_nmax; ++iter)
    {
        double E_new = this->optimize_all();
        diff_E = E_new - Etotal_old;
        Etotal_old = E_new;

        bool conv = this->converge();

        std::cout << "\n******\nniter of rdmft: " << iter 
                    << std::fixed << std::setprecision(10);
        std::cout << "\n\nEtotal_rdmft by opti ONs&NOs: " << E_new
                    << "\ndiff_E: " << diff_E
                    << "\ndiff_occ_num_max: " << this->diff_occ_num_max
                    << "\ndiff_DM_max: " << this->diff_DM_max
                    << "\n\nmax_off_diag_F: " << this->max_off_diag_Fock
                    << std::endl;

        if( iter%10 == 1 )
        {
            rdmft::printMatrix_pointer(this->nk_total, this->rdmft_solver.nbands_total, this->rdmft_solver.occ_number.c, "occ_number", 10);
        }
        std::cout << "******" << std::endl << std::defaultfloat;

        if ( conv && std::abs(diff_E) < 1e-5 )
        {
            break;
        }
    }

    std::cout << "\n***\nthetaR_real(k=0): \n" << this->thetaR_real[0] << "\n***\n" << std::endl;
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch_AD<TK, TR>::optimize_all()
{
    double Etotal = 0.0;
    if( this->R_need_ls )
    {
        Etotal = this->one_optimizer->step( [this]() { return this->trial_E_Egrad_all(true); } ).item().toDouble();

        // because the step size corresponding to the last call of the trial_E_Egrad() function by torch is not necessarily the final step size
        // torch's line search will continue to look for a more suitable step size after first finding one that satisfies the strong Wolfe condition.
        Etotal = this->trial_E_Egrad_all(false).item().toDouble();
    }
    else
    {
        // the optimizer does not have a built-in (not needed) line search, so use this function to perform an update
        Etotal = this->trial_E_Egrad_all().item().toDouble();
        this->one_optimizer->step();
        // Etotal = this->trial_Ex_Egrad().item().toDouble();
    }

    return Etotal;
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch_AD<TK, TR>::trial_E_Egrad_all(bool cal_grad)
{
    // convert data
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        this->update_R_tensor(ik);
    }

    double Etotal = this->cal_Etotal( &this->var_x_tensor, &this->R_tensor, 1, 1 );
    torch::Tensor loss_Etotal = torch::tensor(Etotal, torch::dtype(torch::kDouble).requires_grad(false));

    if( cal_grad )
    {
        this->one_optimizer->zero_grad();

        this->cal_dE_dx( this->dE_dx_tensor );
        this->var_x_tensor.mutable_grad() = this->dE_dx_tensor;

        this->cal_dE_dR_all();
    }

    return loss_Etotal;
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::update_R_tensor(const int ik)
{
    // generate upper triangle index
    auto idx = torch::triu_indices(this->nbands64, this->nbands64);

    // although it is a temporary variable, the release time is managed by torch and does not destroy the calculation graph
    torch::Tensor R_upper = torch::zeros({this->nbands64, this->nbands64}, torch_dtype<TK>());

    // fill thetaR_real and thetaR_imag with upper triangular indices
    if( PARAM.inp.gamma_only )
    {
        R_upper = R_upper.index_put({idx[0], idx[1]}, this->thetaR_real[ik]);
    }
    else
    {
        // torch::Tensor theta_complex = torch::complex(this->thetaR_real, this->thetaR_imag);
        // R_upper = R_upper.index_put({idx[0], idx[1]}, theta_complex);
        R_upper = R_upper.index_put( {idx[0], idx[1]}, torch::complex(this->thetaR_real[ik],  this->thetaR_imag[ik]) );
    }

    // constructing a skew-Hermitian matrix R_skew = R_upper - R_upper^\dagger
    this->R_tensor[ik] = R_upper - R_upper.transpose(0,1).conj();
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch_AD<TK, TR>::trial_ER_Egrad(const int ik, bool cal_grad)
{
    this->update_R_tensor(ik);

    double Ek = this->cal_Etotal( nullptr, &this->R_tensor[ik], 0, 1, ik );
    torch::Tensor loss_Ek = torch::tensor(Ek, torch::dtype(torch::kDouble).requires_grad(false));

    if( cal_grad )
    {
        this->R_optimizer[ik]->zero_grad();
        torch::Tensor temp;
        this->cal_dE_dR( temp, ik );
    }

    return loss_Ek;
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch_AD<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
                                                    const torch::Tensor* R_tensor_ik,
                                                    bool cal_by_occ_num,
                                                    bool cal_by_orb,
                                                    const int ik)
{
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
        // this->wfc_new_tensor[ik] = torch::matmul( torch::matrix_exp(this->R_tensor[ik]), this->wfc_0_tensor[ik] );
        this->wfc_new_tensor[ik] = torch::matmul( torch::matrix_exp(*R_tensor_ik), this->wfc_0_tensor[ik] );
        std::vector<TK> wfc_vec;
        rdmft::tensor2vector( this->wfc_new_tensor[ik], wfc_vec );
        rdmft::vec2psi(ik, this->pv, wfc_vec, this->wfc_new);
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
double ESolver_RDMFT_Torch_AD<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
                                                    const std::vector<torch::Tensor>* R_tensor,
                                                    bool cal_by_occ_num,
                                                    bool cal_by_orb)
{
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
    if( R_tensor != nullptr )
    {   
        std::vector<TK> wfc_vec;
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            this->wfc_new_tensor[ik] = torch::matmul( torch::matrix_exp( (*R_tensor)[ik] ), this->wfc_0_tensor[ik] );
            rdmft::tensor2vector( this->wfc_new_tensor[ik], wfc_vec );
            rdmft::vec2psi(ik, this->pv, wfc_vec, this->wfc_new);
        }
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
void ESolver_RDMFT_Torch_AD<TK, TR>::cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik)
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

    // obtain accurate dE/dR using analytical dE/dwfc and automatic differentiation, and convert the format
    psi::Psi<TK> dE_dwfc;
    std::vector<TK> dE_dwfc_vec;
    this->rdmft_solver.cal_E_grad_wfc(dE_dwfc);
    rdmft::psi2vec( ik, this->pv, dE_dwfc, dE_dwfc_vec );
    this->dE_dwfc_tensor[ik] = torch::from_blob(dE_dwfc_vec.data(), {nbands64, nbasis64}, torch_dtype<TK>()).clone();

    // auto diff
    this->wfc_new_tensor[ik].backward( this->dE_dwfc_tensor[ik] );

    this->has_cal_E_occ_num = false;
}


// void cal_dE_dR_all(const std::vector<torch::Tensor>* R_tensor = nullptr);
template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::cal_dE_dR_all(const std::vector<torch::Tensor>* R_tensor)
{
    if( !this->has_cal_E_wfc )
    {
        if( R_tensor != nullptr )
        {
            this->cal_Etotal(nullptr, R_tensor, 0, 1);
        }
        else
        {
            std::cout << "\n***\n" << "cal_Etotal() was not performed before cal_dE_dR(), and no new iteration point was provided" << "\n***\n" << std::endl;
            assert(0);
        }
    }

    // obtain accurate dE/dR using analytical dE/dwfc and automatic differentiation, and convert the format
    psi::Psi<TK> dE_dwfc;
    this->rdmft_solver.cal_E_grad_wfc(dE_dwfc);

    std::vector<TK> dE_dwfc_vec;
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        rdmft::psi2vec( ik, this->pv, dE_dwfc, dE_dwfc_vec );
        this->dE_dwfc_tensor[ik] = torch::from_blob(dE_dwfc_vec.data(), {nbands64, nbasis64}, torch_dtype<TK>()).clone();

        // auto diff
        this->wfc_new_tensor[ik].backward( this->dE_dwfc_tensor[ik] );
    }

    this->has_cal_E_occ_num = false;
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch_AD<TK, TR>::check_hermi_lambda()
{
    double max_off_diag_Fock = 0.0;
    std::vector<TK> Fock_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);

    for(int ik=0; ik<this->nk_total; ++ik)
    {
        this->rdmft_solver.cal_antisym_lambda(ik, Fock_local, this->grad_factor);
        for(int i=0; i<Fock_local.size(); ++i)
        {
            double norm_Fij = std::abs( Fock_local[i] );
            max_off_diag_Fock = std::max( max_off_diag_Fock, norm_Fij );
        }
    }

    rdmft::reduce_all_max(max_off_diag_Fock);

    return max_off_diag_Fock;
}





    

template class ESolver_RDMFT_Torch_AD<double, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, std::complex<double>>;

}



