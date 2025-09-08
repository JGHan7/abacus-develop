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
    this->classname = "ESolver_RDMFT_Torch";
}


template <typename TK, typename TR>
ESolver_RDMFT_Torch_AD<TK, TR>::~ESolver_RDMFT_Torch_AD()
{
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::init_opti_param()
{
    this->nbands64 = PARAM.inp.nbands;
    this->nbasis64 = this->pv.get_wfc_global_nbasis();

    // ONs
    this->var_x.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    this->dE_dx.resize(this->nk_total*PARAM.inp.nbands, 0.0);
    rdmft::vector2tensor(this->var_x, this->var_x_tensor, { static_cast<int>(this->var_x.size()) });
    this->var_x_tensor.mutable_grad() = torch::zeros_like(this->var_x_tensor);





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
        thetaR_real[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>());
        thetaR_imag[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>());
        this->R_tensor[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        // rotation_R = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        
        wfc_new_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        wfc_0_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        dE_dwfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
    }
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::select_optimizer()
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
            this->thetaR_real[ik].set_requires_grad(true);
            param_R[ik] = { this->thetaR_real[ik] };
        }
        else
        {
            this->thetaR_real[ik].set_requires_grad(true);
            this->thetaR_imag[ik].set_requires_grad(true);
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


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::couple_opti()
{
    
}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::decouple_opti()
{
    
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

    double E = this->cal_Etotal( nullptr, &this->R_tensor[ik], 0, 1, ik );

    if( cal_grad )
    {
        this->R_optimizer[ik]->zero_grad();
    }


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
        this->wfc_new_tensor[ik] = torch::matmul( torch::matrix_exp(this->R_tensor[ik]), this->wfc_0_tensor[ik] );
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
void ESolver_RDMFT_Torch_AD<TK, TR>::cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik)
{
    
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch_AD<TK, TR>::check_hermi_lambda()
{
    
}









    

template class ESolver_RDMFT_Torch_AD<double, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, std::complex<double>>;

}



