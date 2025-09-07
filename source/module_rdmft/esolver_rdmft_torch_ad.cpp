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
    this->opti_deltaR = false;

    this->nbands64 = PARAM.inp.nbands;
    this->nbasis64 = this->pv.get_wfc_global_nbasis();

    for(int ik=0; ik<this->nk_total; ++ik)
    {
        thetaR_real[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<double>());
        wfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
    }

}


template <typename TK, typename TR>
void ESolver_RDMFT_Torch_AD<TK, TR>::select_optimizer()
{

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
void ESolver_RDMFT_Torch_AD<TK, TR>::cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik)
{
    
}


template <typename TK, typename TR>
torch::Tensor ESolver_RDMFT_Torch_AD<TK, TR>::trial_ER_Egrad(const int ik, bool cal_grad)
{
    
}


template <typename TK, typename TR>
double ESolver_RDMFT_Torch_AD<TK, TR>::cal_Etotal(const torch::Tensor* var_x_tensor,
                                                    const torch::Tensor* R_tensor_ik,
                                                    bool cal_by_occ_num,
                                                    bool cal_by_orb,
                                                    const int ik)
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



