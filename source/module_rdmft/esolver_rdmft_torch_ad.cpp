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
    ;
}



























    

template class ESolver_RDMFT_Torch_AD<double, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, double>;
template class ESolver_RDMFT_Torch_AD<std::complex<double>, std::complex<double>>;

}



