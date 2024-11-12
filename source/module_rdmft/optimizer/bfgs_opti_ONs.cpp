//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================


#include "module_rdmft/optimizer/bfgs_opti_ONs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"


namespace rdmft
{


template<typename TK, typename TR>
BFGS_ONs<TK, TR>::BFGS_ONs()
{

}


template<typename TK, typename TR>
BFGS_ONs<TK, TR>::~BFGS_ONs()
{
    
}








template class BFGS_ONs<double, double>;
template class BFGS_ONs<std::complex<double>, double>;
template class BFGS_ONs<std::complex<double>, std::complex<double>>;


}


