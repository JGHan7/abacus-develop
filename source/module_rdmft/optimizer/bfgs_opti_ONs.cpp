//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================


#include "module_rdmft/optimizer/bfgs_opti_ONs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"


namespace rdmft
{


template<typename TX>
BFGS_ONs<TX>::BFGS_ONs()
{

}


template<typename TX>
BFGS_ONs<TX>::~BFGS_ONs()
{
    
}


template<typename TX>
void BFGS_ONs<TX>::init()
{

}





// template class BFGS_ONs<double, double>;
// template class BFGS_ONs<std::complex<double>, double>;
// template class BFGS_ONs<std::complex<double>, std::complex<double>>;

template class BFGS_ONs<double>;

}


