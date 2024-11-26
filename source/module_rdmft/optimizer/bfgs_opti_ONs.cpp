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
void BFGS_ONs<TX>::init(int nk_total_in, int nbands_in)
{
    this->nk_total = nk_total_in;
    this->nbands = nbands_in;

    x0.resize(nk_total*nbands);
    x1.resize(nk_total*nbands);
    diff_x.resize(nk_total*nbands);
    dE_dx0.resize(nk_total*nbands);
    dE_dx1.resize(nk_total*nbands);
    diff_gradient.resize(nk_total*nbands);
    Hk.resize( nk_total*nbands * nk_total*nbands );
}


template<typename TX>
void BFGS_ONs<TX>::optimize(std::vector<TX>& x_new)
{

}


template<typename TX>
void BFGS_ONs<TX>::get_start_guess(std::vector<TX>& x0_in)
{
    x0 = x0_in;

    // identity matrix or other method to initialize H0
    for(int i=0; i<nk_total*nbands; ++i) { Hk[i*(nk_total*nbands) + i] = 1.0; }
}






// template class BFGS_ONs<double, double>;
// template class BFGS_ONs<std::complex<double>, double>;
// template class BFGS_ONs<std::complex<double>, std::complex<double>>;

template class BFGS_ONs<double>;

}


