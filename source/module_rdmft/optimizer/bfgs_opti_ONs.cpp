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
    diff_grad.resize(nk_total*nbands);
    Hk.resize( nk_total*nbands * nk_total*nbands );
    search_direction.resize(nk_total*nbands);

    rho_diffX_diffGrad.resize( nk_total*nbands * nk_total*nbands );
    rho_diffX_diffX_T.resize( nk_total*nbands * nk_total*nbands );
}


template<typename TX>
void BFGS_ONs<TX>::optimize(const std::vector<TX>& dE_dx_in, std::vector<TX>& x_new)
{

    // cal search_direction, p_k = - H_k+1 * dE_dx

    // line search, x_k+1 = x_k + alpha_k * p_k

    // diff_x, s_k = x_k+1 - x_k

    // diff_grad, yk = (dE_dx)_k+1 - (dE_dx)_k




    // cal rho
    rdmft::dgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &this->rho, 1, 1, nk_total*nbands, 'T', 'N');
    this->rho = 1.0/this->rho;

    // cal rho_diffX_diffGrad, I-rho_diffX_diffGrad
    rdmft::dgemm_lapack( this->diff_x.data(), this->diff_grad.data(), this->rho_diffX_diffGrad.data(), nk_total*nbands, nk_total*nbands, 1, 'N', 'T', -(this->rho) );
    for(int i=0; i<nk_total*nbands; ++i) { this->rho_diffX_diffGrad[i*nk_total*nbands + i] += 1.0; }

    // cal rho_diffX_diffX_T
    rdmft::dgemm_lapack( this->diff_x.data(), this->diff_x.data(), this->rho_diffX_diffX_T.data(), nk_total*nbands, nk_total*nbands, 1, 'N', 'T', this->rho );

    // cal H_k+1
    std::vector<TX> H_tmp = this->Hk;
    rdmft::dgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), H_tmp.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands );
    rdmft::dgemm_lapack( H_tmp.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands, 'N', 'T' );
    // rdmft::dgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands );
    // rdmft::dgemm_lapack( this->Hk.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands, 'N', 'T' );
    for(int i=0; i<this->Hk.size(); ++i) { this->Hk[i] += this->rho_diffX_diffX_T[i]; }

}


template<typename TX>
void BFGS_ONs<TX>::get_start_guess(const std::vector<TX>& dE_dx_in, std::vector<TX>& x0_in)
{
    for(int j=0; j<x0_in.size(); ++j) { this->x0[j] = x0_in[j]; }

    // identity matrix or other method to initialize H0
    for(int i=0; i<nk_total*nbands; ++i) { Hk[i*(nk_total*nbands) + i] = 1.0; }


}






// template class BFGS_ONs<double, double>;
// template class BFGS_ONs<std::complex<double>, double>;
// template class BFGS_ONs<std::complex<double>, std::complex<double>>;

template class BFGS_ONs<double>;

}


