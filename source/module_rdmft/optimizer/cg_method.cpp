//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================


#include "module_rdmft/optimizer/cg_method.h"


namespace rdmft
{


template<typename TX>
CG_method<TX>::CG_method()
{

}


template<typename TX>
CG_method<TX>::~CG_method()
{
    
}


template<typename TX>
void CG_method<TX>::init(const int dim_in, const double precond_eps_in)
{
    rdmft::Opti_method<TX>::init(dim_in, precond_eps_in);




}


template<typename TX>
void CG_method<TX>::get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool new_landscape, const std::vector<TX>* d2E_dx2)
{





}




template class CG_method<double>;
template class CG_method<std::complex<double>>;

}