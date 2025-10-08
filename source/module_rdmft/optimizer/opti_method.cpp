//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================


#include "module_rdmft/optimizer/opti_method.h"


namespace rdmft
{


template<typename TX>
Opti_method<TX>::Opti_method()
{

}


template<typename TX>
Opti_method<TX>::~Opti_method()
{
    
}


template<typename TX>
void Opti_method<TX>::init(const int dim_in, const double precond_eps_in)
{
    this->dim = dim_in;
    this->precond_eps = precond_eps_in;
    this->iter = 0;

    dE_dx.resize(this->dim);
    search_direction.resize(this->dim);
}



template class Opti_method<double>;
template class Opti_method<std::complex<double>>;

}