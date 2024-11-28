//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================


#include "module_rdmft/optimizer/line_search.h"


namespace rdmft
{

template<typename TK, typename TR>
LineSearch<TK, TR>::LineSearch()
{
    ;
}


template<typename TK, typename TR>
LineSearch<TK, TR>::~LineSearch()
{
    ;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::init(RDMFT<TK, TR>* rdmft_solver_in)
{
    this->rdmft_solver = rdmft_in;
    this->ebi.init(rdmft_solver->nk_total);
    this->bfgs_rdmft.init(rdmft_solver->nk_total, PARAM.inp.nbands);

    
}

















template class LineSearch<double, double>;
template class LineSearch<std::complex<double>, double>;
template class LineSearch<std::complex<double>, std::complex<double>>;

// template class LineSearch<double, RDMFT<double, double>>;
// template class LineSearch<double, RDMFT<std::complex<double>, double>>;
// template class LineSearch<double, RDMFT<std::complex<double>, std::complex<double>>>;
// template class LineSearch<std::complex<double>, RDMFT<double, double>>;
// template class LineSearch<std::complex<double>, RDMFT<std::complex<double>, double>>;
// template class LineSearch<std::complex<double>, RDMFT<std::complex<double>, std::complex<double>>>;


}

