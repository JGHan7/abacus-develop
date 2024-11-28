//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================


#include "module_rdmft/optimizer/line_search.h"


namespace rdmft
{

template<typename TX, typename T_rdmft>
LineSearch::LineSearch()
{
    ;
}


template<typename TX, typename T_rdmft>
LineSearch::~LineSearch()
{
    ;
}


template<typename TX, typename T_rdmft>
void LineSearch::init(T_rdmft* rdmft_in)
{
    this->rdmft_solver = rdmft_in;
    this->ebi.init(rdmft_solver->nk_total);
    this->bfgs_rdmft.init(rdmft_solver->nk_total, PARAM.inp.nbands);


}


















template class LineSearch<double, RDMFT<double, double>>;
template class LineSearch<double, RDMFT<std::complex<double>, double>>;
template class LineSearch<double, RDMFT<std::complex<double>, std::complex<double>>>;
// template class LineSearch<std::complex<double>, RDMFT<double, double>>;
// template class LineSearch<std::complex<double>, RDMFT<std::complex<double>, double>>;
// template class LineSearch<std::complex<double>, RDMFT<std::complex<double>, std::complex<double>>>;


}

