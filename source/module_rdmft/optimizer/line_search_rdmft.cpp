//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================

#include "module_rdmft/optimizer/line_search_rdmft.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

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
void LineSearch<TK, TR>::init(RDMFT<TK, TR>* rdmft_in)
{
    this->rdmft_solver = rdmft_in;
    this->ebi.init(rdmft_solver->nk_total);
    this->bfgs_rdmft.init(rdmft_solver->nk_total, PARAM.inp.nbands);

    
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::do_line_search(const bool start_guess)
{
    // dealing with the initial situation

    if(start_guess)
    {
        ebi.get_inital_guess(&rdmft_solver->occ_number);
        // ebi.get_inital_guess(); // use the random inital value

    }

    // at the end of the previous step, EBI has got the latest x (rdmft_solver has update occ_num)

    // rdmft cal dE_docc_num

    // EBI: convert dE_docc_num to dE_dx (x in EBI is the latest, that is, it is consistent with dE_dx)

    // get pk: EBI provide var_x and dE_dx to BFGS

    // have pk, use strong wolfe to find step_size, alpha !!!

    // get alpha, update x_k+1 = x_k + alpha * pk

    // convert x_k+1 to occ_num
    // EBI.update_x_occ_num(x_k+1), 

    // rdmft_solver update occ_num, Hk, etc.
    // ModuleBase::matrix occ_num = this->ebi.get_occ_number();
    // this->rdmft_solver->update_elec( &occ_num );


}


template<typename TK, typename TR>
void LineSearch<TK, TR>::strong_wolfe()
{

}


template<typename TK, typename TR>
void LineSearch<TK, TR>::zoom()
{

}


// to be developed
template<typename TK, typename TR>
void LineSearch<TK, TR>::wolfe()
{

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

