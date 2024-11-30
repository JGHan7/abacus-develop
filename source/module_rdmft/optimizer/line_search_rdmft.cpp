//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================

#include "module_rdmft/optimizer/line_search_rdmft.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include <algorithm>

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
    this->bfgs_opti_x.init(rdmft_solver->nk_total, PARAM.inp.nbands);

    this->var_x.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->dE_dx.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->search_direction.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::do_line_search(const bool start_guess)
{
    if(start_guess)
    {
        std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
        if(this->ebi.random_inital)
        {
            ebi.get_inital_guess();
            // update rdmft elec_state
            ModuleBase::matrix occ_number = this->ebi.get_occ_number();
            this->rdmft_solver->update_elec( &occ_number );
        }
        else
        {
            ebi.get_inital_guess(&rdmft_solver->occ_number);
        }
    }

    // rdmft cal dE_docc_num
    this->rdmft_solver->cal_E_grad_wfc_occ_num();

    // EBI: convert dE_docc_num to dE_dx (x in EBI is the latest, that is, it is consistent with dE_dx)
    std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
    this->ebi.get_dE_dx(dE_docc_num, this->dE_dx);

    // get pk: EBI provide var_x and dE_dx to BFGS
    this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, start_guess);

    // have pk, use strong wolfe to find step_size, alpha !!!
    if(1) { this->strong_wolfe(); }
    else { this->wolfe(); }

    // get alpha, update x_k+1 = x_k + alpha * pk
    // var_x += alpha*pk


    // convert x_k+1 to occ_num
    this->ebi.update_x_occ_num(this->var_x);

    // rdmft_solver update occ_num, Hk, etc.
    ModuleBase::matrix occ_number = this->ebi.get_occ_number();
    this->rdmft_solver->update_elec( &occ_number );


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

