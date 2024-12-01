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

    this->ls_c1 = 0.0001;
    this->ls_c2 = 0.9;
    this->ls_condition = "swolfe";
    this->max_step_size = 1000;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::get_start_guess()
{
    std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    if(this->ebi.random_inital)
    {
        ebi.get_inital_guess(this->var_x);
        // update rdmft elec_state
        ModuleBase::matrix occ_number = this->ebi.get_occ_number();
        this->rdmft_solver->update_elec( &occ_number );
    }
    else
    {
        ebi.get_inital_guess(this->var_x, &rdmft_solver->occ_number);
    }
    this->phi_0 = this->rdmft_solver->cal_Energy();
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::do_line_search(const bool start_guess)
{
    if(start_guess) { this->get_start_guess(); }

    // rdmft cal dE_docc_num, EBI convert dE_docc_num to dE_dx
    this->cal_dE_dx(this->dE_dx);

    this->cal_pk_dphi0(start_guess);

    // have pk, use strong wolfe to find step_size, alpha !!!
    if(this->ls_condition == "swolfe") // PARAM.inp.ls_condition == "swolfe"
    {
        this->strong_wolfe();
    }
    else if(this->ls_condition == "wolfe")
    {
        this->wolfe();
    }
    else
    {
        this->strong_wolfe();
    }

    // get alpha, update x_k+1 = x_k + alpha * pk
    // var_x += alpha*pk



    // convert x_k+1 to occ_num, rdmft_solver update occ_num, Hk, etc.
    this->phi_0 = this->cal_phi(this->var_x);


}


template<typename TK, typename TR>
void LineSearch<TK, TR>::strong_wolfe()
{
    // get phi_0, dphi_0
    // this->phi_0 = this->rdmft_solver->cal_Energy();
    rdmft::dgemm_lapack( this->dE_dx.data(), this->search_direction.data(), &this->dphi_0, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');

    // initial step_size before each iteration
    this->step_size = (1.0 < this->max_step_size) ? 1.0 : this->max_step_size/2.0;

    // old: represents the relevant quantity under the last trial_step_size/trial_x
    double step_size_old =0.0;

    std::vector<double> trial_x(this->var_x.size() ,0.0);
    while(1)
    {
        for(int i=0; i<trial_x.size(); ++i)
        {
            trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
        }

        // //! dphi_trial = (dphi/dalpha at alpha_trial) = E'(x_k + alpha_trial*p_k) * p_k^T
        // double dphi_trial = 0.0;

    }

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


template<typename TK, typename TR>
double LineSearch<TK, TR>::cal_phi(const std::vector<double>& x_new)
{
    // convert x_k+1 to occ_num
    this->ebi.update_x_occ_num(x_new);

    // rdmft_solver update occ_num, Hk, etc.
    ModuleBase::matrix occ_number = this->ebi.get_occ_number();
    this->rdmft_solver->update_elec( &occ_number );

    // cal phi(alpha) = E(x_k + alpha * p_k)
    double phi = this->rdmft_solver->cal_Energy();

    return phi;
}


template<typename TK, typename TR>
double LineSearch<TK, TR>::cal_dphi(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr)
{
    this->cal_dE_dx(dE_dx_new, x_new_ptr);

    double dphi = 0.0;
    rdmft::dgemm_lapack( dE_dx_new.data(), this->search_direction.data(), &dphi, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');

    return dphi;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::cal_dE_dx(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr = nullptr)
{
    if( x_new_ptr != nullptr ) { this->cal_phi( *x_new_ptr ); }

    // rdmft cal dE_docc_num
    this->rdmft_solver->cal_E_grad_wfc_occ_num();

    // EBI: convert dE_docc_num to dE_dx (x in EBI is the latest, that is, it is consistent with dE_dx)
    std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
    this->ebi.get_dE_dx(dE_docc_num, dE_dx_new);
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::cal_pk_dphi0(const bool start_guess)
{
    // get pk: EBI provide var_x and dE_dx to BFGS
    this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, start_guess);

    // get dphi_0
    rdmft::dgemm_lapack( dE_dx_new.data(), this->search_direction.data(), &this->dphi_0, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');
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

