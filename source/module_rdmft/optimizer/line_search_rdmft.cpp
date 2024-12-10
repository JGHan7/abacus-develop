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
    this->occ_number.create(rdmft_solver->nk_total, PARAM.inp.nbands);

    this->ls_c1 = 0.0001;
    this->ls_c2 = 0.999;
    this->ls_condition = "swolfe";
    this->max_step_size = 1000;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::get_start_guess()
{
    std::cout << "\n******\n" << "start_guess: ebi, 0.0" << "\n******\n" << std::endl;
    std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    std::cout << "\n******\n" << "start_guess: ebi, 0.01" << "\n******\n" << std::endl;
    if(this->ebi.random_inital)
    {
        ebi.get_inital_guess(this->var_x);
        std::cout << "\n******\n" << "start_guess: ebi, 0.1" << "\n******\n" << std::endl;
        // update rdmft elec_state
        ModuleBase::matrix occ_number = this->ebi.get_occ_number();
        std::cout << "\n******\n" << "start_guess: ebi, 0.15" << "\n******\n" << std::endl;
        this->rdmft_solver->update_elec( &occ_number );
    }
    else
    {
        ebi.get_inital_guess(this->var_x, &rdmft_solver->occ_number);
        std::cout << "\n******\n" << "start_guess: ebi, 0.1" << "\n******\n" << std::endl;
    }
    this->occ_number = this->ebi.get_occ_number();
    this->phi_0 = this->rdmft_solver->cal_Energy();
    std::cout << "\n******\n" << "start_guess: ebi, 0.2" << "\n******\n" << std::endl;
}


template<typename TK, typename TR>
double LineSearch<TK, TR>::do_line_search(const bool start_guess)
{
    if(start_guess) { this->get_start_guess(); }

    std::cout << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;
    // rdmft cal dE_docc_num, EBI convert dE_docc_num to dE_dx
    this->cal_dE_dx(this->dE_dx);
    std::cout << "\n******\n" << "start_guess: ls, 0.1" << "\n******\n" << std::endl;

    this->cal_pk_dphi0(start_guess);
    std::cout << "\n******\n" << "start_guess: ls, 1.0" << "\n******\n" << std::endl;

    // have pk, use strong wolfe to find step_size, update x_k+1 = x_k + step_size * p_k
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

    // convert x_k+1 to occ_num, rdmft_solver update occ_num, Hk, etc.
    this->phi_0 = this->cal_phi(this->var_x);   // has be calculated in swolfe() or zoom() ?

    ModuleBase::matrix diff_occ_num = ( this->occ_number );
    this->occ_number = this->ebi.get_occ_number();
    diff_occ_num -= this->occ_number;
    
    // std::abs( diff_occ_num )
    // return max( diff_occ_num )
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::strong_wolfe()
{
    // initial step_size before each iteration
    this->step_size = (1.0 < this->max_step_size) ? 1.0 : this->max_step_size/2.0;

    // 0: represents the relevant quantity under x_k, that is, var_x
    // phi_0, dphi_0 have obtained

    // old: represents the relevant quantity under the last trial_step_size/trial_x
    double step_size_old =0.0;
    double phi_old = this->phi_0;

    // trial_x = x_k(or this->var_x) + trial_step_size * p_k
    std::vector<double> trial_x(this->var_x.size() ,0.0);
    std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);

    while(1)
    {
        for(int i=0; i<trial_x.size(); ++i)
        {
            trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
        }

        double trial_phi = this->cal_phi(trial_x);
        if( (trial_phi > this->phi_0 + this->ls_c1 * this->step_size * this->dphi_0) || trial_phi > phi_old)
        {
            this->zoom(step_size_old, phi_old, this->step_size);
            break;
        }

        double trial_dphi = this->cal_dphi(trial_dE_dx);
        if( std::abs(trial_dphi) <= -this->ls_c2 * this->dphi_0 )
        {
            for(int i=0; i<trial_x.size(); ++i) { this->var_x[i] = trial_x[i]; }
            break;
        }

        if( trial_dphi >= 0 )
        {
            this->zoom(this->step_size, trial_phi, step_size_old);
            break;
        }

        step_size_old = this->step_size;
        phi_old = trial_phi;

        // needs improvement, using quadratic or cubic, currently using dichotomy
        this->step_size = ( this->step_size + this->max_step_size ) / 2.0;
    }

    // update x_k+1 = x_k + step_size * p_k
    for(int i=0; i<this->var_x.size(); ++i) { this->var_x[i] += this->step_size * this->search_direction[i]; }

}


template<typename TK, typename TR>
void LineSearch<TK, TR>::zoom(double step_size_low, double phi_low, double step_size_high)
{
    double alpha_lo = step_size_low;
    double alpha_hi = step_size_high;
    double f_low = phi_low;

    // trial_x = x_k(or this->var_x) + trial_step_size * p_k
    std::vector<double> trial_x(this->var_x.size() ,0.0);
    std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);

    int times = 0;
    while(1)
    {
        // needs improvement, using quadratic or cubic, currently using dichotomy
        this->step_size = (alpha_lo + alpha_hi)/2.0;

        for(int i=0; i<trial_x.size(); ++i)
        {
            trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
        }

        double trial_phi = this->cal_phi(trial_x);
        if( (trial_phi > this->phi_0 + this->ls_c1 * this->step_size * this->dphi_0) || trial_phi >= f_low )
        {
            alpha_hi = this->step_size;
        }
        else
        {
            double trial_dphi = this->cal_dphi(trial_dE_dx);
            if( std::abs(trial_dphi) <= -this->ls_c2 * this->dphi_0 )
            {
                break;
            }

            if( trial_dphi * (alpha_hi - alpha_lo) >= 0 )
            {
                alpha_hi = alpha_lo;
            }

            alpha_lo = this->step_size;
            f_low = trial_phi;
        }

        if( times >= 10 ) { break; }
    }



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
void LineSearch<TK, TR>::cal_dE_dx(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr)
{
    if( x_new_ptr != nullptr ) { this->cal_phi( *x_new_ptr ); }

    // rdmft cal dE_docc_num
    this->rdmft_solver->cal_E_grad_occ_num();

    std::cout << "\n******\n" << "ls: cal_dE_dx(), 0.1" << "\n******\n" << std::endl;
    // EBI: convert dE_docc_num to dE_dx (x in EBI is the latest, that is, it is consistent with dE_dx)
    std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
    std::cout << "\n******\n" << "ls: cal_dE_dx(), 0.2" << "\n******\n" << std::endl;
    this->ebi.get_dE_dx(dE_docc_num, dE_dx_new);
    std::cout << "\n******\n" << "ls: cal_dE_dx(), 0.3" << "\n******\n" << std::endl;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::cal_pk_dphi0(const bool start_guess)
{
    // get pk: EBI provide var_x and dE_dx to BFGS
    this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, start_guess);

    // get dphi_0
    rdmft::dgemm_lapack( this->dE_dx.data(), this->search_direction.data(), &this->dphi_0, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');
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

