//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================

#include "module_rdmft/optimizer/line_search_rdmft.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include <algorithm>

#include "module_rdmft/rdmft_tools.h" // temp

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
    this->ebi.init(rdmft_solver->nk_total, rdmft_solver->get_kv().get_nkstot_full(), rdmft_solver->get_kv().wk);
    this->bfgs_opti_x.init(rdmft_solver->nk_total, PARAM.inp.nbands);

    this->var_x.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->dE_dx.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->search_direction.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->occ_number.create(rdmft_solver->nk_total, PARAM.inp.nbands);

    this->ls_wolfe_c1 = 0.0001;
    this->ls_wolfe_c2 = 0.999;
    this->ls_armijo_c1 = 0.0001;
    this->ls_armijo_c2 = 0.9;
    this->ls_condition = "swolfe";
    this->max_step_size = 1000;
    this->min_step_size = 1e-10;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::get_start_guess()
{
    std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    if(this->ebi.random_inital)
    {
        ebi.get_inital_guess(this->var_x);

        // // update rdmft elec_state
        // ModuleBase::matrix occ_number( this->ebi.get_occ_number() );
        // this->rdmft_solver->update_elec( &occ_number );
    }
    else
    {
        ebi.get_inital_guess(this->var_x, &rdmft_solver->occ_number);
    }

    this->occ_number = this->ebi.get_occ_number();
    this->rdmft_solver->update_elec( &(this->occ_number) );

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }

    rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, this->occ_number.c, "occ_number in ls", 10);

    this->phi_0 = this->rdmft_solver->cal_Energy();
}


template<typename TK, typename TR>
double LineSearch<TK, TR>::do_line_search(const bool start_guess)
{

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }

    if(start_guess)
    {
        this->get_start_guess();
        std::cout << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;
        out_file << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;
    }

    out_file << "\n******\n" << "ls, before cal_dE_dx()" << "\n******\n" << std::endl;

    // rdmft cal dE_docc_num, EBI convert dE_docc_num to dE_dx
    this->cal_dE_dx(this->dE_dx);

    out_file << "\n******\n" << "ls, after cal_dE_dx()" << "\n******\n" << std::endl;

    this->cal_pk_dphi0(start_guess);
    std::cout << "\n******\n" << "ls, dphi_0: " << this->dphi_0 << "\n******\n" << std::endl;

    out_file << "\n******\n" << "ls, dphi_0: " << this->dphi_0 << "\n******\n" << std::endl;

    std::vector<double> var_x_old = this->var_x;

    out_file << "\n******\n" << "ls, var_x_old" << "\n******\n" << std::endl;

    // have pk, use strong wolfe to find step_size
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

    out_file << "\n******\n" << "ls, after condition" << "\n******\n" << std::endl;

    // update x_k+1 = x_k + step_size * p_k
    for(int i=0; i<this->var_x.size(); ++i)
    {
        this->var_x[i] += this->step_size * this->search_direction[i]; 
    }

    out_file << "\n******\n" << "ls, after step_size * search_direction" << "\n******\n" << std::endl;

    // if( a_equal_b(var_x_old, this->var_x) )
    // {
    //     std::cout << "\n" << "line_search_rdmft: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
    //     return 0.0;
    // }

    double temp_num = a_equal_b(var_x_old, this->var_x);

    Parallel_Reduce::reduce_all(temp_num);

    if( std::abs( temp_num ) > 1e-12 )
    {
        std::cout << "\n" << "line_search_rdmft: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
        return 0.0;
    }


    out_file << "\n******\n" << "ls, after a_equal_b()" << "\n******\n" << std::endl;

    // std::cout << "\n" << "after swolfe, this->step_size: " << this->step_size << "\n" << std::endl;
    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->search_direction.data(), "search_direction");
    // std::cout << "\n******\n += [0]: " << this->step_size * this->search_direction[0] << "\n******\n" << std::endl;




    std::cout << "\n******\n" << "ls, do_line_search: strong_wolfe, 1.0" << "\n******\n" << std::endl;
    out_file << "\n******\n" << "ls, do_line_search: strong_wolfe, 1.0" << "\n******\n" << std::endl;


    // convert x_k+1 to occ_num, rdmft_solver update occ_num, Hk, etc.
    this->phi_0 = this->cal_phi(this->var_x);   // has be calculated in swolfe() or zoom() ?

    std::cout << "\n******\n" << "ls, do_line_search, phi_0: " << this->phi_0 << "\n******\n" << std::endl;
    out_file << "\n******\n" << "ls, do_line_search, phi_0: " << this->phi_0 << "\n******\n" << std::endl;

    out_file.close();

    // ModuleBase::matrix diff_occ_num = ( this->occ_number );
    // this->occ_number = this->ebi.get_occ_number();
    // diff_occ_num -= this->occ_number;
    
    // std::abs( diff_occ_num )
    ModuleBase::matrix temp_occ( this->ebi.get_occ_number() );
    std::vector<double> diff_occ_num(this->rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
    for(int ik=0; ik<temp_occ.nr; ++ik)
    {
        for(int ib=0; ib<temp_occ.nc; ++ib)
        {
            diff_occ_num[ ik*PARAM.inp.nbands + ib ] = std::abs( this->occ_number(ik, ib) - temp_occ(ik, ib) );
        }
    }
    this->occ_number = temp_occ;

    auto diff_occ_num_max = std::max_element(diff_occ_num.begin(), diff_occ_num.end());

    return *diff_occ_num_max;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::strong_wolfe()
{

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }

    std::cout << "\n" << "Enter strong_wolfe()" << "\n" << std::endl;
    out_file << "\n" << "Enter strong_wolfe()" << "\n" << std::endl;
    // initial step_size before each iteration
    this->step_size = (1.0 < this->max_step_size) ? 1.0 : this->max_step_size/2.0;

    out_file << "\nstep_size: " << this->step_size << "\n" << std::endl;

    // 0: represents the relevant quantity under x_k, that is, var_x
    // phi_0, dphi_0 have obtained
    // old: represents the relevant quantity under the last trial_step_size/trial_x
    double step_size_old =0.0;
    double phi_old = this->phi_0;
    double dphi_old = this->dphi_0;

    // trial_x = x_k(or this->var_x) + trial_step_size * p_k
    std::vector<double> trial_x(this->var_x.size() ,0.0);
    std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
    double trial_phi = 0.0;
    double trial_dphi = 0.0;

    int times = 0;
    while(1)
    {
        std::cout << "\n" << "in strong_wolfe(), while" << "\n" << std::endl;
        out_file << "\n" << "in strong_wolfe(), while" << "\n" << std::endl;

        if(times != 0)
        {
            // // since max is too large, the dichotomy is too extreme
            // // and it is easy to fall into a saddle point when solving mu
            // // i.e., the number of particles is not conserved
            // this->step_size = ( this->step_size + this->max_step_size ) / 2.0; // needs improvement, using quadratic or cubic, currently using dichotomy

            std::cout << "\n" << "before quadratic interpolation" << "\n" << std::endl;
            // quadratic interpolation, temp_step = -b/(2a) in quadratic function
            // improved using cubic interpolation ?
            double temp_step = - this->dphi_0 * this->step_size * this->step_size / ( trial_phi - this->phi_0 - this->dphi_0 * this->step_size ) / 2.0;
            this->step_size = ( 1.1 * this->step_size < temp_step ) ? temp_step : 1.1 * this->step_size;
            this->step_size = ( this->step_size < this->max_step_size ) ? this->step_size : this->max_step_size;
            std::cout << "\n" << "quadratic interpolation in strong wolfe, temp_step:" << temp_step << "\n" << std::endl;

            out_file << "\n" << "after quadratic interpolation" << "\n" << std::endl;

        }

        out_file << "\ntimes: " << times << ", step_size: " << this->step_size << "\n" << std::endl;

        rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, this->var_x.data(), "now var_x", 10);

        for(int i=0; i<trial_x.size(); ++i)
        {
            trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
        }

        out_file << "\n" << "SW, after trial_x" << "\n" << std::endl;

        rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, trial_x.data(), "now trial_x", 10);
        rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, this->search_direction.data(), "now search_direction", 10);

        // if( a_equal_b(trial_x, this->var_x) )

        double temp_num = a_equal_b(trial_x, this->var_x);
        Parallel_Reduce::reduce_all(temp_num);
        if( std::abs( temp_num ) > 1e-12 )
        {
            std::cout << "\n" << "line_search_rdmft: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
            out_file << "\n" << "line_search_rdmft: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
            ++times;
            continue;
        }

        out_file << "\n" << "SW, after a_equal_b()" << "\n" << std::endl;


        std::cout << "\n" << "SW, before cal_phi()" << "\n" << std::endl;
        out_file << "\n" << "SW, before cal_phi()" << "\n" << std::endl;

        trial_phi = this->cal_phi(trial_x);

        out_file << "\n" << "SW, after cal_phi()" << "\n" << std::endl;
        out_file << std::fixed << std::setprecision(14) << "\nin SW, trial_phi: " << trial_phi << "\n" << std::endl << std::defaultfloat;


        if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi > phi_old)
        {
            std::cout << "\n" << "Enter SW condition 1" << "\n" << std::endl;
            out_file << "\n" << "Enter SW condition 1" << "\n" << std::endl;
            this->zoom(step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
            break;
        }

        std::cout << "\n" << "before cal_dphi()" << "\n" << std::endl;
        out_file << "\n" << "before cal_dphi()" << "\n" << std::endl;

        trial_dphi = this->cal_dphi(trial_dE_dx);

        out_file << "\nin SW, trial_dphi: " << trial_dphi << "\n" << std::endl;

        if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
        {
            std::cout << "\n" << "Enter SW condition 2" << "\n" << std::endl;
            out_file << "\n" << "Enter SW condition 2" << "\n" << std::endl;
            break;
        }

        if( trial_dphi >= 0 )
        {
            std::cout << "\n" << "Enter SW condition 3" << "\n" << std::endl;
            out_file << "\n" << "Enter SW condition 3" << "\n" << std::endl;
            this->zoom(this->step_size, trial_phi, step_size_old, phi_old, trial_dphi);
            break;
        }

        step_size_old = this->step_size;
        phi_old = trial_phi;
        dphi_old = trial_dphi;

        if( this->step_size > this->max_step_size )
        {
            std::cout << "\n******\n" << "in strong wolfe, step_size > max_step_size" << "\n******\n" << std::endl;
            break;
        }

        ++times;
        if( times>=15 )
        {
            std::cout << "\n******\n" << "strong wolfe times too big: " << times << "\n******\n" << std::endl;
            break;
        }


    }

    out_file.close();

}


template<typename TK, typename TR>
void LineSearch<TK, TR>::zoom(double step_size_low, double phi_low, double step_size_high, double phi_high, double dphi_low)
{
    double alpha_lo = step_size_low;
    double alpha_hi = step_size_high;
    double f_lo = phi_low;
    double f_hi = phi_high;
    double df_lo = dphi_low;

    // trial_x = x_k(or this->var_x) + trial_step_size * p_k
    std::vector<double> trial_x(this->var_x.size() ,0.0);
    std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);

    std::cout << "\n" << "Enter ZOOM()" << "\n" << std::endl;

    int times = 0;
    while(1)
    {
        std::cout << "\n" << "in ZOOM(), while" << "\n" << std::endl;
        // needs improvement, using quadratic or cubic, currently using dichotomy
        // this->step_size = (alpha_lo + alpha_hi)/2.0;

        // improved using cubic interpolation ?
        double diff_alpha = alpha_hi - alpha_lo;
        double incr_alpha = - df_lo * diff_alpha * diff_alpha / ( f_hi - (f_lo + df_lo * diff_alpha) ) / 2.0;
        if( incr_alpha < diff_alpha * this->ls_armijo_c1 )
        {
            incr_alpha = diff_alpha * this->ls_armijo_c1;
        }
        if( incr_alpha > diff_alpha * this->ls_armijo_c2 )
        {
            incr_alpha = diff_alpha * this->ls_armijo_c2;
        }
        this->step_size = alpha_lo + incr_alpha;


        for(int i=0; i<trial_x.size(); ++i)
        {
            trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
        }

        std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
        if (!out_file.is_open())
        {
            std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
        }

        out_file << "\n" << "ZOOM, before cal_phi()" << "\n" << std::endl;

        double trial_phi = this->cal_phi(trial_x);

        out_file << "\n" << "ZOOM, after cal_phi()" << "\n" << std::endl;

        out_file.close();



        if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi >= f_lo )
        {
            std::cout << "\n" << "Enter ZOOM condition 1" << "\n" << std::endl;
            alpha_hi = this->step_size;
        }
        else
        {
            double trial_dphi = this->cal_dphi(trial_dE_dx);
            if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
            {
                std::cout << "\n" << "Enter ZOOM condition 2" << "\n" << std::endl;
                break;
            }

            if( trial_dphi * (alpha_hi - alpha_lo) >= 0 )
            {
                std::cout << "\n" << "Enter ZOOM condition 3, exchange high_low" << "\n" << std::endl;
                alpha_hi = alpha_lo;
            }

            alpha_lo = this->step_size;
            f_lo = trial_phi;
        }

        ++times;
        if( times >= 10 )
        {
            std::cout << "\n******\n" << "zoom times too big: " << times << "\n******\n" << std::endl;
            break;
        }

        if( this->step_size <= this->min_step_size ) { break; }
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

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }

    out_file << "\n" << "Enter LineSearch: cal_phi()" << "\n" << std::endl;

    // convert x_k+1 to occ_num
    this->ebi.update_x_occ_num(x_new);
    out_file << "\n" << "step size now: " << this->step_size << "\n" << std::endl;

    // rdmft_solver update occ_num, Hk, etc.
    // ModuleBase::matrix occ_number( this->ebi.get_occ_number() );
    ModuleBase::matrix occ_number = this->ebi.get_occ_number();

    out_file << "\n" << "ebi.get_occ_number()" << "\n" << std::endl;

    this->rdmft_solver->update_elec( &occ_number );

    out_file << "\n" << "rdmft_solver->update_elec()" << "\n" << std::endl;

    // cal phi(alpha) = E(x_k + alpha * p_k)
    double phi = this->rdmft_solver->cal_Energy();

    out_file << "\n" << "rdmft_solver->cal_Energy()" << "\n" << std::endl;

    out_file.close();

    return phi;
}


template<typename TK, typename TR>
double LineSearch<TK, TR>::cal_dphi(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr)
{

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }

    out_file << "\n" << "ls, cal_dphi, cal_dE_dx: before cal_phi()" << "\n" << std::endl;

    this->cal_dE_dx(dE_dx_new, x_new_ptr);

    out_file << "\n" << "ls, cal_dphi, cal_dE_dx: after cal_phi()" << "\n" << std::endl;




    double dphi = 0.0;
    rdmft::dgemm_lapack( dE_dx_new.data(), this->search_direction.data(), &dphi, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');



    out_file.close();

    return dphi;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::cal_dE_dx(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr)
{

    std::ofstream out_file(this->rdmft_solver->process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << this->rdmft_solver->process_file << std::endl;
    }


    if( x_new_ptr != nullptr ) { this->cal_phi( *x_new_ptr ); }

    std::cout << "\n******\n" << "in cal_dE_dx" << std::endl;

    // rdmft cal dE_docc_num
    this->rdmft_solver->cal_E_grad_occ_num();

    // EBI: convert dE_docc_num to dE_dx (x in EBI is the latest, that is, it is consistent with dE_dx)
    std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
    this->ebi.get_dE_dx(dE_docc_num, dE_dx_new);

    rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, dE_docc_num.data(), "dE_docc_num");
    rdmft::printMatrix_pointer(out_file, this->rdmft_solver->nk_total, PARAM.inp.nbands, dE_dx_new.data(), "dE_dx_new");
    

    rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, dE_dx_new.data(), "dE_dx_new");
    rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->var_x.data(), "now var_x", 10);
    std::cout << "\n******\n" << std::endl;
}


template<typename TK, typename TR>
void LineSearch<TK, TR>::cal_pk_dphi0(const bool start_guess)
{
    // get pk: EBI provide var_x and dE_dx to BFGS
    this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, start_guess);

    rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->search_direction.data(), "search_direction");

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

