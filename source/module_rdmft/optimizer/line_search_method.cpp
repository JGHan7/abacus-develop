//==========================================================
// Author: Jingang Han
// DATE : 2025-09-22
//==========================================================

#include <algorithm>
#include <map>
#include  "module_rdmft/optimizer/line_search_method.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_parameter/parameter.h"

namespace rdmft
{



template<typename TX>
LineSearch<TX>::LineSearch(const int max_ls_in, const double tolerance_change_in)
{
    this->max_ls = max_ls_in;
    this->tolerance_change = tolerance_change_in;
    this->ls_wolfe_c1 = PARAM.inp.ls_wolfe_c1;
    this->ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2;
    this->ls_armijo_c1 = PARAM.inp.ls_armijo_c1;
    this->ls_armijo_c2 = PARAM.inp.ls_armijo_c2;
    this->ls_condition = PARAM.inp.ls_condition;
    this->max_step_size = PARAM.inp.max_step_size;
    this->min_step_size = PARAM.inp.min_step_size; // PARAM.inp.min_step_size, 1e-8, 1e-10?
}


template<typename TX>
LineSearch<TX>::~LineSearch()
{
    ;
}


template<typename TX>
Options LineSearch<TX>::get_options()
{
    return Options(this->ls_wolfe_c1, this->ls_wolfe_c2,
                    this->ls_armijo_c1, this->ls_armijo_c2,
                    this->max_step_size, this->min_step_size,
                    this->ls_condition);
}


template<typename TX>
double LineSearch<TX>::do_line_search(const std::function<double(const double)>& cal_phi,
                                        const std::function<double()>& cal_dphi,
                                        const double phi_0_in,
                                        const double dphi_0_in,
                                        const double max_elem_pk_in,
                                        const double inital_step)
{
    this->phi_0 = phi_0_in;
    this->dphi_0 = dphi_0_in;
    this->max_elem_pk = max_elem_pk_in;
    this->step_size = std::abs(inital_step);

    if(this->ls_condition == "swolfe")
    {
        this->strong_wolfe(cal_phi, cal_dphi);
    }
    else if(this->ls_condition == "wolfe")
    {
        this->wolfe(cal_phi, cal_dphi);
    }
    else if(this->ls_condition == "test")
    {
        this->strong_wolfe2(cal_phi, cal_dphi);
    }
    else // fixed step
    {
        this->step_size = PARAM.inp.ls_fixed_step;
    }
    // this->step_size = std::max(this->step_size, this->min_step_size);
    std::cout << "\n***\nin ls, ls_time: " << this->ls_times  << "\n***\n" << std::endl;

    return this->step_size;
}


template<typename TX>
void LineSearch<TX>::strong_wolfe(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi)
{
    // std::cout << "\n" << "test ls: strong_wolfe()" << "\n" << std::endl;

    // strong wolfe condition
    double step_size_old = 0.0;
    double phi_old = this->phi_0;
    double dphi_old = this->dphi_0;

    bool done = false;
    this->ls_times = 0;
    std::vector<double> bracket, bracket_phi, bracket_dphi;

    double trial_phi = cal_phi(this->step_size);
    double trial_dphi = cal_dphi();
    ++this->num_cal_phi;

    while(this->ls_times < this->max_ls)
    {
        // Armijo condition
        if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 || ( this->ls_times > 1 && (trial_phi >= phi_old) ) )
        {
            bracket = {step_size_old, this->step_size};
            bracket_phi = {phi_old, trial_phi};
            bracket_dphi = {dphi_old, trial_dphi};
            break;
        }

        // curvature condition
        if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 ) // need abs? dphi_0 should be negative, but...
        {
            bracket = {this->step_size, this->step_size};
            bracket_phi = {trial_phi, trial_phi};
            bracket_dphi = {trial_dphi, trial_dphi};
            done = true;
            break;
        }

        if( trial_dphi >= 0 )
        {
            bracket = {step_size_old, this->step_size};
            bracket_phi = {phi_old, trial_phi};
            bracket_dphi = {dphi_old, trial_dphi};
            break;
        }

        // use interpolation algorithm to give the next trial step size, refer to pyTorch
        double min_step = this->step_size + 0.01 * ( this->step_size - step_size_old );
        double max_step = this->step_size * 10.0;
        double temp_step = this->step_size;
        // std::pair<double,double> bounds{this->min_step_size, this->max_step_size}; // error !
        std::pair<double,double> bounds{min_step, max_step};
        this->step_size = rdmft::cubic_interpolate( step_size_old, phi_old, dphi_old,
                                                    this->step_size, trial_phi, trial_dphi,
                                                    &bounds );

        step_size_old = temp_step;
        phi_old = trial_phi;
        dphi_old = trial_dphi;

        trial_phi = cal_phi(this->step_size);
        trial_dphi = cal_dphi();
        ++this->num_cal_phi;
        ++this->ls_times;
    }

    // safe fallback strategy
    if( this->ls_times == this->max_ls )
    {
        bracket = {0.0, this->step_size};
        bracket_phi = {this->phi_0, trial_phi};
        bracket_dphi = {this->dphi_0, trial_dphi};
    }

    // ensure bracket_phi[low_pos] < bracket_phi[high_pos]
    int low_pos = 0;
    int high_pos = 1;
    if( bracket_phi[0] > bracket_phi[1] )
    {
        std::swap(low_pos, high_pos);
    }
    if( !done && (this->ls_times < this->max_ls) )
    {
        // use zoom() to reduce the interval until a step size that satisfies the strong wolfe condition is found
        this->zoom( cal_phi, cal_dphi,
                    bracket[low_pos], bracket_phi[low_pos], bracket_dphi[low_pos],
                    bracket[high_pos], bracket_phi[high_pos], bracket_dphi[high_pos] );
    }
    else
    {
        this->step_size = bracket[low_pos];
    }

}


template<typename TX>
void LineSearch<TX>::zoom(const std::function<double(const double)>& cal_phi,
                            const std::function<double()>& cal_dphi,
                            double step_size_low,
                            double phi_low,
                            double dphi_low,
                            double step_size_high,
                            double phi_high,
                            double dphi_high)
{
    double alpha_lo = step_size_low;
    double alpha_hi = step_size_high;
    double f_lo = phi_low;
    double f_hi = phi_high;
    double df_lo = dphi_low;
    double df_hi = dphi_high;

    bool done = false;
    bool small_progress = false;
    while( !done && this->ls_times < this->max_ls )
    {
        this->step_size = rdmft::cubic_interpolate( alpha_lo, f_lo, df_lo,
                                                        alpha_hi, f_hi, df_hi );

        // prevent interpolation points from being too close to the boundary and causing insufficient progress
        double epsilon = 0.1 * ( alpha_hi - alpha_lo );
        if( std::min(alpha_hi - this->step_size, this->step_size - alpha_lo ) < epsilon )
        {
            if( small_progress || (this->step_size >= alpha_hi) || (this->step_size <= alpha_lo) )
            {
                this->step_size = ( std::abs(this->step_size - alpha_hi) < std::abs(this->step_size - alpha_lo) )
                                    ? alpha_hi - epsilon : alpha_lo + epsilon ;
                small_progress = false;
            }
            else
            {
                small_progress = true;
            }
        }
        else
        {
            small_progress = false;
        }

        double trial_phi = cal_phi(this->step_size);
        double trial_dphi = cal_dphi();
        ++this->num_cal_phi;
        ++this->ls_times;

        if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi >= f_lo )
        {
            alpha_hi = this->step_size;
            f_hi = trial_phi;
            df_hi = trial_dphi;

            if( f_hi < f_lo )
            {
                std::swap(alpha_lo, alpha_hi);
                std::swap(f_lo, f_hi);
                std::swap(df_lo, df_hi);
            }
        }
        else
        {
            if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 ) // need abs? dphi_0 should be negative, but...
            {
                done = true;
            }

            if( trial_dphi * (alpha_hi - alpha_lo) >= 0 )
            {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
                df_hi = df_lo;
            }

            alpha_lo = this->step_size;
            f_lo = trial_phi;
            df_lo = trial_dphi;
        }

        if( std::abs(alpha_hi - alpha_lo) * this->max_elem_pk < this->tolerance_change )
        {
            break;
        }
    }

    this->step_size = alpha_lo;
}


template<typename TX>
void LineSearch<TX>::wolfe(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi)
{
    
}


template<typename TX>
void LineSearch<TX>::strong_wolfe2(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi)
{
    double trial_phi = 0.0;
    double trial_dphi = 0.0;
    this->armijo_step = 0.0;

    // bool small_step = false;
    for(int times=0; times<20; ++times)
    {
        if( this->step_size < 1e-8 )
        {
            std::cout << "\n" << "occNum optimization completed?" << this->step_size << std::endl;
            this->step_size = this->min_step_size;
            return;
        }

        trial_phi = cal_phi(this->step_size);
        if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )
        {
            if( times < 5 )
            {
                this->step_size *= 0.5;
            }
            else
            {
                this->step_size *= 0.2;
            }
            // small_step = true;
            std::cout << "\n" << "initial step size needs to be reduced : " << this->step_size << std::endl;
            // continue;
        }
        else
        {
            break;
        }
    }
    this->step_size = std::max(this->step_size, this->min_step_size);


    // strong wolfe condition
    double step_size_old = 0.0;
    double phi_old = this->phi_0;
    double dphi_old = this->dphi_0;

    double factor1 = 0.0;
    for(int times=0; times<50; ++times)
    {
        if( times != 0 )
        {
            std::cout << "\n" << "step_size now: " << this->step_size << std::endl;
            trial_phi = cal_phi(this->step_size);
            if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )  // && times > 0
            {
                // this->step_size /= factor1;
                // armijo_step = this->step_size;
                // std::cout << "\n" << "find zoom failed! the energy must drop in armijo steps, update as: " << this->step_size << "\n" << std::endl;

                std::cout << "\n" << "Enter SW condition 1, zoom()" << "\n" << std::endl;
                this->zoom2(cal_phi, cal_dphi, step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
                return;
            }
            else
            {
                this->armijo_step = this->step_size;
                std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;
            }
        }

        trial_dphi = cal_dphi();
        if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
        {
            std::cout << "\n" << "Find SW step: " << this->step_size << "\n" << std::endl;
            return;
        }

        if( trial_dphi >= 0 )
        {
            std::cout << "\n" << "Enter SW condition 3, zoom()" << "\n" << std::endl;
            this->armijo_step = this->step_size;
            std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;

            this->zoom2(cal_phi, cal_dphi, this->step_size, trial_phi, step_size_old, phi_old, trial_dphi);  // ? which one ???
            // this->zoom2(cal_phi, cal_dphi, step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
            return;
        }

        step_size_old = this->step_size;
        phi_old = trial_phi;
        dphi_old = trial_dphi;

        // if( times<10 )
        // {
        //     factor1 = 1.1;
        //     this->step_size *= factor1;
        // }
        // else
        {
            factor1 = 1.1;
            this->step_size *= factor1;
        }
        this->step_size = std::min(this->step_size, this->max_step_size);


        // Under the strict control of Armijo condition, is it possible for the situation here to occur?
        // need!
        // If a is too large, it may directly cross the unique minimum point, and the Armijo condition is satisfied in a very wide range.
        // This is prone to problems and the step size should be reduced.
        // Can zoom() accept non-monotonic {alpha_i} ?


        // for(int ik=0; ik<this->rdmft_solver->nk_total; ++ik)
        // {
        //     if( this->rdmft_solver->occ_number(ik, 0) < 0.4 )
        //     {
        //         armijo_step = this->step_size / factor1;
        //         this->step_size /= factor1 * 1.05; // select 1.05 != factor1
        //         // armijo_step = this->step_size;

        //         // this->step_size /= factor1;
        //         // std::cout << "\n" << "exactLS failed! the energy must drop in armijo steps, update as:: " << this->step_size << "\n" << std::endl;
        //         // return;
        //     }
        // }

    }

    this->armijo_step = this->step_size;
    std::cout << "\n" << "Strong Wolfe failed! the energy must drop in armijo steps, update as:: " << this->armijo_step << "\ndphi_0: " << this->dphi_0 << std::endl;

}




template<typename TX>
void LineSearch<TX>::zoom2(const std::function<double(const double)>& cal_phi,
                            const std::function<double()>& cal_dphi,
                            double step_size_low,
                            double phi_low,
                            double step_size_high,
                            double phi_high,
                            double dphi_low)
{
    double alpha_lo = step_size_low;
    double alpha_hi = step_size_high;
    double f_lo = phi_low;
    double f_hi = phi_high;
    double df_lo = dphi_low;

    double df_hi = 0.0;

    int times = 0;
    while(1)
    {
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
        // this->step_size = alpha_lo + incr_alpha;
        std::cout << "\n" << "incr_alpha: " << incr_alpha << "\nnew step_size by incr_alpha: " << alpha_lo + incr_alpha << "\n" << std::endl;


        // needs improvement, using quadratic or cubic, currently using dichotomy
        this->step_size = (alpha_lo + alpha_hi)/2.0;
        std::cout << "\n" << "in ZOOM(), while: " << times << ", step_size: " << this->step_size << "\n" << std::endl;

        double trial_phi = cal_phi(this->step_size);

        if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi >= f_lo )
        {
            alpha_hi = this->step_size;
            f_hi = trial_phi;
            std::cout << "\n" << "Enter ZOOM condition 1\nalpha_hi: " << alpha_hi << std::endl;
        }
        else
        {
            this->armijo_step = this->step_size;
            std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;

            double trial_dphi = cal_dphi();
            if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
            {
                std::cout << "\n" << "ZOOM() get\nstep_size: " << std::endl;
                return;
            }

            if( trial_dphi * (alpha_hi - alpha_lo) >= 0 )
            {
                std::cout << "\n" << "Enter ZOOM condition 3, exchange high_low" << "\n" << std::endl;
                alpha_hi = alpha_lo;
                f_hi = f_lo;

                df_hi = df_lo;
            }

            alpha_lo = this->step_size;
            f_lo = trial_phi;
            df_lo = trial_dphi;
        }

        std::cout << "\nupdate\n" << "alpha_hi: " << alpha_hi << "\nalpha_lo: " << alpha_lo << std::endl;
        std::cout << "\nupdate\n" << "-ls_w_c2 * dphi_0: " << -this->ls_wolfe_c2 * this->dphi_0 << "\n|dphi_low|: " << df_lo << std::endl;

        // // need it?
        // if( alpha_lo > alpha_hi )
        // {
        //     std::swap(alpha_lo, alpha_hi);
        //     std::swap(f_lo, f_hi);
        //     std::swap(df_lo, df_hi);
        // }
        // std::cout << "\n" << "By comparing the size, exchange high_low" << "\nalpha_hi: " << alpha_hi << "\nalpha_lo: " << alpha_lo << std::endl;

        ++times;
        if( times >= 20 || this->step_size <= this->min_step_size )
        {
            // std::cout << "\n******\n" << "zoom times too big: " << times << "\n******\n" << std::endl;
            this->step_size = this->armijo_step;
            std::cout << "\n" << "zoom() failed! the energy must drop in armijo steps, update as:: " << this->armijo_step << "\n" << std::endl;
            return;
        }

    }
}





template class LineSearch<double>;
template class LineSearch<std::complex<double>>;

}

