//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================

#include <algorithm>
#include "module_rdmft/optimizer/line_search_ONs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"
#include "module_rdmft/optimizer/parameterize_ONs/softmax.h"

#include <torch/torch.h>

#include "module_rdmft/rdmft_tools.h" // temp

namespace rdmft
{

template<typename TK, typename TR>
LineSearch_ONs<TK, TR>::LineSearch_ONs()
{
    ;
}


template<typename TK, typename TR>
LineSearch_ONs<TK, TR>::~LineSearch_ONs()
{
    delete this->param_occ_num;
}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::init(const K_Vectors& kv_in, RDMFT<TK, TR>* rdmft_in)
{
    this->rdmft_solver = rdmft_in;
    
    this->bfgs_opti_x.init(rdmft_solver->nk_total*PARAM.inp.nbands, rdmft_solver->nk_total);
    if(PARAM.inp.occ_num_func == "softmax")
    {
        this->param_occ_num = new rdmft::SOFTMAX();
        std::cout << "\n\nSOFTMAX parameterized ONs are only applicable to electron pairing approaches which we have not yet implemented\n\n" << std::endl;
        assert(0);
    }
    else if(PARAM.inp.occ_num_func == "erf")
    {
        this->param_occ_num = new rdmft::EBI();
    }
    else
    {
        std::cout << "\n\n Please select the correct method to parameterize the occupation numbers \n\n" << std::endl;
        assert(0);
    }
    this->param_occ_num->init(rdmft_solver->nk_total, kv_in.get_nkstot_full(), kv_in.wk);

    this->var_x.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->dE_dx.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->search_direction.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->occ_number.create(rdmft_solver->nk_total, PARAM.inp.nbands);

    // this->ls_wolfe_c1 = 0.0001;
    // this->ls_wolfe_c2 = 0.999;
    // this->ls_armijo_c1 = 0.0001;
    // this->ls_armijo_c2 = 0.9;
    // this->ls_condition = "swolfe";
    // this->max_step_size = 1000;
    // this->min_step_size = 1e-6; // 1e-8, 1e-10?

    // this->ls_wolfe_c1 = PARAM.inp.ls_wolfe_c1;
    // this->ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2;
    // this->ls_armijo_c1 = PARAM.inp.ls_armijo_c1;
    // this->ls_armijo_c2 = PARAM.inp.ls_armijo_c2;
    // this->ls_condition = PARAM.inp.ls_condition;
    // this->max_step_size = PARAM.inp.max_step_size;
    // this->min_step_size = PARAM.inp.min_step_size; // PARAM.inp.min_step_size, 1e-8, 1e-10?

}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::restart_opti()
{
    this->iter = 0;
    this->init_step = 1.0;
    this->Etotal_iter.clear();
    this->Etotal_iter.push_back(this->rdmft_solver->Etotal);
    this->phi_0 = this->rdmft_solver->Etotal;
}

template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::get_start_guess()
{
    std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    if(PARAM.inp.random_occ_num)
    {
        this->param_occ_num->get_inital_guess(this->var_x);

        // // update rdmft elec_state
        // ModuleBase::matrix occ_number( this->param_occ_num->get_occ_number() );
        // this->rdmft_solver->update_elec( &occ_number );
    }
    else
    {
        this->param_occ_num->get_inital_guess(this->var_x, &rdmft_solver->occ_number);
    }

    this->occ_number = this->param_occ_num->get_occ_number();
    this->rdmft_solver->update_elec( &(this->occ_number) );
    this->phi_0 = this->rdmft_solver->cal_Energy();
}


template<typename TK, typename TR>
double LineSearch_ONs<TK, TR>::do_line_search(const bool start_guess)
{

    if(start_guess)
    {
        this->get_start_guess();
        std::cout << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;

        this->init_step = 1.0;
        this->Etotal_iter.clear();
        this->phi_0 = this->rdmft_solver->Etotal;
        this->Etotal_iter.push_back(this->rdmft_solver->Etotal);

        // return 0.0; // test !!!!!!!!!!
    }

    // std::cout << "\n******\n" << "iter in occ_num: " << iter << "\n" << std::endl;

    // // test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // this->phi_0 = this->cal_phi(this->var_x);

    // rdmft cal dE_docc_num, PARAM_ONs convert dE_docc_num to dE_dx
    this->cal_dE_dx(this->dE_dx);

    this->cal_pk_dphi0( (start_guess || this->iter == 0) );
    // std::cout << "\n******\n" << "ls, dphi_0: " << this->dphi_0 << "\n******\n" << std::endl;
    // if( std::abs(this->dphi_0) < 1e-8 )
    // {
    //     std::cout << "\n" << "dphi_0 is too small !!!!!!!  occ_num convergence ? " << "\n" << std::endl;
    //     return 0.0;
    // }

    if( this->iter != 0 )
    {
        this->init_step = 1.01 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
        this->init_step = std::min(1.0, this->init_step);
        // this->init_step = 1.0 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
        // std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;
    }

    std::vector<double> var_x_old = this->var_x;

    auto phi = [this](const double trial_alpha)
    {
        std::vector<double> x_new(this->var_x.size(), 0.0);
        this->step_size = trial_alpha;
        this->update_x(x_new);
        double trial_phi = this->cal_phi(x_new);
        return trial_phi;
    };

    auto dphi = [this]()
    {
        std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
        double trial_dphi = this->cal_dphi(trial_dE_dx);
        return trial_dphi;
    };

    double max_elem_pk = 0.0;
    for(int i=0; i<this->search_direction.size(); ++i)
    {
        max_elem_pk = std::max( max_elem_pk, std::abs(this->search_direction[i]) );
    }

    this->step_size = this->ls.do_line_search(phi, dphi, this->phi_0, this->dphi_0, max_elem_pk, this->init_step);

    // // have pk, use strong wolfe to find step_size
    // if(this->ls_condition == "swolfe") // PARAM.inp.ls_condition == "swolfe"
    // {
    //     this->strong_wolfe();
    // }
    // else if(this->ls_condition == "wolfe")
    // {
    //     this->wolfe();
    // }
    // else if(this->ls_condition == "exact")
    // {
    //     // this->exact_ls();
    //     this->strong_wolfe2();
    // }
    // else if(this->ls_condition == "test")
    // {
    //     std::cout << "\n" << "test ls" << "\n" << std::endl;

    //     this->step_size = this->ls.do_line_search(phi, dphi, this->phi_0, this->dphi_0, max_elem_pk, this->init_step);
    // }
    // else // fixed step
    // {
    //     this->step_size = PARAM.inp.ls_fixed_step;
    //     // this->strong_wolfe();
    // }
    // this->step_size = std::max(this->step_size, this->min_step_size);

    // update x_k+1 = x_k + step_size * p_k
    // can't use update_x(), because we need "+=" instead of "="
    for(int i=0; i<this->var_x.size(); ++i)
    {
        this->var_x[i] += this->step_size * this->search_direction[i]; 
    }

    std::cout << "\n" << "the final step_size by lineSearch_ONs: " << this->step_size << "\n" << std::endl;

    // if( a_equal_b(var_x_old, this->var_x) )
    // {
    //     std::cout << "\n" << "line_search_ONs: the increase in var_x is too small !!!!!!!!!!!!!!!" << "\n" << std::endl;
    //     return 0.0;
    // }

    // double temp_num = a_equal_b(var_x_old, this->var_x);
    // Parallel_Reduce::reduce_all(temp_num);
    // if( std::abs( temp_num ) > 1e-12 )
    // {
    //     std::cout << "\n" << "line_search_ONs: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
    //     return 0.0;
    // }

    // std::cout << "\n******\n" << "ls, do_line_search: strong_wolfe, 1.0" << "\n******\n" << std::endl;

    // convert x_k+1 to occ_num, rdmft_solver update occ_num, Hk, etc.
    this->phi_0 = this->cal_phi(this->var_x);   // has be calculated in swolfe() or zoom() ?

    // std::cout << "\n******\n" << "ls, do_line_search, phi_0: " << this->phi_0 << "\n******\n" << std::endl;

    // ModuleBase::matrix diff_occ_num = ( this->occ_number );
    // this->occ_number = this->param_occ_num->get_occ_number();
    // diff_occ_num -= this->occ_number;
    
    // std::abs( diff_occ_num )
    ModuleBase::matrix temp_occ( this->param_occ_num->get_occ_number() );
    std::vector<double> diff_occ_num(this->rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
    std::vector<double> diff_rate(this->rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
    for(int ik=0; ik<temp_occ.nr; ++ik)
    {
        for(int ib=0; ib<temp_occ.nc; ++ib)
        {
            diff_occ_num[ ik*PARAM.inp.nbands + ib ] = std::abs( this->occ_number(ik, ib) - temp_occ(ik, ib) );
                                                        // * (this->param_occ_num->get_num_symm_k())[ik];
            diff_rate[ ik*PARAM.inp.nbands + ib ] = std::abs( diff_occ_num[ ik*PARAM.inp.nbands + ib ] / this->occ_number(ik, ib) );
        }
    }
    this->occ_number = temp_occ;

    auto diff_occ_num_max = std::max_element(diff_occ_num.begin(), diff_occ_num.end());
    auto num = std::max_element(diff_rate.begin(), diff_rate.end());
    this->diff_rate_max = *num;
    
    if( iter%10 == 1 )
    {
        rdmft::printMatrix_pointer(temp_occ.nr, temp_occ.nc, diff_occ_num.data(), "after opti, diff_occ_num", 5);
    }

    this->Etotal_iter.push_back(this->phi_0);
    ++this->iter;

    return *diff_occ_num_max;
}


// There are two disadvantages of pseudo-precise line search:
// 1. armijo_step is too aggressive, consider this->step_size /= factor1^2
// 2. When performing binary search, there may be multiple minimum values ​​(peaks) between large armijo_stpe and 0.0
//    so strict armijo condition restrictions are also required when performing binary search.
//    It is best to make step_low and armijo_stpe exist in the same valley
// template<typename TK, typename TR>
// void LineSearch_ONs<TK, TR>::exact_ls()
// {
//     // trial_x = x_k(or this->var_x) + trial_step_size * p_k
//     std::vector<double> trial_x(this->var_x.size() ,0.0);
//     std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
//     double trial_phi = 0.0;
//     double trial_dphi = 0.0;
//     // double armijo_step = 0.0;
//     this->armijo_step = 0.0;

//     // the inital value of step_size is a hard problem and get (step_low, step_high) is not always correct, the following are just for debugging !!!!!!!
//     this->step_size = 0.01;
//     bool small_step = false;
//     for(int times=0; times<60; ++times)
//     {
//         if( this->step_size < 1e-6 )
//         {
//             std::cout << "\n" << "occNum optimization completed?" << this->step_size << std::endl;
//             this->step_size = this->min_step_size;
//             return;
//         }

//         this->update_x(trial_x);
//         trial_phi = this->cal_phi(trial_x);
//         if( trial_phi - this->phi_0 > 0 )
//         {
//             this->step_size *= 0.5;
//             small_step = true;
//             std::cout << "\n" << "initial step size needs to be reduced : " << this->step_size << std::endl;
//             continue;
//         }
//         else
//         {
//             // If 0.1 is not a very small step size in some cases
//             // then this strategy needs to be improved
//             // if( times>0 )
//             // {
//             //     std::cout << "\n" << "very small steps are required to reduce the energy, step_size: " << this->step_size << std::endl;
//             //     return;
//             // }
//             break;
//         }
//     }

//     // use armijo condition to ensure energy decrease
//     if( !small_step )
//     {
//         // if small_step is true, are armjio or exactLS still necessary???
//         this->step_size = 10.0;
//     }
//     double factor = 0.75;
//     // for(int it=0; it<30; ++it)
//     while(1)
//     {
//         std::cout << "\n" << "in Armijo, step size: " << this->step_size << std::endl;
//         this->update_x(trial_x);
//         trial_phi = this->cal_phi(trial_x);
//         if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )
//         {
//             this->step_size *= factor;
//         }
//         else
//         {
//             armijo_step = this->step_size;
//             std::cout << "\n" << "Armijo step size: " << armijo_step << std::endl;
//             break;
//         }
//     }


//     double step_low = 0.0;
//     double step_high = 0.0;

//     // to find zoom to use dichotomy
//     // might also update armijo_step!!!
//     bool find_zoom = false;
//     double factor1 = 0.0;
//     for(int times=0; times<60; ++times)
//     {
//         // this->update_x(trial_x);
//         // std::cout << "\n" << "step_size now: " << this->step_size << std::endl;
//         // trial_phi = this->cal_phi(trial_x);

//         // if( trial_phi - this->Etotal.back() > 0 )
//         // {
//         //     this->step_size *= 0.5;
//         //     std::cout << "\n" << "this->step_size: " << this->step_size << std::endl;
//         //     continue;
//         // }

//         trial_dphi = this->cal_dphi(trial_dE_dx);

//         if( trial_dphi > 0 )
//         {
//             step_high = this->step_size;
//             find_zoom = true;
//             break;
//         }
//         else
//         {
//             step_low = this->step_size;
//             if( times<10 )
//             {
//                 factor1 = 1.5;
//                 this->step_size *= factor1;
//             }
//             else
//             {
//                 factor1 = 1.1;
//                 this->step_size *= factor1;
//             }
//             std::cout << "\n" << "step_low: " << step_low << std::endl;
//         }

//         std::cout << "\n" << "step_size now: " << this->step_size << std::endl;
//         this->update_x(trial_x);
//         trial_phi = this->cal_phi(trial_x);

//         if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )
//         {
//             this->step_size /= factor1;
//             armijo_step = this->step_size;
//             std::cout << "\n" << "find zoom failed! the energy must drop in armijo steps, update as: " << this->step_size << "\n" << std::endl;
//             return;
//         }
//         else
//         {
//             armijo_step = this->step_size;
//         }
        
//         // Under the strict control of Armijo condition, is it possible for the situation here to occur?
//         for(int ik=0; ik<this->rdmft_solver->nk_total; ++ik)
//         {
//             if( this->rdmft_solver->occ_number(ik, 0) < 0.2 )
//             {
//                 this->step_size /= factor1;
//                 armijo_step = this->step_size;
//                 std::cout << "\n" << "exactLS failed! the energy must drop in armijo steps, update as:: " << this->step_size << "\n" << std::endl;
//                 return;
//             }
//         }

//     }

//     std::cout << "\n" << "Armijo step size: " << armijo_step << std::endl;
//     std::cout << "\n" << "step_low: " << step_low << "\nstep_high: " << step_high << "\ndphi_0: " << this->dphi_0 << std::endl;









//     // if find_zoom failed, the step_size now satisfies Armijo condition
//     if(find_zoom)
//     {
//         // (step_low + step_high)/2.0
//         for(int it=0; it<200; ++it)
//         {
//             this->step_size = ( step_low + step_high ) / 2.0;
//             std::cout << "\n" << "in dichotomy, step size: " << this->step_size << std::endl;
//             this->update_x(trial_x);
//             trial_phi = this->cal_phi(trial_x);
//             trial_dphi = this->cal_dphi(trial_dE_dx);

//             if( trial_dphi > 0 )
//             {
//                 step_high = this->step_size;
//             }
//             else
//             {
//                 step_low = this->step_size;
//             }

//             if( std::abs( step_high - step_low ) < 1e-3 )
//             {
//                 break;
//             }

//             if( it == 199 )
//             {
//                 std::cout << "\n******\n" << "exact line search, times too big: " << it << "\n******\n" << std::endl;
//             }
//         }
//         std::cout << "\n" << "step_size by exact line search: " << this->step_size << "\n" << std::endl;
//         // std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;
//     }

//     // if( !find_zoom || ( trial_phi - this->phi_0 ) > 0 )
//     // {
//     //     // this->step_size = 1.0;
//     //     // for(int it=0; it<30; ++it)
//     //     // {
//     //     //     this->update_x(trial_x);
//     //     //     trial_phi = this->cal_phi(trial_x);
            
//     //     //     if( trial_phi - this->Etotal.back() > 0 )
//     //     //     {
//     //     //         this->step_size *= 0.5;
//     //     //     }
//     //     //     else
//     //     //     {
//     //     //         std::cout << "\n" << "The energy must drop in small steps: " << this->step_size << "\n" << std::endl;
//     //     //         std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;
//     //     //         break;
//     //     //     }

//     //     //     if( it == 29 )
//     //     //     {
//     //     //         std::cout << "\n" << "!!!!!!!! The is something wrong in exact line search: Optimization completed?" << this->step_size << "\n" << std::endl;
//     //     //         this->step_size = this->min_step_size;
//     //     //         // assert(0);
//     //     //     }
//     //     // }

//     //     this->step_size = armijo_step;
//     //     std::cout << "\n" << "exactLS failed! the energy must drop in armijo steps: " << this->step_size << "\n" << std::endl;
//     // }



// }


// template<typename TK, typename TR>
// void LineSearch_ONs<TK, TR>::strong_wolfe2()
// {
//     // trial_x = x_k(or this->var_x) + trial_step_size * p_k
//     std::vector<double> trial_x(this->var_x.size() ,0.0);
//     std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
//     double trial_phi = 0.0;
//     double trial_dphi = 0.0;
//     // double armijo_step = 0.0;
//     this->armijo_step = 0.0;

//     // the inital value of step_size is a hard problem and get (step_low, step_high) is not always correct, the following are just for debugging !!!!!!!

//     this->step_size = this->init_step; // 1.0
//     // bool small_step = false;
//     for(int times=0; times<20; ++times)
//     {
//         if( this->step_size < 1e-8 )
//         {
//             std::cout << "\n" << "occNum optimization completed?" << this->step_size << std::endl;
//             this->step_size = this->min_step_size;
//             return;
//         }

//         this->update_x(trial_x);
//         trial_phi = this->cal_phi(trial_x);
//         if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )
//         {
//             if( times < 5 )
//             {
//                 this->step_size *= 0.5;
//             }
//             else
//             {
//                 this->step_size *= 0.2;
//             }
//             // small_step = true;
//             std::cout << "\n" << "initial step size needs to be reduced : " << this->step_size << std::endl;
//             // continue;
//         }
//         else
//         {
//             break;
//         }
//     }
//     this->step_size = std::max(this->step_size, this->min_step_size);
    
//     // strong wolfe condition
//     double step_size_old = 0.0;
//     double phi_old = this->phi_0;
//     double dphi_old = this->dphi_0;

//     double factor1 = 0.0;
//     for(int times=0; times<50; ++times)
//     {
//         if( times != 0 )
//         {
//             std::cout << "\n" << "step_size now: " << this->step_size << std::endl;
//             this->update_x(trial_x);
//             trial_phi = this->cal_phi(trial_x);
//             if( trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0 )  // && times > 0
//             {
//                 // this->step_size /= factor1;
//                 // armijo_step = this->step_size;
//                 // std::cout << "\n" << "find zoom failed! the energy must drop in armijo steps, update as: " << this->step_size << "\n" << std::endl;

//                 std::cout << "\n" << "Enter SW condition 1, zoom()" << "\n" << std::endl;
//                 this->zoom(step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
//                 return;
//             }
//             else
//             {
//                 this->armijo_step = this->step_size;
//                 std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;
//             }
//         }

//         trial_dphi = this->cal_dphi(trial_dE_dx);
//         if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
//         {
//             std::cout << "\n" << "Find SW step: " << this->step_size << "\n" << std::endl;
//             return;
//         }

//         if( trial_dphi >= 0 )
//         {
//             std::cout << "\n" << "Enter SW condition 3, zoom()" << "\n" << std::endl;
//             this->armijo_step = this->step_size;
//             std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;

//             this->zoom(this->step_size, trial_phi, step_size_old, phi_old, trial_dphi);  // ? which one ???
//             // this->zoom(step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
//             return;
//         }

//         step_size_old = this->step_size;
//         phi_old = trial_phi;
//         dphi_old = trial_dphi;

//         // if( times<10 )
//         // {
//         //     factor1 = 1.1;
//         //     this->step_size *= factor1;
//         // }
//         // else
//         {
//             factor1 = 1.1;
//             this->step_size *= factor1;
//         }
//         this->step_size = std::min(this->step_size, this->max_step_size);


//         // Under the strict control of Armijo condition, is it possible for the situation here to occur?
//         // need!
//         // If a is too large, it may directly cross the unique minimum point, and the Armijo condition is satisfied in a very wide range.
//         // This is prone to problems and the step size should be reduced.
//         // Can zoom() accept non-monotonic {alpha_i} ?
//         for(int ik=0; ik<this->rdmft_solver->nk_total; ++ik)
//         {
//             if( this->rdmft_solver->occ_number(ik, 0) < 0.4 )
//             {
//                 armijo_step = this->step_size / factor1;
//                 this->step_size /= factor1 * 1.05; // select 1.05 != factor1
//                 // armijo_step = this->step_size;

//                 // this->step_size /= factor1;
//                 // std::cout << "\n" << "exactLS failed! the energy must drop in armijo steps, update as:: " << this->step_size << "\n" << std::endl;
//                 // return;
//             }
//         }

//     }

//     this->armijo_step = this->step_size;
//     std::cout << "\n" << "Strong Wolfe failed! the energy must drop in armijo steps, update as:: " << this->armijo_step << "\ndphi_0: " << this->dphi_0 << std::endl;

// }





// template<typename TK, typename TR>
// void LineSearch_ONs<TK, TR>::strong_wolfe()
// {
//     // std::cout << "\n" << "Enter strong_wolfe()" << "\n" << std::endl;

//     // big problem here, in theory, step_size_0 should -> 0 !!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//     // initial step_size before each iteration
//     this->step_size = (0.02 < this->max_step_size) ? 0.02 : this->max_step_size/2.0;
//     // this->step_size = (0.1 < this->max_step_size) ? 0.1 : this->max_step_size/2.0;

//     // 0: represents the relevant quantity under x_k, that is, var_x
//     // phi_0, dphi_0 have obtained
//     // old: represents the relevant quantity under the last trial_step_size/trial_x
//     double step_size_old =0.0;
//     double phi_old = this->phi_0;
//     double dphi_old = this->dphi_0;

//     // trial_x = x_k(or this->var_x) + trial_step_size * p_k
//     std::vector<double> trial_x(this->var_x.size() ,0.0);
//     std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
//     double trial_phi = 0.0;
//     double trial_dphi = 0.0;

//     for(int times=0; times<=20; ++times)
//     {
//         std::cout << "\n" << "in strong_wolfe(), while: " << times << ", step_size: " << this->step_size << "\n" << std::endl;
        
//         if( this->step_size > this->max_step_size )
//         {
//             std::cout << "\n******\n" << "in strong wolfe, step_size > max_step_size" << "\n******\n" << std::endl;
//             break;
//         }

//         if(times != 1)
//         {
//             // // since max is too large, the dichotomy is too extreme
//             // // and it is easy to fall into a saddle point when solving mu
//             // // i.e., the number of particles is not conserved
//             // this->step_size = ( this->step_size + this->max_step_size ) / 2.0; // needs improvement, using quadratic or cubic, currently using dichotomy

//             // std::cout << "\n" << "before quadratic interpolation" << "\n" << std::endl;
//             // quadratic interpolation, temp_step = -b/(2a) in quadratic function
//             // improved using cubic interpolation ?
//             double temp_step = - this->dphi_0 * this->step_size * this->step_size / ( trial_phi - this->phi_0 - this->dphi_0 * this->step_size ) / 2.0;
//             this->step_size = ( 1.1 * this->step_size < temp_step ) ? temp_step : 1.1 * this->step_size;
//             // this->step_size = ( 1.5 * this->step_size < temp_step ) ? temp_step : 1.5 * this->step_size;
//             this->step_size = ( this->step_size < this->max_step_size ) ? this->step_size : this->max_step_size;
//             std::cout << "\n" << "quadratic interpolation in strong wolfe, temp_step:" << temp_step << "\n" << std::endl;

//         }

//         for(int i=0; i<trial_x.size(); ++i)
//         {
//             trial_x[i] = this->var_x[i] + this->step_size * this->search_direction[i];
//         }

//         // double temp_num = a_equal_b(trial_x, this->var_x);
//         // Parallel_Reduce::reduce_all(temp_num);
//         // if( std::abs( temp_num ) > 1e-12 )
//         if( a_equal_b(trial_x, this->var_x) )
//         {
//             std::cout << "\n" << "line_search_ONs: the increase in var_x is too small !!!!!!!!!!!!" << "\n" << std::endl;
//             continue;
//         }

//         // std::cout << "\n" << "SW, before cal_phi()" << "\n" << std::endl;

//         trial_phi = this->cal_phi(trial_x);

//         if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi > phi_old)
//         {
//             std::cout << "\n" << "Enter SW condition 1" << "\n" << std::endl;
//             this->zoom(step_size_old, phi_old, this->step_size, trial_phi, dphi_old);
//             break;
//         }

//         // std::cout << "\n" << "before cal_dphi()" << "\n" << std::endl;

//         trial_dphi = this->cal_dphi(trial_dE_dx);

//         if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
//         {
//             std::cout << "\n" << "Enter SW condition 2" << "\n" << std::endl;
//             break;
//         }

//         if( trial_dphi >= 0 )
//         {
//             std::cout << "\n" << "Enter SW condition 3" << "\n" << std::endl;
//             this->zoom(this->step_size, trial_phi, step_size_old, phi_old, trial_dphi);
//             break;
//         }

//         step_size_old = this->step_size;
//         phi_old = trial_phi;
//         dphi_old = trial_dphi;

//         if( times==20 )
//         {
//             std::cout << "\n******\n" << "strong wolfe times too big: " << times << "\n******\n" << std::endl;
//         }
//     }
// }


// // zoom() does not require step_size_high>step_size_low to work
// // but the gradient dphi_low used for acceleration calculation must correspond to step_size_low
// template<typename TK, typename TR>
// void LineSearch_ONs<TK, TR>::zoom(double step_size_low, double phi_low, double step_size_high, double phi_high, double dphi_low)
// {
//     double alpha_lo = step_size_low;
//     double alpha_hi = step_size_high;
//     double f_lo = phi_low;
//     double f_hi = phi_high;
//     double df_lo = dphi_low;

//     double df_hi = 0.0;

//     // trial_x = x_k(or this->var_x) + trial_step_size * p_k
//     std::vector<double> trial_x(this->var_x.size() ,0.0);
//     std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);

//     int times = 0;
//     while(1)
//     {
//         // improved using cubic interpolation ?
//         double diff_alpha = alpha_hi - alpha_lo;
//         double incr_alpha = - df_lo * diff_alpha * diff_alpha / ( f_hi - (f_lo + df_lo * diff_alpha) ) / 2.0;
//         if( incr_alpha < diff_alpha * this->ls_armijo_c1 )
//         {
//             incr_alpha = diff_alpha * this->ls_armijo_c1;
//         }
//         if( incr_alpha > diff_alpha * this->ls_armijo_c2 )
//         {
//             incr_alpha = diff_alpha * this->ls_armijo_c2;
//         }
//         // this->step_size = alpha_lo + incr_alpha;
//         std::cout << "\n" << "incr_alpha: " << incr_alpha << "\nnew step_size by incr_alpha: " << alpha_lo + incr_alpha << "\n" << std::endl;


//         // needs improvement, using quadratic or cubic, currently using dichotomy
//         this->step_size = (alpha_lo + alpha_hi)/2.0;
//         std::cout << "\n" << "in ZOOM(), while: " << times << ", step_size: " << this->step_size << "\n" << std::endl;

//         this->update_x(trial_x);
//         double trial_phi = this->cal_phi(trial_x);

//         if( (trial_phi > this->phi_0 + this->ls_wolfe_c1 * this->step_size * this->dphi_0) || trial_phi >= f_lo )
//         {
//             alpha_hi = this->step_size;
//             f_hi = trial_phi;
//             std::cout << "\n" << "Enter ZOOM condition 1\nalpha_hi: " << alpha_hi << std::endl;
//         }
//         else
//         {
//             this->armijo_step = this->step_size;
//             std::cout << "\n" << "Armijo step size: " << this->armijo_step << std::endl;

//             double trial_dphi = this->cal_dphi(trial_dE_dx);
//             if( std::abs(trial_dphi) <= -this->ls_wolfe_c2 * this->dphi_0 )
//             {
//                 std::cout << "\n" << "ZOOM() get\nstep_size: " << std::endl;
//                 return;
//             }

//             if( trial_dphi * (alpha_hi - alpha_lo) >= 0 )
//             {
//                 std::cout << "\n" << "Enter ZOOM condition 3, exchange high_low" << "\n" << std::endl;
//                 alpha_hi = alpha_lo;
//                 f_hi = f_lo;

//                 df_hi = df_lo;
//             }

//             alpha_lo = this->step_size;
//             f_lo = trial_phi;
//             df_lo = trial_dphi;
//         }

//         std::cout << "\nupdate\n" << "alpha_hi: " << alpha_hi << "\nalpha_lo: " << alpha_lo << std::endl;
//         std::cout << "\nupdate\n" << "-ls_w_c2 * dphi_0: " << -this->ls_wolfe_c2 * this->dphi_0 << "\n|dphi_low|: " << df_lo << std::endl;

//         // // need it?
//         // if( alpha_lo > alpha_hi )
//         // {
//         //     std::swap(alpha_lo, alpha_hi);
//         //     std::swap(f_lo, f_hi);
//         //     std::swap(df_lo, df_hi);
//         // }
//         // std::cout << "\n" << "By comparing the size, exchange high_low" << "\nalpha_hi: " << alpha_hi << "\nalpha_lo: " << alpha_lo << std::endl;

//         ++times;
//         if( times >= 20 || this->step_size <= this->min_step_size )
//         {
//             // std::cout << "\n******\n" << "zoom times too big: " << times << "\n******\n" << std::endl;
//             this->step_size = this->armijo_step;
//             std::cout << "\n" << "zoom() failed! the energy must drop in armijo steps, update as:: " << this->armijo_step << "\n" << std::endl;
//             return;
//         }

//     }

// }


// // to be developed
// template<typename TK, typename TR>
// void LineSearch_ONs<TK, TR>::wolfe()
// {

// }


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::update_x(std::vector<double>& x_new)
{
    std::fill(x_new.begin(), x_new.end(), 0.0);
    for(int i=0; i<x_new.size(); ++i)
    {
        x_new[i] = this->var_x[i] + this->step_size * this->search_direction[i];
    }
}

template<typename TK, typename TR>
double LineSearch_ONs<TK, TR>::cal_phi(std::vector<double>& x_new)
{
    // convert x_k+1 to occ_num
    this->param_occ_num->update_x_occ_num(x_new);

    // rdmft_solver update occ_num, Hk, etc.
    // ModuleBase::matrix occ_number( this->param_occ_num->get_occ_number() );
    ModuleBase::matrix occ_number = this->param_occ_num->get_occ_number();

    this->rdmft_solver->update_elec( &occ_number );

    // cal phi(alpha) = E(x_k + alpha * p_k)
    double phi = this->rdmft_solver->cal_Energy();

    return phi;
}


template<typename TK, typename TR>
double LineSearch_ONs<TK, TR>::cal_dphi(std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr)
{

    this->cal_dE_dx(dE_dx_new, x_new_ptr);

    double dphi = 0.0;
    rdmft::Tgemm_lapack( dE_dx_new.data(), this->search_direction.data(), &dphi, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');

    return dphi;
}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::cal_dE_dx(std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr)
{
    if( x_new_ptr != nullptr ) { this->cal_phi( *x_new_ptr ); }

    // std::cout << "\n******\n" << "in cal_dE_dx" << std::endl;

    // rdmft cal dE_docc_num
    this->rdmft_solver->cal_E_grad_occ_num();

    // PARAM_ONs: convert dE_docc_num to dE_dx (x in PARAM_ONs is the latest, that is, it is consistent with dE_dx)
    std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
    this->param_occ_num->get_dE_dx(dE_docc_num, dE_dx_new);
    
    // test
    double norm_dE_dx = 0.0;
    for(int i=0; i<dE_dx_new.size(); ++i)
    {
        norm_dE_dx += dE_dx_new[i] * dE_dx_new[i];
    }
    norm_dE_dx = std::sqrt(norm_dE_dx);
    // std::cout << "\n***\nin ls, norm_dE_dx: " << norm_dE_dx  << "\n***\n" << std::endl;

    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, dE_dx_new.data(), "look, dE_dx_new");
    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->var_x.data(), "now var_x", 10);
    // std::cout << "\n******\n" << std::endl;
}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::cal_pk_dphi0(const bool new_landscape)
{
    // get pk: PARAM_ONs provide var_x and dE_dx to BFGS
    if( PARAM.inp.precond_occ_num && this->iter > 5 ) // && this->iter > 5 
    {
        std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
        std::vector<double> d2E_dx2(rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
        this->param_occ_num->get_d2E_dx2(dE_docc_num, d2E_dx2);

        this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, new_landscape, &d2E_dx2);
    }
    else
    {
        this->bfgs_opti_x.get_pk(this->dE_dx, this->var_x, this->search_direction, new_landscape);
    }

    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->search_direction.data(), "search_direction");

    // get dphi_0
    rdmft::Tgemm_lapack( this->dE_dx.data(), this->search_direction.data(), &this->dphi_0, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');
}



template class LineSearch_ONs<double, double>;
template class LineSearch_ONs<std::complex<double>, double>;
template class LineSearch_ONs<std::complex<double>, std::complex<double>>;

// template class LineSearch_ONs<double, RDMFT<double, double>>;
// template class LineSearch_ONs<double, RDMFT<std::complex<double>, double>>;
// template class LineSearch_ONs<double, RDMFT<std::complex<double>, std::complex<double>>>;
// template class LineSearch_ONs<std::complex<double>, RDMFT<double, double>>;
// template class LineSearch_ONs<std::complex<double>, RDMFT<std::complex<double>, double>>;
// template class LineSearch_ONs<std::complex<double>, RDMFT<std::complex<double>, std::complex<double>>>;


}

