//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================

#include <algorithm>
#include "module_rdmft/optimizer/line_search_ONs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"
#include "module_rdmft/optimizer/parameterize_ONs/softmax.h"
#include "module_rdmft/optimizer/bfgs_method.h"
#include "module_rdmft/optimizer/cg_method.h"

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

    if( PARAM.inp.precond_occ_num )
    {
        this->precond_bfgs = std::make_unique< rdmft::BFGS_method<double> >();
        this->precond_bfgs->init(rdmft_solver->nk_total*PARAM.inp.nbands);
    }

    if( PARAM.inp.occ_num_opti == "cg" )
    {
        this->x_optimizer = std::make_unique< rdmft::CG_method<double> >();
        this->ls.get_options().ls_wolfe_c2 = PARAM.inp.ls_wolfe_c2_cg;
        // if( PARAM.inp.precond_occ_num )
        // {
        //     this->precond_bfgs = std::make_unique< rdmft::BFGS_method<double> >();
        //     this->precond_bfgs->init(rdmft_solver->nk_total*PARAM.inp.nbands);
        // }
    }
    else
    {
        this->x_optimizer = std::make_unique< rdmft::BFGS_method<double> >();
    }
    this->x_optimizer->init(rdmft_solver->nk_total*PARAM.inp.nbands, 1e-8);
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
    this->scaling_P.resize(rdmft_solver->nk_total * PARAM.inp.nbands);
    this->scaling_P_old.resize(rdmft_solver->nk_total * PARAM.inp.nbands);

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
    ++this->num_restart;
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
    this->restart_opti();
}


template<typename TK, typename TR>
double LineSearch_ONs<TK, TR>::do_line_search(const bool start_guess)
{

    if(start_guess)
    {
        this->get_start_guess();
        std::cout << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;

        // return 0.0; // test !!!!!!!!!!
    }

    // std::cout << "\n******\n" << "iter in occ_num: " << iter << "\n" << std::endl;

    // // test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // this->phi_0 = this->cal_phi(this->var_x);

    bool new_landscape = ( start_guess || this->iter == 0);
    // bool new_landscape = ( start_guess || this->iter == 0 || (PARAM.inp.precond_type != 1 && this->iter%50 == 0) ) ? true: false;

    // rdmft cal dE_docc_num, PARAM_ONs convert dE_docc_num to dE_dx
    this->cal_dE_dx(this->dE_dx);

    this->cal_pk_dphi0( new_landscape );
    // std::cout << "\n******\n" << "ls, dphi_0: " << this->dphi_0 << "\n******\n" << std::endl;

    if( this->iter != 0 )
    {
        this->init_step = 1.01 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
        this->init_step = std::min(1.0, std::abs(this->init_step));
        // this->init_step = 1.0 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
        // std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;
    }

    // std::vector<double> var_x_old = this->var_x;

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

    // this->step_size = std::max(this->step_size, this->min_step_size);

    // update x_k+1 = x_k + step_size * p_k
    // can't use update_x(), because we need "+=" instead of "="
    for(int i=0; i<this->var_x.size(); ++i)
    {
        this->var_x[i] += this->step_size * this->search_direction[i]; 
    }

    std::cout << "\n" << "the final step_size by lineSearch_ONs: " << this->step_size << "\n" << std::endl;


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
    
    // if( iter%10 == 1 )
    // {
    //     rdmft::printMatrix_pointer(temp_occ.nr, temp_occ.nc, diff_occ_num.data(), "after opti, diff_occ_num", 5);
    // }

    this->Etotal_iter.push_back(this->phi_0);
    ++this->iter;

    return *diff_occ_num_max;
}


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
    
    // // test
    // double norm_dE_dx = 0.0;
    // for(int i=0; i<dE_dx_new.size(); ++i)
    // {
    //     norm_dE_dx += dE_dx_new[i] * dE_dx_new[i];
    // }
    // norm_dE_dx = std::sqrt(norm_dE_dx);
    // std::cout << "\n***\nin ls, norm_dE_dx: " << norm_dE_dx  << "\n***\n" << std::endl;

    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, dE_dx_new.data(), "look, dE_dx_new");
    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->var_x.data(), "now var_x", 10);
    // std::cout << "\n******\n" << std::endl;
}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::cal_pk_dphi0(const bool new_landscape)
{
    //
    this->cal_grad_norm();

    // get pk: PARAM_ONs provide var_x and dE_dx to BFGS
    if( PARAM.inp.precond_occ_num ) // && this->iter > 5 
    {
        std::vector<double> dE_docc_num = this->rdmft_solver->get_dE_docc_num();
        std::vector<double> d2E_dx2(rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
        this->param_occ_num->get_d2E_dx2(dE_docc_num, d2E_dx2);

        // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, d2E_dx2.data(), "d2E_dx2", 10);

        // double min_grad2 = *std::min_element(d2E_dx2.begin(), d2E_dx2.end());

        // // double second_min_grad2 = std::numeric_limits<double>::infinity();
        // // for (int j=0; j<d2E_dx2.size(); ++j)
        // // {
        // //     if (d2E_dx2[j] > min_grad2 && d2E_dx2[j] < second_min_grad2)
        // //     {
        // //         second_min_grad2 = d2E_dx2[j];
        // //     }
        // // }

        // double average_grad2 = 0.0;
        // if( min_grad2 < 0 )
        // {
        //     for(int i=0; i<d2E_dx2.size(); ++i)
        //     {
        //         d2E_dx2[i] -= min_grad2;
        //         // d2E_dx2[i] = std::max( d2E_dx2[i], second_min_grad2 - min_grad2 );
        //         average_grad2 += std::abs(d2E_dx2[i]);
        //     }
        // }
        // average_grad2 /= d2E_dx2.size();
        // if( min_grad2 < 0 )
        // {
        //     for(int i=0; i<d2E_dx2.size(); ++i)
        //     {
        //         d2E_dx2[i] += average_grad2 * 0.001;
        //     }
        // }

        // double min_shift_grad2 = compute_min_shift(d2E_dx2, 2.0);
        // rdmft::shift_precond( d2E_dx2 , min_shift_grad2, 0.001);
        // std::cout << "\n\nmin_shift_grad2 in ONs-opti: " << min_shift_grad2 << "\n" << std::endl;



        rdmft::shift_precond( d2E_dx2 , 1e-6, 0.0 );

        // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, d2E_dx2.data(), "d2E_dx2 after shifting", 10);

        if( PARAM.inp.precond_type == 1 )
        {
            if( PARAM.inp.occ_num_opti == "cg" ) // || PARAM.inp.occ_num_opti == "bfgs" is test
            {
                std::vector<double> diag_Bk(rdmft_solver->nk_total * PARAM.inp.nbands, 0.0);
                this->precond_bfgs->get_diag_Bk(this->dE_dx, this->var_x, diag_Bk, new_landscape);
                const double g_factor = PARAM.inp.precond_g;

                // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, diag_Bk.data(), "diag_Bk", 10);

                // double min_Bk = *std::min_element(diag_Bk.begin(), diag_Bk.end());

                // // double second_min_Bk = std::numeric_limits<double>::infinity();
                // // for (int j=0; j<diag_Bk.size(); ++j)
                // // {
                // //     if (diag_Bk[j] > min_Bk && diag_Bk[j] < second_min_Bk)
                // //     {
                // //         second_min_Bk = diag_Bk[j];
                // //     }
                // // }
            
                // double average_Bk = 0.0;
                // if( min_Bk < 0 )
                // {
                //     for(int i=0; i<diag_Bk.size(); ++i)
                //     {
                //         diag_Bk[i] -= min_Bk;
                //         // diag_Bk[i] = std::max(  diag_Bk[i], second_min_Bk - min_Bk );
                //         average_Bk += std::abs(diag_Bk[i]);
                //     }
                // }
                // average_Bk /= diag_Bk.size();

                // if( min_Bk < 0 )
                // {
                //     for(int i=0; i<diag_Bk.size(); ++i)
                //     {
                //         diag_Bk[i] += average_Bk * 0.001;
                //     }
                // }
                // // // print

                // // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, diag_Bk.data(), "diag_Bk after shifting", 10);
                // double min_shift_Bk = compute_min_shift(diag_Bk, 2.0);
                // rdmft::shift_precond( diag_Bk , min_shift_Bk, 0.001);
                // std::cout << "\n\nmin_shift_Bk in ONs-opti: " << min_shift_Bk << "\n" << std::endl;


                rdmft::shift_precond( diag_Bk , 1e-6, 0.0 );

                for(int i=0; i<diag_Bk.size(); ++i)
                {
                    // d2E_dx2[i] = g_factor * diag_Bk[i] + (1 - g_factor) * d2E_dx2[i];
                    d2E_dx2[i] = g_factor * std::abs(diag_Bk[i]) + (1 - g_factor) * std::abs(d2E_dx2[i]);
                }
            }

            this->x_optimizer->get_pk(this->dE_dx, this->var_x, this->search_direction, new_landscape, &d2E_dx2);
        }
        else
        {
            if( new_landscape || this->iter%5 == 0 )
            {
                for(int i=0; i<this->scaling_P.size(); ++i)
                {
                    this->scaling_P[i] = std::sqrt( std::max( 1e-8, std::abs(d2E_dx2[i]) ) );
                }
            }

            if( !new_landscape && this->iter%5 == 0 )
            {
                // get transport mat: T = S_k+1 Sk^-1
                for(int i=0; i<this->scaling_P.size(); ++i)
                {
                    this->scaling_P_old[i] = this->scaling_P[i] / this->scaling_P_old[i];
                }
                this->x_optimizer->transport(this->scaling_P_old);
            }
            this->scaling_P_old = this->scaling_P;

            std::vector<double> var_u= this->var_x;
            std::vector<double> dE_du= this->dE_dx;

            for(int i=0; i<this->scaling_P.size(); ++i)
            {
                var_u[i] *= this->scaling_P[i];
                dE_du[i] /= this->scaling_P[i];
            }
            this->x_optimizer->get_pk(dE_du, var_u, this->search_direction, new_landscape);
            for(int i=0; i<this->search_direction.size(); ++i)
            {
                this->search_direction[i] /= this->scaling_P[i];
            }
        }
    }
    else
    {
        this->x_optimizer->get_pk(this->dE_dx, this->var_x, this->search_direction, new_landscape);
    }

    // rdmft::printMatrix_pointer(this->rdmft_solver->nk_total, PARAM.inp.nbands, this->search_direction.data(), "search_direction");

    // get dphi_0
    rdmft::Tgemm_lapack( this->dE_dx.data(), this->search_direction.data(), &this->dphi_0, 1, 1, rdmft_solver->nk_total * PARAM.inp.nbands , 'T');
}


template<typename TK, typename TR>
void LineSearch_ONs<TK, TR>::cal_grad_norm()
{
    this->grad_norm = 0.0;
    for(int i=0; i<this->dE_dx.size(); ++i)
    {
        this->grad_norm += std::norm(this->dE_dx[i]);
    }
    this->grad_norm = std::sqrt(this->grad_norm);
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

