//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================


#include "module_rdmft/optimizer/cg_method.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

#include "module_rdmft/rdmft_tools.h" // temp
#include <iostream> // temp

namespace rdmft
{


template<typename TX>
CG_method<TX>::CG_method()
{

}


template<typename TX>
CG_method<TX>::~CG_method()
{
    
}


template<typename TX>
void CG_method<TX>::init(const int dim_in, const double precond_eps_in)
{
    rdmft::Opti_method<TX>::init(dim_in, precond_eps_in);

    this->precond_grad.resize(this->dim, 0.0);
    this->precond_grad_old.resize(this->dim, 1.0);
    this->diff_precond_grad.resize(this->dim, 0.0);
}


template<typename TX>
void CG_method<TX>::get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool new_landscape, const std::vector<TX>* d2E_dx2)
{
    const bool precond = (d2E_dx2 == nullptr) ? false : true;
    if( precond )
    {
        for(int i=0; i<this->dim; ++i)
        {
            this->precond_grad[i] = dE_dx_new[i] / std::max( this->precond_eps, std::abs((*d2E_dx2)[i]) );
            // this->precond_grad[i] = dE_dx_new[i] / (*d2E_dx2)[i];

            // TX num_temp = this->precond_eps > std::abs((*d2E_dx2)[i]) ? this->precond_eps: (*d2E_dx2)[i];
            // this->precond_grad[i] = dE_dx_new[i] / num_temp;
        }
    }
    else
    {
        this->precond_grad = dE_dx_new;
    }

    this->cal_beta(dE_dx_new, new_landscape, precond);

    // get new pk
    for(int i=0; i<this->dim; ++i)
    {
        pk[i] = - (this->precond_grad[i] + this->beta * this->search_direction[i]);

        this->search_direction[i] = - pk[i];
        this->precond_grad_old[i] = this->precond_grad[i];
        this->dE_dx[i] = dE_dx_new[i];
    }

    ++this->iter;
}


template<typename TX>
void CG_method<TX>::cal_beta(const std::vector<TX>& dE_dx_new, const bool new_landscape, const bool precond)
{
    if( new_landscape )
    {
        this->beta = 0.0;
        this->iter = 0;
        return;
    }
    else
    {
        const char op_tran = std::is_same<TX, std::complex<double>>::value ? 'C' : 'T';

        // determine whether to restart optimization
        if( precond )
        {
            // cal something
            TX pT_g = 0.0;
            TX pT_g_old = 0.0;
            // TX nega_one = -1.0;
            TX posi_one = 1.0;
            rdmft::Tgemm_lapack( this->search_direction.data(), dE_dx_new.data(), &pT_g, 1, 1, this->dim, op_tran, 'N', posi_one );
            rdmft::Tgemm_lapack( this->search_direction.data(), this->dE_dx.data(), &pT_g_old, 1, 1, this->dim, op_tran, 'N', posi_one );

            // if( std::abs( std::real(pT_g) ) > 0.2 * std::abs( std::real(pT_g_old) ) )
            if( std::abs( std::real(pT_g) ) > 0.2 * std::real(pT_g_old) )
            {
                // print something 
                std::cout << "\n\nrestart CG: \npT_g: " << pT_g << "\npT_g_old: " << pT_g_old << "\n" << std::endl;

                this->beta = 0.0;
                ++this->num_restart_skip;
                return;
            }
        }

        // cal beta
        if( this->beta_type == "PR" )
        {
            // cal diff_z = z_k - z_k-1
            for(int i=0; i<this->dim; ++i)
            {
                this->diff_precond_grad[i] = this->precond_grad[i] - this->precond_grad_old[i];
            }

            // cal beta_k = g_k^T * diff_z / g_k-1^T * z_k-1
            TX gT_diff_z = 0.0;
            TX gT_z_old = 0.0;
            rdmft::Tgemm_lapack( dE_dx_new.data(), this->diff_precond_grad.data(), &gT_diff_z, 1, 1, this->dim, op_tran, 'N' );
            rdmft::Tgemm_lapack( this->dE_dx.data(), this->precond_grad_old.data(), &gT_z_old, 1, 1, this->dim, op_tran, 'N' );

            this->beta = std::real( gT_diff_z / gT_z_old );
        }
        else
        {
            std::cout << "\n\n CG_method only support beta_type == 'PR' now \n\n" << std::endl;
            assert(0);
        }


    }


}




template class CG_method<double>;
template class CG_method<std::complex<double>>;

}