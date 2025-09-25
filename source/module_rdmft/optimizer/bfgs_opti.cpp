//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================


#include "module_rdmft/optimizer/bfgs_opti.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

#include "module_rdmft/rdmft_tools.h" // temp
#include <iostream> // temp

namespace rdmft
{


template<typename TX>
BFGS_Opti<TX>::BFGS_Opti()
{

}


template<typename TX>
BFGS_Opti<TX>::~BFGS_Opti()
{
    
}


template<typename TX>
void BFGS_Opti<TX>::init(const int dim_in, const int nk_total_in)
{

    this->dim = dim_in;

    // temp
    this->nk_total = nk_total_in;
    this->nbands = dim_in / nk_total_in;

    // x0.resize(this->dim);
    // x1.resize(this->dim);
    var_x.resize(this->dim, 0.0);
    diff_x.resize(this->dim, 0.0);
    // dE_dx0.resize(this->dim);
    // dE_dx1.resize(this->dim);
    dE_dx.resize(this->dim);
    diff_grad.resize(this->dim);
    Hk.resize( this->dim * this->dim, 0.0 );
    search_direction.resize(this->dim);

    rho_diffX_diffGrad.resize( this->dim * this->dim );
    rho_diffX_diffX_T.resize( this->dim * this->dim );
}


template<typename TX>
void BFGS_Opti<TX>::get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool new_landscape, const std::vector<TX>* d2E_dx2)
{
    // set some vars zero?

    // get Hk
    this->cal_Hk(dE_dx_new, x_new, new_landscape);
    // if( new_landscape || a_equal_b(x_new, this->var_x) )
    // {
    //     std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    //     std::fill(this->dE_dx.begin(), this->dE_dx.end(), 0.0);
    //     std::fill(this->search_direction.begin(), this->search_direction.end(), 0.0);
    //     // identity matrix or other method to initialize H0
    //     std::fill(Hk.begin(), Hk.end(), 0.0);
    //     for(int i=0; i<this->dim; ++i) { Hk[i*(this->dim) + i] = 1.0; }

    //     // delete in the future
    //     if( !new_landscape )
    //     {
    //         std::cout << "\n" << "BFGS: the increase in var_x is too small !!!!!!!!" << "\n" << std::endl;
    //         // assert(0);
    //     }
    // }
    // else
    // {
    //     for(int j=0; j<x_new.size(); ++j)
    //     {
    //         // diff_x, sk = x_k+1 - x_k
    //         this->diff_x[j] = x_new[j] - this->var_x[j];
    //         // diff_grad, yk = (dE_dx)_k+1 - (dE_dx)_k
    //         this->diff_grad[j] = dE_dx_new[j] - this->dE_dx[j];
    //     }

    //     if( PARAM.inp.print_BFGS_Hk )
    //     {
    //         rdmft::printMatrix_pointer(nk_total, nbands, this->diff_x.data(), "in BFGS_Opti, diff_x");
    //         rdmft::printMatrix_pointer(nk_total, nbands, this->diff_grad.data(), "in BFGS_Opti, diff_grad");
    //     }

    //     char op_tran = 'T';
    //     if constexpr (std::is_same<TX, std::complex<double>>::value)
    //     {
    //         op_tran = 'C';
    //     }

    //     // cal rho = 1/( diffGrad^T * diffX )
    //     TX rho_temp = 0.0;
    //     rdmft::Tgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &rho_temp, 1, 1, this->dim, op_tran, 'N');
    //     this->rho = 1.0/std::real(rho_temp);

    //     if( std::real(rho_temp) < 1e-10 ) { } //////////////////////////

    //     std::cout << "******\n" << "in BFGS_Opti::get_pk(), 1/rho = y^/dagger s: " << rho_temp << "\n******" << std::endl;
    //     std::cout << "******\n" << "in BFGS_Opti::get_pk(), rho: " << this->rho << "\n******" << std::endl;

    //     rho_temp = this->rho;
    //     // cal rho * diffX * diffGrad^T, I - rho * diffX * diffGrad^T
    //     rdmft::Tgemm_lapack( this->diff_x.data(), this->diff_grad.data(), this->rho_diffX_diffGrad.data(), this->dim, this->dim, 1, 'N', op_tran, -(rho_temp) );
    //     for(int i=0; i<this->dim; ++i) { this->rho_diffX_diffGrad[i*this->dim + i] += 1.0; }

    //     // cal rho * diffX * diffX^T
    //     rdmft::Tgemm_lapack( this->diff_x.data(), this->diff_x.data(), this->rho_diffX_diffX_T.data(), this->dim, this->dim, 1, 'N', op_tran, rho_temp );

    //     // cal H_k+1 = (I - rho * diffX * diffGrad^T) * H_k * (I - rho * diffX * diffGrad^T)^T + rho * diffX * diffX^T
    //     std::vector<TX> H_tmp = this->Hk;
    //     rdmft::Tgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), H_tmp.data(), this->dim, this->dim, this->dim );
    //     rdmft::Tgemm_lapack( H_tmp.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), this->dim, this->dim, this->dim, 'N', op_tran );
    //     // rdmft::Tgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), this->Hk.data(), this->dim, this->dim, this->dim );
    //     // rdmft::Tgemm_lapack( this->Hk.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), this->dim, this->dim, this->dim, 'N', op_tran );
    //     for(int i=0; i<this->Hk.size(); ++i) { this->Hk[i] += this->rho_diffX_diffX_T[i]; }

    // }

    if( PARAM.inp.print_BFGS_Hk )
    {
        rdmft::printMatrix_pointer(this->dim, this->dim, this->Hk.data(), "BFGS: Hk", 10);
    }

    // cal search_direction, p_k+1 = -H_k+1 * (dE_dx)_k+1
    // property: H = H^T, also depends on the initial guess H0!
    TX nega_one = -1.0;
    if( d2E_dx2 == nullptr )
    {
        rdmft::Tgemm_lapack( this->Hk.data(), dE_dx_new.data(), this->search_direction.data(), this->dim, 1, this->dim, 'N', 'N', nega_one );
    }
    else
    {
        std::vector<TX> dE_dx_precond = dE_dx_new;
        for(int i=0; i<dE_dx_precond.size(); ++i)
        {
            // dE_dx_precond[i] /= std::max( 1e-12, std::abs((*d2E_dx2)[i]) );
            dE_dx_precond[i] /= std::sqrt( std::max( 1e-12, std::abs((*d2E_dx2)[i]) ) );
        }
        rdmft::Tgemm_lapack( this->Hk.data(), dE_dx_precond.data(), this->search_direction.data(), this->dim, 1, this->dim, 'N', 'N', nega_one );
    }

    // update x, dE_dx, pass search_direction
    for(int j=0; j<x_new.size(); ++j)
    {
        this->var_x[j] = x_new[j];
        this->dE_dx[j] = dE_dx_new[j];
        pk[j] = this->search_direction[j];
    }

}


// template<typename TX>
// void BFGS_Opti<TX>::get_start_guess(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk)
// {
//     // identity matrix or other method to initialize H0
//     for(int i=0; i<this->dim; ++i) { Hk[i*(this->dim) + i] = 1.0; }

//     // cal search_direction, p_k+1 = -H_k+1 * (dE_dx)_k+1
//     // property: H = H^T, also depends on the initial guess H0!
//     rdmft::Tgemm_lapack( this->Hk.data(), dE_dx_new.data(), this->search_direction.data(), this->dim, 1, this->dim, 'N', 'N', -1.0 );

//     // update x, dE_dx, pass search_direction
//     for(int j=0; j<x_new.size(); ++j)
//     {
//         this->var_x[j] = x_new[j];
//         this->dE_dx[j] = dE_dx_new[j];
//         pk[j] = this->search_direction[j];
//     }

// }


template<typename TX>
void BFGS_Opti<TX>::cal_Hk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, const bool new_landscape)
{
    if( new_landscape )
    {
        std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
        std::fill(this->dE_dx.begin(), this->dE_dx.end(), 0.0);
        std::fill(this->search_direction.begin(), this->search_direction.end(), 0.0);
        // identity matrix or other method to initialize H0
        std::fill(Hk.begin(), Hk.end(), 0.0);
        for(int i=0; i<this->dim; ++i)
        {
            Hk[i*(this->dim) + i] = 1.0;
        }
    }
    else
    {
        for(int j=0; j<x_new.size(); ++j)
        {
            // diff_x, sk = x_k+1 - x_k
            this->diff_x[j] = x_new[j] - this->var_x[j];
            // diff_grad, yk = (dE_dx)_k+1 - (dE_dx)_k
            this->diff_grad[j] = dE_dx_new[j] - this->dE_dx[j];
        }

        if( PARAM.inp.print_BFGS_Hk )
        {
            rdmft::printMatrix_pointer(nk_total, nbands, this->diff_x.data(), "in BFGS_Opti, diff_x");
            rdmft::printMatrix_pointer(nk_total, nbands, this->diff_grad.data(), "in BFGS_Opti, diff_grad");
        }

        const char op_tran = std::is_same<TX, std::complex<double>>::value ? 'C' : 'T';

        // cal rho = 1/( diffGrad^T * diffX )
        TX rho_temp = 0.0;
        rdmft::Tgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &rho_temp, 1, 1, this->dim, op_tran, 'N');
        if( std::real(rho_temp) < 1e-10 )
        {
            return;
        }
        this->rho = 1.0/std::real(rho_temp);

        std::cout << "******\n" << "in BFGS_Opti::get_pk(), 1.0/rho = y^/dagger s: " << rho_temp << "\n******" << std::endl;
        std::cout << "******\n" << "in BFGS_Opti::get_pk(), rho: " << this->rho << "\n******" << std::endl;

        rho_temp = this->rho;
        // cal rho * diffX * diffGrad^T, I - rho * diffX * diffGrad^T
        rdmft::Tgemm_lapack( this->diff_x.data(), this->diff_grad.data(), this->rho_diffX_diffGrad.data(), this->dim, this->dim, 1, 'N', op_tran, -(rho_temp) );
        for(int i=0; i<this->dim; ++i)
        {
            this->rho_diffX_diffGrad[i*this->dim + i] += 1.0;
        }

        // cal rho * diffX * diffX^T
        rdmft::Tgemm_lapack( this->diff_x.data(), this->diff_x.data(), this->rho_diffX_diffX_T.data(), this->dim, this->dim, 1, 'N', op_tran, rho_temp );

        // cal H_k+1 = (I - rho * diffX * diffGrad^T) * H_k * (I - rho * diffX * diffGrad^T)^T + rho * diffX * diffX^T
        std::vector<TX> H_tmp = this->Hk;
        rdmft::Tgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), H_tmp.data(), this->dim, this->dim, this->dim );
        rdmft::Tgemm_lapack( H_tmp.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), this->dim, this->dim, this->dim, 'N', op_tran );
        // rdmft::Tgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), this->Hk.data(), this->dim, this->dim, this->dim );
        // rdmft::Tgemm_lapack( this->Hk.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), this->dim, this->dim, this->dim, 'N', op_tran );
        for(int i=0; i<this->Hk.size(); ++i)
        {
            this->Hk[i] += this->rho_diffX_diffX_T[i];
        }
    }

}






template class BFGS_Opti<double>;
template class BFGS_Opti<std::complex<double>>;

}


