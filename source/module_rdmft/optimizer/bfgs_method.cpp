//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================


#include "module_rdmft/optimizer/bfgs_method.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

#include "module_rdmft/rdmft_tools.h" // temp
#include <iostream> // temp

namespace rdmft
{


template<typename TX>
BFGS_method<TX>::BFGS_method()
{

}


template<typename TX>
BFGS_method<TX>::~BFGS_method()
{
    
}


template<typename TX>
void BFGS_method<TX>::init(const int dim_in, const double precond_eps_in)
{

    // this->dim = dim_in;
    // this->precond_eps = precond_eps_in;
    // this->iter = 0;

    // this->dE_dx.resize(this->dim);
    // this->search_direction.resize(this->dim);

    rdmft::Opti_method<TX>::init(dim_in, precond_eps_in);

    this->scaling_H0 = false;
    // x0.resize(this->dim);
    // x1.resize(this->dim);
    var_x.resize(this->dim, 0.0);
    diff_x.resize(this->dim, 0.0);
    // dE_dx0.resize(this->dim);
    // dE_dx1.resize(this->dim);

    diff_grad.resize(this->dim);
    rho_diffX_diffGrad.resize( this->dim * this->dim );
    rho_diffX_diffX_T.resize( this->dim * this->dim );
    Hk.resize( this->dim * this->dim, 0.0 );
    if( PARAM.inp.precond_occ_num && (PARAM.inp.occ_num_opti == "cg" || PARAM.inp.occ_num_opti == "bfgs") ) // "bfgs" is test!
    {
        this->Bk.resize( this->dim * this->dim, 0.0 );
        this->y_yT.resize( this->dim * this->dim, 0.0 );
        this->B_ssT_BT.resize( this->dim * this->dim, 0.0 );
    }

    transport_mat.resize( this->dim * this->dim, 0.0 ); // test

}


template<typename TX>
void BFGS_method<TX>::get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool new_landscape, const std::vector<TX>* d2E_dx2)
{
    // set some vars zero?

    // get Hk
    this->cal_Hk(dE_dx_new, x_new, new_landscape, d2E_dx2);

    if( PARAM.inp.print_BFGS_Hk )
    {
        rdmft::printMatrix_pointer(this->dim, this->dim, this->Hk.data(), "BFGS: Hk", 10);
    }

    // cal search_direction, p_k+1 = -H_k+1 * (dE_dx)_k+1
    // property: H = H^T, also depends on the initial guess H0!
    TX nega_one = -1.0;
    std::vector<TX> dE_dx_precond = dE_dx_new;
    if( d2E_dx2 != nullptr )
    {
        for(int i=0; i<dE_dx_precond.size(); ++i)
        {
            // dE_dx_precond[i] /= std::max( this->precond_eps, std::abs((*d2E_dx2)[i]) );
            dE_dx_precond[i] /= std::sqrt( std::max( this->precond_eps, std::abs((*d2E_dx2)[i]) ) );
        }
    }
    rdmft::Tgemm_lapack( this->Hk.data(), dE_dx_precond.data(), this->search_direction.data(), this->dim, 1, this->dim, 'N', 'N', nega_one );

    // consider whether to keep this code, is it necessary to keep it in the preprocessing case?
    // determine whether the search direction pk is in the descending direction, if not, restart
    TX gT_pk = 0.0;
    const char op_tran = std::is_same<TX, std::complex<double>>::value ? 'C' : 'T';
    rdmft::Tgemm_lapack( dE_dx_new.data(), this->search_direction.data(), &gT_pk, 1, 1, this->dim, op_tran, 'N' );
    if( std::real(gT_pk) >= 0 )
    {
        std::cout << "******\n" << "test !!!: in BFGS_method::get_pk(), real(gT_pk) >= 0: " << gT_pk << "\n******" << std::endl;
        this->cal_Hk(dE_dx_new, x_new, true, d2E_dx2);
        for(int j=0; j<x_new.size(); ++j)
        {
            this->search_direction[j] = -dE_dx_precond[j];
        }
        ++this->num_restart_skip;
    }

    // update x, dE_dx, pass search_direction
    for(int j=0; j<x_new.size(); ++j)
    {
        this->var_x[j] = x_new[j];
        this->dE_dx[j] = dE_dx_new[j];
        pk[j] = this->search_direction[j];
    }
    ++this->iter;

}


template<typename TX>
void BFGS_method<TX>::cal_Hk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, const bool new_landscape, const std::vector<TX>* d2E_dx2)
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
            // // consider and test again !!!
            // if( d2E_dx2 == nullptr )
            // {
            //     Hk[i*(this->dim) + i] = 1.0 / std::max( this->precond_eps, std::abs((*d2E_dx2)[i]) );
            // }
            // else
            {
                Hk[i*(this->dim) + i] = 1.0;
            }
        }
        this->iter = 0;
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
            rdmft::printMatrix_pointer(this->dim, 1, this->diff_x.data(), "in BFGS_method, diff_x", 10);
            rdmft::printMatrix_pointer(this->dim, 1, this->diff_grad.data(), "in BFGS_method, diff_grad", 10);
        }

        const char op_tran = std::is_same<TX, std::complex<double>>::value ? 'C' : 'T';

        // cal rho = 1/( diffGrad^T * diffX )
        TX rho_temp = 0.0;
        rdmft::Tgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &rho_temp, 1, 1, this->dim, op_tran, 'N');
        if( std::real(rho_temp) < 1e-10 )
        {   
            // considering damped BFGS !!!
            std::cout << "******\n" << "test !!!: in BFGS_method::get_pk(), std::real(rho_temp) < 1e-10: " << rho_temp << "\n******" << std::endl;
            ++this->num_restart_skip;
            return;
        }
        this->rho = 1.0/std::real(rho_temp);

        // // scaling H0
        // if( this->scaling_H0 && this->iter == 1)
        // {
        //     rdmft::Tgemm_lapack(this->diff_grad.data(), this->diff_grad.data(), &this->scaling_gamma0, 1, 1, this->dim, op_tran, 'N');
        //     this->scaling_gamma0 = std::real(rho_temp) / this->scaling_gamma0;

        //     for(int i=0; i<this->dim; ++i)
        //     {
        //         if( d2E_dx2 == nullptr )
        //         {
        //             Hk[i*(this->dim) + i] *= this->scaling_gamma0;
        //         }
        //         else
        //         {
        //             Hk[i*(this->dim) + i] *= this->scaling_gamma0 / std::max( this->precond_eps, std::abs((*d2E_dx2)[i]) );
        //         }
        //     }
        //     std::cout << "******\n" << "in BFGS_method::get_pk(), this->scaling_gamma0: " << this->scaling_gamma0 << "\n******" << std::endl;
        // }

        std::cout << "******\n" << "in BFGS_method::get_pk(), 1.0/rho = y^/dagger s: " << rho_temp << "\n******" << std::endl;
        // std::cout << "******\n" << "in BFGS_method::get_pk(), rho: " << this->rho << "\n******" << std::endl;

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


template<typename TX>
void BFGS_method<TX>::get_diag_Bk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& diag_Bk, const bool new_landscape)
{
    if( new_landscape )
    {
        std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
        std::fill(this->dE_dx.begin(), this->dE_dx.end(), 0.0);
        std::fill(Bk.begin(), Bk.end(), 0.0);
        for(int i=0; i<this->dim; ++i)
        {
            Bk[i*(this->dim) + i] = 1.0;
        }
        this->iter = 0;
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

        const char op_tran = std::is_same<TX, std::complex<double>>::value ? 'C' : 'T';
        // this->B_ssT_BT.resize( this->dim * this->dim, 0.0 );

        // cal rho = 1/( diffGrad^T * diffX )
        TX rho_temp = 0.0;
        rdmft::Tgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &rho_temp, 1, 1, this->dim, op_tran, 'N');
        // if( std::real(rho_temp) < 1e-10 )
        // {   
        //     // considering damped BFGS !!!
        //     std::cout << "******\n" << "test !!!: in BFGS_method::get_diag_Bk, std::real(rho_temp) < 1e-10: " << rho_temp << "\n******" << std::endl;
        //     ++this->num_restart_skip;
        // }
        // else
        {
            this->rho = 1.0/std::real(rho_temp);
            rho_temp = this->rho;
            rdmft::Tgemm_lapack( this->diff_grad.data(), this->diff_grad.data(), this->y_yT.data(), this->dim, this->dim, 1, 'N', op_tran, rho_temp );

            TX sT_B_s = 0.0;
            std::vector<TX> B_s(this->dim, 0.0);
            rdmft::Tgemm_lapack( this->Bk.data(), this->diff_x.data(), B_s.data(), this->dim, 1, this->dim, 'N', 'N' );
            rdmft::Tgemm_lapack( this->diff_x.data(), B_s.data(), &sT_B_s, 1, 1, this->dim, op_tran, 'N' );
            sT_B_s = 1.0/std::real(sT_B_s);

            std::vector<TX> B_s_sT(this->dim * this->dim, 0.0);
            rdmft::Tgemm_lapack( this->diff_x.data(), this->diff_x.data(), this->B_ssT_BT.data(), this->dim, this->dim, 1, 'N', op_tran );
            rdmft::Tgemm_lapack( this->Bk.data(), this->B_ssT_BT.data(), B_s_sT.data(), this->dim, this->dim, this->dim, 'N', 'N' );
            rdmft::Tgemm_lapack( B_s_sT.data(), this->Bk.data(), this->B_ssT_BT.data(), this->dim, this->dim, this->dim, 'N', op_tran, sT_B_s );

            for(int j=0; j<this->Bk.size(); ++j)
            {
                this->Bk[j] += this->y_yT[j] - this->B_ssT_BT[j];
            }
        }

    }

    for(int i=0; i<this->dim; ++i)
    {
       diag_Bk[i] = Bk[i*(this->dim) + i];
    }

    // update x, dE_dx, pass search_direction
    for(int j=0; j<x_new.size(); ++j)
    {
        this->var_x[j] = x_new[j];
        this->dE_dx[j] = dE_dx_new[j];
    }
    ++this->iter;
}


template<typename TX>
void BFGS_method<TX>::transport(const std::vector<TX>& diag_T)
{
    for(int j=0; j<this->dim; ++j)
    {
        this->var_x[j] *= diag_T[j];
        this->dE_dx[j] /= diag_T[j];
        transport_mat[j*(this->dim) + j] = 1.0 / diag_T[j];
    }
    std::vector<TX> H_tmp = this->Hk;
    rdmft::Tgemm_lapack( this->transport_mat.data(), this->Hk.data(), H_tmp.data(), this->dim, this->dim, this->dim );
    rdmft::Tgemm_lapack( H_tmp.data(), this->transport_mat.data(), this->Hk.data(), this->dim, this->dim, this->dim );
}


// template<typename TX>
// void BFGS_method<TX>::get_start_guess(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk)
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



template class BFGS_method<double>;
template class BFGS_method<std::complex<double>>;

}


