//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================


#include "module_rdmft/optimizer/bfgs_opti_ONs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

#include "module_rdmft/rdmft_tools.h" // temp
#include <iostream> // temp

namespace rdmft
{


template<typename TX>
BFGS_ONs<TX>::BFGS_ONs()
{

}


template<typename TX>
BFGS_ONs<TX>::~BFGS_ONs()
{
    
}


template<typename TX>
void BFGS_ONs<TX>::init(int nk_total_in, int nbands_in)
{
    this->nk_total = nk_total_in;
    this->nbands = nbands_in;

    // x0.resize(nk_total*nbands);
    // x1.resize(nk_total*nbands);
    var_x.resize(nk_total*nbands);
    diff_x.resize(nk_total*nbands);
    // dE_dx0.resize(nk_total*nbands);
    // dE_dx1.resize(nk_total*nbands);
    dE_dx.resize(nk_total*nbands);
    diff_grad.resize(nk_total*nbands);
    Hk.resize( nk_total*nbands * nk_total*nbands, 0.0 );
    search_direction.resize(nk_total*nbands);

    rho_diffX_diffGrad.resize( nk_total*nbands * nk_total*nbands );
    rho_diffX_diffX_T.resize( nk_total*nbands * nk_total*nbands );
}


template<typename TX>
void BFGS_ONs<TX>::get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool start_guess)
{

#ifdef __MPI
    // just for debug, print in different processes
    int rank_now;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_now);
    std::stringstream temp_str;
    temp_str << "process_" << rank_now << ".txt";
    std::string process_file = temp_str.str();
#endif

    std::ofstream out_file(process_file, std::ios::app);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << process_file << std::endl;
    }

    // set some vars zero?

    // get Hk
    if( start_guess )
    {
        // identity matrix or other method to initialize H0
        for(int i=0; i<nk_total*nbands; ++i) { Hk[i*(nk_total*nbands) + i] = 1.0; }
    }
    else
    {
        for(int j=0; j<x_new.size(); ++j)
        {
            // if( a_equal_b(x_new, this->var_x) ) { return; }

            double temp_num = a_equal_b(x_new, this->var_x);
            Parallel_Reduce::reduce_all(temp_num);
            if( std::abs( temp_num ) > 1e-12 )
            {
                std::cout << "\n" << "line_search_rdmft: the increase in var_x is too small !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << "\n" << std::endl;
                return;
            }


            // diff_x, sk = x_k+1 - x_k
            this->diff_x[j] = x_new[j] - this->var_x[j];
            // diff_grad, yk = (dE_dx)_k+1 - (dE_dx)_k
            this->diff_grad[j] = dE_dx_new[j] - this->dE_dx[j];
        }

        if( PARAM.inp.print_BFGS_Hk )
        {
            rdmft::printMatrix_pointer(nk_total, nbands, this->diff_x.data(), "in BFGS_ONs, diff_x");
            rdmft::printMatrix_pointer(nk_total, nbands, this->diff_grad.data(), "in BFGS_ONs, diff_grad");
        }

        // cal rho = 1/( diffGrad^T * diffX )
        rdmft::dgemm_lapack(this->diff_grad.data(), this->diff_x.data(), &this->rho, 1, 1, nk_total*nbands, 'T', 'N');
        std::cout << "******\n" << "in BFGS_ONs::get_pk(), 1.0/rho: " << this->rho << "\n******" << std::endl;
        this->rho = 1.0/this->rho;

        std::cout << "******\n" << "in BFGS_ONs::get_pk(), rho: " << this->rho << "\n******" << std::endl;

        // cal rho * diffX * diffGrad^T, I - rho * diffX * diffGrad^T
        rdmft::dgemm_lapack( this->diff_x.data(), this->diff_grad.data(), this->rho_diffX_diffGrad.data(), nk_total*nbands, nk_total*nbands, 1, 'N', 'T', -(this->rho) );
        for(int i=0; i<nk_total*nbands; ++i) { this->rho_diffX_diffGrad[i*nk_total*nbands + i] += 1.0; }

        // cal rho * diffX * diffX^T
        rdmft::dgemm_lapack( this->diff_x.data(), this->diff_x.data(), this->rho_diffX_diffX_T.data(), nk_total*nbands, nk_total*nbands, 1, 'N', 'T', this->rho );

        // cal H_k+1 = (I - rho * diffX * diffGrad^T) * H_k * (I - rho * diffX * diffGrad^T)^T + rho * diffX * diffX^T
        std::vector<TX> H_tmp = this->Hk;
        rdmft::dgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), H_tmp.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands );
        rdmft::dgemm_lapack( H_tmp.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands, 'N', 'T' );
        // rdmft::dgemm_lapack( this->rho_diffX_diffGrad.data(), this->Hk.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands );
        // rdmft::dgemm_lapack( this->Hk.data(), this->rho_diffX_diffGrad.data(), this->Hk.data(), nk_total*nbands, nk_total*nbands, nk_total*nbands, 'N', 'T' );
        for(int i=0; i<this->Hk.size(); ++i) { this->Hk[i] += this->rho_diffX_diffX_T[i]; }

    }

    if( PARAM.inp.print_BFGS_Hk )
    {
        rdmft::printMatrix_pointer(nk_total*nbands, nk_total*nbands, this->Hk.data(), "BFGS: Hk", 10);
    }

    // rdmft::printMatrix_pointer(out_file, nk_total*nbands, nk_total*nbands, this->Hk.data(), "BFGS: Hk", 10);

    out_file.close();

    // cal search_direction, p_k+1 = -H_k+1 * (dE_dx)_k+1
    // property: H = H^T, also depends on the initial guess H0!
    rdmft::dgemm_lapack( this->Hk.data(), dE_dx_new.data(), this->search_direction.data(), nk_total*nbands, 1, nk_total*nbands, 'N', 'N', -1.0 );

    // update x, dE_dx, pass search_direction
    for(int j=0; j<x_new.size(); ++j)
    {
        this->var_x[j] = x_new[j];
        this->dE_dx[j] = dE_dx_new[j];
        pk[j] = this->search_direction[j];
    }

}


// template<typename TX>
// void BFGS_ONs<TX>::get_start_guess(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk)
// {
//     // identity matrix or other method to initialize H0
//     for(int i=0; i<nk_total*nbands; ++i) { Hk[i*(nk_total*nbands) + i] = 1.0; }

//     // cal search_direction, p_k+1 = -H_k+1 * (dE_dx)_k+1
//     // property: H = H^T, also depends on the initial guess H0!
//     rdmft::dgemm_lapack( this->Hk.data(), dE_dx_new.data(), this->search_direction.data(), nk_total*nbands, 1, nk_total*nbands, 'N', 'N', -1.0 );

//     // update x, dE_dx, pass search_direction
//     for(int j=0; j<x_new.size(); ++j)
//     {
//         this->var_x[j] = x_new[j];
//         this->dE_dx[j] = dE_dx_new[j];
//         pk[j] = this->search_direction[j];
//     }

// }






// template class BFGS_ONs<double, double>;
// template class BFGS_ONs<std::complex<double>, double>;
// template class BFGS_ONs<std::complex<double>, std::complex<double>>;

template class BFGS_ONs<double>;

}


