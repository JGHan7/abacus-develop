//==========================================================
// Author: Jingang Han
// DATE : 2024-11-09
//==========================================================

#include "module_rdmft/optimizer/optimizer_tools.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <random>
#include <algorithm>

namespace rdmft
{


template <>
void antisymm_mat<double>(const Parallel_2D* para_mat,
                            const int gloabl_row_mat,
                            const double* mat, 
                            double* asym_mat,
                            double alpha)
{
    for(int i=0; i<para_mat->get_local_size(); ++i) asym_mat[i] = mat[i];

    const double a = -alpha,  b = alpha;
    const int one_int = 1;
    pdtran_(&gloabl_row_mat, &gloabl_row_mat, &a, mat, &one_int, &one_int, para_mat->desc, &b, asym_mat, &one_int, &one_int, para_mat->desc);

}



// void pdsyev_(const char* jobz, const char* uplo, const int* n, double* A, const int* ia, const int* ja, 
//             const int* desca, double* w, double* z, const int* iz, const int* jz, const int* descz,
//             double* work, int* lwork, int* info);
template <>
void pdiag_scalapack<double>(const Parallel_2D* para_mat,
                                const int global_row_mat,
                                double* mat,
                                double* egienvalue,
                                double* egienvector,
                                bool get_egivector)
{
    char jobz = 'V';
    if(!get_egivector) jobz = 'N';

    const char uplo = 'U';
    const int one_int = 1;
    int info = 0;

    // these settings refer to the scalapack source code documentation
    int tmp0 = global_row_mat*( global_row_mat>2 ? global_row_mat: 2 );
    int tmp_num = tmp0 > (2*global_row_mat-2) ? tmp0: (2*global_row_mat-2);
    int lwork = static_cast<int>( ( global_row_mat*( 5 + para_mat->get_col_size()) + tmp_num + 1) * 1.1 );

    std::vector<double> work(lwork, 0);

#ifdef __MPI
    pdsyev_(&jobz,
            &uplo,
            &global_row_mat,
            mat,
            &one_int,
            &one_int,
            para_mat->desc,
            egienvalue,
            egienvector,
            &one_int,
            &one_int,
            para_mat->desc,
            work.data(),
            &lwork,
            &info);
#endif

    if( info ) { std::cout << "\n***\n" << "there is something wrong when calling pzheev_()" << "\n***\n" << std::endl; }

}


template <>
void GkPsi<double>(const Parallel_2D* para_mat, 
                    const Parallel_Orbitals* ParaV, 
                    const double& G, 
                    const double& wfc, 
                    double& G_wfc)
{
    const int one_int = 1;
    const double one_double = 1.0;
    const double zero_double = 0.0;
    const char N_char = 'N';
    // const char T_char = 'T';

#ifdef __MPI
    const int nbasis = ParaV->desc[2];
    const int nbands = ParaV->desc_wfc[3];

    // cpp perspective: G(nbands, nbands') * wfc(nbands', nbasis) 
    // = fortran perspective: wfc(nbasis, nbands') * G(nbands', nbands)
    pdgemm_( &N_char, &N_char, &nbasis, &nbands, &nbands, &one_double, &wfc, &one_int, &one_int, ParaV->desc_wfc,
        &G, &one_int, &one_int, para_mat->desc, &zero_double, &G_wfc, &one_int, &one_int, ParaV->desc_wfc );
#endif
}


template <>
void pTgemm_scalapack<double>(const Parallel_2D* para_A,
                                const Parallel_2D* para_B,
                                const Parallel_2D* para_C,
                                const double* A,
                                const double* B,
                                double* C,
                                const int global_row_C,
                                const int global_col_C,
                                const int global_contract_index,
                                const char op_A,
                                const char op_B,
                                double alpha,
                                double beta)
{
    const int one_int = 1;

    pdgemm_( &op_A, &op_B, &global_row_C, &global_col_C, &global_contract_index, &alpha, A, &one_int, &one_int, para_A->desc,
            &B, &one_int, &one_int, para_B->desc, &beta, C, &one_int, &one_int, para_C->desc );
}


// to compute C = alpha * A.? * B.? + beta * C
// all use the fortran perspective, not cpp
void dgemm_lapack(const double* A,
                    const double* B,
                    double* C, 
                    const int row_C, 
                    const int col_C, 
                    const int contract_index,
                    const char op_A,
                    const char op_B,
                    const double alpha,
                    const double beta)
{
    const int lda = (op_A == 'N') ? row_C : contract_index;
    const int ldb = (op_B == 'N') ? contract_index : col_C;
    const int one_int = 1;

    dgemm_( &op_A, &op_B, &row_C, &col_C, &contract_index, &alpha, A, &lda, B, &ldb, &beta, C, &row_C );
}


/********* the following function is used by the BFGS_opti_ONs method *********/






/********* the following function is just used by the EBI_constraint method *********/

// check this approximation using std::erf( erf_inv_own(x) ) - x
double erf_inv_own(double x) 
{
    if (x < -1.0 || x > 1.0) {
        throw std::domain_error("Input out of range: erf_inv(x) requires -1 <= x <= 1.");
    }

    if (x == 0.0) return 0.0;
    if (x == 1.0) return std::numeric_limits<double>::infinity();
    if (x == -1.0) return -std::numeric_limits<double>::infinity();

    // High-precision initial approximation
    double w, p;
    if (std::abs(x) <= 0.7) {
        w = 0.5 * (1 - x);
        p = std::sqrt(-2.0 * std::log(w));
        // Abramowitz & Stegun (26.2.23) initial Approximation
        p = (((-0.140543331 * p + 0.914624893) * p - 1.645349621) * p + 0.886226899) /
            ((((0.012229801 * p - 0.329097515) * p + 1.442710462) * p - 2.118377725) * p + 1.0);
    } else {
        w = std::sqrt(-std::log((1.0 - std::abs(x)) / 2.0));
        p = (((1.641345311 * w + 3.429567803) * w - 1.62490649) * w - 1.970840454) /
            ((1.637067800 * w + 3.543889200) * w + 1.0);
        if (x < 0) p = -p;
    }

    // Newton-Raphson iterative Improvement
    for (int i = 0; i < 5; ++i) {
        double err = std::erf(p) - x; // current Error
        double deriv = 2.0 / std::sqrt(M_PI) * std::exp(-p * p); // derivative of erf
        p -= err / deriv; // update
    }

    return p;
}


double erf_der1(double x)
{
    double y = ( 2 * std::exp( -std::pow(x, 2) ) )/std::sqrt(ModuleBase::PI);
    return y;
}


double erf_der2(double x)
{
    double y = ( -4 * x * std::exp( -std::pow(x, 2) ) )/std::sqrt(ModuleBase::PI);
    return y;
}


void random_descend(std::vector<double>& num, const double* value, int* location)
{
    // random seed, requires hardware support
    std::random_device rd;
    // mersenne Twister engine
    std::mt19937 gen(rd());     // std::mt19937 gen(42);
    // a uniform distribution in the range (0.0, 1.0)
    std::uniform_real_distribution<> dis( std::nextafter(0.0, 1.0), 1.0 );

    for(int i=0; i<num.size(); ++i) { num[i] = dis(gen); }
    // sort descending
    std::sort(num.begin(), num.end(), std::greater<>());

    if( value != nullptr && location != nullptr )
    {
        double sum_num = 0.0;
        for(int j=0; j<num.size(); ++j)
        {
            sum_num += num[j];
            if( sum_num > *value )
            {
                *location = j;
                break;
            }
        }
    }
}




}
