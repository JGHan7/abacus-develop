//==========================================================
// Author: Jingang Han
// DATE : 2024-11-09
//==========================================================

#include "module_rdmft/optimizer/optimizer_tools.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

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







// check this approximation using std::erf( erf_inv_own(x) ) - x
double erf_inv_own(double x) 
{
    // if (x < -1 || x > 1) {
    //     throw std::domain_error("Input for erf_inv must be in the range [-1, 1]");
    // }

    // if (x == 0) return 0;
    // if (x == 1) return std::numeric_limits<double>::infinity();
    // if (x == -1) return -std::numeric_limits<double>::infinity();

    // // Constants used in the approximation
    // const double a[] = { 0.886226899, -1.645349621, 0.914624893, -0.140543331 };
    // const double b[] = { -2.118377725, 1.442710462, -0.329097515, 0.012229801 };
    // const double c[] = { -1.970840454, -1.62490649, 3.429567803, 1.641345311 };
    // const double d[] = { 3.543889200, 1.637067800 };

    // double result;
    // double abs_x = std::abs(x);

    // // Approximation for |x| <= 0.7
    // if (abs_x <= 0.7) {
    //     double z = x * x;
    //     result = x * (((a[3] * z + a[2]) * z + a[1]) * z + a[0]) /
    //                   ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
    // }
    // // Approximation for |x| > 0.7
    // else {
    //     double z = std::sqrt(-std::log((1.0 - abs_x) / 2.0));
    //     result = (((c[3] * z + c[2]) * z + c[1]) * z + c[0]) /
    //                   ((d[1] * z + d[0]) * z + 1.0);
    //     if (x < 0) result = -result;
    // }

    // return result;






    if (x < -1.0 || x > 1.0) {
        throw std::domain_error("Input out of range: erf_inv(x) requires -1 <= x <= 1.");
    }

    if (x == 0.0) return 0.0;
    if (x == 1.0) return std::numeric_limits<double>::infinity();
    if (x == -1.0) return -std::numeric_limits<double>::infinity();

    // 高精度初始近似
    double w, p;
    if (std::abs(x) <= 0.7) {
        w = 0.5 * (1 - x);
        p = std::sqrt(-2.0 * std::log(w));
        // Abramowitz & Stegun (26.2.23) 初始近似
        p = (((-0.140543331 * p + 0.914624893) * p - 1.645349621) * p + 0.886226899) /
            ((((0.012229801 * p - 0.329097515) * p + 1.442710462) * p - 2.118377725) * p + 1.0);
    } else {
        w = std::sqrt(-std::log((1.0 - std::abs(x)) / 2.0));
        p = (((1.641345311 * w + 3.429567803) * w - 1.62490649) * w - 1.970840454) /
            ((1.637067800 * w + 3.543889200) * w + 1.0);
        if (x < 0) p = -p;
    }

    // Newton-Raphson 迭代改进
    for (int i = 0; i < 5; ++i) {
        double err = std::erf(p) - x; // 当前误差
        double deriv = 2.0 / std::sqrt(M_PI) * std::exp(-p * p); // erf 的导数
        p -= err / deriv; // 更新
    }

    return p;

}





}
