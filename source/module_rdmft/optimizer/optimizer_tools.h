//==========================================================
// Author: Jingang Han
// DATE : 2024-11-09
//==========================================================
#ifndef OPTIMIZER_TOOLS_H
#define OPTIMIZER_TOOLS_H

#include "module_base/blas_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_basis/module_ao/parallel_2d.h"
// #include "module_basis/module_ao/parallel_orbitals.h"
// #include "module_base/parallel_reduce.h"
// #include "module_lr/utils/lr_util.h"
// #include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace rdmft
{


// anti-symmetrize mat
// asym_mat = alpha * ( mat - mat^dagger )
template <typename TK>
void antisymm_mat(const Parallel_2D* para_mat,
                    const int gloabl_row_mat,
                    const TK* mat, 
                    TK* asym_mat,
                    double alpha = 0.5)
{
    for(int i=0; i<para_mat->get_local_size(); ++i) asym_mat[i] = mat[i];

    const TK a = -alpha,  b = alpha;
    const int one_int = 1;
    pztranc_(&gloabl_row_mat, &gloabl_row_mat, &a, mat, &one_int, &one_int, para_mat->desc, &b, asym_mat, &one_int, &one_int, para_mat->desc);

}


template <>
void antisymm_mat<double>(const Parallel_2D* para_mat,
                            const int gloabl_row_mat,
                            const double* mat, 
                            double* asym_mat,
                            double alpha = 0.5);



// !!! Note: the upper triangular part of mat will be destroyed
// the scale of egienvalue is global, mat and egienvector are local (2d-block)
// c++ perspective: each row of the egienvector-matrix is ​​an eigenvector
template <typename TK>
void pdiag_scalapack(const Parallel_2D* para_mat,
                        const int global_row_mat,
                        TK* mat,
                        double* egienvalue,
                        TK* egienvector)
{
    const char jobz = 'V', uplo = 'U';
    const int one_int = 1;
    int info = 0;

    // these settings refer to the scalapack source code documentation
    int tmp_num = para_mat->get_row_size() + para_mat->get_col_size() + PARAM.inp.nb2d;
    int lwork = static_cast<int>( (tmp_num*PARAM.inp.nb2d + 3*global_row_mat + std::pow(global_row_mat, 2)) * 1.1 );
    int lrwork = static_cast<int>( (4*global_row_mat - 2) * 1.1 );

    // std::vector<std::complex<double>> work(lwork, 0);
    std::vector<TK> work(lwork, 0);
    std::vector<double> rwork(lrwork, 0);

#ifdef __MPI
    pzheev_(&jobz,
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
            rwork.data(),
            &lrwork,
            &info);
#endif

    if( info ) { std::cout << "\n***\n" << "there is something wrong when calling pzheev_()" << "\n***\n" << std::endl; }

}


// !!! Note: the upper triangular part of mat will be destroyed
template <>
void pdiag_scalapack<double>(const Parallel_2D* para_mat,
                                const int global_row_mat,
                                double* mat,
                                double* egienvalue,
                                double* egienvector);




// // wfc and H_wfc need to be k_firest and provide wfc(ik, 0, 0) and H_wfc(ik, 0, 0)
// // contraction index: nbasis
// template <typename TK>
// void HkPsi(const Parallel_Orbitals* ParaV, const TK& HK, const TK& wfc, TK& H_wfc)
// {

//     const int one_int = 1;
//     //const double one_double = 1.0, zero_double = 0.0;
//     const std::complex<double> one_complex = {1.0, 0.0};
//     const std::complex<double> zero_complex = {0.0, 0.0};
//     const char N_char = 'N';
//     const char C_char = 'C';    // Using 'C' is consistent with the formula

// #ifdef __MPI
//     const int nbasis = ParaV->desc[2];
//     const int nbands = ParaV->desc_wfc[3];

//     //because wfc(bands, basis'), H(basis, basis'), we do wfc*H^T(in the perspective of cpp, not in fortran). And get H_wfc(bands, basis) is correct.
//     pzgemm_( &C_char, &N_char, &nbasis, &nbands, &nbasis, &one_complex, &HK, &one_int, &one_int, ParaV->desc,
//         &wfc, &one_int, &one_int, ParaV->desc_wfc, &zero_complex, &H_wfc, &one_int, &one_int, ParaV->desc_wfc );
// #endif
// }







// template <>
// void HkPsi<double>(const Parallel_Orbitals* ParaV, const double& HK, const double& wfc, double& H_wfc);



}

#endif
