//==========================================================
// Author: Jingang Han
// DATE : 2024-11-09
//==========================================================
#ifndef OPTIMIZER_TOOLS_H
#define OPTIMIZER_TOOLS_H

#include "module_base/blas_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_base/parallel_2d.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_parameter/parameter.h"
#include "module_base/constants.h"
#include "module_base/parallel_reduce.h"
#include "module_base/matrix.h"
#include "module_cell/klist.h"
// #include "module_lr/utils/lr_util.h"
// #include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_rdmft/rdmft_tools.h" // temp

#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

#include <torch/torch.h>

namespace rdmft
{

// learn from module_base/parallel_reduce.h
//! obtain the maximum value of object between different processes and broadcasts it
template <typename T>
void reduce_all_max(T& object);


template <typename T>
void reduce_all_max(T* object, const int n);


//! determine whether a is numerically equal to b
template <typename TK>
bool a_equal_b(const std::vector<TK>& a, const std::vector<TK>& b)
{
    double diff_sum = 0.0;
    for(int i=0; i<a.size(); ++i)
    {
        diff_sum += std::abs( a[i] - b[i] );
    }

    if( diff_sum < 1e-16 ) // 1e-16 would be error in different processors?
    {
        return true;
    }
    else
    {
        return false;
    }
}


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

#ifdef __MPI
    pztranc_(&gloabl_row_mat, &gloabl_row_mat, &a, mat, &one_int, &one_int, para_mat->desc, &b, asym_mat, &one_int, &one_int, para_mat->desc);
#endif
}


template <>
void antisymm_mat<double>(const Parallel_2D* para_mat,
                            const int gloabl_row_mat,
                            const double* mat, 
                            double* asym_mat,
                            double alpha);


//! check the Hermitian property or symmetry of a matrix
//! return the max value of std::abs( Mij-conj(Mji) )
template <typename TK>
double check_hermi(const Parallel_2D* para_mat, const std::vector<TK>& mat, const int global_row_mat)
{
    double max_Mij = 0.0;
    std::vector<TK> zero_mat(mat.size(), 0.0);
    rdmft::antisymm_mat(para_mat, global_row_mat, mat.data(), zero_mat.data(), 1.0);
    
    for(int iloc=0; iloc<zero_mat.size(); ++iloc)
    {
        max_Mij = std::max( max_Mij, std::abs(zero_mat[iloc]) );
    }

    rdmft::reduce_all_max(max_Mij);

    return max_Mij;
}


template<typename TK>
void get_identi_mat(const Parallel_2D* para_mat, std::vector<TK>& iden_mat)
{
    if( iden_mat.size() != para_mat->get_local_size() ) { iden_mat.resize(para_mat->get_local_size(), static_cast<TK>(0.0)); }

    const int nrow = para_mat->get_row_size();
    const int ncol = para_mat->get_col_size();

    for(int i=0; i<nrow; ++i)
    {
        const int i_global = para_mat->local2global_row(i);
        for(int j=0; j<ncol; ++j)
        {
            int j_global = para_mat->local2global_col(j);
            if( i_global == j_global )
            {
                iden_mat[ i+j*nrow ] = static_cast<TK>(1.0);
            }
            else
            {
                iden_mat[ i+j*nrow ] = static_cast<TK>(0.0);
            }
        }
    }
}


//! The diagonal is 1, and the rest is 0
//! Multiplying exch_mat on the left will reverse the order of the rows of the matrix
//! Multiplying exch_mat on the right will reverse the order of the columns of the matrix
template<typename TK>
void get_exchange_mat(const Parallel_2D* para_mat, std::vector<TK>& exch_mat)
{
    if( exch_mat.size() != para_mat->get_local_size() ) { exch_mat.resize(para_mat->get_local_size(), static_cast<TK>(0.0)); }

    const int nrow = para_mat->get_row_size();
    const int ncol = para_mat->get_col_size();
    const int global_row = para_mat->get_global_row_size();

    for(int i=0; i<nrow; ++i)
    {
        const int i_global = para_mat->local2global_row(i);
        for(int j=0; j<ncol; ++j)
        {
            int j_global = para_mat->local2global_col(j);
            if( (global_row - i_global - 1) == j_global )
            {
                exch_mat[ i+j*nrow ] = static_cast<TK>(1.0);
            }
            else
            {
                exch_mat[ i+j*nrow ] = static_cast<TK>(0.0);
            }
        }
    }
}


// !!! Note: the upper triangular part of mat will be destroyed
// the scale of egienvalue is global, mat and egienvector are local (2d-block)
// c++ perspective: each row of the egienvector-matrix is ​​an eigenvector
template <typename TK>
void pdiag_scalapack(const Parallel_2D* para_mat,
                        const int global_row_mat,
                        TK* mat,
                        double* egienvalue,
                        TK* egienvector,
                        const bool get_egivector = true)
{
    char jobz = 'V';
    if(!get_egivector) jobz = 'N';

    const char uplo = 'U';
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

    if( info ) { std::cout << "\n***\n" << "there is something wrong when calling pzheev_(), info: " << info << "\n***\n" << std::endl; }
    assert( info == 0 );

}


// !!! Note: the upper triangular part of mat will be destroyed
template <>
void pdiag_scalapack<double>(const Parallel_2D* para_mat,
                                const int global_row_mat,
                                double* mat,
                                double* egienvalue,
                                double* egienvector,
                                const bool get_egivector);


// contraction index: nbands
template <typename TK>
void GkPsi(const Parallel_2D* para_mat, 
            const Parallel_Orbitals* ParaV, 
            const TK& G, 
            const TK& wfc, 
            TK& G_wfc)
{
    const int one_int = 1;
    // const std::complex<double> one_complex = {1.0, 0.0};
    // const std::complex<double> zero_complex = {0.0, 0.0};
    const TK one_complex = 1.0;
    const TK zero_complex = 0.0;
    const char N_char = 'N';
    // const char T_char = 'T';

#ifdef __MPI
    const int nbasis = ParaV->desc[2];
    const int nbands = ParaV->desc_wfc[3];

    // cpp perspective: G(nbands, nbands') * wfc(nbands', nbasis) 
    // = fortran perspective: wfc(nbasis, nbands') * G(nbands', nbands)
    pzgemm_( &N_char, &N_char, &nbasis, &nbands, &nbands, &one_complex, &wfc, &one_int, &one_int, ParaV->desc_wfc,
        &G, &one_int, &one_int, para_mat->desc, &zero_complex, &G_wfc, &one_int, &one_int, ParaV->desc_wfc );
#endif
}


template <>
void GkPsi<double>(const Parallel_2D* para_mat, 
                    const Parallel_Orbitals* ParaV, 
                    const double& G, 
                    const double& wfc, 
                    double& G_wfc);


//! to compute C = alpha * A.? * B.? + beta * C, op_ = 'N' or 'T' or 'C', in the case of MPI
//! When para_B and para_C use nullptr, A, B, and C are square matrices of the same dimension.
//! all use the fortran perspective, not cpp
template <typename TK>
void pTgemm_scalapack(const Parallel_2D* para_A,
                        const TK* A,
                        const TK* B,
                        TK* C,
                        const int global_row_C,
                        const int global_col_C,
                        const int global_contract_index,
                        const char op_A = 'N',
                        const char op_B = 'N',
                        const Parallel_2D* para_B = nullptr,
                        const Parallel_2D* para_C = nullptr,
                        TK alpha = 1.0,
                        TK beta = 0.0)
{
    const int one_int = 1;
    if( para_B == nullptr ) { para_B = para_A; }
    if( para_C == nullptr ) { para_C = para_A; }

#ifdef __MPI
    pzgemm_( &op_A, &op_B, &global_row_C, &global_col_C, &global_contract_index, &alpha, A, &one_int, &one_int, para_A->desc,
            B, &one_int, &one_int, para_B->desc, &beta, C, &one_int, &one_int, para_C->desc );
#endif
}


template <>
void pTgemm_scalapack<double>(const Parallel_2D* para_A,
                                const double* A,
                                const double* B,
                                double* C,
                                const int global_row_C,
                                const int global_col_C,
                                const int global_contract_index,
                                const char op_A,
                                const char op_B,
                                const Parallel_2D* para_B,
                                const Parallel_2D* para_C,
                                double alpha,
                                double beta);



//! to compute C = alpha * A.? * B.? + beta * C, op_ = 'N' or 'T'
//! all use the fortran perspective, not cpp
void dgemm_lapack(const double* A,
                    const double* B,
                    double* C, 
                    const int row_C, 
                    const int col_C, 
                    const int contract_index,
                    const char op_A = 'N',
                    const char op_B = 'N',
                    double alpha = 1.0,
                    double beta = 0.0);



/********* the following function is used by the BFGS_opti_ONs method *********/






/********* the following function is just used by the EBI_constraint method *********/

double erf_inv_own(double x);


//! the first derivative of the erf function
double erf_der1(double x);


//! the second derivative of the erf function
double erf_der2(double x);


//! generate a set of random numbers with a uniform distribution in the range (0.0, 1.0), in descending order
//! optional: loc is the smallest number satisfying \sum_{p=1}^{loc} num_{p} >= value
// void random_descend(std::vector<double>& num, const double* value = nullptr, int* location = nullptr, const std::vector<double>* num_symm_k = nullptr);
void random_descend(std::vector<double>& num, double max = 1.0);


//! return location which is the smallest number( = ib*nk_total + ik) satisfying \sum_{p=1}^{location} num_{p} >= value
//! according to the same bands, different k order sum, that is, first sum k, then sum band
int smallest_loc_big_value(const int nk_nospin,
                            const int nbands,
                            const double value,
                            const std::vector<double>& num,
                            const double* num_symm_k);




/********* the following function is used by mixing in rdmft *********/


void occ_num2wg(const K_Vectors* kv, const ModuleBase::matrix& occ_num, ModuleBase::matrix& wg);


void wg2occ_num(const K_Vectors* kv, const ModuleBase::matrix& wg, ModuleBase::matrix& occ_num);


//! convert the local density matrix to the global density matrix
//! dm_local<ik, <miu1_loc, miu2_loc>>, dm_global<ik, <miu1, miu2>>
template <typename TK>
void dm_local2global(const Parallel_2D* para_D,
                        const std::vector< std::vector<TK> >& dm_local,
                        std::vector<TK>& dm_global,
                        const int global_dim)
{
    // malloc and ensure the initial value is 0.0
    int temp_size = global_dim * global_dim;
    if( dm_global.size() != temp_size * dm_local.size() )
    {
        dm_global.assign(temp_size * dm_local.size(), static_cast<TK>(0.0));
    }
    else
    {
        std::fill(dm_global.begin(), dm_global.end(), static_cast<TK>(0.0));
    }

    // convert
    for(int ik=0; ik<dm_local.size(); ++ik)
    {
        for(int iu1=0; iu1<para_D->get_col_size(); ++iu1)
        {
            const int iu1_global = para_D->local2global_col(iu1);
            for(int iu2=0; iu2<para_D->get_row_size(); ++iu2)
            {
                const int iu2_global = para_D->local2global_row(iu2);
                dm_global[ ik * temp_size + iu1_global * global_dim + iu2_global ] = dm_local[ik][ iu1 * para_D->get_row_size() + iu2 ];
            }
        }
    }

    // collecting data
    Parallel_Reduce::reduce_all( dm_global.data(), dm_global.size() );

}


//! convert the global density matrix to the local density matrix
//! dm_local<ik, <miu1_loc, miu2_loc>>, dm_global<ik, <miu1, miu2>>
template <typename TK>
void dm_global2local(const Parallel_2D* para_D,
                        const std::vector<TK>& dm_global,
                        std::vector< std::vector<TK> >& dm_local,
                        const int global_dim)
{
    // malloc and ensure the initial value is 0.0
    for(int ik=0; ik<dm_local.size(); ++ik)
    {
        if( dm_local[ik].size() != para_D->nloc )
        {
            dm_local[ik].assign(dm_local[ik].size(), static_cast<TK>(0.0));
        }
        else
        {
            std::fill(dm_local[ik].begin(), dm_local[ik].end(), static_cast<TK>(0.0));
        }
    }

    // convert
    int temp_size = global_dim * global_dim;
    for(int ik=0; ik<dm_local.size(); ++ik)
    {
        for(int iu1=0; iu1<para_D->get_col_size(); ++iu1)
        {
            const int iu1_global = para_D->local2global_col(iu1);
            for(int iu2=0; iu2<para_D->get_row_size(); ++iu2)
            {
                const int iu2_global = para_D->local2global_row(iu2);
                dm_local[ik][ iu1 * para_D->get_row_size() + iu2 ] = dm_global[ ik * temp_size + iu1_global * global_dim + iu2_global ];
            }
        }
    }
}



//! Cholesky decomposition: A = L * L^dagger
//! fortran perspective: L, the lower triangular part of A is destroyed. U, the upper triangular part of A is destroyed.
template <typename TK>
void cholesky_decom(const Parallel_2D* para_A, TK* A_mat, TK* L_mat, const char uplo_in = 'L')
{
    const char uplo = uplo_in;
    const int one_int = 1;
    int info = 0;
    const int global_row_A = para_A->get_global_row_size();

    // Cholesky decomposition, the lower triangular part of A is L
    if constexpr (std::is_same<TK, double>::value)
    {
        pdpotrf_( &uplo, &global_row_A, A_mat, &one_int, &one_int, para_A->desc, &info );
    }
    else if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        pzpotrf_( &uplo, &global_row_A, A_mat, &one_int, &one_int, para_A->desc, &info );
    }

    if( info ) { std::cout << "\n***\n" << "there is something wrong when calling pzpotrf_(), info: " << info << "\n***\n" << std::endl; }
    assert( info == 0 );

    // copy the lower triangular or upper triangular part of A to L
    for(int ic=0; ic<para_A->get_col_size(); ++ic)
    {
        const int ic_global = para_A->local2global_col(ic);
        for(int ir=0; ir<para_A->get_row_size(); ++ir)
        {
            const int ir_global = para_A->local2global_row(ir);
            if(uplo == 'L')
            {
                if( ir_global >= ic_global )
                {
                    L_mat[ ir + ic*para_A->get_row_size() ] = A_mat[ ir + ic*para_A->get_row_size() ];
                }
                else
                {
                    L_mat[ ir + ic*para_A->get_row_size() ] = static_cast<TK>(0.0);
                }
            }
            else
            {
                if( ir_global <= ic_global )
                {
                    L_mat[ ir + ic*para_A->get_row_size() ] = A_mat[ ir + ic*para_A->get_row_size() ];
                }
                else
                {
                    L_mat[ ir + ic*para_A->get_row_size() ] = static_cast<TK>(0.0);
                }
            }

        }
    }
}


template <typename TK>
int get_work_size_getri(const int global_row,
                        const int* desc,
                        std::vector<int>& ipiv,
                        int& lwork_out,
                        int& liwork_out)
{
    int info = 0;
    int ia = 1, ja = 1;

    std::vector<TK> dummy_mat(1, static_cast<TK>(0));
    TK work = static_cast<TK>(0.0);
    int iwork = 0;
    int lwork = -1;
    int liwork = -1;

    if constexpr (std::is_same<TK, double>::value)
    {
        pdgetri_(&global_row, dummy_mat.data(), &ia, &ja, desc, ipiv.data(), &work, &lwork, &iwork, &liwork, &info);
        lwork_out = static_cast<int>(work);
    }
    else if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        pzgetri_(&global_row, dummy_mat.data(), &ia, &ja, desc, ipiv.data(), &work, &lwork, &iwork, &liwork, &info);
        lwork_out = static_cast<int>(std::real(work));
    }

    liwork_out = iwork;

    return info;

    if (info != 0)
    {
        std::cout << "Query pd/zgetri_ failed with info = " << info << std::endl;
    }
    assert( info == 0 );

}


//! inv matrix, output inv_A_mat will overwrite A_mat
template <typename TK>
void inv_matrix(const Parallel_2D* para_A, TK* A_mat)
{
    const int global_dim = para_A->get_global_row_size();
    const int one_int = 1;
    int info1 = 0;
    int info2 = 0;

    std::vector<int> ipiv(para_A->get_row_size(), 0);

    int lwork = 0;
    int liwork = 0;
    int info3 = get_work_size_getri<TK>(global_dim, para_A->desc, ipiv, lwork, liwork);
    if( info3 != 0 )
    {
        lwork = std::max( 4*global_dim*PARAM.inp.nb2d, 1024 );
        liwork = std::max( 2*para_A->get_row_size(), 64 );
    }
    std::vector<TK> work(lwork, static_cast<TK>(0.0));
    std::vector<int> iwork(liwork, 0.0);

    if constexpr (std::is_same<TK, double>::value)
    {
        // LU decomposition
        pdgetrf_( &global_dim, &global_dim, A_mat, &one_int, &one_int, para_A->desc, ipiv.data(), &info1 );
        // inverse
        pdgetri_( &global_dim, A_mat, &one_int, &one_int, para_A->desc, ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info2 );
    }
    else if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        // LU decomposition
        pzgetrf_( &global_dim, &global_dim, A_mat, &one_int, &one_int, para_A->desc, ipiv.data(), &info1 );
        // inverse
        pzgetri_( &global_dim, A_mat, &one_int, &one_int, para_A->desc, ipiv.data(), work.data(), &lwork, iwork.data(), &liwork, &info2 );
    }

    if( info1 != 0 || info2 != 0 )
    {
        std::cout << "\n***\n" << "there is something wrong when calling pzgetrf_()/pzgetri_(), info1: " << info1 << ", info2: " << info2 << "\n***\n" << std::endl;
    }
    assert( info1 == 0 && info2 == 0 );

}


//! decompose the density matrix DM into C^dagger*F*C, given that C*C^dagger=I
//! the code assumes that the number of eigenvectors obtained = DMk's dimension
template <typename TK>
void decom_dm(const Parallel_2D* para_D,
                const std::vector<TK>& DMk,
                double* wg,
                TK* wfc)
{

    const int dim = para_D->get_global_row_size(); // global_nbasis

    // to reorder
    std::vector<double> temp_wg(dim, 0.0);
    std::vector<TK> temp_wfc(para_D->nloc, 0.0);

    // diga(D): D = C^dagger * wg * C
    std::vector<TK> iden_mat(para_D->get_local_size(), 0.0);
    std::vector<TK> temp_DM(para_D->nloc, 0.0);
    get_identi_mat(para_D, iden_mat);
    pTgemm_scalapack(para_D, DMk.data(), iden_mat.data(), temp_DM.data(), dim, dim, dim, 'N', 'N');
    pdiag_scalapack(para_D, dim, temp_DM.data(), temp_wg.data(), temp_wfc.data(), true);

    // rdmft::printMatrix_pointer(para_D->get_row_size(), para_D->get_col_size(), temp_wfc.data(), "temp_wfc in decom_dm", 10);

    // convert
    for(int ib=0; ib<para_D->get_global_col_size(); ++ib)
    {
        // sort the natural occupation numbers from largest to smallest
        wg[ib] = temp_wg[para_D->get_global_col_size() - ib - 1];
    }

    // change the order of the eigenvectors wfc so that they correspond to the eigenvalues ​​wg
    std::vector<TK> exch_mat(para_D->get_local_size(), 0.0);
    std::vector<TK> temp_wfc2(para_D->nloc, 0.0);
    get_exchange_mat(para_D, exch_mat);
    pTgemm_scalapack(para_D, temp_wfc.data(), exch_mat.data(), temp_wfc2.data(), dim, dim, dim, 'N', 'N');

    // rdmft::printMatrix_pointer(para_D->get_row_size(), para_D->get_col_size(), temp_wfc2.data(), "temp_wfc2 in decom_dm", 10);

    for(int iloc=0; iloc<para_D->nloc; ++iloc)
    {
        wfc[iloc] = temp_wfc2[iloc];
    }

    // rdmft::printMatrix_pointer(para_D->get_row_size(), para_D->get_col_size(), wfc, "wfc in decom_dm", 10);

    // recode: could instead the above code which reorder the wfc!
    // for(int ib=0; ib<para_D->get_col_size(); ++ib)
    // {
    //     const int loc_size = para_D->get_row_size();
    //     for(int ibasis=0; ibasis<loc_size; ++ibasis)
    //     {
    //         wfc[ibasis + ib*loc_size] = temp_wfc2[ ibasis + ib*loc_size ];
    //     }
    // }

}


//! decompose the density matrix DM into C^dagger*F*C, given that C*S*C^dagger=I
template <typename TK>
void decom_dm(const Parallel_2D* ParaV,
                const std::vector<TK>& DMk,
                double* wg,
                TK* wfc,
                const Parallel_Orbitals* para_wfc,
                TK* Sk)
{
    std::vector<TK> L_mat(ParaV->nloc, 0.0);
    std::vector<TK> M_mat(ParaV->nloc, 0.0);
    std::vector<TK> wfc_X(ParaV->nloc, 0.0);
    const int dim = ParaV->get_global_row_size(); // global_nbasis

    // used when global_nbands != global_nbasis
    std::vector<double> temp_wg(dim, 0.0);
    std::vector<TK> temp_wfc(ParaV->nloc, 0.0);

    // get L matrix
    cholesky_decom(ParaV, Sk, L_mat.data());

    // M = L^dagger * DM * L
    std::vector<TK> temp_mat(ParaV->nloc, 0.0);
    pTgemm_scalapack(ParaV, L_mat.data(), DMk.data(), temp_mat.data(), dim, dim, dim, 'C', 'N');
    pTgemm_scalapack(ParaV, temp_mat.data(), L_mat.data(), M_mat.data(), dim, dim, dim, 'N', 'N');
    // // test1
    // pTgemm_scalapack(ParaV, L_mat.data(), DMk.data(), temp_mat.data(), dim, dim, dim, 'N', 'N');
    // pTgemm_scalapack(ParaV, temp_mat.data(), L_mat.data(), M_mat.data(), dim, dim, dim, 'N', 'C');

    // diga(M): M = X^dagger * wg * X
    pdiag_scalapack(ParaV, dim, M_mat.data(), temp_wg.data(), wfc_X.data(), true);

    // inv(L), L_mat_inv will overwrite L_mat
    inv_matrix(ParaV, L_mat.data());

    // wfc = X * inv(L)
    pTgemm_scalapack(ParaV, wfc_X.data(), L_mat.data(), temp_wfc.data(), dim, dim, dim, 'N', 'N');
    // // test1
    // pTgemm_scalapack(ParaV, wfc_X.data(), L_mat.data(), temp_wfc.data(), dim, dim, dim, 'N', 'C');
    // // test2
    // pTgemm_scalapack(ParaV, L_mat.data(), wfc_X.data(), temp_wfc.data(), dim, dim, dim, 'N', 'N');




    // isolated atom case: diga(D): D = C^dagger * wg * C
    std::vector<TK> iden_mat(ParaV->get_local_size(), 0.0);
    get_identi_mat(ParaV, iden_mat);
    pTgemm_scalapack(ParaV, DMk.data(), iden_mat.data(), M_mat.data(), dim, dim, dim, 'N', 'N');
    pdiag_scalapack(ParaV, dim, M_mat.data(), temp_wg.data(), temp_wfc.data(), true);




    // convert
    for(int ib=0; ib<para_wfc->get_wfc_global_nbands(); ++ib)
    {
        // sort the natural occupation numbers from largest to smallest
        // note that the corresponding wfc also needs to change the order
        wg[ib] = temp_wg[para_wfc->get_wfc_global_nbands() - ib - 1];

        // wg[ib] = temp_wg[ib];
    }

    // change the order of the eigenvectors wfc so that they correspond to the eigenvalues ​​wg
    std::vector<TK> exch_mat(ParaV->get_local_size(), 0.0);
    std::vector<TK> temp_wfc2(ParaV->nloc, 0.0);
    get_exchange_mat(ParaV, exch_mat);
    pTgemm_scalapack(ParaV, temp_wfc.data(), exch_mat.data(), temp_wfc2.data(), dim, dim, dim, 'N', 'N');

    // rdmft::printMatrix_pointer(ParaV->get_row_size(), ParaV->get_col_size(), exch_mat.data(), "exch_mat", 10);

    if( para_wfc->get_wfc_global_nbands() == dim )
    {
        for(int iloc=0; iloc<para_wfc->nloc; ++iloc)
        {
            wfc[iloc] = temp_wfc2[iloc];
        }
    }
    else
    {
        for(int ib=0; ib<para_wfc->ncol_bands; ++ib)
        {
            const int loc_size = para_wfc->get_row_size();
            for(int ibasis=0; ibasis<loc_size; ++ibasis)
            {
                wfc[ibasis + ib*loc_size] = temp_wfc2[ ibasis + ib*loc_size ];
            }
        }
    }



}




//! get exp(R) from the skew-Hermitian matrix R
template <typename TK>
void exp_skew_hermi_mat(const Parallel_2D* para_mat, const std::vector<TK>& R, std::vector<TK>& exp_R)
{
    if( exp_R.size() != R.size() )
    {
        exp_R.resize(R.size(), 0.0);
    }

    // hermi_mat A = i * M
    std::vector<std::complex<double>> hermi_mat(R.size());
    std::complex<double> imag_one(0.0, 1.0);
    for(int i=0; i<R.size(); ++i)
    {
        hermi_mat[i] = imag_one * R[i];
    }

    const int global_row = para_mat->get_global_row_size();
    std::vector<double> diag_elem(global_row, 0.0);
    std::vector<std::complex<double>> exp_diag(para_mat->get_row_size() * para_mat->get_col_size(), 0.0);
    std::vector<std::complex<double>> egi_vector(para_mat->get_row_size() * para_mat->get_col_size(), 0.0);

    // A = V * diag_e * V^dagger
    rdmft::pdiag_scalapack( para_mat, global_row, hermi_mat.data(), diag_elem.data(), egi_vector.data() );

    std::complex<double> imag_nega_one(0.0, -1.0);
    int nrow = para_mat->get_row_size();
    for(int ic=0; ic<para_mat->get_col_size(); ++ic)
    {
        const int ic_global = para_mat->local2global_col(ic);
        for(int ir=0; ir<nrow; ++ir)
        {
            int ir_global = para_mat->local2global_row(ir);
            if( ic_global == ir_global )
            {
                exp_diag[ir+ic*nrow] = std::exp( imag_nega_one * diag_elem[ic_global] );
            }
        }
    }

    // exp(R) =  V * exp(-i * diag_e) * V^dagger
    std::vector<std::complex<double>> temp_mat(para_mat->get_row_size() * para_mat->get_col_size(), 0.0);
    rdmft::pTgemm_scalapack( para_mat, egi_vector.data(), exp_diag.data(),
                            temp_mat.data(), global_row, global_row, global_row, 'N', 'N' );
    rdmft::pTgemm_scalapack( para_mat, temp_mat.data(), egi_vector.data(),
                            exp_diag.data(), global_row, global_row, global_row, 'N', 'C' );
    if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        exp_R = exp_diag;
    }
    else
    {
        for(int i=0; i<exp_diag.size(); ++i)
        {
            exp_R[i] = std::real( exp_diag[i] );
        }
    }
}









/********* the following function is used in rdmft with libTorch *********/


// // convert vector to tensor
// template <typename T>
// void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor, const std::vector<int64_t>& shape)
// {
//     int64_t total = 1;
//     for (auto s : shape) total *= s;
//     assert(static_cast<int64_t>(vec.size()) == total);

//     torch::Dtype dtype;
//     if constexpr (std::is_same<T, double>::value)
//     {
//         dtype = torch::kDouble;
//         if (!tensor.defined() || tensor.sizes() != torch::IntArrayRef(shape) || tensor.dtype() != dtype)
//         {
//             tensor = torch::empty(shape, dtype);
//         }

//         std::memcpy(tensor.data_ptr<double>(), vec.data(), sizeof(double) * vec.size());
//     }
//     else if constexpr (std::is_same<T, std::complex<double>>::value)
//     {
//         dtype = torch::kComplexDouble;
//         if (!tensor.defined() || tensor.sizes() != torch::IntArrayRef(shape) || tensor.dtype() != dtype)
//         {
//             tensor = torch::empty(shape, dtype);
//         }

//         // convert std::complex<double> to c10::complex<double>
//         auto* data_ptr = tensor.data_ptr<c10::complex<double>>();
//         for (int64_t i = 0; i < total; ++i)
//         {
//             data_ptr[i] = c10::complex<double>(vec[i].real(), vec[i].imag());
//         }
//     }
//     else
//     {
//         static_assert(!sizeof(T), "unsupported type in vector to tensor");
//     }
// }


// template <typename T>
// void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor, std::initializer_list<int> shape_list)
// {
//     std::vector<int64_t> shape(shape_list.begin(), shape_list.end());
//     vector2tensor(vec, tensor, shape);
// }


// template <typename T>
// void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor, const std::vector<int>& shape_vec)
// {
//     std::vector<int64_t> shape(shape_vec.begin(), shape_vec.end());
//     vector2tensor(vec, tensor, shape);
// }


// convert vector to tensor
template <typename T>
void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor,
                   const std::vector<int64_t>& shape, bool need_grad = true)
{
    int64_t total = 1;
    for (auto s : shape) total *= s;
    assert(static_cast<int64_t>(vec.size()) == total);

    torch::Dtype dtype;
    if constexpr (std::is_same<T, double>::value)
    {
        dtype = torch::kDouble;
        if (!tensor.defined() || tensor.sizes() != torch::IntArrayRef(shape) || tensor.dtype() != dtype)
        {
            tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).requires_grad(need_grad));
        }
        else
        {
            tensor.set_requires_grad(need_grad);
        }

        std::memcpy(tensor.data_ptr<double>(), vec.data(), sizeof(double) * vec.size());
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        dtype = torch::kComplexDouble;
        if (!tensor.defined() || tensor.sizes() != torch::IntArrayRef(shape) || tensor.dtype() != dtype)
        {
            tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).requires_grad(need_grad));
        }
        else
        {
            tensor.set_requires_grad(need_grad);
        }

        // convert std::complex<double> to c10::complex<double>
        auto* data_ptr = tensor.data_ptr<c10::complex<double>>();
        for (int64_t i = 0; i < total; ++i)
        {
            data_ptr[i] = c10::complex<double>(vec[i].real(), vec[i].imag());
        }
    }
    else
    {
        static_assert(!sizeof(T), "unsupported type in vector to tensor");
    }
}


template <typename T>
void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor,
                   std::initializer_list<int> shape_list, bool need_grad = true)
{
    std::vector<int64_t> shape(shape_list.begin(), shape_list.end());
    vector2tensor(vec, tensor, shape, need_grad);
}


template <typename T>
void vector2tensor(const std::vector<T>& vec, torch::Tensor& tensor,
                   const std::vector<int>& shape_vec, bool need_grad = true)
{
    std::vector<int64_t> shape(shape_vec.begin(), shape_vec.end());
    vector2tensor(vec, tensor, shape, need_grad);
}



// convert tensor to vector
template <typename T>
void tensor2vector(const torch::Tensor& tensor, std::vector<T>& vec)
{
    auto flat = tensor.flatten();
    int64_t N = flat.numel();
    vec.resize(N);

    if constexpr (std::is_same<T, double>::value)
    {
        std::memcpy(vec.data(), flat.data_ptr<double>(), sizeof(double) * N);
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        const c10::complex<double>* data_ptr = flat.data_ptr<c10::complex<double>>();
        for (int64_t i = 0; i < N; ++i)
        {
            vec[i] = std::complex<double>(data_ptr[i].real(), data_ptr[i].imag());
        }
    }
    else
    {
        static_assert(!sizeof(T), "unsupported type in tensor to vector");
    }
}


//! copy the real and imaginary parts from a complex tensor
void split_complex_tensor(const torch::Tensor& R, torch::Tensor& R_real, torch::Tensor& R_imag);


//! recover a tensor from the real and imaginary parts
void merge_complex_tensor(const torch::Tensor& R_real, const torch::Tensor& R_imag, torch::Tensor& R);


// void torch_set_lr(torch::optim::LBFGS& opt, double new_lr);


template <typename OptimizerType>
void torch_set_lr(OptimizerType& opt, double new_lr)
{
    for (auto& group : opt.param_groups())
    {
        using OptionsType = typename std::decay<decltype(group.options())>::type;
        auto& options = static_cast<OptionsType&>(group.options());
        options.lr(new_lr);
    }
}


// temp
void torch_print_options(const torch::optim::LBFGS& opt);





}

#endif
