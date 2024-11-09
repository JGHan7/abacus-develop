//==========================================================
// Author: Jingang Han
// DATE : 2024-11-09
//==========================================================

#include "module_rdmft/optimizer/optimizer_tools.h"


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
                                double* egienvector)
{
    const char jobz = 'V', uplo = 'U';
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













}
