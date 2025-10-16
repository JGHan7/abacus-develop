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

#ifdef __MPI
#include <mpi.h>
#endif
#include "module_base/parallel_comm.h"

namespace rdmft
{

template <>
void reduce_all_max<int>(int& object)
{
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &object, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    return;
}


template <>
void reduce_all_max<double>(double& object)
{
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &object, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    return;
}


template <>
void reduce_all_max<int>(int* object, const int n)
{
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, object, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    return;
}


template <>
void reduce_all_max<double>(double* object, const int n)
{
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, object, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    return;
}


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

#ifdef __MPI
    pdtran_(&gloabl_row_mat, &gloabl_row_mat, &a, mat, &one_int, &one_int, para_mat->desc, &b, asym_mat, &one_int, &one_int, para_mat->desc);
#endif

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
                                const bool get_egivector)
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
    assert( info == 0 );

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
                                double beta)
{
    const int one_int = 1;
    if( para_B == nullptr ) { para_B = para_A; }
    if( para_C == nullptr ) { para_C = para_A; }

#ifdef __MPI
    pdgemm_( &op_A, &op_B, &global_row_C, &global_col_C, &global_contract_index, &alpha, A, &one_int, &one_int, para_A->desc,
            B, &one_int, &one_int, para_B->desc, &beta, C, &one_int, &one_int, para_C->desc );
#endif
}


// to compute C = alpha * A.? * B.? + beta * C
// all use the fortran perspective, not cpp
template <>
void Tgemm_lapack<double>(const double* A,
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


/********* the following function is used by the BFGS_method *********/


//! implementation of cubic interpolation
//! learned from pyTorch with minor modifications: https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/optim/lbfgs.cpp
double cubic_interpolate(double x1, double f1, double g1,
                            double x2, double f2, double g2,
                            std::pair<double, double>* bounds)
{
    double x_min = 0.0;
    double x_max = 0.0;
    if(bounds == nullptr)
    {
        auto p = std::minmax({x1, x2});
        x_min = p.first;
        x_max = p.second;
    }
    else
    {
        x_min = std::min(bounds->first, bounds->second);
        x_max = std::max(bounds->first, bounds->second);
    }

    double trial_x = 0.0;
    double d1 = (g1 + g2) - (3 * (f1 - f2) / (x1 - x2));
    double d2_square = std::pow(d1, 2) - g1 * g2;
    if (d2_square >= 0)
    {
        double d2 = std::sqrt(d2_square);
        // minimum point of the fitted cubic polynomial
        double min_point = 0.0;
        if (x1 <= x2)
        {
            min_point = x2 - ((x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2)));
        }
        else
        {
            min_point = x1 - ((x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)));
        }
        trial_x =  std::min(std::max(min_point, x_min), x_max);
    }
    else
    {
        trial_x =  (x_min + x_max) / 2.0;
    }

    return trial_x;
}



/********* the following function is just used by the EBI_constraint method *********/

// check this approximation using std::erf( erf_inv_own(x) ) - x
// double erf_inv_own(double x) 
// {
//     if (x < -1.0 || x > 1.0) {
//         throw std::domain_error("Input out of range: erf_inv(x) requires -1 <= x <= 1.");
//     }

//     if (x == 0.0) return 0.0;
//     if (x == 1.0) return std::numeric_limits<double>::infinity();
//     if (x == -1.0) return -std::numeric_limits<double>::infinity();

//     // High-precision initial approximation
//     double w, p;
//     if (std::abs(x) <= 0.7) 
//     {
//         w = 0.5 * (1 - x);
//         p = std::sqrt(-2.0 * std::log(w));
//         // Abramowitz & Stegun (26.2.23) initial Approximation
//         p = (((-0.140543331 * p + 0.914624893) * p - 1.645349621) * p + 0.886226899) /
//             ((((0.012229801 * p - 0.329097515) * p + 1.442710462) * p - 2.118377725) * p + 1.0);
//     }
//     else
//     {
//         w = std::sqrt(-std::log((1.0 - std::abs(x)) / 2.0));
//         p = (((1.641345311 * w + 3.429567803) * w - 1.62490649) * w - 1.970840454) /
//             ((1.637067800 * w + 3.543889200) * w + 1.0);
//         if (x < 0) p = -p;
//     }

//     // Newton-Raphson iterative Improvement
//     for (int i = 0; i < 5; ++i) {
//         double err = std::erf(p) - x; // current Error
//         double deriv = 2.0 / std::sqrt(M_PI) * std::exp(-p * p); // derivative of erf
//         p -= err / deriv; // update
//     }

//     return p;
// }

// check this approximation using std::erf( erf_inv_own(x) ) - x
double erf_inv_own(double x) 
{
    if (x < -1.0 || x > 1.0)
    {
        throw std::domain_error("Input out of range: erf_inv(x) requires -1 <= x <= 1.");
    }

    if (x == 0.0) return 0.0;
    if (x == 1.0) return std::numeric_limits<double>::infinity();
    if (x == -1.0) return -std::numeric_limits<double>::infinity();

    double result; 
    double abs_x = std::abs(x);

    // Piecewise approach
    if (abs_x <= 0.4)
    {
        // Taylor expansion for |x| < 0.5
        const double p = x * x;
        result = x * (1.0 + p * (1.0 / 3.0 + p * (1.0 / 10.0 + p * (1.0 / 42.0))));
    }
    else
    {
        double w = 0.0;
        // Abramowitz and Stegun method for |x| > 0.5
        if (abs_x > 0.7)
        {
            w = std::sqrt(-std::log((1.0 - abs_x) / 2.0));
            // More accurate initial guess
            result = (((1.641345311 * w + 3.429567803) * w - 1.62490649) * w - 1.970840454) / 
                     ((1.637067800 * w + 3.543889200) * w + 1.0);
        }
        else
        {
            w = std::sqrt(-2.0 * std::log( 0.5 * (1 - x) ));
            // Intermediate region initial guess
            result = x * (((-0.140543331 * w + 0.914624893) * w - 1.645349621) * w + 0.886226899) /
                         ((((0.012229801 * w - 0.329097515) * w + 1.442710462) * w - 2.118377725) * w + 1.0);
        }

        if (x < 0) result = -result;
    }

    // Newton-Raphson refinement
    for (int i = 0; i < 4; ++i) {
        double err = std::erf(result) - x; 
        double deriv = 2.0 / std::sqrt(M_PI) * std::exp(-result * result); 
        result -= err / deriv; 
    }

    return result;
}


double erf_der1(double x)
{
    double y = ( 2.0 * std::exp( -std::pow(x, 2) ) )/std::sqrt(ModuleBase::PI);
    return y;
}


double erf_der2(double x)
{
    double y = ( -4.0 * x * std::exp( -std::pow(x, 2) ) )/std::sqrt(ModuleBase::PI);
    return y;
}


// void random_descend(std::vector<double>& num, const double* value, int* location, const std::vector<double>* num_symm_k = nullptr)
// {
//     // random seed, requires hardware support
//     std::random_device rd;
//     // mersenne Twister engine
//     std::mt19937 gen(rd());     // std::mt19937 gen(42);
//     // a uniform distribution in the range (0.0, 1.0)
//     std::uniform_real_distribution<> dis( std::nextafter(0.0, 1.0), 1.0 );

//     for(int i=0; i<num.size(); ++i) { num[i] = dis(gen); }
//     // sort descending
//     std::sort(num.begin(), num.end(), std::greater<>());

//     if( value != nullptr && location != nullptr )
//     {
//         double sum_num = 0.0;
//         for(int j=0; j<num.size(); ++j)
//         {
//             sum_num += num[j];
//             if( sum_num > *value )
//             {
//                 *location = j;
//                 break;
//             }
//         }
//     }
// }
void random_descend(std::vector<double>& num, double max)
{
    // random seed, requires hardware support
    std::random_device rd;
    // mersenne Twister engine
    std::mt19937 gen(rd());     // std::mt19937 gen(42);
    // a uniform distribution in the range (0.0, 1.0)
    std::uniform_real_distribution<> dis( std::nextafter(0.0, max), max );

    for(int i=0; i<num.size(); ++i) { num[i] = dis(gen); }
    // sort descending
    std::sort(num.begin(), num.end(), std::greater<>());
}


int smallest_loc_big_value(const int nk_nospin,
                            const int nbands,
                            const double value,
                            const std::vector<double>& num,
                            const double* num_symm_k)
{
    int location = 0;
    double sum = 0.0;
    for(int ib=0; ib<nbands; ++ib)
    {
        for(int ik=0; ik<nk_nospin; ++ik)
        {
            sum += num[ib*nk_nospin + ik] * num_symm_k[ik];
            if( sum > value )
            {
                location = ib*nk_nospin + ik;
                return location;
            }
        }
    }

    std::cout << "\n\n*******\nPossible error in rdmft calculation: EBI\n*******\n" << std::endl;
    return 0;
}




/********* the following function is used by mixing in rdmft *********/


void occ_num2wg(const K_Vectors* kv, const ModuleBase::matrix& occ_num, ModuleBase::matrix& wg)
{
    wg = (occ_num);
    for(int ik=0; ik < wg.nr; ++ik)
    {
        for(int inb=0; inb < wg.nc; ++inb)
        {
            wg(ik, inb) *= kv->wk[ik];
        }
    }
}


void wg2occ_num(const K_Vectors* kv, const ModuleBase::matrix& wg, ModuleBase::matrix& occ_num)
{
    occ_num = (wg);
    for(int ik=0; ik < wg.nr; ++ik)
    {
        for(int inb=0; inb < wg.nc; ++inb)
        {
            occ_num(ik, inb) /= kv->wk[ik];
        }
    }
}







// double
template<>
torch::TensorOptions torch_dtype<double>()
{
    return torch::dtype(torch::kDouble);
}

// std::complex<double>
template<>
torch::TensorOptions torch_dtype<std::complex<double>>()
{
    return torch::dtype(torch::kComplexDouble);
}


//! copy the real and imaginary parts from a complex tensor
void split_complex_tensor(const torch::Tensor& R, torch::Tensor& R_real, torch::Tensor& R_imag)
{
    // TORCH_CHECK(R.dtype() == torch::kComplexDouble, "Input tensor must be complex double");

    if ( R.is_complex() )
    {
        R_real = torch::real(R).clone().detach().set_requires_grad(true);
        R_imag = torch::imag(R).clone().detach().set_requires_grad(true);
    }
    else
    {
        R_real = R.clone().detach().set_requires_grad(true);
        R_imag = torch::zeros_like(R_real).detach().set_requires_grad(true);
    }
}


//! recover a tensor from the real and imaginary parts
void merge_complex_tensor(const torch::Tensor& R_real, const torch::Tensor& R_imag, torch::Tensor& R)
{
    TORCH_CHECK(R_real.sizes() == R_imag.sizes(), "Real and Imag tensors must have the same shape");
    TORCH_CHECK(R_real.dtype() == torch::kDouble && R_imag.dtype() == torch::kDouble, "Real and Imag tensors must be double");
    TORCH_CHECK(R.is_complex(), "Target tensor R must be complex");
    TORCH_CHECK(R.sizes() == R_real.sizes(), "Target tensor R must have the same shape as real/imag parts");

    R = torch::complex(R_real, R_imag);

    // does not change the memory address and other properties of R
    // auto R_tmp = torch::complex(R_real, R_imag);
    // R.copy_(R_tmp);

}


// void torch_set_lr(torch::optim::LBFGS& opt, double new_lr)
// {
//     for (auto& group : opt.param_groups())
//     {
//         auto& options = static_cast<torch::optim::LBFGSOptions&>(group.options());
//         options.lr(new_lr);
//     }
// }


void torch_print_options(const torch::optim::LBFGS& opt)
{
    int i = 0;
    for (const auto& group : opt.param_groups()) {
        const auto& options = static_cast<const torch::optim::LBFGSOptions&>(group.options());

        std::cout << "Param group " << i++ << ":\n";
        std::cout << "  lr              = " << options.lr() << "\n";
        std::cout << "  max_iter        = " << options.max_iter() << "\n";
        std::cout << "  tolerance_grad  = " << options.tolerance_grad() << "\n";
        std::cout << "  tolerance_change= " << options.tolerance_change() << "\n";
        std::cout << "  history_size    = " << options.history_size() << "\n";
        std::cout << "  line_search_fn  = "
                << options.line_search_fn().value_or("none")
                << "\n";
        std::cout << std::endl;
    }
}


template <>
double median_sorted(std::vector<std::complex<double>> data)
{
    if (data.empty())
    {
        throw std::domain_error("median of empty vector");
    }

    std::vector<double> data_dou(data.size());
    for(int i=0; i<data.size(); ++i)
    {
        data_dou[i] = std::real(data[i]);
    }

    std::sort(data_dou.begin(), data_dou.end());
    int n = data_dou.size();
    if (n % 2 == 1)
    {
        return data_dou[n / 2];
    }
    else
    {
        return 0.5 * (data_dou[n / 2 - 1] + data_dou[n / 2]);
    }
}



}
