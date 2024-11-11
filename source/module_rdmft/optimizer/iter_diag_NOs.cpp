//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_lr/utils/lr_util.h"
// #include "module_base/blas_connector.h"
// #include "module_base/scalapack_connector.h"

// #include "module_psi/psi.h"

namespace rdmft
{


template <typename TK, typename TR>
IterDiag_NOs<TK, TR>::IterDiag_NOs()
{

}


template <typename TK, typename TR>
IterDiag_NOs<TK, TR>::~IterDiag_NOs()
{

}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::init(const int nk_total_in, const Parallel_2D& para_Fij_in, const Parallel_Orbitals& ParaV_in)
{
    this->nk_total = nk_total_in;
    this->nbands_total = PARAM.inp.nbands;
    this->para_Fij = &para_Fij_in;
    this->ParaV = &ParaV_in;
    // identi_mat = get_identi_mat(this->para_Fij); // temporary
    this->new_wfc.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->scale_zeta = 0.01; // PARAM.inp.scale_zeta_rdmft?

    this->lambda.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    for(int ik=0; ik<nk_total; ++ik)
    {
        diag_Fii[ik].resize(nbands_total);
        lambda[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size() );
    }
    this->Fock_like_mat = this->lambda;
    this->nos_rep_wfc = this->lambda;
    this->nos_rep_wfc0 = this->lambda;
}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::before_inner_loop(int* scale_factor)
{
    if( scale_factor != nullptr && scale_factor > 0 ) this->scale_zeta = *scale_factor;
    this->energy_drop = 0;
    this->energy_rise = 0;
}


template<typename TK, typename TR>
double IterDiag_NOs<TK, TR>::optimize_orb(RDMFT<TK, TR>& rdmft_solver)
{
    this->etotal_old = this->etotal;

    // this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);
    this->get_lambda(rdmft_solver.occ_number, rdmft_solver.fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);

    this->get_Fock();

    // diag(Fock)
    for(int ik=0; ik<nk_total; ++ik)
    {
        // get Fii and new_wfc in NOs
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, this->Fock_like_mat[ik].data(), diag_Fii[ik].data(), nos_rep_wfc[ik].data());
        // get new_wfc in NAOs
        rdmft::GkPsi( this->para_Fij, this->ParaV, nos_rep_wfc[ik][0], rdmft_solver.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
    }

    rdmft_solver.update_elec( nullptr, &(this->new_wfc) );
    this->etotal = rdmft_solver.cal_Energy();

    double diff_e = this->etotal - this->etotal_old;
    if(diff_e < 0) { ++this->energy_drop; }
    else { ++this->energy_rise; }

    this->nos_rep_wfc0 = this->nos_rep_wfc; // nos_rep_wfc0 *= nos_rep_wfc ?
    this->new_wfc.zero_out();

    return diff_e;
}



template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_start_guess(RDMFT<TK, TR>& rdmft_solver)
{
    // this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);
    this->get_lambda(rdmft_solver.occ_number, rdmft_solver.fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);

    // get start_Fock = symm_lambda
    std::vector< std::vector<TK> > symm_lambda = lambda;
    symmetr_lambda(this->para_Fij, this->lambda, symm_lambda);

    // diag(symm_lambda)
    for(int ik=0; ik<nk_total; ++ik)
    {
        // get start_Fii and start_wfc in NOs
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, symm_lambda[ik].data(), diag_Fii[ik].data(), nos_rep_wfc[ik].data());

        // get start_wfc in NAOs
        rdmft::GkPsi( this->para_Fij, this->ParaV, nos_rep_wfc[ik][0], rdmft_solver.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
    }

    rdmft_solver.update_elec( nullptr, &(this->new_wfc) );

    this->nos_rep_wfc0 = this->nos_rep_wfc;
    this->new_wfc.zero_out();
}



template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_lambda(const ModuleBase::matrix& wg, 
                                        const ModuleBase::matrix& wk_fun_occNum, 
                                        const std::vector< std::vector<TK> >& H_no_exx, 
                                        const std::vector< std::vector<TK> >& H_exx)
{
    // for(int i=0; i<lambda.size(); ++i) { lambda[i] = lambda_in[i]; }
    
    // times occNum
    for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    {
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            // use wg or occ_number???
            const double wg_local = wg(ik, para_Fij->local2global_col(ic));
            const double wk_fun_local = wk_fun_occNum(ik, para_Fij->local2global_col(ic));

            for(int ir=0; ir<nrow; ++ir)
            {
                this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*wg_local 
                                                + H_exx[ik][ir + ic*nrow]*wk_fun_local;
            }
        }
    } 
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_Fock()
{
    for(int ik=0; ik<this->nk_total; ++ik)
    {   
        // c++ perspective: only the upper triangle of F is correct (excluding the diagonal)
        antisymm_mat(this->para_Fij, nbands_total, this->lambda[ik].data(), this->Fock_like_mat[ik].data(), 1.0);

        // fortran perspective: only the lower triangle of F is correct (excluding the diagonal)
        // now make the upper triangle and diagonal elements correct as well
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = para_Fij->local2global_row(ir);

                if(ic_global > ir_global) 
                {
                    // the upper triangle
                    this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                }
                else if (ic_global == ir_global)
                {
                    // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
                    this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global];
                }
            }
        }
        set_zero_vector(diag_Fii[ik]);
    }

    // this->scale_Fock();

    // rotate Fock?
    this->check_hermi(this->Fock_like_mat);
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::scale_Fock()
{
    this->adjust_scale();

    for(int ik=0; ik<this->nk_total; ++ik)
    {
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = para_Fij->local2global_row(ir);
                
                if(ic_global != ir_global) 
                {
                    double c_rc = this->scale_zeta/std::abs(this->Fock_like_mat[ik][ir+ic*nrow]);
                    if( c_rc < 1.0 ) this->Fock_like_mat[ik][ir+ic*nrow] *= c_rc;
                }
            }
        }
    }
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::adjust_scale()
{
    // refer to octopus
    if( this->energy_drop > static_cast<int>(1.5*this->energy_rise) ) this->scale_zeta *= 1.01;
    else if( this->energy_drop < static_cast<int>(1.1*this->energy_rise) ) this->scale_zeta *= 0.95;

    // if( this->energy_drop > static_cast<int>(1.5*this->energy_rise) ) this->scale_zeta *= 1.05;
    // else if( this->energy_drop < static_cast<int>(1.1*this->energy_rise) ) this->scale_zeta *= 0.95;
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::rotate_Fock()
{

}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::check_hermi(std::vector< std::vector<TK> >& mat)
{
    for(int ik=0; ik<mat.size(); ++ik)
    {
        std::vector<TK> zero_mat(mat[ik].size(), 0.0);
        rdmft::antisymm_mat(this->para_Fij, PARAM.inp.nbands, mat[ik].data(), zero_mat.data(), 1.0);
        
        for(int iloc=0; iloc<zero_mat.size(); ++iloc)
        {
            if( std::abs(zero_mat[iloc]) > 1e-12 ) std::cout << "\n\n******\n" << "Fock_like_mat is not Hermitian" << "\n******\n" << std::endl;
        }
    }
}



template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::symmetr_lambda(const Parallel_2D* para_mat, 
                                            const std::vector< std::vector<TK> >& lambda, 
                                            std::vector< std::vector<TK> >& symm_lambda)
{
    for(int ik=0; ik<lambda.size(); ++ik)
    {
        LR_Util::matsym(lambda[ik].data(), PARAM.inp.nbands, *para_mat, symm_lambda[ik].data());
    }
}


// template <typename TK, typename TR>
// void IterDiag_NOs<TK, TR>::fill_diag_elem(const Parallel_2D* para_mat, 
//                                             const std::vector< std::vector<TK> >& mat_filling, 
//                                             std::vector< std::vector<TK> >& mat_filled)
// {
//     for(int ik=0; ik<mat_filled.size(); ++ik)
//     {
//         const int nrow = para_mat->get_row_size();
//         const int ncol = para_mat->get_col_size();
        
//         for(int i=0; i<nrow; ++i)
//         {
//             int i_global = para_mat->local2global_row(i);
//             for(int j=0; j<ncol; ++j)
//             {
//                 int j_global = para_mat->local2global_col(j);
//                 if( i_global == j_global ) { mat_filled[ i+j*nrow ] = mat_filling[ i+j*nrow ]; }
//             }
//         }
//     }
// }




// template<typename TK, typename TR>
// std::vector<TK> IterDiag_NOs<TK, TR>::get_identi_mat(const Parallel_2D* para_mat)
// {
//     std::vector<TK> tmp_mat(para_mat->get_local_size(), TK(0.0));
//     const int nrow = para_mat->get_row_size();
//     const int ncol = para_mat->get_col_size();

//     for(int i=0; i<nrow; ++i)
//     {
//         const int i_global = para_mat->local2global_row(i);
//         for(int j=0; j<ncol; ++j)
//         {
//             int j_global = para_mat->local2global_col(j);
//             if( i_global == j_global ) { tmp_mat[ i+j*nrow ] = 1.0; }
//         }
//     }
    
//     return tmp_mat;
// }




template class IterDiag_NOs<double, double>;
template class IterDiag_NOs<std::complex<double>, double>;
template class IterDiag_NOs<std::complex<double>, std::complex<double>>;




}


