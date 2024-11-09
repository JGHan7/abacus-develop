//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================

// #include "module_rdmft/rdmft.h"
// #include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_lr/utils/lr_util.h"
// #include "module_base/blas_connector.h"
// #include "module_base/scalapack_connector.h"


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
void IterDiag_NOs<TK, TR>::optimize_orb(RDMFT<TK, TR>& rdmft_solver)
{
    this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);

}



template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_start_guess(RDMFT<TK, TR>& rdmft_solver)
{
    this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);
    // std::vector< std::vector<TK> > symm_lambda( this->lambda.size(), std::vector<TK>( this->lambda[0].size() ) );
    std::vector< std::vector<TK> > symm_lambda = lambda;
    symmetr_lambda(this->para_Fij, this->lambda, symm_lambda);

    // diag(symmlambda), get start_Fii and start_NOs
    for(int ik=0; ik<nk_total; ++ik)
    {
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, symm_lambda[ik].data, diag_Fii[ik].data(), nos_rep_wfc[ik].data());
    }

    // rdmft_solver.update_elec(nullptr, start_NOs);

    // fill_diag_elem(this->para_Fij,  , this->Fock_like_mat);
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
    for(int ik=0; ik<nk_total; ++ik)
    {   
        // c++ perspective: only the upper triangle of F is correct (excluding the diagonal)
        antisymm_mat(this->para_Fij, nbands_total, this->lambda[ik].data(), this->Fock_like_mat[ik].data(), 1.0);

        // fortran perspective: only the lower triangle of F is correct (excluding the diagonal)
        // now make the upper triangle correct as well
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = para_Fij->local2global_row(ir);
                
                if(ic_global > ir_global) this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
            }
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


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::fill_diag_elem(const Parallel_2D* para_mat, 
                                            const std::vector< std::vector<TK> >& mat_filling, 
                                            std::vector< std::vector<TK> >& mat_filled)
{
    for(int ik=0; ik<mat_filled.size(); ++ik)
    {
        const int nrow = para_mat->get_row_size();
        const int ncol = para_mat->get_col_size();
        
        for(int i=0; i<nrow; ++i)
        {
            int i_global = para_mat->local2global_row(i);
            for(int j=0; j<ncol; ++j)
            {
                int j_global = para_mat->local2global_col(j);
                if( i_global == j_global ) { mat_filled[ i+j*nrow ] = mat_filling[ i+j*nrow ]; }
            }
        }
    }
}








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


