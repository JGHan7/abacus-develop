//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_base/blas_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_lr/utils/lr_util.h"

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
void IterDiag_NOs<TK, TR>::init(int nk_total_in, const Parallel_2D* para_Fij_in, const Parallel_Orbitals* ParaV_in)
{
    this->nk_total = nk_total_in;
    this->para_Fij = para_Fij_in;
    this->ParaV = ParaV_in;

    this->lambda.resize(nk_total);
    this->Fock_like_mat.resize(nk_total);
    for(int ik=0; ik<nk_total; ++ik)
    {
        lambda[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size() );
        Fock_like_mat[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size() );
    }
}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::optimize_orb(RDMFT<TK, TR>& rdmft_solver)
{
    this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);

}



template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_lambda(ModuleBase::matrix& wg, 
                                        ModuleBase::matrix& wk_fun_occNum, 
                                        std::vector< std::vector<TK> >& H_no_exx, 
                                        std::vector< std::vector<TK> >& H_exx)
{
    // for(int i=0; i<lambda.size(); ++i) { lambda[i] = lambda_in[i]; }
    
    // times occNum
    for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    {
        int nb_col = para_Fij->get_col_size();
        for(int inb_col=0; inb_col<para_Fij->nb_col; ++inb_col)
        {
            // use wg or occ_number???
            int wg_local = wg(ik, para_Fij->local2global_col(inb_col));
            int wk_fun_local = wk_fun_occNum(ik, para_Fij->local2global_col(inb_col));

            for(int inb_row=0; inb_row<para_Fij->get_row_size(); ++inb_row)
            {
                this->lambda[ik][inb_row*nb_col + inb_col] = H_no_exx[ik][inb_row*nb_col + inb_col]*wg_local 
                                                        + H_exx[ik][inb_row*nb_col + inb_col]*wk_fun_local;
            }
        }
    } 
}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_start_guess(RDMFT<TK, TR>& rdmft_solver)
{
    this->get_lambda(rdmft_solver.wg, rdmft_solver.wk_fun_occNum, rdmft_solver.Hij_no_exx, rdmft_solver.Hij_exx);
    std::vector< std::vector<TK> > symm_lambda = lambda;
    symmetr_lambda(this->para_Fij, this->lambda, symm_lambda);

    // diag(symmlambda), get start_Fii and start_NOs 

    // fill_diag_elem(this->para_Fij,  , this->Fock_like_mat);
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::symmetr_lambda(Parallel_2D* para_mat, 
                                            std::vector< std::vector<TK> >& lambda, 
                                            std::vector< std::vector<TK> >& symm_lambda)
{
    for(int ik=0; ik<lambda.size(); ++ik)
    {
        LR_Util::matsym(lambda[ik].data(), PARAM.inp.nbands, *para_mat, symm_lambda[ik].data());
    }
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::fill_diag_elem(Parallel_2D* para_mat, 
                                            std::vector< std::vector<TK> >& mat_filling, 
                                            std::vector< std::vector<TK> >& mat_filled)
{
    for(int ik=0; ik<mat_filled.size(); ++ik)
    {
        const int nrow = para_mat.get_row_size();
        const int ncol = para_mat.get_col_size();
        
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








}


