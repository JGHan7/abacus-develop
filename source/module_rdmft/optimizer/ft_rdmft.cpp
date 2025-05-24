//==========================================================
// Author: Jingang Han
// DATE : 2025-05-23
//==========================================================


#include "module_rdmft/optimizer/ft_rdmft.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

namespace rdmft
{




template <typename TK, typename TR>
void FT_RDMFT<TK, TR>::init(const int nk_total_in,
                            const K_Vectors& kv_in,
                            const Parallel_2D& para_Fij_in,
                            const Parallel_Orbitals& ParaV_in,
                            RDMFT<TK, TR>* rdmft_solver_in,
                            const ModuleBase::matrix& ekb_in)
{
    rdmft::IDMFT<TK, TR>::init(nk_total_in, kv_in, para_Fij_in, ParaV_in, rdmft_solver_in);

    for(int ik=0; ik<this->diag_Fii.size(); ++ik)
    {
        for(int ib=0; ib<this->diag_Fii[ik].size(); ++ib)
        {
            this->diag_Fii[ik][ib] = ekb_in(ik, ib);
        }
    }
}


template <typename TK, typename TR>
void FT_RDMFT<TK, TR>::get_Fock()
{

    // modify digaonal elements of the Fock matrix
    this->rdmft_solver->cal_E_grad_occ_num();

    for(int ik=0; ik<this->Fock_like_mat.size(); ++ik)
    {
        std::fill(this->Fock_like_mat[ik].begin(), this->Fock_like_mat[ik].end(), 0.0);

        // // idmft: Fock matrix
        // for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
        // {
        //     this->Fock_like_mat[ik][iloc] = (this->rdmft_solver->Hij_no_exx[ik][iloc] + this->rdmft_solver->Hij_exx[ik][iloc]); // * wk_nospin[ik] ,no need to multiply k-point weight and spin weight
        // }

        // iterDiag: Fock-like matrix
        std::vector<TK> lambda = this->Fock_like_mat[ik];
        std::fill(lambda.begin(), lambda.end(), 0.0);
        int nrow = this->para_Fij->get_row_size();
        for(int ic=0; ic<this->para_Fij->get_col_size(); ++ic)
        {
            // use wg or occ_number??? occ_num!
            const double occ_num_local = this->rdmft_solver->wg(ik, this->para_Fij->local2global_col(ic));
            const double fun_occNum_local = this->rdmft_solver->wk_fun_occNum(ik, this->para_Fij->local2global_col(ic));

            for(int ir=0; ir<nrow; ++ir)
            {
                lambda[ir + ic*nrow] = this->rdmft_solver->Hij_no_exx[ik][ir + ic*nrow]*occ_num_local 
                                                + this->rdmft_solver->Hij_exx[ik][ir + ic*nrow]*fun_occNum_local;
            }
        }

        antisymm_mat(this->para_Fij, this->nbands, lambda.data(), this->Fock_like_mat[ik].data(), 1.0);
        // int nrow = this->para_Fij->get_row_size();
        for(int ic=0; ic<this->para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = this->para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = this->para_Fij->local2global_row(ir);

                if( ic_global == ir_global )
                {
                    // // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
                    // this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global];
                }
                else
                {
                    // double norm_Fij = std::abs( this->Fock_like_mat[ik][ir+ic*nrow] );
                    // this->max_off_diag_F = std::max(this->max_off_diag_F, norm_Fij);

                    if(ic_global > ir_global) 
                    {
                        // the upper triangle
                        this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                    }

                }
            }
        }


        // int nrow = this->para_Fij->get_row_size();
        for(int ic=0; ic<this->para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = this->para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = this->para_Fij->local2global_row(ir);

                if( ic_global == ir_global )
                {
                    // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
                    // 
                    double num = this->rdmft_solver->occ_number(ik, ic_global);
                    if( std::abs(1.0 - num) < 1e-16 )
                    {
                        num = 1.0 - 1e-16;
                    }
                    else if( num < 1e-20 )
                    {
                        num = 1e-20;
                    }

                    // this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global] + this->rdmft_solver->occNum_wfcHamiltWfc(ik, ic_global);
                    this->Fock_like_mat[ik][ir+ic*nrow] = this->rdmft_solver->occNum_wfcHamiltWfc(ik, ic_global) + this->kappa * std::log( (1 - num)/num );
                }
            }
        }

    }

    std::cout << "\n\n******\n" << "new get_Fock()" << "\n******\n" << std::endl;

    if(PARAM.inp.rotate_fock)
    {
        this->rotate_Fock();
    }

}



template <typename TK, typename TR>
void FT_RDMFT<TK, TR>::optimize_kappa()
{
    this->dE_dk = 0.0;
    this->rdmft_solver->cal_E_grad_occ_num();
    const ModuleBase::matrix& occ_num = this->rdmft_solver->occ_number;

    for(int ik=0; ik<this->nk_total; ++ik)
    {
        for(int ib=0; ib<this->nbands; ++ib)
        {
            double num_ik_ib = occ_num(ik, ib);
            if( std::abs(1.0 - num_ik_ib) < 1e-16 )
            {
                num_ik_ib = 1.0 - 1e-16;
            }
            else if( num_ik_ib < 1e-20 )
            {
                num_ik_ib = 1e-20;
            }

            double num = num_ik_ib * (1 - num_ik_ib) * std::log( (1 - num_ik_ib)/num_ik_ib );
            this->dE_dk += this->rdmft_solver->occNum_wfcHamiltWfc(ik, ib) * num / this->kappa;
        }
    }

    // could use line search combining CG, BFGS methods, etc.
    this->kappa += -this->dE_dk;

    if( this->kappa < 0 )
    {
        this->kappa = 1e-6;
    }

    std::cout << "\n" << "optimize kappa: \ndelta_kappa = " << -this->dE_dk << "\nnew kappa = " << this->kappa << "\n" << std::endl;

    // optimize the occ_number
    ModuleBase::matrix occ_num_pass(this->nk_total, this->nbands);
    rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, this->rdmft_solver->occ_number.c, "occ_number before opti_kappa", 10);
    this->opti_occ_num(occ_num_pass);
    rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_number after opti_kappa", 10);

    // update the occupation number
    this->rdmft_solver->update_elec( &occ_num_pass );


}














template class FT_RDMFT<double, double>;
template class FT_RDMFT<std::complex<double>, double>;
template class FT_RDMFT<std::complex<double>, std::complex<double>>;


}
