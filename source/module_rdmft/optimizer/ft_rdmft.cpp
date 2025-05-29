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

    this->dE_dk.resize(this->nk_total*this->nbands, 0.0);
    this->pk.resize(this->nk_total*this->nbands, 0.0);
    this->bfgs_opti_k.init(this->nk_total, this->nbands);
    this->kappa_tensor.resize(this->nk_total*this->nbands, this->kappa);
    this->average_k = this->kappa;

}


template <typename TK, typename TR>
void FT_RDMFT<TK, TR>::get_Fock()
{

    // modify digaonal elements of the Fock matrix
    this->rdmft_solver->cal_E_grad_occ_num();
    ModuleBase::matrix dE_docc_num = this->rdmft_solver->occNum_wfcHamiltWfc;

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
                    else if( num < 1e-20 ) // 1e-12?
                    {
                        num = 1e-20;
                    }

                    // test
                    if( num < 1e-5 && GlobalC::exx_info.info_global.cal_exx && PARAM.inp.rdmft_power_alpha != 1.0 )
                    {
                        dE_docc_num(ik, ic_global) = 0.0;
                    }

                    // this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global] + this->rdmft_solver->occNum_wfcHamiltWfc(ik, ic_global);
                    // this->Fock_like_mat[ik][ir+ic*nrow] = this->rdmft_solver->occNum_wfcHamiltWfc(ik, ic_global) + this->kappa * std::log( (1 - num)/num );

            // this->kappa or this->kappa_tensor[ij] ???????!!!!!!!!!!!!!!!!!!!!!!

                    // this->Fock_like_mat[ik][ir+ic*nrow] = dE_docc_num(ik, ic_global) + this->kappa * std::log( (1 - num)/num );
                    // this->Fock_like_mat[ik][ir+ic*nrow] = dE_docc_num(ik, ic_global) + this->kappa_tensor[ik*this->nbands + ic_global] * std::log( (1 - num)/num );
                    this->Fock_like_mat[ik][ir+ic*nrow] = dE_docc_num(ik, ic_global) + this->average_k * std::log( (1 - num)/num );
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
    this->dE_dk_sum = 0.0;
    this->dE_dk.resize(this->dE_dk.size(), 0.0);
    this->rdmft_solver->cal_E_grad_occ_num();
    
    const ModuleBase::matrix& occ_num = this->rdmft_solver->occ_number;
    ModuleBase::matrix wk_wfc_Vnoexx_wfc(this->nk_total, this->nbands);
    ModuleBase::matrix wk_wfc_Vexx_wfc(this->nk_total, this->nbands);
    this->rdmft_solver->get_wk_wfcHwfc(wk_wfc_Vnoexx_wfc, wk_wfc_Vexx_wfc);

    for(int ik=0; ik<this->nk_total; ++ik)
    {
        for(int ib=0; ib<this->nbands; ++ib)
        {
            double num_ik_ib = occ_num(ik, ib);
            if( std::abs(1.0 - num_ik_ib) < 1e-16 )
            {
                num_ik_ib = 1.0 - 1e-16;
            }
            else if( num_ik_ib < 1e-12 )
            {
                num_ik_ib = 1e-12;
            }

            // double num = num_ik_ib * (1 - num_ik_ib) * std::log( (1 - num_ik_ib)/num_ik_ib );
            // this->dE_dk_sum += this->rdmft_solver->occNum_wfcHamiltWfc(ik, ib) * num / this->kappa;
            // this->dE_dk[ik*this->nbands + ib] = this->rdmft_solver->occNum_wfcHamiltWfc(ik, ib) * num / this->kappa_tensor[ik*this->nbands + ib];

            double alpha = PARAM.inp.rdmft_power_alpha;
            double factor_no_exx = num_ik_ib * (1 - num_ik_ib) * std::log( (1 - num_ik_ib)/num_ik_ib );
            double factor_exx = alpha * std::pow(num_ik_ib, alpha) * (1 - num_ik_ib) * std::log( (1 - num_ik_ib)/num_ik_ib );
            this->dE_dk[ik*this->nbands + ib] = ( wk_wfc_Vnoexx_wfc(ik, ib)*factor_no_exx + wk_wfc_Vexx_wfc(ik, ib)*factor_exx )/ this->kappa_tensor[ik*this->nbands + ib];
        }
    }

    if( this->kappa < 0 )
    {
        this->kappa = 1e-6;
    }

    if( first_opti_k )
    {
        this->bfgs_opti_k.get_pk(this->dE_dk, this->kappa_tensor, this->pk, first_opti_k);
        this->first_opti_k =false;
    }
    else
    {
        this->bfgs_opti_k.get_pk(this->dE_dk, this->kappa_tensor, this->pk);
    }



    // could use line search combining CG, BFGS methods, etc.
    // this->kappa += -this->dE_dk_sum;
    this->max_diff_kappa = 0.0;
    for(int i=0; i<this->kappa_tensor.size(); ++i)
    {
        this->max_diff_kappa = std::max(this->max_diff_kappa, std::abs(this->pk[i]));
    }

    this->average_k = 0.0;
    int num_kappa = this->kappa_tensor.size();
    for(int i=0; i<this->kappa_tensor.size(); ++i)
    {
        this->kappa_tensor[i] += this->pk[i] / ( this->max_diff_kappa * 0.2 );
        if( this->kappa_tensor[i] < 0 )
        {
            this->kappa_tensor[i] = 1e-4; // how much is proper?
            num_kappa -= 1;
        }
        else
        {
            this->average_k += this->kappa_tensor[i];
        }
    }
    // this->average_k /= this->kappa_tensor.size();
    this->average_k /= num_kappa;

    // std::cout << "\n" << "optimize kappa: \ndelta_kappa = " << -this->dE_dk_sum << "\nnew kappa = " << this->kappa << "\n" << std::endl;
    rdmft::printMatrix_pointer(this->nk_total, this->nbands, this->rdmft_solver->occNum_wfcHamiltWfc.c, "dE_dn", 10);
    rdmft::printMatrix_pointer(this->nk_total, this->nbands, this->dE_dk.data(), "dE_dkappa", 10);
    rdmft::printMatrix_pointer(this->nk_total, this->nbands, this->pk.data(), "pk", 10);
    rdmft::printMatrix_pointer(this->nk_total, this->nbands, this->kappa_tensor.data(), "new kappa-tensor", 10);

    // optimize the occ_number
    ModuleBase::matrix occ_num_pass(this->nk_total, this->nbands);
    rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, this->rdmft_solver->occ_number.c, "occ_number before opti_kappa", 10);
    this->opti_occ_num(occ_num_pass);
    rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_number after opti_kappa", 10);

    // update the occupation number
    this->rdmft_solver->update_elec( &occ_num_pass );


}



template <typename TK, typename TR>
double FT_RDMFT<TK, TR>::cal_occ_num(const int is)
{
    double tot_occ_num = 0.0;

    for(int ik=0; ik<this->nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            
            this->occ_number[is][ik*this->nbands + ib] = 1.0 / ( 1 + std::exp( ( this->diag_Fii[ is*this->nk_nospin + ik][ib] - this->mu[is] ) / this->kappa_tensor[(is*this->nk_nospin + ik)*this->nbands + ib] ) ) ;

            tot_occ_num += this->occ_number[is][ik*this->nbands + ib] * this->num_symm_k[is*this->nk_nospin + ik];
        }
    }

    return tot_occ_num;
}










template class FT_RDMFT<double, double>;
template class FT_RDMFT<std::complex<double>, double>;
template class FT_RDMFT<std::complex<double>, std::complex<double>>;


}
