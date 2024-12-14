//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================

#include <algorithm>
// #include "module_rdmft/rdmft.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_lr/utils/lr_util.h"
#include "module_base/parallel_reduce.h"
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
    this->naos_rep_wfc1.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->scale_zeta = 0.01; // PARAM.inp.scale_zeta_rdmft?
    this->scale_zeta_vector.resize(nk_total);

    // malloc
    this->lambda.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    this->rotation_mat.resize(nk_total);
    for(int ik=0; ik<nk_total; ++ik)
    {
        diag_Fii[ik].resize(nbands_total, 0.0);
        lambda[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );

        // or write in before_opti(), if you update the fixed NOs representation after each ONs optimization
        rdmft::get_identi_mat( para_Fij, this->rotation_mat[ik] );
    }
    this->Fock_like_mat = this->lambda;
    this->nos_rep_wfc = this->lambda;

    // temp
    this->if_rotate_Fock = false;
}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::before_opti(int* scale_factor)
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

    // // diag(Fock)
    // for(int ik=0; ik<nk_total; ++ik)
    // {
    //     // std::fill( nos_rep_wfc[ik].begin(), nos_rep_wfc[ik].end(), 0.0 );
    //     // std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);

    //     // get Fii and new_wfc in NOs
    //     rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, this->Fock_like_mat[ik].data(),
    //                                 this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());
    //     // get new_wfc in NAOs
    //     rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], rdmft_solver.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );

    // }

    // test T
    Parallel_2D para_wfc;
    #ifdef __MPI
        para_wfc.set(this->ParaV->desc[2], nbands_total, this->ParaV->nb, this->ParaV->blacs_ctxt); // maybe in default, PARAM.inp.nb2d = 0, can't be used
    #endif

    for(int ik=0; ik<nk_total; ++ik)
    {
        // std::fill( nos_rep_wfc[ik].begin(), nos_rep_wfc[ik].end(), 0.0 );
        // std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);gg

        // get Fii and new_wfc in NOs
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, this->Fock_like_mat[ik].data(),
                                    this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());
        // rdmft::printMatrix_pointer(para_Fij->get_row_size(), para_Fij->get_col_size(), this->Fock_like_mat[ik].data(), "Fock mat", 10); // test
        // get new_wfc in NAOs
        if( !this->if_get_wfc1 )
        {
            // std::cout << "\n******\n" << "iterDiag: 0.1, once" << "\n******\n" << std::endl;
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], rdmft_solver.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );

            // // test T
            // rdmft::pTgemm_scalapack( &para_wfc, &(rdmft_solver.wfc(ik, 0, 0)), this->nos_rep_wfc[ik].data(), &(this->new_wfc(ik, 0, 0)),
            //                             this->ParaV->desc[2], nbands_total, nbands_total, 'N', 'N', this->para_Fij, &para_wfc );
        }
        else
        {
            // std::cout << "\n******\n" << "iterDiag: 0.2, many" << "\n******\n" << std::endl;
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );

            // std::vector<TK> mat_temp = this->rotation_mat[ik];
            // rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->nos_rep_wfc[ik].data(),
            //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );
        }
        
        // get rotation_mat, rotation_mat_t-step = G_t * G_t-1 * ... * G_1
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->nos_rep_wfc[ik].data(),
        //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );
        std::vector<TK> mat_temp = this->rotation_mat[ik];
        rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->nos_rep_wfc[ik].data(),
                                    this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );

        // test T

    }

    rdmft_solver.update_elec( nullptr, &(this->new_wfc) );
    this->etotal = rdmft_solver.cal_Energy();

    double diff_e = this->etotal - this->etotal_old;
    if(diff_e < 0) { ++this->energy_drop; }
    else { ++this->energy_rise; }

    if( !this->if_get_wfc1 && this->if_rotate_Fock )
    {
        // rotate the Fock-like matrix to the first step NOs representation,
        // then new_wfc = nos_rep_wfc * ( nos_rep_wfc1 * NAOs_rep_wfc0) for each iteration

        // get NAOs_rep_wfc1 = nos_rep_wfc1 * NAOs_rep_wfc0
        TK* p_new_wfc = &this->new_wfc(0, 0, 0);
        TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
        for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

        // TK* p_new_wfc = &( rdmft_solver.wfc(0, 0, 0) );
        // TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
        // for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

        this->if_get_wfc1 = true;
        // std::cout << "\n******\n" << "iterDiag: 0.3, once" << "\n******\n" << std::endl;
    }
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
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, symm_lambda[ik].data(),
                                    this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());

        // get start_wfc in NAOs
        rdmft::GkPsi( this->para_Fij, this->ParaV, nos_rep_wfc[ik][0], rdmft_solver.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );

        // temp
        // get rotation_mat, rotation_mat_t-step = G_t * G_t-1 * ... * G_1
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->nos_rep_wfc[ik].data(),
        //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );
        // std::vector<TK> mat_temp = this->rotation_mat[ik];
        // rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->nos_rep_wfc[ik].data(),
        //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );
    }

    // if( !this->if_get_wfc1 && this->if_rotate_Fock )
    // {
    //     // rotate the Fock-like matrix to the first step NOs representation,
    //     // then new_wfc = nos_rep_wfc * ( nos_rep_wfc1 * NAOs_rep_wfc0) for each iteration

    //     // get NAOs_rep_wfc1 = nos_rep_wfc1 * NAOs_rep_wfc0
    //     TK* p_new_wfc = &this->new_wfc(0, 0, 0);
    //     TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
    //     for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

    //     // TK* p_new_wfc = &( rdmft_solver.wfc(0, 0, 0) );
    //     // TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
    //     // for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

    //     this->if_get_wfc1 = true;
    //     std::cout << "\n******\n" << "iterDiag: 0.3, once" << "\n******\n" << std::endl;
    // }

    rdmft_solver.update_elec( nullptr, &(this->new_wfc) );

    this->new_wfc.zero_out();
}



template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_lambda(const ModuleBase::matrix& wg, 
                                        const ModuleBase::matrix& wk_fun_occNum, 
                                        std::vector< std::vector<TK> >& H_no_exx, 
                                        std::vector< std::vector<TK> >& H_exx)
{
    // for(int i=0; i<lambda.size(); ++i) { lambda[i] = lambda_in[i]; }

    // // test transpose in potential matrix
    // std::vector<TK> iden_mat(this->para_Fij->get_local_size(), 0.0);
    // std::vector<TK> temp_mat(this->para_Fij->get_local_size(), 0.0);
    // rdmft::get_identi_mat(this->para_Fij, iden_mat);
    // for(int ik=0; ik<nk_total; ++ik)
    // {
    //     // std::vector<TK> temp_mat = this->lambda[ik];
    //     // rdmft::pTgemm_scalapack(this->para_Fij, temp_mat.data(), iden_mat.data(),
    //     //                             this->lambda[ik].data(), nbands_total, nbands_total, nbands_total, 'T', 'N');

    //     temp_mat = H_no_exx[ik];
    //     rdmft::pTgemm_scalapack(this->para_Fij, temp_mat.data(), iden_mat.data(),
    //                                 H_no_exx[ik].data(), nbands_total, nbands_total, nbands_total, 'T', 'N');
    //     temp_mat = H_exx[ik];
    //     rdmft::pTgemm_scalapack(this->para_Fij, temp_mat.data(), iden_mat.data(),
    //                                 H_exx[ik].data(), nbands_total, nbands_total, nbands_total, 'T', 'N');
    // }

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

    // // test T
    // std::vector<TK> iden_mat(this->para_Fij->get_local_size(), 0.0);
    // rdmft::get_identi_mat(this->para_Fij, iden_mat);
    // for(int ik=0; ik<nk_total; ++ik)
    // {
    //     std::vector<TK> temp_mat = this->lambda[ik];
    //     rdmft::pTgemm_scalapack(this->para_Fij, temp_mat.data(), iden_mat.data(),
    //                                 this->lambda[ik].data(), nbands_total, nbands_total, nbands_total, 'T', 'N');
    // }

    // // test transpose in potential matrix
    // // // times occNum
    // for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    // {
    //     int ncol = para_Fij->get_col_size();
    //     int nrow = para_Fij->get_row_size();
    //     for(int ir=0; ir<para_Fij->get_row_size(); ++ir)
    //     {
    //         // use wg or occ_number???
    //         const double wg_local = wg(ik, para_Fij->local2global_row(ir));
    //         const double wk_fun_local = wk_fun_occNum(ik, para_Fij->local2global_row(ir));

    //         for(int ic=0; ic<ncol; ++ic)
    //         {
    //             this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*wg_local 
    //                                             + H_exx[ik][ir + ic*nrow]*wk_fun_local;
    //         }
    //     }
    // }
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_Fock()
{
    this->max_off_diag_F = 0.0;

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


                // if(ic_global > ir_global) 
                // {
                //     // the upper triangle
                //     this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                // }
                // // // test
                // // if(ic_global < ir_global) 
                // // {
                // //     // the upper triangle
                // //     this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                // // }
                // else if (ic_global == ir_global)
                // {
                //     // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
                //     this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global];
                // }

                if( ic_global == ir_global )
                {
                    // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
                    this->Fock_like_mat[ik][ir+ic*nrow] = this->diag_Fii[ik][ic_global];
                }
                else
                {
                    double norm_Fij = std::abs( this->Fock_like_mat[ik][ir+ic*nrow] );
                    this->max_off_diag_F = std::max(this->max_off_diag_F, norm_Fij);

                    if(ic_global > ir_global) 
                    {
                        // the upper triangle
                        this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                    }
                }
            }
        }
        // here or other place? 
        std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);
    }

    // get the max value of std::abs(Fij) in a global sense
    rdmft::reduce_all_max(this->max_off_diag_F);

    // this->scale_Fock();

    if(if_rotate_Fock)
    {
        this->rotate_Fock();
    }

    // check the Hermitian property of Fock
    double max_Fij = 0.0;
    for(int ik=0; ik<this->Fock_like_mat.size(); ++ik)
    {
        double fix_k_max = rdmft::check_hermi(this->para_Fij, this->Fock_like_mat[ik], PARAM.inp.nbands);
        max_Fij = std::max(max_Fij, fix_k_max);
    }
    if( max_Fij > 1e-12 )
    {
        std::cout << "\n\n******\n" << "Fock_like_mat is not Hermitian" << "\n******\n" << std::endl;
    }
    // this->check_hermi(this->Fock_like_mat); // delete in the futuregggggG
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
                    double c_rc = this->scale_zeta/std::abs(this->Fock_like_mat[ik][ir+ic*nrow]); // choose one of the two

                    // double c_rc = this->scale_zeta_vector[ik]/std::abs(this->Fock_like_mat[ik][ir+ic*nrow]); // choose one of the two
                    if( c_rc < 1.0 ) this->Fock_like_mat[ik][ir+ic*nrow] *= c_rc;
                }
            }
        }
    }
    std::fill(this->scale_zeta_vector.begin(), this->scale_zeta_vector.end(), 0.0);
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::rotate_Fock()
{
    std::vector<TK> iden_mat(this->para_Fij->get_local_size(), 0.0);
    std::vector<TK> G_Gdagger(this->para_Fij->get_local_size(), 0.0);
    std::vector<TK> Gdagger_G(this->para_Fij->get_local_size(), 0.0);
    rdmft::get_identi_mat(this->para_Fij, iden_mat);

    // rotate the Fock-like matrix to the first step NOs representation
    // known F = F^dagger. F_1-NOs = R_t-1 * F_t-NOs * (R_t-1)^dagger
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        // rdmft::pTgemm_scalapack( this->para_Fij, this->Fock_like_mat[ik].data(), this->rotation_mat[ik].data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->Fock_like_mat[ik].data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );

        std::vector<TK> mat_temp(this->Fock_like_mat[ik].size(), 0.0);

        // // known F = F^dagger. F_1-NOs = R_t-1 * F_t-NOs * (R_t-1)^dagger
        // rdmft::pTgemm_scalapack( this->para_Fij, this->Fock_like_mat[ik].data(), this->rotation_mat[ik].data(),
        //                             mat_temp.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), mat_temp.data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );

        // // F_1-NOs = R_t-1 * F_t-NOs * (R_t-1)^dagger
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->Fock_like_mat[ik].data(),
        //                             mat_temp.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
        // rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->rotation_mat[ik].data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
        
        // work to local minimum ??????!!!
        // without any T, may be right ?!
        rdmft::pTgemm_scalapack( this->para_Fij, this->Fock_like_mat[ik].data(), this->rotation_mat[ik].data(),
                                    mat_temp.data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), mat_temp.data(),
                                    this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'C', 'N' );

        // // without any T
        // rdmft::pTgemm_scalapack( this->para_Fij, this->Fock_like_mat[ik].data(), this->rotation_mat[ik].data(),
        //                             mat_temp.data(), nbands_total, nbands_total, nbands_total, 'T', 'N' );
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), mat_temp.data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'C', 'N' );

        // // test T
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->Fock_like_mat[ik].data(),
        //                             mat_temp.data(), nbands_total, nbands_total, nbands_total, 'T', 'N' );
        // rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->rotation_mat[ik].data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );


        // test: verify the unitarity of the rotation matrix
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->rotation_mat[ik].data(),
                                    G_Gdagger.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->rotation_mat[ik].data(),
                                    Gdagger_G.data(), nbands_total, nbands_total, nbands_total, 'C', 'N' );
        for(int iloc=0; iloc<G_Gdagger.size(); ++iloc)
        {
            TK ver1 = G_Gdagger[iloc] - iden_mat[iloc];
            TK ver2 = iden_mat[iloc] - Gdagger_G[iloc];
            if( std::abs(ver1) > 1e-13 )
            {
                std::cout << "\n******\n" << "false: the unitarity of the rotation matrix. G_Gdagger, ik = " << ik  << ", diff = " << ver1 << "\n******\n" << std::endl;
            }
            if( std::abs(ver2) > 1e-13 )
            {
                std::cout << "\n******\n" << "false: the unitarity of the rotation matrix. Gdagger_G, ik = " << ik  << ", diff = " << ver2 << "\n******\n" << std::endl;
            }
        }
    }
}


template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::adjust_scale()
{
    // std::vector<double> scale_vector(nk_total);s
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
                    this->scale_zeta_vector[ik] += std::abs( this->Fock_like_mat[ik][ir+ic*nrow] );
                }
            }
        }
        Parallel_Reduce::reduce_all(this->scale_zeta_vector[ik]);
        this->scale_zeta_vector[ik] /= nbands_total*(nbands_total-1);
        if( this->scale_zeta_vector[ik] > std::abs(this->diag_Fii[ik][0])/200.0 ) this->scale_zeta_vector[ik] = std::abs(this->diag_Fii[ik][0])/200.0;


        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n******\nik: " << ik << ",   avar_off_diag: " << scale_zeta_vector[ik] << ",    F00: " << this->diag_Fii[ik][0] << "\n******\n" 
                    << std::endl << std::defaultfloat;
    }

    // refer to octopus
    if( this->energy_drop > static_cast<int>(1.5*this->energy_rise) ) this->scale_zeta *= 1.01;
    else if( this->energy_drop < static_cast<int>(1.1*this->energy_rise) ) this->scale_zeta *= 0.95;

    // if( this->energy_drop > static_cast<int>(1.5*this->energy_rise) ) this->scale_zeta *= 1.05;
    // else if( this->energy_drop < static_cast<int>(1.1*this->energy_rise) ) this->scale_zeta *= 0.95;
}


// template <typename TK, typename TR>
// void IterDiag_NOs<TK, TR>::check_hermi(std::vector< std::vector<TK> >& mat)
// {
//     for(int ik=0; ik<mat.size(); ++ik)
//     {
//         std::vector<TK> zero_mat(mat[ik].size(), 0.0);
//         rdmft::antisymm_mat(this->para_Fij, PARAM.inp.nbands, mat[ik].data(), zero_mat.data(), 1.0);
        
//         for(int iloc=0; iloc<zero_mat.size(); ++iloc)
//         {
//             if( std::abs(zero_mat[iloc]) > 1e-12 ) std::cout << "\n\n******\n" << "Fock_like_mat is not Hermitian" << "\n******\n" << std::endl;
//         }
//     }
// }



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


