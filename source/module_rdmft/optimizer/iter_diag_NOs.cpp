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
#include "module_base/lapack_connector.h" // temp

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
void IterDiag_NOs<TK, TR>::init(const int nk_total_in, const int nkstot_full_in, const Parallel_2D& para_Fij_in, const Parallel_Orbitals& ParaV_in, RDMFT<TK, TR>* rdmft_solver_in)
{
    this->rdmft_solver = rdmft_solver_in;
    this->nk_total = nk_total_in;
    this->nk_nospin = this->nk_total / PARAM.inp.nspin;
    this->nbands_total = PARAM.inp.nbands;
    this->kv = this->rdmft_solver->kv;
    int nkstot_full = nkstot_full_in;
    this->para_Fij = &para_Fij_in;
    this->ParaV = &ParaV_in;
    // identi_mat = get_identi_mat(this->para_Fij); // temporary

    this->init_orb_by_lambda = PARAM.inp.init_orb_by_lambda;
    this->scale_F = PARAM.inp.scale_fock;
    this->scale_zeta = PARAM.inp.scale_zeta;
    // this->scale_zeta = 0.01; // PARAM.inp.scale_zeta_rdmft?
    this->scale_zeta_vector.resize(nk_total);

    dEee_docc_num.create(nk_total, nbands_total);

    // malloc
    this->new_wfc.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->naos_rep_wfc1.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->lambda.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    this->diag_num.resize(nk_total);
    this->Fock_like_mat.resize(nk_total);
    this->nos_rep_wfc.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    this->rotation_mat.resize(nk_total);
    this->DM.resize(nk_total, std::vector<TK>(this->ParaV->nloc, 0.0));
    for(int ik=0; ik<nk_total; ++ik)
    {
        diag_Fii[ik].resize(nbands_total, 0.0);
        diag_num[ik].resize(nbands_total, 0.0);
        lambda[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->Fock_like_mat[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->nos_rep_wfc[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );

        // or write in restart_opti(), if you update the fixed NOs representation after each ONs optimization
        rdmft::get_identi_mat( para_Fij, this->rotation_mat[ik] );
    }

    if( PARAM.inp.mixing_rdmft )
    {
        this->dmk_in.resize(PARAM.globalv.nlocal*PARAM.globalv.nlocal*this->nk_total, 0.0);
        this->dmk_out.resize(this->dmk_in.size(), 0.0);
        this->mixing_dmk.init( PARAM.inp.mixing_mode, PARAM.inp.mixing_beta, PARAM.inp.mixing_ndim, this->dmk_in.size() );
        if( PARAM.inp.rotate_fock )
        {
            this->DM_nos_rep.resize(nk_total, std::vector<TK>(this->para_Fij->nloc, 0.0));
        }

        // test mixing diag_Fii
        this->diag_Fii_in.resize(this->nk_total * this->nbands_total, 0.0);
        this->diag_Fii_out.resize(this->nk_total * this->nbands_total, 0.0);
        this->mixing_Fii.init( PARAM.inp.mixing_mode, PARAM.inp.mixing_beta, PARAM.inp.mixing_ndim, this->diag_Fii_in.size() );
    }

    if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        this->grad.resize(nk_total);
        this->moment_m.resize(nk_total);
        this->moment_v.resize(nk_total);
        this->vhat_max.resize(nk_total);
        for(int ik=0; ik<nk_total; ++ik)
        {
            this->grad[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->moment_m[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->moment_v[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
            this->vhat_max[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        }

        this->m_hat.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->v_hat.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->skew_hermi_mat.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->adam_rotation.resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );

        this->learn_rate = PARAM.inp.adam_learn_rate;
    }

    // get the number of symmetric k-points
    this->num_symm_k.resize(this->kv->wk.size());
    for(int iks=0; iks<this->num_symm_k.size(); ++iks)
    {
        this->num_symm_k[iks] = this->kv->wk[iks] * nkstot_full;
    }

    this->sys_nelec_spin.resize(PARAM.inp.nspin);
    if( PARAM.inp.nspin == 1 )
    {
        // remove the weight of spin
        for(int iks=0; iks<this->kv->wk.size(); ++iks)
        {
            // this->wk_nospin[iks] /= 2.0;
            this->num_symm_k[iks] /= 2.0;
        }

        this->sys_nelec_spin[0] = (PARAM.inp.nelec / 2.0) * nkstot_full;
        // std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->sys_nelec_spin[0] = ((PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0) * nkstot_full;
        this->sys_nelec_spin[1] = ((PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0) * nkstot_full;
        // std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "this->sys_nelec_spin[1]: " << this->sys_nelec_spin[1] << "\n******\n" << std::endl;
    }


}


template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::restart_opti(hamilt::Hamilt<TK>* p_hamilt_in, int* scale_factor)
{
    if( scale_factor != nullptr && scale_factor > 0 ) this->scale_zeta = *scale_factor;
    this->energy_drop = 0;
    this->energy_rise = 0;

    // if( PARAM.inp.rotate_fock )
    // {
    //     rdmft::get_identi_mat( para_Fij, this->rotation_mat[ik] );
    //     this->if_get_wfc1 = false;
    // }

    this->iter_step = 0;
    if( PARAM.inp.mixing_rdmft )
    {
        this->mixing_dmk.reset();
        this->mixing_Fii.reset(); // test !!!!!!!!!!!!!!!!!
    }
    this->p_hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(p_hamilt_in);

    // use the passed in orbital or KS-orbital as the initial natural orbital
    if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        for(int ik=0; ik<nk_total; ++ik)
        {
            rdmft::get_identi_mat( para_Fij, this->nos_rep_wfc[ik] );
            std::fill(this->moment_m[ik].begin(), this->moment_m[ik].end(), 0.0);
            std::fill(this->moment_v[ik].begin(), this->moment_v[ik].end(), 0.0);
            std::fill(this->vhat_max[ik].begin(), this->vhat_max[ik].end(), 0.0);
        }
    }
}


template<typename TK, typename TR>
double IterDiag_NOs<TK, TR>::optimize_orb(RDMFT<TK, TR>& rdmft_solver_in)
{
    this->etotal_old = this->etotal;

    this->get_lambda(rdmft_solver_in.wg, rdmft_solver_in.wk_fun_occNum, rdmft_solver_in.Hij_no_exx, rdmft_solver_in.Hij_exx);
    // this->get_lambda(rdmft_solver_in.occ_number, rdmft_solver_in.fun_occNum, rdmft_solver_in.Hij_no_exx, rdmft_solver_in.Hij_exx);

    if( PARAM.inp.rdmft_orb_opti == "iter_diag" )
    {
        this->get_Fock();

        // when mixing the DMk in certain fixed NOs
        // the condition of iter_step should be consistent with which step's NOs is used as the representation of the Fock matrix
        this->start_mixing = ( PARAM.inp.mixing_rdmft && this->iter_step > 0 );

        if( this->start_mixing )
        {
            if( PARAM.inp.rotate_fock)
            {
                rdmft::dm_local2global(this->para_Fij, this->DM_nos_rep, this->dmk_in, this->nbands_total);
            }
            else
            {
                rdmft::dm_local2global(this->ParaV, this->DM, this->dmk_in, PARAM.globalv.nlocal);
            }

            // test mixing diag_Fii
            for(int ik=0; ik<this->nk_total; ++ik)
            {
                for(int ib=0; ib<this->nbands_total; ++ib)
                {
                    this->diag_Fii_in[ik*this->nbands_total + ib] = this->diag_Fii[ik][ib];
                }
            }
        }

        for(int ik=0; ik<nk_total; ++ik)
        {
            std::fill( nos_rep_wfc[ik].begin(), nos_rep_wfc[ik].end(), 0.0 );
            std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);

            // get Fii and new_wfc in NOs
            rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, this->Fock_like_mat[ik].data(),
                                        this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());

            if( !this->if_get_wfc1 )
            {
                // std::cout << "\n******\n" << "iterDiag: 0.1, once" << "\n******\n" << std::endl;
                rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], rdmft_solver_in.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
            }
            else
            {
                // std::cout << "\n******\n" << "iterDiag: 0.2, many" << "\n******\n" << std::endl;
                // right?
                rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );
            }

            // update the rotation matrix
            if( PARAM.inp.rotate_fock ) // && !this->start_mixing , add this condition when only use DM to determine whether it converges
            {
                // test, rotation = egienvector?
                this->rotation_mat[ik] = this->nos_rep_wfc[ik];
            }
        }

        // update from 1-step_wfc?
        if( !this->if_get_wfc1 && PARAM.inp.rotate_fock )
        {
            // rotate the Fock-like matrix to the first step NOs representation,
            // then new_wfc = nos_rep_wfc * NAOs_rep_wfc0 for each iteration
            TK* p_wfc = &this->rdmft_solver->wfc(0, 0, 0);
            TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
            for(int i=0; i<this->naos_rep_wfc1.size(); ++i)
            {
                p_naos_rep_wfc1[i] = p_wfc[i];
            }
            this->if_get_wfc1 = true;
        }

    }
    else if( PARAM.inp.rdmft_orb_opti == "adam" )
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            std::fill(this->grad[ik].begin(), this->grad[ik].end(), 0.0);

            // the factor is 1.0, 2.0, or 4.0 ?
            // antisymm_mat(this->para_Fij, nbands_total, this->lambda[ik].data(), this->grad[ik].data(), this->kv->wk[ik]); // ? 1.0, 2.0, 4.0?
            antisymm_mat(this->para_Fij, nbands_total, this->lambda[ik].data(), this->grad[ik].data(), 2.0);



            for(int i=0; i<this->grad[ik].size(); ++i)
            {
                this->moment_m[ik][i] = PARAM.inp.adam_beta1 * this->moment_m[ik][i] + (1.0 - PARAM.inp.adam_beta1) * this->grad[ik][i];
                this->moment_v[ik][i] = PARAM.inp.adam_beta2 * this->moment_v[ik][i] + (1.0 - PARAM.inp.adam_beta2) * std::norm(this->grad[ik][i]);
                this->m_hat[i] = this->moment_m[ik][i] / (1.0 - PARAM.inp.adam_beta1);
                this->v_hat[i] = this->moment_v[ik][i] / (1.0 - PARAM.inp.adam_beta2);
                this->vhat_max[ik][i] = std::max(this->vhat_max[ik][i], this->v_hat[i]);
                // this->vhat_max[ik][i] = this->v_hat[i];
                this->skew_hermi_mat[i] = - this->learn_rate * this->m_hat[i] / std::sqrt( this->vhat_max[ik][i] + 1e-16 );
            }

            // rdmft::printMatrix_pointer(nbands_total, nbands_total, this->v_hat.data(), "v_hat", 10);

            // vhat_max is obtained by std::max(), which will affect the skew-Hermitian properties of the matrix y (i.e. skew_hermi_mat)
            // v_hat and vhat_max must > 0, sostd::max() don't affect the skew-Hermitian properties of the matrix y

            // adam_rotation = exp( skew_hermi_mat )
            this->get_adam_rotation(this->skew_hermi_mat);

            // // NOs = NOs * adam_rotation
            // std::vector<TK> temp_mat = this->nos_rep_wfc[ik];
            // rdmft::pTgemm_scalapack( this->para_Fij, temp_mat.data(), this->adam_rotation.data(),
            //                         this->nos_rep_wfc[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );

            // // new_wfc = new_NOs * this->rdmft_solver.wfc
            // rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], rdmft_solver_in.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
            
            // rdmft::GkPsi( this->para_Fij, this->ParaV, this->adam_rotation[0], rdmft_solver_in.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
            
            // correct?
            rdmft::pTgemm_scalapack( this->para_Fij, &(this->rdmft_solver->wfc(ik, 0, 0)), this->adam_rotation.data(),
                                        &(this->new_wfc(ik, 0, 0)), nbands_total, nbands_total, nbands_total, 'N', 'N' );

            // rdmft::pTgemm_scalapack( this->para_Fij, this->adam_rotation.data(), &(this->rdmft_solver->wfc(ik, 0, 0)),
            //                             &(this->new_wfc(ik, 0, 0)), nbands_total, nbands_total, nbands_total, 'N', 'N' );

            // std::vector<TK> U_Udagger(this->adam_rotation.size(), 0.0);
            // std::vector<TK> Udagger_U(this->adam_rotation.size(), 0.0);
            // std::vector<TK> iden_mat(this->adam_rotation.size(), 0.0);
            // rdmft::get_identi_mat(this->para_Fij, iden_mat);

            // // test: verify the unitarity of the rotation matrix
            // rdmft::pTgemm_scalapack( this->para_Fij, this->adam_rotation.data(), this->adam_rotation.data(),
            //                             U_Udagger.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
            // rdmft::pTgemm_scalapack( this->para_Fij, this->adam_rotation.data(), this->adam_rotation.data(),
            //                             Udagger_U.data(), nbands_total, nbands_total, nbands_total, 'C', 'N' );
            // for(int iloc=0; iloc<Udagger_U.size(); ++iloc)
            // {
            //     TK ver1 = U_Udagger[iloc] - iden_mat[iloc];
            //     TK ver2 = iden_mat[iloc] - Udagger_U[iloc];
            //     if( std::abs(ver1) > 1e-13 )
            //     {
            //         std::cout << "\n******\n" << "false: the unitarity of the rotation matrix. U_Udagger, ik = " << ik  << ", diff = " << ver1 << "\n******\n" << std::endl;
            //     }
            //     if( std::abs(ver2) > 1e-13 )
            //     {
            //         std::cout << "\n******\n" << "false: the unitarity of the rotation matrix. Udagger_U, ik = " << ik  << ", diff = " << ver2 << "\n******\n" << std::endl;
            //     }
            // }

        }

    }

    // cal DM_out
    ModuleBase::matrix temp_wg(this->rdmft_solver->wg);
    std::vector< std::vector<TK> > DM_new(nk_total, std::vector<TK>(this->ParaV->nloc, 0.0));
    rdmft::cal_special_DM(this->ParaV, temp_wg, this->new_wfc, DM_new);
    if( PARAM.inp.mixing_rdmft && PARAM.inp.rotate_fock )
    {
        // get the NOs representation of DMk
        psi::Psi<TK> nos_wfc;
        nos_wfc.resize(nk_total, this->para_Fij->ncol, this->para_Fij->nrow);
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            for(int iloc=0; iloc<this->para_Fij->nloc; ++iloc)
            {
                TK* p_nos_wfc = &nos_wfc(ik, 0, 0);
                p_nos_wfc[iloc] = this->nos_rep_wfc[ik][iloc];
            }
        }
        rdmft::cal_special_DM(this->para_Fij, temp_wg, nos_wfc, this->DM_nos_rep);
    }

    // diff_DM
    this->diff_DM_max = 0.0;
    for(int ik=0; ik<nk_total; ++ik)
    {
        for(int iloc=0; iloc<DM_new[ik].size(); ++iloc)
        {
            double diff_DM = std::abs( this->DM[ik][iloc] - DM_new[ik][iloc] );
            if( diff_DM > this->diff_DM_max )
            {
                this->diff_DM_max = diff_DM;
            }
        }
    }
    rdmft::reduce_all_max(this->diff_DM_max);

    // if convergence, don't mixing
    if( this->diff_DM_max > PARAM.inp.scf_thr )
    {
        // mixing DM, get the mixed wg and wfc
        if( this->start_mixing )
        {
            if( PARAM.inp.rotate_fock )
            {
                this->do_mixing(this->DM_nos_rep);
            }
            else
            {
                this->do_mixing(DM_new);
            }
        }
    }
    this->DM = DM_new;

    this->rdmft_solver->update_elec( nullptr, &(this->new_wfc) );
    this->etotal = this->rdmft_solver->cal_Energy();
    this->diff_Etotal = this->etotal - this->etotal_old;
    this->new_wfc.zero_out();
    
    ++this->iter_step;
    
    if(this->diff_Etotal < 0) { ++this->energy_drop; }
    else { ++this->energy_rise; }

    return this->diff_Etotal;
}



template<typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_start_guess(RDMFT<TK, TR>& rdmft_solver_in, const bool conver_initial_value)
{
    this->get_lambda(rdmft_solver_in.wg, rdmft_solver_in.wk_fun_occNum, rdmft_solver_in.Hij_no_exx, rdmft_solver_in.Hij_exx);
    // this->get_lambda(rdmft_solver_in.occ_number, rdmft_solver_in.fun_occNum, rdmft_solver_in.Hij_no_exx, rdmft_solver_in.Hij_exx);

    // get start_Fock = symm_lambda
    std::vector< std::vector<TK> > symm_lambda = lambda;
    symmetr_lambda(this->para_Fij, this->lambda, symm_lambda);

    // diag(symm_lambda)
    for(int ik=0; ik<nk_total; ++ik)
    {
        // get start_Fii and start_wfc in NOs
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands_total, symm_lambda[ik].data(),
                                    this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());

        // // get start_wfc in NAOs
        // rdmft::GkPsi( this->para_Fij, this->ParaV, nos_rep_wfc[ik][0], rdmft_solver_in.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );

        // temp
        // get rotation_mat, rotation_mat_t-step = G_t * G_t-1 * ... * G_1
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->nos_rep_wfc[ik].data(),
        //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );
        // std::vector<TK> mat_temp = this->rotation_mat[ik];
        // rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->nos_rep_wfc[ik].data(),
        //                             this->rotation_mat[ik].data(), nbands_total, nbands_total, nbands_total );

        // rdmft::printMatrix_pointer(1, nbands_total, this->diag_Fii[ik].data(), "diag of symm_lambda", 5);
    }

    // if( !this->if_get_wfc1 && PARAM.inp.rotate_fock )
    // {
    //     // rotate the Fock-like matrix to the first step NOs representation,
    //     // then new_wfc = nos_rep_wfc * ( nos_rep_wfc1 * NAOs_rep_wfc0) for each iteration

    //     // get NAOs_rep_wfc1 = nos_rep_wfc1 * NAOs_rep_wfc0
    //     TK* p_new_wfc = &this->new_wfc(0, 0, 0);
    //     TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
    //     for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

    //     // TK* p_new_wfc = &( rdmft_solver_in.wfc(0, 0, 0) );
    //     // TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
    //     // for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_new_wfc[i]; }

    //     this->if_get_wfc1 = true;
    //     std::cout << "\n******\n" << "iterDiag: 0.3, once" << "\n******\n" << std::endl;
    // }

    if( this->init_orb_by_lambda )
    {
        for(int ik=0; ik<nk_total; ++ik)
        {
            // get start_wfc in NAOs
            rdmft::GkPsi( this->para_Fij, this->ParaV, nos_rep_wfc[ik][0], rdmft_solver_in.wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
        }
        rdmft_solver_in.update_elec( nullptr, &(this->new_wfc) );
    }

    this->new_wfc.zero_out();
}



template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::get_lambda(const ModuleBase::matrix& occ_num, 
                                        const ModuleBase::matrix& fun_occNum, 
                                        const std::vector< std::vector<TK> >& H_no_exx, 
                                        const std::vector< std::vector<TK> >& H_exx)
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

    // rdmft::printMatrix_pointer(occ_num.nr, occ_num.nc, occ_num.c, "occ_number used in lambda", 10);
    // rdmft::printMatrix_pointer(occ_num.nr, occ_num.nc, fun_occNum.c, "fun_occ_number used in lambda", 10);

    // times occNum
    for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    {
        std::fill(this->lambda[ik].begin(), this->lambda[ik].end(), 0.0);
        // the first right one !!!!!!!!!!!!!!!!!!!!!!!!!!!
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            // use wg or occ_number??? occ_num!
            const double occ_num_local = occ_num(ik, para_Fij->local2global_col(ic));
            const double fun_occNum_local = fun_occNum(ik, para_Fij->local2global_col(ic));

            for(int ir=0; ir<nrow; ++ir)
            {
                this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*occ_num_local + H_exx[ik][ir + ic*nrow]*fun_occNum_local;
                // this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*occ_num_local + H_exx[ik][ir + ic*nrow]*occ_num_local;
                // this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*occ_num_local;
            }
        }

        // // test formula
        // int nrow = para_Fij->get_row_size();
        // for(int ir=0; ir<nrow; ++ir)
        // {
        //     // use wg or occ_number???
        //     const double occ_num_local = occ_num(ik, para_Fij->local2global_row(ir));
        //     const double fun_occNum_local = fun_occNum(ik, para_Fij->local2global_row(ir));

        //     for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        //     {
        //         this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*occ_num_local 
        //                                         + H_exx[ik][ir + ic*nrow]*fun_occNum_local;
        //     }
        // }

        if(PARAM.inp.print_fock)
        {
            std::cout << "\nik: " << ik << std::endl;
            rdmft::printMatrix_pointer(para_Fij->get_row_size(), para_Fij->get_col_size(), this->lambda[ik].data(), "lambda[ik]", 10);
        }

        // std::cout << "\n******\ncheck lambda hermi: " << rdmft::check_hermi(this->para_Fij, this->lambda[ik], this->nbands_total) << "\n******\n" << std::endl;

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
    //         const double occ_num_local = wg(ik, para_Fij->local2global_row(ir));
    //         const double fun_occNum_local = wk_fun_occNum(ik, para_Fij->local2global_row(ir));

    //         for(int ic=0; ic<ncol; ++ic)
    //         {
    //             this->lambda[ik][ir + ic*nrow] = H_no_exx[ik][ir + ic*nrow]*occ_num_local 
    //                                             + H_exx[ik][ir + ic*nrow]*fun_occNum_local;
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
        std::fill(this->Fock_like_mat[ik].begin(), this->Fock_like_mat[ik].end(), 0.0);

        // the first right one !!!!!!!!!!!!!!!!!!!!!!!!!!!
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

                    int occupied_state = static_cast<int>( std::ceil( this->sys_nelec_spin[ik] ) );
                    if( (std::abs( PARAM.inp.level_shifting ) > 1e-12) && 
                        (std::abs( this->diag_Fii[ik][occupied_state] - this->diag_Fii[ik][occupied_state-1] ) < std::abs(PARAM.inp.level_shifting)) )
                    {
                        // currently only applicable to molecular computing
                        // ik now is ispin
                        if( ic_global < occupied_state )
                        {
                            this->Fock_like_mat[ik][ir+ic*nrow] -= PARAM.inp.level_shifting;
                        }
                        else
                        {
                            this->Fock_like_mat[ik][ir+ic*nrow] += PARAM.inp.level_shifting;
                        }
                    }
                }
                else
                {
                    double norm_Fij = std::abs( this->Fock_like_mat[ik][ir+ic*nrow] );
                    this->max_off_diag_F = std::max(this->max_off_diag_F, norm_Fij);

                    // the first right one !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if(ic_global > ir_global) 
                    {
                        // the upper triangle
                        this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                    }

                    // // test
                    // if(ic_global < ir_global) 
                    // {
                    //     // fortran perspective: only the upper triangle of F is correct (excluding the diagonal)
                    //     // now make the lower triangle and diagonal elements correct as well
                    //     // the lower triangle
                    //     this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                    // }

                    // // test formula
                    // if(ic_global > ir_global) 
                    // {
                    //     // the upper triangle
                    //     this->Fock_like_mat[ik][ir+ic*nrow] = -( this->Fock_like_mat[ik][ir+ic*nrow] );
                    // }

                }
            }
        }
        // // here or other place? 
        // std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);

        if(PARAM.inp.print_fock)
        {
            std::cout << "\nik: " << ik << std::endl;
            rdmft::printMatrix_pointer(para_Fij->get_row_size(), para_Fij->get_col_size(), this->Fock_like_mat[ik].data(), "Fock[ik]", 10);
        }

    }

    // get the max value of std::abs(Fij) in a global sense
    rdmft::reduce_all_max(this->max_off_diag_F);

    // if( !this->start_mixing && PARAM.inp.mixing_rdmft )
    // {
    //     if( std::abs(this->diff_Etotal)<1e-4 ) { this->start_mixing = true; }
    // }
    
    // if( PARAM.inp.mixing_rdmft && this->start_mixing ) // !test!!!!!!!!!!!!!!
    // {
    //     this->mixing();
    // }

    // // test !!!!!!!!!!!!!!!!!!!!!!!!!!!
    // if( std::abs(this->max_off_diag_F / this->diag_Fii[0][0]) > 1.0 )
    // {
    //     for(int ik=0; ik<this->nk_total; ++ik)
    //     {
    //         int nrow = para_Fij->get_row_size();
    //         for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
    //         {
    //             const int ic_global = para_Fij->local2global_col(ic);
    //             for(int ir=0; ir<nrow; ++ir)
    //             {
    //                 int ir_global = para_Fij->local2global_row(ir);

    //                 if( ic_global == ir_global )
    //                 {
    //                     // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
    //                     this->Fock_like_mat[ik][ir+ic*nrow] = 0.0;
    //                 }
    //             }
    //         }
    //     }
    // }

    this->rdmft_solver->cal_E_grad_occ_num();

    // // test !!!!!!!!!!!!!!!!
    // for(int ik=0; ik<this->nk_total; ++ik)
    // {
    //     for(int ib=0; ib<diag_num[ik].size(); ++ib)
    //     {
    //         double num = this->rdmft_solver->occ_number(ik, ib);
    //         if( std::abs(1.0 - num) < 1e-16 )
    //         {
    //             num = 1.0 - 1e-16;
    //         }
    //         else if( num < 1e-20 )
    //         {
    //             num = 1e-20;
    //         }
            
    //         // diag_num[ik][ib] = -std::log(num) + std::log(1.0-num);
    //         diag_num[ik][ib] = this->diag_Fii[ik][ib] - 
    //                             ( this->rdmft_solver->occNum_wfc_Vee_wfc(ik, ib) - this->dEee_docc_num(ik, ib) ) * this->rdmft_solver->occ_number(ik, ib);
    //         this->dEee_docc_num(ik, ib) = this->rdmft_solver->occNum_wfc_Vee_wfc(ik, ib);
    //     }

    //     double factor = std::abs( this->diag_Fii[ik][0] / diag_num[ik][0] );
    //     int nrow = para_Fij->get_row_size();
    //     for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
    //     {
    //         const int ic_global = para_Fij->local2global_col(ic);
    //         for(int ir=0; ir<nrow; ++ir)
    //         {
    //             int ir_global = para_Fij->local2global_row(ir);

    //             if( ic_global == ir_global )
    //             {
    //                 // use the eigenvalues ​​of the last diag(F) to form the diagonal elements of this F
    //                 // this->Fock_like_mat[ik][ir+ic*nrow] = diag_num[ik][ic_global] * factor;
    //                 this->Fock_like_mat[ik][ir+ic*nrow] = diag_num[ik][ic_global];
    //             }
    //         }
    //     }

    //     rdmft::printMatrix_pointer(this->nk_total, diag_num[ik].size(), diag_num[ik].data(), "diag_num[ik] from ni", 10);
    //     rdmft::printMatrix_pointer(this->nk_total, diag_num[ik].size(), this->rdmft_solver->occNum_wfc_Vee_wfc.c, "dEee/dni", 10);

    //     // for(int ib=0; ib<diag_num[ik].size(); ++ib)
    //     // {
    //     //     diag_num[ik][ib] *= factor;
    //     // }
    //     rdmft::printMatrix_pointer(this->nk_total, diag_num[ik].size(), this->diag_Fii[ik].data(), "diag_Fii", 10);
    //     // rdmft::printMatrix_pointer(this->nk_total, diag_num[ik].size(), diag_num[ik].data(), "diag_num[ik] from ni, * factor", 10);
        

    //     // here or other place? 
    //     std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);

    // }


    for(int ik=0; ik<this->nk_total; ++ik)
    {
        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = para_Fij->local2global_row(ir);

                if( ic_global == ir_global )
                {
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
                    this->Fock_like_mat[ik][ir+ic*nrow] = this->rdmft_solver->occNum_wfcHamiltWfc(ik, ic_global) 
                                                            + PARAM.inp.idmft_kappa * std::log( (1 - num)/num );
                }
            }
        }
    }


    // if(PARAM.inp.scale_fock)
    if( this->scale_F )
    {
        this->scale_Fock();
    }

    if( PARAM.inp.rotate_fock )
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

        // // work to local minimum ??????!!!
        // // without any T, may be right ?!
        // rdmft::pTgemm_scalapack( this->para_Fij, this->Fock_like_mat[ik].data(), this->rotation_mat[ik].data(),
        //                             mat_temp.data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );
        // rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), mat_temp.data(),
        //                             this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'C', 'N' );


        // test Fji fortran, G in cpp, the right one?
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->Fock_like_mat[ik].data(),
                                    mat_temp.data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );
        rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->rotation_mat[ik].data(),
                                    this->Fock_like_mat[ik].data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );


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
void IterDiag_NOs<TK, TR>::get_adam_rotation(std::vector<TK>& skew_hermi_m)
{
    // hermi_mat A = i * M
    std::vector<std::complex<double>> hermi_mat(skew_hermi_m.size());
    std::complex<double> imag_one(0.0, 1.0);
    for(int i=0; i<skew_hermi_m.size(); ++i)
    {
        hermi_mat[i] = imag_one * skew_hermi_m[i];
    }

    std::vector<double> diag_elem(this->nbands_total, 0.0);
    std::vector<std::complex<double>> exp_diag(para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0);
    std::vector<std::complex<double>> egi_vector(para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0);

    // A = V * diag_e * V^dagger
    rdmft::pdiag_scalapack( this->para_Fij, this->nbands_total, hermi_mat.data(), diag_elem.data(), egi_vector.data() );

    std::complex<double> imag_nega_one(0.0, -1.0);
    int nrow = this->para_Fij->get_row_size();
    for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
    {
        const int ic_global = this->para_Fij->local2global_col(ic);
        for(int ir=0; ir<nrow; ++ir)
        {
            int ir_global = this->para_Fij->local2global_row(ir);
            if( ic_global == ir_global )
            {
                exp_diag[ir+ic*nrow] = std::exp( imag_nega_one * diag_elem[ic_global] );
            }
        }
    }

    // adam_rotation = exp(skew_hermi_m) = V * exp(-i * diag_e) * V^dagger
    std::vector<std::complex<double>> temp_mat(para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0);
    rdmft::pTgemm_scalapack( this->para_Fij, egi_vector.data(), exp_diag.data(),
                            temp_mat.data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );
    rdmft::pTgemm_scalapack( this->para_Fij, temp_mat.data(), egi_vector.data(),
                            exp_diag.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );
    if constexpr (std::is_same<TK, std::complex<double>>::value)
    {
        this->adam_rotation = exp_diag;
    }
    else
    {
        for(int i=0; i<exp_diag.size(); ++i)
        {
            this->adam_rotation[i] = std::real( exp_diag[i] );
        }
    }




    // std::vector<std::complex<double>> diag_mat(para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0);
    // for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
    // {
    //     const int ic_global = this->para_Fij->local2global_col(ic);
    //     for(int ir=0; ir<nrow; ++ir)
    //     {
    //         int ir_global = this->para_Fij->local2global_row(ir);
    //         if( ic_global == ir_global )
    //         {
    //             diag_mat[ir+ic*nrow] = imag_nega_one * diag_elem[ic_global];
    //         }
    //     }
    // }

    // std::fill(temp_mat.begin(), temp_mat.end(), 0.0);
    // std::fill(exp_diag.begin(), exp_diag.end(), 0.0);
    // rdmft::pTgemm_scalapack( this->para_Fij, egi_vector.data(), diag_mat.data(),
    //                         temp_mat.data(), nbands_total, nbands_total, nbands_total, 'N', 'N' );
    // rdmft::pTgemm_scalapack( this->para_Fij, temp_mat.data(), egi_vector.data(),
    //                         exp_diag.data(), nbands_total, nbands_total, nbands_total, 'N', 'C' );

    // bool pass = true;
    // for(int iloc=0; iloc<exp_diag.size(); ++iloc)
    // {
    //     std::complex<double> ver1 = exp_diag[iloc] - skew_hermi_m[iloc];
    //     if( std::abs(ver1) > 1e-13 )
    //     {
    //         pass = false;
    //         std::cout << "\n******\n" << "false: the unitarity of get_adam_rotation(), iloc: " << iloc << "\n******\n" << std::endl;
    //     }
    // }

    // if(!pass)
    // {
    //     std::cout << "\n******\n" << "get_adam_rotation() is error" << "\n******\n" << std::endl;
    //     rdmft::printMatrix_pointer(nbands_total, nbands_total, exp_diag.data(), "V * egivalue * V^dagger", 10);
    //     rdmft::printMatrix_pointer(nbands_total, nbands_total, skew_hermi_m.data(), "skew_hermi_m", 10);
    // }
    // else
    // {
    //     std::cout << "\n******\n" << "get_adam_rotation() is correct" << "\n******\n" << std::endl;
    // }

}





template <typename TK, typename TR>
void IterDiag_NOs<TK, TR>::do_mixing(std::vector< std::vector<TK> >& DMk)
{
    // mixing
    rdmft::dm_local2global(this->ParaV, DMk, this->dmk_out, PARAM.inp.rotate_fock ? this->nbands_total : PARAM.globalv.nlocal);
    this->mixing_dmk.push_data(this->dmk_in.data(), this->dmk_out.data());
    this->mixing_dmk.cal_coef();
    this->mixing_dmk.mix_dmk(this->dmk_out.data());

    // // test mixing diag_Fii
    // for(int ik=0; ik<this->nk_total; ++ik)
    // {
    //     for(int ib=0; ib<this->nbands_total; ++ib)
    //     {
    //         this->diag_Fii_out[ik*this->nbands_total + ib] = this->diag_Fii[ik][ib];
    //     }
    // }
    // this->mixing_Fii.push_data(this->diag_Fii_in.data(), this->diag_Fii_out.data());
    
    // // calculate the mixing coefficient of Fii independently, or use the mixing coefficient of DMk
    // // this->mixing_Fii.cal_coef();
    // this->mixing_Fii.get_coef() = this->mixing_dmk.get_coef();

    // this->mixing_Fii.mix_dmk(this->diag_Fii_out.data());
    // for(int ik=0; ik<this->nk_total; ++ik)
    // {
    //     for(int ib=0; ib<this->nbands_total; ++ib)
    //     {
    //         this->diag_Fii[ik][ib] = this->diag_Fii_out[ik*this->nbands_total + ib];
    //     }
    // }


    // convert and decompose DMk to get wg and wfc
    rdmft::dm_global2local(this->ParaV, this->dmk_out, DMk, PARAM.inp.rotate_fock ? this->nbands_total : PARAM.globalv.nlocal);
    ModuleBase::matrix temp_wg(nk_nospin*PARAM.inp.nspin, this->nbands_total);
    temp_wg.zero_out();
    this->new_wfc.zero_out();

    // mixing DMk in NOs representation
    if( PARAM.inp.rotate_fock )
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            // decompose the DMk to obtain "wfc" and wg
            rdmft::decom_dm(this->para_Fij,
                            DMk[ik],
                            &temp_wg(ik, 0),
                            this->nos_rep_wfc[ik].data());

            // get new_wfc in NAOs
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );

            // the rotation matrix depends only on the orthogonal wfc that is actually used to update the elec_state at each step.
            this->rotation_mat[ik] = this->nos_rep_wfc[ik];
        }
    }
    else // mixing DMk in NAOs representation
    {
        for(int ik=0; ik<this->nk_total; ++ik)
        {
            this->p_hamilt_lcao->updateSk(ik);
            TK* p_sk = this->p_hamilt_lcao->getSk();

            // decompose the DMk to obtain wfc and wg
            rdmft::decom_dm(this->ParaV,
                            DMk[ik], 
                            &temp_wg(ik, 0),
                            &this->new_wfc(ik, 0, 0),
                            this->ParaV,
                            p_sk);
        }
    }

    // could be deleted, because iterDiag does not optimize occ_number
    // just verify
    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, this->nbands_total);
    rdmft::wg2occ_num(this->kv, temp_wg, occ_num_pass);
    double tot_occ_num = 0.0;
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        tot_occ_num = 0.0;
        for(int ik=0; ik<this->nk_nospin; ++ik)
        {
            for(int ib=0; ib<this->nbands_total; ++ib)
            {
                if( is == 0 )
                {
                    tot_occ_num += occ_num_pass(ik, ib) * this->num_symm_k[ik];
                }
                else
                {
                    tot_occ_num += occ_num_pass(is*this->nk_nospin + ik, ib) * this->num_symm_k[is*this->nk_nospin + ik];
                }
            }
        }

        double occ_num_error = tot_occ_num - this->sys_nelec_spin[is];

        std::cout << "\n******\nafter mixing:\nis=" << is << ", total_elec_num: " << tot_occ_num << std::endl;
        std::cout << "is=" << is << ", mixing_occ_num_error: " << occ_num_error << "\n******\n" << std::endl;

    }
}



template <typename TK, typename TR>
double IterDiag_NOs<TK, TR>::check_hermi_lambda()
{
    this->get_lambda(this->rdmft_solver->wg, this->rdmft_solver->wk_fun_occNum, this->rdmft_solver->Hij_no_exx, this->rdmft_solver->Hij_exx);

    this->max_off_diag_F = 0.0;
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        // get Fock_like_mat = lambda_ij -lambda*_ji
        std::fill(this->Fock_like_mat[ik].begin(), this->Fock_like_mat[ik].end(), 0.0);
        antisymm_mat(this->para_Fij, nbands_total, this->lambda[ik].data(), this->Fock_like_mat[ik].data(), 1.0);

        int nrow = para_Fij->get_row_size();
        for(int ic=0; ic<para_Fij->get_col_size(); ++ic)
        {
            const int ic_global = para_Fij->local2global_col(ic);
            for(int ir=0; ir<nrow; ++ir)
            {
                int ir_global = para_Fij->local2global_row(ir);
                if( ic_global >= ir_global )
                {
                    double norm_Fij = std::abs( this->Fock_like_mat[ik][ir+ic*nrow] );
                    this->max_off_diag_F = std::max(this->max_off_diag_F, norm_Fij);
                }
            }
        }
    }

    // get the max value of std::abs(Fij) in a global sense
    rdmft::reduce_all_max(this->max_off_diag_F);

    return this->max_off_diag_F;
}


// template <typename TK, typename TR>
// void IterDiag_NOs<TK, TR>::mixing()
// {
//     // if(iter_step < this->mixing_step-1)
//     // {
//     //     // store the Fock matrix of the previous (mixing_step-1) step
//     //     auto pair_Fock = this->Fock_record.find(this->iter_step);
//     //     if( pair_Fock != Fock_record.end() )
//     //     {
//     //         std::vector< std::vector<TK> > & Fock_ith = pair_Fock->second;
//     //         for(int ik=0; ik<this->nk_total; ++ik)
//     //         {
//     //             for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
//     //             {
//     //                 Fock_ith[ik][iloc] = this->Fock_like_mat[ik][iloc];
//     //             }
//     //         }
//     //     }
//     // }

//     // store the Fock matrix of the i-step
//     auto pair_Fock = this->Fock_record.find(this->iter_step % this->mixing_step);
//     std::vector< std::vector<TK> > & Fock_ith = pair_Fock->second;
//     for(int ik=0; ik<this->nk_total; ++ik)
//     {
//         // std::fill(Fock_ith[ik].begin(), Fock_ith[ik].end(), 0.0);
//         for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
//         {
//             Fock_ith[ik][iloc] = this->Fock_like_mat[ik][iloc];
//         }
//     }

//     // else
//     if( iter_step >= this->mixing_step-1 )
//     {
//         for(int ik=0; ik<this->nk_total; ++ik)
//         {
//             std::fill(Fock_like_mat[ik].begin(), Fock_like_mat[ik].end(), 0.0);
//             for(int j=0; j<this->mixing_step; ++j)
//             {
//                 std::vector< std::vector<TK> > & Fock_jth = this->Fock_record.find( (this->iter_step + j) % this->mixing_step )->second;
//                 for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
//                 {
//                     this->Fock_like_mat[ik][iloc] += Fock_jth[ik][iloc] * this->mixing_coef[(j+this->mixing_step-1) % this->mixing_step];
//                 }

//             }

//             for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
//             {
//                 Fock_ith[ik][iloc] = this->Fock_like_mat[ik][iloc];
//             }
//         }
//     }

//     ++this->iter_step;
// }


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


        // std::cout << std::fixed << std::setprecision(6);
        // std::cout << "\n******\nik: " << ik << ",   avar_off_diag: " << scale_zeta_vector[ik] << ",    F00: " << this->diag_Fii[ik][0] << "\n******\n" 
        //             << std::endl << std::defaultfloat;
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


