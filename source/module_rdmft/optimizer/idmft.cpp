//==========================================================
// Author: Jingang Han
// DATE : 2025-03-11
//==========================================================

#include <algorithm>
// #include "module_rdmft/rdmft.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_rdmft/optimizer/idmft.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_lr/utils/lr_util.h"
#include "module_base/parallel_reduce.h"
#include "module_base/matrix.h"
// #include "module_base/blas_connector.h"
// #include "module_base/scalapack_connector.h"

// #include "module_psi/psi.h"

namespace rdmft
{


template <typename TK, typename TR>
IDMFT<TK, TR>::IDMFT()
{

}


template <typename TK, typename TR>
IDMFT<TK, TR>::~IDMFT()
{

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::init(const int nk_total_in,
                            const K_Vectors& kv_in,
                            const Parallel_2D& para_Fij_in,
                            const Parallel_Orbitals& ParaV_in,
                            RDMFT<TK, TR>* rdmft_solver_in)
{
    this->nk_total = nk_total_in;
    this->nk_nospin = this->nk_total / PARAM.inp.nspin;
    this->kv = &kv_in;
    this->nbands = PARAM.inp.nbands;
    // this->wk_nospin = this->kv->wk;
    int nkstot_full = this->kv->get_nkstot_full();

    this->para_Fij = &para_Fij_in;
    this->ParaV = &ParaV_in;
    this->rdmft_solver = rdmft_solver_in;

    // malloc
    this->new_wfc.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->naos_rep_wfc1.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->Fock_like_mat.resize(nk_total);
    this->nos_rep_wfc.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    this->rotation_mat.resize(nk_total);
    this->DM.resize(nk_total, std::vector<TK>(this->ParaV->nloc, 0.0));
    for(int ik=0; ik<nk_total; ++ik)
    {
        this->diag_Fii[ik].resize(nbands, 0.0);
        this->Fock_like_mat[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );
        this->nos_rep_wfc[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );

        // or write in before_opti(), if you update the fixed NOs representation after each ONs optimization
        rdmft::get_identi_mat( para_Fij, this->rotation_mat[ik] );
    }
    
    this->occ_number.resize(PARAM.inp.nspin);
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        this->occ_number[is].resize(nk_nospin*nbands);
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
    }

    // this->mixing_rdmft = PARAM.inp.mixing_rdmft;
    // if(this->mixing_rdmft)
    // {
    //     for(int i=0; i<this->mixing_step; ++i)
    //     {
    //         this->Fock_record[i] = std::vector<std::vector<TK>>(nk_total, std::vector<TK>( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 ));
    //     }
    // }

    // get the number of symmetric k-points
    this->num_symm_k.resize(this->kv->wk.size());
    for(int iks=0; iks<this->num_symm_k.size(); ++iks)
    {
        this->num_symm_k[iks] = this->kv->wk[iks] * nkstot_full;
    }

    // get the total number of electrons
    this->sys_nelec_spin.resize(PARAM.inp.nspin);
    this->mu.resize(PARAM.inp.nspin, 0.0);
    if( PARAM.inp.nspin == 1 )
    {
        // remove the weight of spin
        for(int iks=0; iks<this->kv->wk.size(); ++iks)
        {
            // this->wk_nospin[iks] /= 2.0;
            this->num_symm_k[iks] /= 2.0;
        }

        this->sys_nelec_spin[0] = (PARAM.inp.nelec / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->sys_nelec_spin[0] = ((PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0) * nkstot_full;
        this->sys_nelec_spin[1] = ((PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
        std::cout << "\n******\n" << "this->sys_nelec_spin[1]: " << this->sys_nelec_spin[1] << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "PARAM.inp.nelec: " << PARAM.inp.nelec << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "PARAM.inp.nupdown: " << PARAM.inp.nupdown << "\n******\n" << std::endl;
    }

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::before_opti(const std::vector< std::vector<TK> >& DM_in, hamilt::Hamilt<TK>* p_hamilt_in)
{
    // get the initial guess of DM
    this->DM = DM_in;   // could be delete now
    this->iter_step = 0;
    this->p_hamilt_lcao = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(p_hamilt_in);
}


template <typename TK, typename TR>
double IDMFT<TK, TR>::optimize()
{
    this->etotal_old = this->etotal;

    this->get_Fock();

    // the condition of iter_step should be consistent with which step's NOs is used as the representation of the Fock matrix ?
    this->start_mixing = ( PARAM.inp.mixing_rdmft && this->iter_step > 0 );

    if( this->start_mixing )
    {
        if( PARAM.inp.rotate_fock)
        {
            rdmft::dm_local2global(this->para_Fij, this->DM_nos_rep, this->dmk_in, this->nbands);
        }
        else
        {
            rdmft::dm_local2global(this->ParaV, this->DM, this->dmk_in, PARAM.globalv.nlocal);
        }
    }

    for(int ik=0; ik<nk_total; ++ik)
    {
        std::fill( nos_rep_wfc[ik].begin(), nos_rep_wfc[ik].end(), 0.0 );
        std::fill(diag_Fii[ik].begin(), diag_Fii[ik].end(), 0.0);

        // get Fii and new_wfc in NOs
        rdmft::pdiag_scalapack(this->para_Fij, this->nbands, this->Fock_like_mat[ik].data(),
                                    this->diag_Fii[ik].data(), this->nos_rep_wfc[ik].data());

        // get new_wfc in NAOs
        if( !this->if_get_wfc1 )
        {
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->rdmft_solver->wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
        }
        // else if( !this->start_mixing )
        else
        {
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );
        }

        // update the rotation matrix
        if( PARAM.inp.rotate_fock )
        {
            // test, rotation = egienvector?
            this->rotation_mat[ik] = this->nos_rep_wfc[ik];
        }
    }

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

        rdmft::printMatrix_pointer(ParaV->get_row_size(), ParaV->get_col_size(), p_naos_rep_wfc1, "naos_rep_wfc1", 10);
    }


    // optimize the occ_number
    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);
    this->opti_occ_num(occ_num_pass);

    // cal DM_out
    ModuleBase::matrix temp_wg(nk_nospin*PARAM.inp.nspin, nbands);
    rdmft::occ_num2wg(this->kv, occ_num_pass, temp_wg);
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
            // for(int ib=0; ib<this->para_Fij->ncol; ++ib)
            // {
            //     for(int ibs=0; ibs<this->para_Fij->nrow; ++ibs)
            //     {
            //         nos_wfc(ik, ib, ibs) = this->nos_rep_wfc[ik][ib*this->para_Fij->nrow + ibs];
            //     }
            // }
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

    // // if converage, don't mixing
    // if( this->diff_DM_max < PARAM.inp.scf_thr )
    // {
    //     // update the occupation number and wfc
    //     this->rdmft_solver->update_elec( &occ_num_pass, &(this->new_wfc) );
    //     this->etotal = this->rdmft_solver->cal_Energy();
    //     this->new_wfc.zero_out();

    //     this->diff_Etotal = this->etotal - this->etotal_old;

    //     if( std::abs(this->diff_Etotal) < PARAM.inp.iter_diag_ethr )
    //     {
    //         this->DM = DM_new;
    //         return this->diff_Etotal;
    //     }
    // }

    // rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_num_pass before mixing", 10);

    // if converage, don't mixing
    if( this->diff_DM_max > PARAM.inp.scf_thr )
    {
        // mixing DM, get the mixed wg and wfc
        if( this->start_mixing )
        {
            if( PARAM.inp.rotate_fock )
            {
                this->do_mixing(this->DM_nos_rep, occ_num_pass);
            }
            else
            {
                this->do_mixing(DM_new, occ_num_pass);
            }
        }
    }

    // rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_num_pass after mixing", 10);

    this->DM = DM_new;

    

    std::cout << "\n***\n in idmft, 0.8 \n***\n" << std::endl;

    // update the occupation number and wfc
    this->rdmft_solver->update_elec( &occ_num_pass, &(this->new_wfc) );

    std::cout << "\n***\n in idmft, 0.9 \n***\n" << std::endl;

    this->etotal = this->rdmft_solver->cal_Energy();

    std::cout << "\n***\n in idmft, 1.0 \n***\n" << std::endl;
    
    this->new_wfc.zero_out();

    this->diff_Etotal = this->etotal - this->etotal_old;

    ++this->iter_step;

    return this->diff_Etotal;

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::opti_occ_num(ModuleBase::matrix& occ_num_pass)
{
    this->solving_mu();

    // ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);
    for(int ik=0; ik<occ_num_pass.nr; ++ik)
    {
        for(int ib=0; ib<occ_num_pass.nc; ++ib)
        {
            if( ik < occ_num_pass.nr/PARAM.inp.nspin )
            {
                occ_num_pass(ik, ib) = this->occ_number[0][ik*nbands + ib];
            }
            else
            {
                occ_num_pass(ik, ib) = this->occ_number[PARAM.inp.nspin-1][ik*nbands + ib];
            }
        }
    }

    // this->rdmft_solver->update_elec( &occ_num_pass );
    // std::cout << std::scientific << std::setprecision(6) << std::endl;
    // rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_number", 10);
    // std::cout << std::defaultfloat;

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::get_Fock()
{
    for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    {
        std::fill(this->Fock_like_mat[ik].begin(), this->Fock_like_mat[ik].end(), 0.0);
        for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
        {
            this->Fock_like_mat[ik][iloc] = (this->rdmft_solver->Hij_no_exx[ik][iloc] + this->rdmft_solver->Hij_exx[ik][iloc]); // * wk_nospin[ik] ,no need to multiply k-point weight and spin weight
        }
    }

    // if( !this->start_mixing && this->mixing_rdmft )
    // {
    //     if( std::abs(this->diff_Etotal)<1e-4 ) { this->start_mixing = true; }
    // }
    
    // if( this->mixing_rdmft && this->start_mixing ) // !test!!!!!!!!!!!!!!
    // {
    //     this->mixing();
    // }

    if(PARAM.inp.rotate_fock)
    {
        this->rotate_Fock();
    }
}


template <typename TK, typename TR>
void IDMFT<TK, TR>::do_mixing(std::vector< std::vector<TK> >& DMk, ModuleBase::matrix& occ_num_pass)
{
    // mixing
    rdmft::dm_local2global(this->ParaV, DMk, this->dmk_out, PARAM.inp.rotate_fock ? this->nbands : PARAM.globalv.nlocal);
    this->mixing_dmk.push_data(this->dmk_in.data(), this->dmk_out.data());
    this->mixing_dmk.cal_coef();
    this->mixing_dmk.mix_dmk(this->dmk_out.data());

    // convert and decompose DMk to get wg and wfc
    rdmft::dm_global2local(this->ParaV, this->dmk_out, DMk, PARAM.inp.rotate_fock ? this->nbands : PARAM.globalv.nlocal);
    ModuleBase::matrix temp_wg(nk_nospin*PARAM.inp.nspin, nbands);
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

            // rdmft::printMatrix_pointer(temp_wg.nr, temp_wg.nc, temp_wg.c, "temp_wg in  mixing", 10);

            // rdmft::printMatrix_pointer(ParaV->get_row_size(), ParaV->get_col_size(), this->nos_rep_wfc[ik].data(), "nos_rep_wfc[0]", 10);

            // get new_wfc in NAOs
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );

            this->rotation_mat[ik] = this->nos_rep_wfc[ik]; // ???????????????????????????

            // rdmft::printMatrix_pointer(ParaV->get_row_size(), ParaV->get_col_size(), &this->new_wfc(ik, 0, 0), "this->new_wfc", 10);
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

    rdmft::wg2occ_num(this->kv, temp_wg, occ_num_pass);

    // rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_num_pass in mixing", 10);

    double tot_occ_num = 0.0;
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            tot_occ_num += occ_num_pass(ik, ib) * this->num_symm_k[ik];
        }
    }
    double occ_num_error = tot_occ_num - this->sys_nelec_spin[0];

    std::cout << "\n******\nafter mixing:\n" << "total_elec_num: " << tot_occ_num << std::endl;
    std::cout << "mixing_occ_num_error: " << occ_num_error << "\n******\n" << std::endl;


    // // not mixed occupation number?
    // if( std::abs(occ_num_error) > PARAM.inp.tot_nelec_thr )
    // {
    //     this->opti_occ_num(occ_num_pass);
    // }

}




// template <typename TK, typename TR>
// void IDMFT<TK, TR>::mixing()
// {
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

//     // when enough steps of Fock are stored, start mixing
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
void IDMFT<TK, TR>::rotate_Fock()
{

    std::vector<TK> iden_mat(this->para_Fij->get_local_size(), 0.0);
    std::vector<TK> G_Gdagger(this->para_Fij->get_local_size(), 0.0);
    std::vector<TK> Gdagger_G(this->para_Fij->get_local_size(), 0.0);
    rdmft::get_identi_mat(this->para_Fij, iden_mat);


    // rotate the Fock-like matrix to the first step NOs representation
    // known F = F^dagger. F_1-NOs = R_t-1 * F_t-NOs * (R_t-1)^dagger
    for(int ik=0; ik<this->nk_total; ++ik)
    {

        std::vector<TK> mat_temp(this->Fock_like_mat[ik].size(), 0.0);


        // test Fji fortran, G in cpp, the right one?
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->Fock_like_mat[ik].data(),
                                    mat_temp.data(), nbands, nbands, nbands, 'N', 'N' );
        rdmft::pTgemm_scalapack( this->para_Fij, mat_temp.data(), this->rotation_mat[ik].data(),
                                    this->Fock_like_mat[ik].data(), nbands, nbands, nbands, 'N', 'C' );


        // test: verify the unitarity of the rotation matrix
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->rotation_mat[ik].data(),
                                    G_Gdagger.data(), nbands, nbands, nbands, 'N', 'C' );
        rdmft::pTgemm_scalapack( this->para_Fij, this->rotation_mat[ik].data(), this->rotation_mat[ik].data(),
                                    Gdagger_G.data(), nbands, nbands, nbands, 'C', 'N' );
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
void IDMFT<TK, TR>::solving_mu()
{
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        double low_mu = 0.0;
        double high_mu = 0.0;
        double error_old = 1.0;
        double error_new = 1.0;

        // use mu from the previous step as the initial value
        // find low_mu and high_mu
        int times = 0;
        double sign_kappa = 1.0;
        if( PARAM.inp.idmft_kappa < 0 ) { sign_kappa = -1.0; }

        while( error_old * error_new > 0 || times <= 1 )
        {
            ++times;
            error_old = error_new;
            error_new = this->cal_occ_num(is) - this->sys_nelec_spin[is];

            if( error_new < 0 )
            {
                low_mu = this->mu[is];
                this->mu[is] += 1.0 * sign_kappa;
            }
            else if( error_new > 0 )
            {
                high_mu = this->mu[is];
                this->mu[is] -= 1.0 * sign_kappa;
            }

        }

        double solve_mu_times = 0;
        double total_elec_num = 0.0;
        double occ_num_error = 0.0;
        // use dichotomy
        while( solve_mu_times < 200 )
        {
            ++solve_mu_times;
            this->mu[is] = (low_mu + high_mu) / 2.0;

            // occ_num_error = this->cal_occ_num(is) - this->sys_nelec_spin[is];
            total_elec_num = this->cal_occ_num(is);
            occ_num_error = total_elec_num - this->sys_nelec_spin[is];

            if( std::abs(occ_num_error) < PARAM.inp.tot_nelec_thr ) { break; }
            
            if( occ_num_error < 0 )
            {
                low_mu = this->mu[is];
            }
            else if( occ_num_error > 0 )
            {
                high_mu = this->mu[is];
            }
        }

        if( solve_mu_times >= 200 )
        {
            std::cout << "\n" << "solve_mu_times is too big: " << solve_mu_times << "\n" << std::endl;
            std::cout << "\n" << "electron number is not conserved !!!!!!!!!!! " << "\n" << std::endl;
            assert( solve_mu_times <= 200 );
        }

        std::cout << "\n" << "mu: " << this->mu[is] << std::endl;
        std::cout << "\n" << "total_elec_num: " << total_elec_num << std::endl;
        std::cout << "\n" << "occ_num_error: " << occ_num_error << "\n" << std::endl;

    }
}


template <typename TK, typename TR>
double IDMFT<TK, TR>::cal_occ_num(const int is)
{
    double tot_occ_num = 0.0;

    for(int ik=0; ik<nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            
            this->occ_number[is][ik*nbands + ib] = 1.0 / ( 1 + std::exp( ( this->diag_Fii[ is*this->nk_nospin + ik][ib] - this->mu[is] ) / PARAM.inp.idmft_kappa ) ) ;

            tot_occ_num += this->occ_number[is][ik*nbands + ib] * this->num_symm_k[is*this->nk_nospin + ik];
        }
    }

    return tot_occ_num;
}


// template <typename TK, typename TR>
// void IDMFT<TK, TR>::solve_zero_occ_num()
// {
//     ModuleBase::matrix temp_occ_num = this->rdmft_solver->occ_number;
//     for(int ik=0; ik<temp_occ_num.nr; ++ik)
//     {
//         int zero_num = 0;
//         for(int ib=0; ib<temp_occ_num.nc; ++ib)
//         {
//             if( std::abs(temp_occ_num(ik, ib)) < 1e-16 )
//             {
//                 ++zero_num;
//                 temp_occ_num(ik, ib) = 1e-16;
//             }
//         }
//         temp_occ_num(ik, 0) -= zero_num * 1e-16;
//     }
//     this->rdmft_solver->update_elec( &temp_occ_num );
// }






template class IDMFT<double, double>;
template class IDMFT<std::complex<double>, double>;
template class IDMFT<std::complex<double>, std::complex<double>>;


}

