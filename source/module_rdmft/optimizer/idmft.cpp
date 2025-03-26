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

    // temp
    // this->if_rotate_Fock = false;
    this->if_rotate_Fock = PARAM.inp.rotate_fock;

    // get the number of symmetric k-points
    this->num_symm_k.resize(this->kv->wk.size());
    for(int iks; iks<this->num_symm_k.size(); ++iks)
    {
        this->num_symm_k[iks] = this->kv->wk[iks] * nkstot_full;
    }

    // get the total number of electrons
    this->sys_nelec_spin.resize(PARAM.inp.nspin);
    this->mu.resize(PARAM.inp.nspin, 0.0);
    if( PARAM.inp.nspin == 1 )
    {
        // remove the weight of spin
        for(int iks; iks<this->kv->wk.size(); ++iks)
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
double IDMFT<TK, TR>::optimize()
{
    this->etotal_old = this->etotal;

    this->get_Fock();

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
        else
        {
            rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->naos_rep_wfc1(ik, 0, 0), this->new_wfc(ik, 0, 0) );
        }

        // update the rotation matrix
        if( this->if_rotate_Fock )
        {
            // test, rotation = egienvector?
            this->rotation_mat[ik] = this->nos_rep_wfc[ik];
        }
    }

    // update from 1-step_wfc?
    if( !this->if_get_wfc1 && this->if_rotate_Fock )
    {
        // rotate the Fock-like matrix to the first step NOs representation,
        // then new_wfc = nos_rep_wfc * ( nos_rep_wfc1 * NAOs_rep_wfc0) for each iteration

        // get NAOs_rep_wfc1 = nos_rep_wfc1 * NAOs_rep_wfc0
        TK* p_wfc = &this->rdmft_solver->wfc(0, 0, 0);
        TK* p_naos_rep_wfc1 = &this->naos_rep_wfc1(0, 0, 0);
        for(int i=0; i<this->new_wfc.size(); ++i) { p_naos_rep_wfc1[i] = p_wfc[i]; }

        this->if_get_wfc1 = true;
    }

    // optimize the occ_number
    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);
    this->opti_occ_num(occ_num_pass);

    // update the occupation number and wfc
    this->rdmft_solver->update_elec( &occ_num_pass, &(this->new_wfc) );
    this->etotal = this->rdmft_solver->cal_Energy();
    this->new_wfc.zero_out();

    // cal new DM and diff_DM
    std::vector< std::vector<TK> > DM_new(nk_total, std::vector<TK>(this->ParaV->nloc));
    this->rdmft_solver->cal_DM_XC(this->rdmft_solver->wg, DM_new);
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

            this->DM[ik][iloc] = DM_new[ik][iloc];
        }
    }
    rdmft::reduce_all_max(this->diff_DM_max);

    this->diff_Etotal = this->etotal - this->etotal_old;

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

    if(this->if_rotate_Fock)
    {
        this->rotate_Fock();
    }
}


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

