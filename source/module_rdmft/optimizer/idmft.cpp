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
    this->wk_nospin = this->kv->wk;
    int nkstot_full = this->kv->get_nkstot_full();

    this->para_Fij = &para_Fij_in;
    this->ParaV = &ParaV_in;
    this->rdmft_solver = rdmft_solver_in;

    // malloc
    this->new_wfc.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->naos_rep_wfc1.resize(nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->Fock_like_mat.resize(nk_total);
    this->diag_Fii.resize(nk_total);
    // this->rotation_mat.resize(nk_total);
    for(int ik=0; ik<nk_total; ++ik)
    {
        this->diag_Fii[ik].resize(nbands, 0.0);
        this->Fock_like_mat[ik].resize( para_Fij->get_row_size() * para_Fij->get_col_size(), 0.0 );

        // // or write in before_opti(), if you update the fixed NOs representation after each ONs optimization
        // rdmft::get_identi_mat( para_Fij, this->rotation_mat[ik] );
    }
    this->nos_rep_wfc = this->Fock_like_mat;
    
    this->occ_number.resize(PARAM.inp.nspin);
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        this->occ_number[is].resize(nk_nospin*nbands);
    }

    // // temp
    // // this->if_rotate_Fock = false;
    // this->if_rotate_Fock = PARAM.inp.rotate_fock;

    this->sys_nelec_spin.resize(PARAM.inp.nspin);
    this->mu.resize(PARAM.inp.nspin, 0.0);
    if( PARAM.inp.nspin == 1 )
    {
        // remove the weight of spin
        for(int ik; ik<this->wk_nospin.size(); ++ik)
        {
            this->wk_nospin[ik] /= 2.0;
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
    }

    // get the number of symmetric k-points
    this->num_symm_k.resize(this->wk_nospin.size());
    for(int iks; iks<this->num_symm_k.size(); ++iks)
    {
        this->num_symm_k[iks] = this->wk_nospin[iks] * nkstot_full;
    }

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::get_Fock()
{
    for(int ik=0; ik<Fock_like_mat.size(); ++ik)
    {
        std::fill(this->Fock_like_mat[ik].begin(), this->Fock_like_mat[ik].end(), 0.0);
        for(int iloc=0; iloc<this->Fock_like_mat[ik].size(); ++iloc)
        {
            this->Fock_like_mat[ik][iloc] = (this->rdmft_solver->Hij_no_exx[ik][iloc] + this->rdmft_solver->Hij_exx[ik][iloc]) * wk_nospin[ik];
        }
    }
}


template <typename TK, typename TR>
double IDMFT<TK, TR>::optimize_orb()
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
        rdmft::GkPsi( this->para_Fij, this->ParaV, this->nos_rep_wfc[ik][0], this->rdmft_solver->wfc(ik, 0, 0), this->new_wfc(ik, 0, 0) );
    }

    this->rdmft_solver->update_elec( nullptr, &(this->new_wfc) );
    this->etotal = this->rdmft_solver->cal_Energy();
    this->new_wfc.zero_out();

    this->diff_Etotal = this->etotal - this->etotal_old;

    return this->diff_Etotal;

}


template <typename TK, typename TR>
void IDMFT<TK, TR>::opti_occ_num()
{
    this->solving_mu();

    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);
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

    this->rdmft_solver->update_elec( &occ_num_pass );

    std::cout << std::scientific << std::setprecision(6) << std::endl;
    rdmft::printMatrix_pointer(occ_num_pass.nr, occ_num_pass.nc, occ_num_pass.c, "occ_number", 10);
    std::cout << std::defaultfloat;

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
        while( error_old * error_new > 0 || times <= 1 )
        {
            ++times;
            error_old = error_new;
            error_new = this->cal_occ_num(is) - this->sys_nelec_spin[is];

            // assume kappa is greater than 0
            if( error_new < 0 )
            {
                low_mu = this->mu[is];
                this->mu[is] += 1.0;
            }
            else if( error_new > 0 )
            {
                high_mu = this->mu[is];
                this->mu[is] -= 1.0;
            }

        }

        double solve_mu_times = 0;
        // use dichotomy
        while( solve_mu_times < 200 )
        {
            ++solve_mu_times;
            this->mu[is] = (low_mu + high_mu) / 2.0;
            double occ_num_error = this->cal_occ_num(is) - this->sys_nelec_spin[is];
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

        if( solve_mu_times == 200 )
        {
            std::cout << "\n" << "solve_mu_times is too big: " << solve_mu_times << "\n" << std::endl;
            std::cout << "\n" << "electron number is not conserved !!!!!!!!!!! " << "\n" << std::endl;
        }

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


template <typename TK, typename TR>
void IDMFT<TK, TR>::solve_zero_occ_num()
{
    ModuleBase::matrix& temp_occ_num = this->rdmft_solver->occ_number;
    for(int ik=0; ik<temp_occ_num.nr; ++ik)
    {
        int zero_num = 0;
        for(int ib=0; ib<temp_occ_num.nc; ++ib)
        {
            if( std::abs(temp_occ_num(ik, ib)) < 1e-14 )
            {
                ++zero_num;
                temp_occ_num(ik, ib) = 1e-14;
            }
        }
        temp_occ_num(ik, 0) -= zero_num * 1e-14;
    }
}






template class IDMFT<double, double>;
template class IDMFT<std::complex<double>, double>;
template class IDMFT<std::complex<double>, std::complex<double>>;


}

