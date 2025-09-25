//==========================================================
// Author: Jingang Han
// DATE : 2025-09-24
//==========================================================

#include <algorithm>
#include "module_rdmft/optimizer/line_search_NOs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"

// #include <torch/torch.h>

#include "module_rdmft/rdmft_tools.h" // temp

namespace rdmft
{

template<typename TK, typename TR>
LineSearch_NOs<TK, TR>::LineSearch_NOs()
{
    ;
}


template<typename TK, typename TR>
LineSearch_NOs<TK, TR>::~LineSearch_NOs()
{
    ;
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::init(RDMFT<TK, TR>* rdmft_in)
{
    this->rdmft_solver = rdmft_in;
    this->nk_total = this->rdmft_solver->nk_total;
    this->ParaV = this->rdmft_solver->ParaV;
    this->para_Fij = &this->rdmft_solver->para_Eij;

    this->nbands64 = PARAM.inp.nbands;
    this->nbasis64 = this->ParaV->get_wfc_global_nbasis();
    
    this->opti_deltaR = PARAM.inp.small_rotation;

    this->R_optimizer.resize(this->nk_total);
    this->thetaR.resize(nk_total);
    this->R_tensor.resize(this->nk_total);
    this->dE_dR_global.resize(this->nk_total);
    this->dE_dthetaR_global.resize(this->nk_total);
    this->search_direction.resize(this->nk_total);

    this->Ek_iter.resize(this->nk_total);
    this->step_size_k.resize(this->nk_total, 1.0);
    this->phi_0_k.resize(this->nk_total, 0.0);
    this->dphi_0_k.resize(this->nk_total, 0.0);
    this->init_step_k.resize(this->nk_total, 1.0);

    wfc_new_tensor.resize(this->nk_total);
    wfc_0_tensor.resize(this->nk_total);
    dE_dwfc_tensor.resize(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        // thetaR_real[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        // thetaR_imag[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        this->thetaR[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>().requires_grad(true));
        this->R_tensor[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        this->dE_dthetaR_tensor[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());

        wfc_new_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        wfc_0_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        dE_dwfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());

        this->R_optimizer[ik] = std::make_unique< rdmft::BFGS_Opti<TK> >();
        this->R_optimizer[ik]->init( PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, this->nk_total );

        this->dE_dR_global[ik].resize(PARAM.inp.nbands * PARAM.inp.nbands, 0.0);
        this->dE_dthetaR_global[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
        this->search_direction[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
    }

    this->wfc_new.resize(this->nk_total, this->ParaV->ncol_bands, this->ParaV->nrow);
    this->wfc_new.zero_out();
    this->wfc_old = this->wfc_new;

}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::restart_opti()
{
    this->iter = 0;
    this->init_step = 1.0;
    this->phi_0 = this->rdmft_solver->Etotal;
    this->Etotal_iter.clear();
    this->Etotal_iter.push_back(this->rdmft_solver->Etotal);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        this->Ek_iter[ik].clear();
        this->Ek_iter[ik].push_back(this->rdmft_solver->Etotal);
    }
}

template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::get_start_guess()
{
    // get wfc0 by tensor type
    std::vector<std::vector<TK>> wfc_vec(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        rdmft::psi2vec(ik, *this->ParaV, this->rdmft_solver->wfc, wfc_vec[ik]);
        rdmft::vector2tensor( wfc_vec[ik], this->wfc_0_tensor[ik], {nbands64, nbasis64} );
    }


    // std::fill(this->var_x.begin(), this->var_x.end(), 0.0);
    // if(PARAM.inp.random_occ_num)
    // {
    //     this->param_occ_num->get_inital_guess(this->var_x);

    //     // // update rdmft elec_state
    //     // ModuleBase::matrix occ_number( this->param_occ_num->get_occ_number() );
    //     // this->rdmft_solver->update_elec( &occ_number );
    // }
    // else
    // {
    //     this->param_occ_num->get_inital_guess(this->var_x, &rdmft_solver->occ_number);
    // }

    // this->occ_number = this->param_occ_num->get_occ_number();
    // this->rdmft_solver->update_elec( &(this->occ_number) );
    // this->phi_0 = this->rdmft_solver->cal_Energy();
}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::do_line_search(const bool start_guess)
{
    if( start_guess )
    {
        this->restart_opti();
    }
    bool new_landscape = (this->iter == 0) ? true: false;











    // if(start_guess)
    // {
    //     this->get_start_guess();
    //     std::cout << "\n******\n" << "start_guess: ls, 0.0" << "\n******\n" << std::endl;

    //     this->init_step = 1.0;
    //     this->Etotal_iter.clear();
    //     this->phi_0 = this->rdmft_solver->Etotal;
    //     this->Etotal_iter.push_back(this->rdmft_solver->Etotal);

    //     // return 0.0; // test !!!!!!!!!!
    // }

    // std::cout << "\n******\n" << "iter in occ_num: " << iter << "\n" << std::endl;

    // // // test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // // this->phi_0 = this->cal_phi(this->var_x);

    // // rdmft cal dE_docc_num, PARAM_ONs convert dE_docc_num to dE_dx
    // this->cal_dE_dx(this->dE_dx);

    // this->cal_pk_dphi0( (start_guess || this->iter == 0) );
    // std::cout << "\n******\n" << "ls, dphi_0: " << this->dphi_0 << "\n******\n" << std::endl;

    // if( this->iter != 0 )
    // {
    //     this->init_step = 1.01 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
    //     this->init_step = std::min(1.0, this->init_step);
    //     // this->init_step = 1.0 * 2.0 * ( this->Etotal_iter.back() - this->Etotal_iter[this->Etotal_iter.size() - 2] ) / this->dphi_0;
    //     std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;
    // }

    // std::vector<double> var_x_old = this->var_x;

    // auto phi = [this](const double trial_alpha)
    // {
    //     std::vector<double> x_new(this->var_x.size(), 0.0);
    //     this->step_size = trial_alpha;
    //     this->update_x(x_new);
    //     double trial_phi = this->cal_phi(x_new);
    //     return trial_phi;
    // };

    // auto dphi = [this]()
    // {
    //     std::vector<double> trial_dE_dx(this->dE_dx.size(), 0.0);
    //     double trial_dphi = this->cal_dphi(trial_dE_dx);
    //     return trial_dphi;
    // };

    // double max_elem_pk = 0.0;
    // for(int i=0; i<this->search_direction.size(); ++i)
    // {
    //     max_elem_pk = std::max( max_elem_pk, std::abs(this->search_direction[i]) );
    // }

    // this->step_size = this->ls.do_line_search(phi, dphi, this->phi_0, this->dphi_0, max_elem_pk, this->init_step);

    // // update x_k+1 = x_k + step_size * p_k
    // // can't use update_x(), because we need "+=" instead of "="
    // for(int i=0; i<this->var_x.size(); ++i)
    // {
    //     this->var_x[i] += this->step_size * this->search_direction[i]; 
    // }

    // std::cout << "\n" << "the final step_size by lineSearch_NOs: " << this->step_size << "\n" << std::endl;

    if( this->opti_deltaR )
    {
        for (int ik=0; ik < this->nk_total; ++ik)
        {
            this->wfc_0_tensor[ik] = this->wfc_new_tensor[ik].detach();
        }
    }


}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::update_R_wfc(const int* ik)
{
    for(int jk=0; jk<this->nk_total; ++jk)
    {
        if( ik != nullptr )
        {
            if( jk != *ik )
            {
                continue;
            }
        }

        // generate upper triangle index
        auto idx = torch::triu_indices(this->nbands64, this->nbands64);

        // although it is a temporary variable, the release time is managed by torch and does not destroy the calculation graph
        torch::Tensor R_upper = torch::zeros({this->nbands64, this->nbands64}, torch_dtype<TK>());

        // fill thetaR with upper triangular indices
        R_upper = R_upper.index_put({idx[0], idx[1]}, this->thetaR[jk]);

        // constructing a skew-Hermitian matrix R_skew = R_upper - R_upper^\dagger
        // this->R_tensor[jk] = ( R_upper - R_upper.transpose(0,1).conj() ) * this->scaling_R;
        this->R_tensor[jk] = R_upper - R_upper.transpose(0,1).conj();

        // get new wfc
        this->wfc_new_tensor[jk] = torch::matmul( torch::matrix_exp(this->R_tensor[jk]), this->wfc_0_tensor[jk] );
        std::vector<TK> wfc_vec;
        rdmft::tensor2vector( this->wfc_new_tensor[jk], wfc_vec );
        rdmft::vec2psi(jk, *this->ParaV, wfc_vec, this->wfc_new);

        if( this->opti_deltaR )
        {
            this->thetaR[jk].data().zero_();
        }
    }
}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::cal_phi(const int* ik)
{
    // update wfc_new
    // for(int jk=0; jk<this->nk_total; ++jk)
    // {
    //     if( ik != nullptr )
    //     {
    //         if( jk != *ik )
    //         {
    //             continue;
    //         }
    //     }
    //     this->update_R_wfc(jk);
    // }
    this->update_R_wfc(ik);

    // cal energy
    this->rdmft_solver->update_elec( nullptr, &(this->wfc_new) );
    double phi = this->rdmft_solver->cal_Energy();

    return phi;
}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::cal_dphi(const int* ik)
{
    this->cal_dE_dR(ik);
    // double dphi = 0.0;
    TK dphi = 0.0;

    for(int jk=0; jk<this->nk_total; ++jk)
    {
        if( ik != nullptr )
        {
            if( jk != *ik )
            {
                continue;
            }
        }
        rdmft::Tgemm_lapack( this->dE_dthetaR_global[jk].data(), this->search_direction[jk].data(), &dphi, 1, 1, this->nk_total * PARAM.inp.nbands , 'T');
    }

    return std::real(dphi);
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::cal_pk_dphi0(const bool new_landscape, const int* ik)
{
    std::vector<TK> d2E_dR2_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
    std::vector<TK> d2E_dR2_global(PARAM.inp.nbands * PARAM.inp.nbands, 0.0);
    torch::Tensor d2E_dR2_tensor;

    torch::Tensor d2E_dthetaR_2_tensor = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
    std::vector<TK> d2E_dthetaR_2_global(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);

    std::vector<TK> thetaR_vec(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
    for(int jk=0; jk<this->nk_total; ++jk)
    {
        if( ik != nullptr )
        {
            if( jk != *ik )
            {
                continue;
            }
        }

        rdmft::tensor2vector(this->thetaR[jk], thetaR_vec);
        // get pk: provide thetaR and dE_dthetaR to BFGS
        if( PARAM.inp.precond_orb )
        {
            this->rdmft_solver->cal_E_grad2_Rpq(jk, d2E_dR2_local);
            rdmft::collect_vec(this->para_Fij, d2E_dR2_local, d2E_dR2_global);
            rdmft::vector2tensor(d2E_dR2_global, d2E_dR2_tensor, {nbands64, nbands64});

            // generate upper triangle index
            auto idx = torch::triu_indices(this->nbands64, this->nbands64);

            // fill upper triangle dE_dthetaR_tensor
            d2E_dthetaR_2_tensor = d2E_dR2_tensor.index({idx[0], idx[1]}) 
                                            - d2E_dR2_tensor.index({idx[1], idx[0]}).conj();
            
            // convert data formats
            rdmft::tensor2vector(d2E_dthetaR_2_tensor, d2E_dthetaR_2_global);

            this->R_optimizer[jk]->get_pk(this->dE_dthetaR_global[jk], thetaR_vec, this->search_direction[jk], new_landscape, &d2E_dthetaR_2_global);
        }
        else
        {
            this->R_optimizer[jk]->get_pk(this->dE_dthetaR_global[jk], thetaR_vec, this->search_direction[jk], new_landscape);
        }

        TK dphi_0_temp = 0.0;
        rdmft::Tgemm_lapack( this->dE_dthetaR_global[jk].data(), this->search_direction[jk].data(), &dphi_0_temp, 1, 1, this->nk_total * PARAM.inp.nbands , 'T');
        this->dphi_0_k[jk] = std::real(dphi_0_temp);
    }
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::cal_dE_dR(const int* ik)
{
    std::vector<TK> dE_dR_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
    // std::vector<TK> d2E_dR2_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
    torch::Tensor dE_dR_tensor_jk;

    for(int jk=0; jk<this->nk_total; ++jk)
    {
        if( ik != nullptr )
        {
            if( jk != *ik )
            {
                continue;
            }
        }

        if( PARAM.inp.rdmft_auto_diff )
        {
            // obtain accurate dE/dR using analytical dE/dwfc and automatic differentiation, and convert the format
            psi::Psi<TK> dE_dwfc;
            std::vector<TK> dE_dwfc_vec;
            this->rdmft_solver->cal_E_grad_wfc(dE_dwfc);
            rdmft::psi2vec( jk, *this->ParaV, dE_dwfc, dE_dwfc_vec );
            this->dE_dwfc_tensor[jk] = torch::from_blob(dE_dwfc_vec.data(), {nbands64, nbasis64}, torch_dtype<TK>()).clone();

            // auto diff
            this->wfc_new_tensor[jk].backward( this->dE_dwfc_tensor[jk] );

            // get dE_dthetaR
            this->dE_dthetaR_tensor[jk] = this->thetaR[jk].grad().clone();
        }
        else
        {
            // antisymm lambda to get dE_dR_local
            this->rdmft_solver->cal_antisym_lambda(jk, dE_dR_local, this->grad_factor);

            // convert data formats
            rdmft::collect_vec(this->para_Fij, dE_dR_local, dE_dR_global[jk]);
            rdmft::vector2tensor(dE_dR_global[jk], dE_dR_tensor_jk, { PARAM.inp.nbands * PARAM.inp.nbands });

            // generate upper triangle index
            auto idx = torch::triu_indices(this->nbands64, this->nbands64);
            
            // fill upper triangle dE_dthetaR_tensor
            this->dE_dthetaR_tensor[jk] = dE_dR_tensor_jk.index({idx[0], idx[1]}) 
                                            - dE_dR_tensor_jk.index({idx[1], idx[0]}).conj();
        }

        // convert data formats
        rdmft::tensor2vector( this->dE_dthetaR_tensor[jk], this->dE_dthetaR_global[jk] );
    }
}








template class LineSearch_NOs<double, double>;
template class LineSearch_NOs<std::complex<double>, double>;
template class LineSearch_NOs<std::complex<double>, std::complex<double>>;



}

