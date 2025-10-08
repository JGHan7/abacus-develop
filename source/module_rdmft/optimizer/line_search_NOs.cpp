//==========================================================
// Author: Jingang Han
// DATE : 2025-09-24
//==========================================================

#include <algorithm>
#include "module_rdmft/optimizer/line_search_NOs.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_rdmft/optimizer/bfgs_method.h"
#include "module_rdmft/optimizer/cg_method.h"

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

    this->scaling_P_old.resize(this->nk_total);

    // this->var_thetaR.resize(this->nk_total);
    this->var_thetaR_tensor.resize(this->nk_total);
    this->var_thetaR_for_bfgs.resize(this->nk_total);

    this->R_optimizer.resize(this->nk_total);
    this->ls.resize(this->nk_total);
    this->thetaR.resize(nk_total);
    this->R_tensor.resize(this->nk_total);
    this->dE_dthetaR_tensor.resize(this->nk_total);
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
        // this->var_thetaR[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
        this->var_thetaR_tensor[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
        this->var_thetaR_for_bfgs[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());

        // thetaR_real[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        // thetaR_imag[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        this->thetaR[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>().requires_grad(true));
        this->R_tensor[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        this->dE_dthetaR_tensor[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());

        wfc_new_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        wfc_0_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        dE_dwfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());

        this->ls[ik] = std::make_unique< rdmft::LineSearch<double> >();
        if( PARAM.inp.rdmft_orb_opti == "cg" )
        {
            this->R_optimizer[ik] = std::make_unique< rdmft::CG_method<TK> >();
            
            // test
            this->ls[ik]->get_options().ls_wolfe_c2 = 0.1;
        }
        else
        {
            this->R_optimizer[ik] = std::make_unique< rdmft::BFGS_method<TK> >();
        }
        this->R_optimizer[ik]->init( PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 1e-10 );

        this->dE_dR_global[ik].resize(PARAM.inp.nbands * PARAM.inp.nbands, 0.0);
        this->dE_dthetaR_global[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
        this->search_direction[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);

        this->scaling_P_old[ik].resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
    }

    this->scaling_P.resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
    
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
        this->init_step_k[ik] = 1.0;
        this->phi_0_k[ik] = this->rdmft_solver->Etotal;
        this->Ek_iter[ik].clear();
        this->Ek_iter[ik].push_back(this->rdmft_solver->Etotal);
    }
    ++this->num_restart;
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

    // perform a forward calculation before all optimizations begin
    if( PARAM.inp.rdmft_auto_diff )
    {
        this->cal_phi(nullptr);
    }

    this->restart_opti();
}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::do_line_search(const bool start_guess)
{
    // if( start_guess )
    // {
    //     this->restart_opti();
    // }

    bool new_landscape = (this->iter == 0) ? true: false;
    // bool new_landscape = (this->iter == 0 || (PARAM.inp.precond_type != 1 && this->iter%10 == 0) ) ? true: false;


    for(int ik=0; ik<this->nk_total; ++ik)
    {

        // for multiple k-point problems, we need to consider whether these two functions should be inside or outside the loop !!!!!!!!!!!!!!!!!
        // as well as the state update of rdmft_solver (k-point update or overall update) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        this->cal_dE_dR(&ik);
        this->cal_pk_dphi0(new_landscape, &ik);
        // std::cout << "\n******\n" << "ls, dphi_0_k: " << this->dphi_0_k[ik] << "\n******\n" << std::endl;

        if( this->iter != 0 )
        {
            this->init_step_k[ik] = 1.01 * 2.0 * ( this->Ek_iter[ik].back() - this->Ek_iter[ik][this->Ek_iter[ik].size() - 2] ) / this->dphi_0_k[ik];
            this->init_step_k[ik] = std::min(1.0, std::abs(init_step_k[ik]));
            std::cout << "\n" << "init_step by quadratic: " << this->init_step << "\n" << std::endl;

            // // test
            // if( PARAM.inp.rdmft_orb_opti == "cg" )
            // {
            //     this->init_step_k[ik] = 1.0;
            // }
        }

        auto phi = [this, ik](const double trial_alpha)
        {
            this->step_size_k[ik] = trial_alpha;
            this->update_thetaR(&ik);
            double trial_phi = this->cal_phi(&ik);
            return trial_phi;
        };

        auto dphi = [this, ik]()
        {
            double trial_dphi = this->cal_dphi(&ik);;
            return trial_dphi;
        };

        double max_elem_pk = 0.0;
        for(int j=0; j<this->search_direction[ik].size(); ++j)
        {
            max_elem_pk = std::max( max_elem_pk, std::abs(this->search_direction[ik][j]) );
        }

        // do line search
        this->step_size_k[ik] = this->ls[ik]->do_line_search(phi, dphi, this->phi_0_k[ik], this->dphi_0_k[ik], max_elem_pk, this->init_step_k[ik]);

        // use the final_step_size to update 
        this->phi_0_k[ik] = phi(this->step_size_k[ik]);
        this->Ek_iter[ik].push_back(this->phi_0_k[ik]);
        std::cout << "\n" << "the final step_size by lineSearch_NOs: " << step_size_k[ik] << "\n" << std::endl;

        // 
        torch::Tensor pk_tensor = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
        rdmft::vector2tensor(this->search_direction[ik], pk_tensor, { nbands64*(nbands64 + 1) / 2 });
        this->var_thetaR_for_bfgs[ik] = this->var_thetaR_for_bfgs[ik] + this->step_size_k[ik] * pk_tensor;

        if( this->opti_deltaR )
        {

            this->wfc_0_tensor[ik] = this->wfc_new_tensor[ik].detach().clone();
            this->var_thetaR_tensor[ik].data().zero_();
        }
        else
        {
            this->var_thetaR_tensor[ik] = this->thetaR[ik].detach().clone();
        }

    }

    ++this->iter;
    return this->rdmft_solver->Etotal;

}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::update_thetaR(const int* ik)
{
    torch::Tensor pk_tensor = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
    for(int jk=0; jk<this->nk_total; ++jk)
    {
        if( ik != nullptr )
        {
            if( jk != *ik )
            {
                continue;
            }
        }

        // the curly braces are required
        {
            // in this scope, tracing operations are prohibited (for thetaR)
            torch::NoGradGuard no_grad;

            rdmft::vector2tensor(this->search_direction[jk], pk_tensor, { nbands64*(nbands64 + 1) / 2 });
            this->thetaR[jk] = this->var_thetaR_tensor[jk] + this->step_size_k[jk] * pk_tensor;
        }
        this->thetaR[jk].set_requires_grad(true);
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

        // if( this->opti_deltaR )
        // {
        //     this->thetaR[jk].data().zero_();
        // }
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
        rdmft::Tgemm_lapack( this->dE_dthetaR_global[jk].data(), this->search_direction[jk].data(), &dphi, 1, 1, PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 'T');
    }

    return std::real(dphi);
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::cal_pk_dphi0(const bool new_landscape, const int* ik)
{
    // std::vector<TK> d2E_dR2_local(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
    // std::vector<TK> d2E_dR2_global(PARAM.inp.nbands * PARAM.inp.nbands, 0.0);
    // torch::Tensor d2E_dR2_tensor;
    // torch::Tensor d2E_dthetaR_2_tensor = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
    // std::vector<TK> d2E_dthetaR_2_global(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);

    std::vector<TK> d2E_dR2_local;
    std::vector<TK> d2E_dR2_global;
    torch::Tensor d2E_dR2_tensor;
    torch::Tensor d2E_dthetaR_2_tensor;
    std::vector<TK> d2E_dthetaR_2_global;
    std::vector<TK> dE_dthetaR_u; // test
    // std::vector<TK> transport; // test
    if( PARAM.inp.precond_orb )
    {
        d2E_dR2_local.resize(this->para_Fij->get_row_size() * this->para_Fij->get_col_size(), 0.0);
        d2E_dR2_global.resize(PARAM.inp.nbands * PARAM.inp.nbands, 0.0);
        d2E_dthetaR_2_tensor = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>());
        d2E_dthetaR_2_global.resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
        dE_dthetaR_u.resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
        // transport.resize(PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 0.0);
    }

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

        // rdmft::tensor2vector(this->thetaR[jk], thetaR_vec); this->var_thetaR_for_bfgs[ik]
        rdmft::tensor2vector(this->var_thetaR_for_bfgs[jk], thetaR_vec);
        // get pk: provide thetaR and dE_dthetaR to BFGS
        if( PARAM.inp.precond_orb ) // && this->iter >= 5
        {
            // get d2E_dR2
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

            if( PARAM.inp.precond_type == 1 )
            {
                this->R_optimizer[jk]->get_pk(this->dE_dthetaR_global[jk], thetaR_vec, this->search_direction[jk], new_landscape, &d2E_dthetaR_2_global);
            }
            else
            {
                if( new_landscape || this->iter%5 == 0 )
                {
                    for(int i=0; i<this->scaling_P.size(); ++i)
                    {
                        this->scaling_P[i] = std::sqrt( std::max( 1e-10, std::abs(d2E_dthetaR_2_global[i]) ) );
                    }
                }

                if( !new_landscape && this->iter%5 == 0 )
                {
                    // get transport mat: T = S_k+1 Sk^-1
                    for(int i=0; i<this->scaling_P.size(); ++i)
                    {
                        this->scaling_P_old[jk][i] = this->scaling_P[i] / this->scaling_P_old[jk][i];
                    }
                    this->R_optimizer[jk]->transport(this->scaling_P_old[jk]);
                }
                this->scaling_P_old[jk] = this->scaling_P;

                for(int i=0; i<this->scaling_P.size(); ++i)
                {
                    thetaR_vec[i] *= this->scaling_P[i];
                    dE_dthetaR_u[i] = this->dE_dthetaR_global[jk][i] / this->scaling_P[i];
                }
                this->R_optimizer[jk]->get_pk(dE_dthetaR_u, thetaR_vec, this->search_direction[jk], new_landscape);
                for(int i=0; i<this->search_direction[jk].size(); ++i)
                {
                    this->search_direction[jk][i] /= this->scaling_P[i];
                }
            }
        }
        else
        {
            this->R_optimizer[jk]->get_pk(this->dE_dthetaR_global[jk], thetaR_vec, this->search_direction[jk], new_landscape);
        }

        TK dphi_0_temp = 0.0;
        rdmft::Tgemm_lapack( this->dE_dthetaR_global[jk].data(), this->search_direction[jk].data(), &dphi_0_temp, 1, 1, PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, 'T');
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
            rdmft::vector2tensor(dE_dR_global[jk], dE_dR_tensor_jk, {nbands64, nbands64});

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

