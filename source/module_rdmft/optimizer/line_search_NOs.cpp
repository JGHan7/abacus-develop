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
    this->nbands64 = PARAM.inp.nbands;
    this->nbasis64 = this->rdmft_solver->ParaV->get_wfc_global_nbasis();
    
    // this->opti_deltaR = PARAM.inp.small_rotation;

    this->R_optimizer.resize(this->nk_total);
    this->thetaR.resize(nk_total);
    this->R_tensor.resize(this->nk_total);
    this->Ek_iter.resize(this->nk_total);

    // wfc_new_tensor.resize(this->nk_total);
    // wfc_0_tensor.resize(this->nk_total);
    // dE_dwfc_tensor.resize(this->nk_total);
    for(int ik=0; ik<this->nk_total; ++ik)
    {
        // thetaR_real[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        // thetaR_imag[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<double>().requires_grad(true));
        this->thetaR[ik] = torch::zeros({ nbands64*(nbands64 + 1) / 2 }, torch_dtype<TK>().requires_grad(true));
        this->R_tensor[ik] = torch::zeros({nbands64, nbands64}, torch_dtype<TK>());
        
        // wfc_new_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        // wfc_0_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());
        // dE_dwfc_tensor[ik] = torch::zeros({nbands64, nbasis64}, torch_dtype<TK>());

        this->R_optimizer[ik] = std::make_unique< rdmft::BFGS_Opti<TK> >();
        this->R_optimizer[ik]->init( PARAM.inp.nbands*(PARAM.inp.nbands + 1) / 2, this->nk_total );
    }

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


}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::cal_phi(const int* ik, std::vector<double>& x_new)
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




    }
}


template<typename TK, typename TR>
double LineSearch_NOs<TK, TR>::cal_dphi(const int* ik, std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr)
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




    }
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::cal_pk_dphi0(const bool new_landscape)
{
    
}


template<typename TK, typename TR>
void LineSearch_NOs<TK, TR>::cal_dE_dR(const int* ik, std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr)
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




    }
}










template class LineSearch_NOs<double, double>;
template class LineSearch_NOs<std::complex<double>, double>;
template class LineSearch_NOs<std::complex<double>, std::complex<double>>;



}

