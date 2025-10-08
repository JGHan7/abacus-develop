//==========================================================
// Author: Jingang Han
// DATE : 2025-09-24
//==========================================================
#ifndef LINE_SEARCH_NOS_H
#define LINE_SEARCH_NOS_H

// temp
#include <torch/torch.h>

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/opti_method.h"
#include  "module_rdmft/optimizer/line_search_method.h"


namespace rdmft
{

  //! @brief physical notation: natural orbitals = ONs

template<typename TK, typename TR>
class LineSearch_NOs
{

  public:

    LineSearch_NOs();
    ~LineSearch_NOs();

    void init(RDMFT<TK, TR>* rdmft_in);

    //! generate start guess
    virtual void get_start_guess();

    //! use an approximate line search method to find a suitable step size
    double do_line_search(const bool start_guess = false);

    void restart_opti();

    std::vector<double> Etotal_iter;
    std::vector<std::vector<double>> Ek_iter;
    int iter = 0;
    int num_restart = 0;

    int get_num_bfgs_restart() { return this->R_optimizer[0]->num_restart_skip; }


  protected:

    //! calculate phi(alpha) = E(x_k + alpha * p_k) and return it. x_new = x_k + alpha*p_k
    virtual double cal_phi(const int* ik);

    //! calculate dE/dx, dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T and return dphi.
    //! in the current framework (thetaR separates k-point storage), k must be specified when calling
    virtual double cal_dphi(const int* ik);

    //! calculate the direction of the line search: pk, and dphi_0
    virtual void cal_pk_dphi0(const bool new_landscape = false, const int* ik = nullptr);

    //! calculate dE/dx
    //! R_new = R_k + alpha*p_k, if R is the same as the R in the last call to cal_phi(), nullptr is used.
    virtual void cal_dE_dR(const int* ik);

    RDMFT<TK, TR>* rdmft_solver = nullptr;

    std::vector< std::unique_ptr<rdmft::Opti_method<TK>> > R_optimizer;

    // rdmft::LineSearch<TK> ls;
    std::vector< std::unique_ptr<rdmft::LineSearch<double>> > ls;

    std::vector<std::vector<TK>> search_direction;

    //! step_size, alpha: x_k+1 = x_k + alpha * p_k
    double step_size = 1.0;
    std::vector<double> step_size_k;

    //! phi(alpha) = E(x_k + alpha * p_k), phi_0 = E(x_k)
    double phi_0 = 0.0;
    std::vector<double> phi_0_k;

    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T, dphi_0 = E'(x_k) * p_k^T
    double dphi_0 = 0.0;
    std::vector<double> dphi_0_k;

    // delete in the future ?
    double init_step = 1.0;
    std::vector<double> init_step_k;

  private:

    int64_t nbands64 = 0;
    int64_t nbasis64 = 0;
    int nk_total = 0;

    bool opti_deltaR = true;

    const double grad_factor = 1.0;

    std::vector<TK> scaling_P; // test
    std::vector< std::vector<TK> > scaling_P_old; // test

    //! the actual optimized parameters, only the optimized results of each step are stored (not affected by the intermediate values ​​of the line search)
    // std::vector< std::vector<TK> > var_thetaR;
    std::vector<torch::Tensor> var_thetaR_tensor;
    std::vector<torch::Tensor> var_thetaR_for_bfgs;

    //! thetaR is the leaf tensor of automatic differentiation (the starting point of the computational graph)
    std::vector<torch::Tensor> thetaR;
    std::vector<torch::Tensor> R_tensor;
    // std::vector<torch::Tensor> dE_dR_tensor;
    std::vector<torch::Tensor> dE_dthetaR_tensor;

    std::vector< std::vector<TK> > dE_dR_global;
    std::vector< std::vector<TK> > dE_dthetaR_global;

    std::vector<torch::Tensor> wfc_new_tensor;
    std::vector<torch::Tensor> wfc_0_tensor;
    std::vector<torch::Tensor> dE_dwfc_tensor;

    psi::Psi<TK> wfc_new;
    psi::Psi<TK> wfc_old;

    const Parallel_Orbitals* ParaV = nullptr;

    const Parallel_2D* para_Fij = nullptr;

    void update_R_wfc(const int* ik = nullptr);

    void update_thetaR(const int* ik = nullptr);


};

}


#endif