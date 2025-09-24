//==========================================================
// Author: Jingang Han
// DATE : 2025-09-24
//==========================================================
#ifndef LINE_SEARCH_NOS_H
#define LINE_SEARCH_NOS_H

// temp
#include <torch/torch.h>

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/bfgs_opti.h"
#include  "module_rdmft/optimizer/line_search_method.h"


namespace rdmft
{


template<typename TK, typename TR>
class LineSearch_NOs
{

  public:

    LineSearch_NOs();
    ~LineSearch_NOs();

    void init(RDMFT<TK, TR>* rdmft_in);

    //! use an approximate line search method to find a suitable step size
    double do_line_search(const bool start_guess = false);

    void restart_opti();

    std::vector<double> Etotal_iter;
    std::vector<std::vector<double>> Ek_iter;
    int iter = 0;


  protected:

    //! generate start guess
    virtual void get_start_guess();

    //! calculate phi(alpha) = E(x_k + alpha * p_k) and return it. x_new = x_k + alpha*p_k
    virtual double cal_phi(const int* ik, std::vector<double>& x_new);

    //! calculate dE/dx, dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T and return dphi.
    //! x_new = x_k + alpha*p_k, if x is the same as the x in the last call to cal_phi(), nullptr is used.
    virtual double cal_dphi(const int* ik, std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr = nullptr);  // is dE_dx_new useful here? Consider deleting the outgoing

    //! calculate the direction of the line search: pk, and dphi_0
    virtual void cal_pk_dphi0(const bool new_landscape = false);

    //! calculate dE/dx
    //! R_new = R_k + alpha*p_k, if R is the same as the R in the last call to cal_phi(), nullptr is used.
    virtual void cal_dE_dR(const int* ik, std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr = nullptr);

    RDMFT<TK, TR>* rdmft_solver = nullptr;

    std::vector< std::unique_ptr<rdmft::BFGS_Opti<TK>> > R_optimizer;

    rdmft::LineSearch<TK> ls;

    //! step_size, alpha: x_k+1 = x_k + alpha * p_k
    double step_size = 1.0;

    //! phi(alpha) = E(x_k + alpha * p_k), phi_0 = E(x_k)
    double phi_0 = 0.0;

    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T, dphi_0 = E'(x_k) * p_k^T
    double dphi_0 = 0.0;

    // delete in the future ?
    double init_step = 1.0;

  private:

    int64_t nbands64 = 0;
    int64_t nbasis64 = 0;
    int nk_total = 0;

    bool opti_deltaR = true;

    std::vector<torch::Tensor> thetaR;
    std::vector<torch::Tensor> R_tensor;
    // std::vector<torch::Tensor> wfc_new_tensor;
    // std::vector<torch::Tensor> wfc_0_tensor;
    // std::vector<torch::Tensor> dE_dwfc_tensor;




};

}


#endif