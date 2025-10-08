//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================
#ifndef LINE_SEARCH_ONS_H
#define LINE_SEARCH_ONS_H

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/parameterize_ONs/param_ONs.h"
#include "module_rdmft/optimizer/bfgs_method.h"
#include  "module_rdmft/optimizer/line_search_method.h"

namespace rdmft
{

template<typename TK, typename TR>
class LineSearch_ONs
{

  public:

    LineSearch_ONs();
    ~LineSearch_ONs();

    void init(const K_Vectors& kv_in, RDMFT<TK, TR>* rdmft_in);

    //! use an approximate line search method to find a suitable step size
    double do_line_search(const bool start_guess = false);

    void restart_opti();

    std::vector<double> Etotal_iter;
    int iter = 0;
    int num_restart = 0;

    int get_num_bfgs_restart() { return this->x_optimizer->num_restart_skip; }

    // temp
    double diff_rate_max = 1.0;

    // temp?
    std::vector<double>* get_var_x() { return &this->var_x; }
    ModuleBase::matrix get_occ_num() { return this->occ_number; }

    // //! parameters in strong wolfe or wolfe conditions
    // double ls_wolfe_c1 = 0.0;
    // double ls_wolfe_c2 = 0.0;

    // //! parameters in Armijo-Goldstein conditions
    // double ls_armijo_c1 = 0.0;
    // double ls_armijo_c2 = 0.0;

    // std::string ls_condition;

    // double max_step_size = 0.0;
    // double min_step_size = 0.0;


  protected:

    // //! Strong Wolfe condition, get the appropriate step length
    // void strong_wolfe();

    // //! Strong Wolfe condition, get the appropriate step length. By myself
    // void strong_wolfe2();

    // //! exact line search, get the most appropriate step length, just be used to test or solve simple problem
    // void exact_ls();

    // //! Wolfe condition
    // void wolfe();

    // //! used in Strong Wolfe condition, get the appropriate step length
    // void zoom(double step_size_low, double phi_low, double step_size_high, double phi_high, double dphi_low);

    //! calculate the new x based on the current step size
    void update_x(std::vector<double>& x_new);

    //! calculate phi(alpha) = E(x_k + alpha * p_k) and return it. x_new = x_k + alpha*p_k
    virtual double cal_phi(std::vector<double>& x_new);

    //! calculate dE/dx, dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T and return dphi.
    //! x_new = x_k + alpha*p_k, if x is the same as the x in the last call to cal_phi(), nullptr is used.
    virtual double cal_dphi(std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr = nullptr);  // is dE_dx_new useful here? Consider deleting the outgoing

    //! calculate the direction of the line search: pk, and dphi_0
    virtual void cal_pk_dphi0(const bool new_landscape = false);

    //! generate start guess
    virtual void get_start_guess();

    //! calculate dE/dx
    //! x_new = x_k + alpha*p_k, if x is the same as the x in the last call to cal_phi(), nullptr is used.
    virtual void cal_dE_dx(std::vector<double>& dE_dx_new, std::vector<double>* x_new_ptr = nullptr);

    //! x_k, (dE_dx)_k, search direction p_k and occupation numbers
    std::vector<double> var_x;
    std::vector<double> dE_dx;
    std::vector<double> search_direction;
    ModuleBase::matrix occ_number;

    //! step_size, alpha: x_k+1 = x_k + alpha * p_k
    double step_size = 1.0;

    //! the largest possible but appropriate step that satisfies the Armijo condition
    double armijo_step = 0.0;

    double init_step = 1.0;

    //! phi(alpha) = E(x_k + alpha * p_k), phi_0 = E(x_k)
    double phi_0 = 0.0;

    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T, dphi_0 = E'(x_k) * p_k^T
    double dphi_0 = 0.0;

  private:

    //! objective function E(occ_num)=E(x): provides Etotal_rdmft and first-order gradient
    RDMFT<TK, TR>* rdmft_solver = nullptr;

  public: // temp public
    //! handle constraints: convert natural occupation numbers and var_x
    // rdmft::EBI ebi;
    rdmft::PARAM_ONs* param_occ_num = nullptr;

  private:
    //! optimizer: use the BFGS method to get the search direction, p_k
    std::unique_ptr< rdmft::BFGS_method<double> > x_optimizer;


    rdmft::LineSearch<double> ls;

    std::vector<double> scaling_P; // test
    std::vector<double> scaling_P_old; // test

};

}


#endif