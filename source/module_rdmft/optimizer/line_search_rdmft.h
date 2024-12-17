//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================
#ifndef LINE_SEARCH_RDMFT_H
#define LINE_SEARCH_RDMFT_H

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/ebi_constraint.h"
#include "module_rdmft/optimizer/bfgs_opti_ONs.h"


namespace rdmft
{

template<typename TK, typename TR>
class LineSearch
{

  public:

    LineSearch();
    ~LineSearch();

    void init(RDMFT<TK, TR>* rdmft_in);

    //! use an approximate line search method to find a suitable step size
    double do_line_search(const bool start_guess = false);

    // temp?
    std::vector<double>* get_var_x() { return &this->var_x; }

    //! parameters in strong wolfe or wolfe conditions
    double ls_wolfe_c1 = 0.0;
    double ls_wolfe_c2 = 0.0;

    //! parameters in Armijo-Goldstein conditions
    double ls_armijo_c1 = 0.0;
    double ls_armijo_c2 = 0.0;

    std::string ls_condition;

    double max_step_size = 0.0;


  protected:

    //! Strong Wolfe condition
    void strong_wolfe();

    //! Wolfe condition
    void wolfe();

    //! used in Strong Wolfe condition
    void zoom(double step_size_low, double phi_low, double step_size_high, double phi_high, double dphi_low);

    //! calculate phi(alpha) = E(x_k + alpha * p_k) and return it. x_new = x_k + alpha*p_k
    virtual double cal_phi(const std::vector<double>& x_new);

    //! calculate dE/dx, dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T and return dphi.
    //! x_new = x_k + alpha*p_k, if x is the same as the x in the last call to cal_phi(), nullptr is used.
    virtual double cal_dphi(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr = nullptr);  // is dE_dx_new useful here? Consider deleting the outgoing

    //! calculate the direction of the line search: pk, and dphi_0
    virtual void cal_pk_dphi0(const bool start_guess = false);

    //! generate start guess
    virtual void get_start_guess();

    //! calculate dE/dx
    //! x_new = x_k + alpha*p_k, if x is the same as the x in the last call to cal_phi(), nullptr is used.
    virtual void cal_dE_dx(std::vector<double>& dE_dx_new, const std::vector<double>* x_new_ptr = nullptr);

    //! x_k, (dE_dx)_k, search direction p_k and occupation numbers
    std::vector<double> var_x;
    std::vector<double> dE_dx;
    std::vector<double> search_direction;
    ModuleBase::matrix occ_number;

    //! step_size, alpha: x_k+1 = x_k + alpha * p_k
    double step_size = 1.0;

    //! phi(alpha) = E(x_k + alpha * p_k), phi_0 = E(x_k)
    double phi_0 = 0.0;

    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T, dphi_0 = E'(x_k) * p_k^T
    double dphi_0 = 0.0;

  private:

    //! objective function E(occ_num)=E(x): provides Etotal_rdmft and first-order gradient
    RDMFT<TK, TR>* rdmft_solver = nullptr;

  public: // temp public
    //! handle constraints: convert natural occupation numbers and var_x
    rdmft::EBI ebi;

  private:
    //! optimizer: use the BFGS method to get the search direction, p_k
    rdmft::BFGS_ONs<double> bfgs_opti_x;





};

}


#endif