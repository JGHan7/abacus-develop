//==========================================================
// Author: Jingang Han
// DATE : 2025-09-22
//==========================================================
#ifndef LINE_SEARCH_METHOD_H
#define LINE_SEARCH_METHOD_H


#include <complex>
#include <string>
#include <vector>
#include <functional>

// #include "module_rdmft/rdmft.h"
// #include "module_rdmft/optimizer/parameterize_ONs/param_ONs.h"
// #include "module_rdmft/optimizer/bfgs_opti.h"

namespace rdmft
{

    struct Options
    {
        double& ls_wolfe_c1;
        double& ls_wolfe_c2;
        double& ls_armijo_c1;
        double& ls_armijo_c2;
        double& max_step_size;
        double& min_step_size;
        std::string& ls_condition;

        Options(double& w1, double& w2,
                double& a1, double& a2,
                double& maxs, double& mins,
                std::string& cond)
            : ls_wolfe_c1(w1), ls_wolfe_c2(w2),
              ls_armijo_c1(a1), ls_armijo_c2(a2),
              max_step_size(maxs), min_step_size(mins),
              ls_condition(cond) {}
    };


template<typename TX>
class LineSearch
{

  public:

    LineSearch(int max_ls_in = 25);
    ~LineSearch();

    //! use an approximate line search method to find a suitable step size
    //! the only required parameter for cal_phi and cal_dphi is, const double trial_step_size 
    //! line search can be implemented without explicitly knowing the search direction and other details,
    //! everything related can be packaged in cal_phi/dphi (so there is no need to consider whether the matrix calculation is multi-process)
    double do_line_search(const std::function<double(const double)>& cal_phi,
                            const std::function<double()>& cal_dphi,
                            const double phi_0_in,
                            const double dphi_0_in,
                            const double max_elem_pk_in = 1.0,
                            const double inital_step = 1.0);


    // example
    // std::vector<double> trial_x(n, 0.0);
    // auto f = [this, &trial_x]() { cal_Etotal(trial_x, this->x0 ); };

    int max_ls = 25;


    // 
    double tolerance_change = 1e-9;

    Options get_options()
    {
        return Options(this->ls_wolfe_c1, this->ls_wolfe_c2,
                       this->ls_armijo_c1, this->ls_armijo_c2,
                       this->max_step_size, this->min_step_size,
                       this->ls_condition);
    }


  protected:

    //! Strong Wolfe condition, get the appropriate step length
    //! The framework is based on J. Nocedal's "Numerical Optimization."
    //! For handling extreme values, refer to pyTorch: https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/optim/lbfgs.cpp
    void strong_wolfe(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi);

    //! Wolfe condition
    void wolfe(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi);

    //! used in Strong Wolfe condition, get the appropriate step length
    void zoom(const std::function<double(const double)>& cal_phi,
                const std::function<double()>& cal_dphi,
                double step_size_low,
                double phi_low,
                double dphi_low,
                double step_size_high,
                double phi_high,
                double dphi_high);

    //! Strong Wolfe condition, get the appropriate step length
    void strong_wolfe2(const std::function<double(const double)>& cal_phi, const std::function<double()>& cal_dphi);
    
    //! used in Strong Wolfe condition, get the appropriate step length
    void zoom2(const std::function<double(const double)>& cal_phi,
                const std::function<double()>& cal_dphi,
                double step_size_low,
                double phi_low,
                double step_size_high,
                double phi_high,
                double dphi_low);

    //! parameters in strong wolfe or wolfe conditions
    double ls_wolfe_c1 = 0.0;
    double ls_wolfe_c2 = 0.0;

    //! parameters in Armijo-Goldstein conditions
    double ls_armijo_c1 = 0.0;
    double ls_armijo_c2 = 0.0;

    //! conditions for selecting the step size used in inexact line search
    std::string ls_condition = "swolfe";

    //! the maximum and minimum step sizes allowed for line search
    double max_step_size = 0.0;
    double min_step_size = 0.0;

    // the element with the largest modulus of the search direction vector pk
    double max_elem_pk = 1.0;

    //! step_size, alpha: x_k+1 = x_k + alpha * p_k
    double step_size = 1.0;

    //! the largest possible but appropriate step that satisfies the Armijo condition
    double armijo_step = 0.0;

    // double init_step = 1.0;

    //! phi(alpha) = E(x_k + alpha * p_k), phi_0 = E(x_k)
    double phi_0 = 0.0;

    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T, dphi_0 = E'(x_k) * p_k^T
    double dphi_0 = 0.0;

  private:

    //！ the total number of times the cal_phi() function is called during the entire optimization period
    int num_cal_phi = 0;

    //! the number of times cal_phi() is called during a line search.
    int ls_times = 0;


};

}


#endif