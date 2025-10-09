//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef BFGS_METHOD_H
#define BFGS_METHOD_H


#include "module_rdmft/optimizer/opti_method.h"


namespace rdmft
{


template<typename TX>
class BFGS_method: public rdmft::Opti_method<TX>
{
  public:
    
    BFGS_method();
    ~BFGS_method();

    void init(const int dim_in, const double precond_eps_in = 1e-8) override;

    //! pk is the search direction
    void get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk,
                const bool new_landscape = false, const std::vector<TX>* d2E_dx2 = nullptr) override;

    // // ! initialize H0 and obtain p0
    // void get_start_guess(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk);

    //! 
    void get_diag_Bk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& diag_Bk, const bool new_landscape = false);

    // std::vector<TX> x0, x1, diff_x;
    // std::vector<TX> dE_dx0, dE_dx1, diff_grad;
    void transport(const std::vector<TX>& diag_T) override;

    // int iter = 0;
    // int num_restart_skip = 0;

    bool scaling_H0 = false; // delete?
    TX scaling_gamma0 = 1.0;


  protected:

    // // Parallel_2D* para_mat = nullptr;
    // int dim = 0;

    // double precond_eps = 1e-8;

    // //! first-order gradient
    // std::vector<TX> dE_dx;

    // //! in the quasi-Newton method, the search direction p_k = - Hk * dE_dx
    // std::vector<TX> search_direction;
    
    //! the optimized variable x, and diff_x = x_k+1 - x_k
    std::vector<TX> var_x, diff_x;

    //! diff_grad = (dE_dx)_k+1 - (dE_dx)_k
    std::vector<TX> diff_grad;

    //! approximate inverse Hessian matrix
    std::vector<TX> Hk;

    //! approximate Hessian matrix
    std::vector<TX> Bk;

    std::vector<TX> y_yT;
    std::vector<TX> B_ssT_BT;

    //! rho_k = 1.0/(diff_grad^T * diff_x)
    double rho;


  private:

    // process variables, class members or on-the-fly, which one is better?
    std::vector<TX> rho_diffX_diffGrad, rho_diffX_diffX_T;


    void cal_Hk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, const bool new_landscape = false, const std::vector<TX>* d2E_dx2 = nullptr);


    // test
    std::vector<TX> transport_mat;


};





}


#endif