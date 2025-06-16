//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef BFGS_OPTI_ONS_H
#define BFGS_OPTI_ONS_H


#include "module_rdmft/rdmft.h"
// #include "module_base/matrix.h"

#include <vector>

namespace rdmft
{

//! @brief physical notation: natural occupation numbers = ONs

//! currently only supports TX=double
//! currently only unconstrained optimization of the occupancy numbers is considered 
//! which means that in rdmft problem the EBI method must be used
template<typename TX>
class BFGS_ONs
{
  public:
    
    BFGS_ONs();
    ~BFGS_ONs();

    void init(int nk_total_in, int nbands_in);

    //! pk is the search direction
    void get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk, const bool new_landscape = false);

    // // ! initialize H0 and obtain p0
    // void get_start_guess(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk);

    // std::vector<TX> x0, x1, diff_x;
    // std::vector<TX> dE_dx0, dE_dx1, diff_grad;



  protected:


    int nk_total = 0;
    int nbands = 0;

    //! the optimized variable x, and diff_x = x_k+1 - x_k
    std::vector<TX> var_x, diff_x;

    //! first-order gradient, and diff_grad = (dE_dx)_k+1 - (dE_dx)_k
    std::vector<TX> dE_dx, diff_grad;

    //! approximate Hessian matrix
    std::vector<TX> Hk;

    //! in the quasi-Newton method, the search direction p_k = - Hk * dE_dx
    std::vector<TX> search_direction;

    //! rho_k = 1.0/(diff_grad^T * diff_x)
    double rho;



  private:

    // process variables, class members or on-the-fly, which one is better?
    std::vector<TX> rho_diffX_diffGrad, rho_diffX_diffX_T;










};





}


#endif