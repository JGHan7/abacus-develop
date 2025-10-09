//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================
#ifndef OPTI_METHOD_H
#define OPTI_METHOD_H


#include <complex>
#include <vector>


namespace rdmft
{


template<typename TX>
class Opti_method
{
  public:
    
    Opti_method();
    virtual ~Opti_method();

    virtual void init(const int dim_in, const double precond_eps_in = 1e-8);

    //! pk is the search direction
    virtual void get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk,
                const bool new_landscape = false, const std::vector<TX>* d2E_dx2 = nullptr) = 0;

    int iter = 0;
    int num_restart_skip = 0;

    //! 
    // virtual void get_diag_Bk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& diag_Bk, const bool new_landscape = false) {};

    // test
    virtual void transport(const std::vector<TX>& diag_T) {};

  protected:

    // Parallel_2D* para_mat = nullptr;
    int dim = 0;

    double precond_eps = 1e-8;

    //! first-order gradient
    std::vector<TX> dE_dx;

    //! the search direction
    std::vector<TX> search_direction;


};





}


#endif