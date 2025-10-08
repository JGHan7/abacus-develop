//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================
#ifndef CG_METHOD_H
#define CG_METHOD_H


#include <string>
#include "module_rdmft/optimizer/opti_method.h"


namespace rdmft
{


template<typename TX>
class CG_method: public rdmft::Opti_method<TX>
{
  public:
    
    CG_method();
    ~CG_method();

    void init(const int dim_in, const double precond_eps_in = 1e-8) override;

    //! pk is the search direction
    void get_pk(const std::vector<TX>& dE_dx_new, const std::vector<TX>& x_new, std::vector<TX>& pk,
                const bool new_landscape = false, const std::vector<TX>* d2E_dx2 = nullptr) override;



  protected:

    void cal_beta(const std::vector<TX>& dE_dx_new, const bool new_landscape = false, const bool precond = false);

    double beta = 0.0;

    std::string beta_type = "PR";

  private:

    //! gradient after precond: z_k, z_k-1
    std::vector<TX> precond_grad;
    std::vector<TX> precond_grad_old;

    //! diff_z = z_k - z_k-1
    std::vector<TX> diff_precond_grad;




};





}


#endif