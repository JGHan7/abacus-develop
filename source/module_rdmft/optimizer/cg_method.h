//==========================================================
// Author: Jingang Han
// DATE : 2025-10-08
//==========================================================
#ifndef CG_METHOD_H
#define CG_METHOD_H


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




  private:




};





}


#endif