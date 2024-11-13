//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef BFGS_OPTI_ONS_H
#define BFGS_OPTI_ONS_H


#include "module_rdmft/rdmft.h"
// #include "module_base/matrix.h"

namespace rdmft
{


template<typename TK, typename TR>
class BFGS_ONs
{
  public:
    
    BFGS_ONs();
    ~BFGS_ONs();









  protected:








  private:

    ModuleBase::matrix x0, x1, diff_x;
    ModuleBase::matrix dE_dx0, dE_dx1, diff_gradient;
    double rho;










};





}


#endif