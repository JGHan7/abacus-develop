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

//! currently only supports TX=double
//! currently only unconstrained optimization of the occupancy numbers is considered 
//! which means that the EBI method must be used
template<typename TX>
class BFGS_ONs
{
  public:
    
    BFGS_ONs();
    ~BFGS_ONs();

    void init();









  protected:








  private:

    std::vector<TX> x0, x1, diff_x;
    std::vector<TX> dE_dx0, dE_dx1, diff_gradient;
    double rho;










};





}


#endif