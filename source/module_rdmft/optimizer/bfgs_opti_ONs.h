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
//! which means that the EBI method must be used
template<typename TX>
class BFGS_ONs
{
  public:
    
    BFGS_ONs();
    ~BFGS_ONs();

    void init(int nk_total_in, int nbands_in);

    double optimize(std::vector<TX>& x_new);

    void get_start_guess(std::vector<TX>& x0_in);

    std::vector<TX> x0, x1, diff_x;
    std::vector<TX> dE_dx0, dE_dx1, diff_gradient;
    std::vector<TX> Hk;
    double rho;







  protected:


  int nk_total = 0;
  int nbands = 0;





  private:

    // std::vector<TX> x0, x1, diff_x;
    // std::vector<TX> dE_dx0, dE_dx1, diff_gradient;
    // double rho;










};





}


#endif