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
    void do_line_search(const bool start_guess = false);



  protected:

    //! Strong Wolfe condition
    void strong_wolfe();

    //! Wolfe condition
    void wolfe();

    //! used in Strong Wolfe condition
    void zoom();






  private:

    //! objective function E(x): provides Etotal_rdmft and first-order gradient
    RDMFT<TK, TR>* rdmft_solver = nullptr;

    //! handle constraints: convert natural occupation numbers and var_x
    rdmft::EBI ebi;

    //! optimizer: use the BFGS method to get the search direction, p_k
    rdmft::BFGS_ONs<double> bfgs_rdmft;

    //! x_k+1 = x_k + alpha * p_k, E(x), phi(alpha) = E(x_k + alpha * p_k)
    //! dphi/dalpha = E'(x_k + alpha * p_k) * p_k^T
    //! dphi_0 = (dphi/dalpha at alpha=0) = E'(x_k) * p_k^T
    double dphi_0;

    std::vector<double> 





};

}


#endif