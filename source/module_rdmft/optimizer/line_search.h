//==========================================================
// Author: Jingang Han
// DATE : 2024-11-28
//==========================================================
#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/ebi_constraint.h"
#include "module_rdmft/optimizer/bfgs_opti_ONs.h"


namespace rdmft
{

//! currently only supports TX=double
template<typename TX, typename T_rdmft>
class LineSearch
{

  public:

    LineSearch();
    ~LineSearch();

    init(T_rdmft* rdmft_solver_in);








  protected:








  private:


    T_rdmft* rdmft_solver = nullptr;

    rdmft::EBI ebi;

    rdmft::BFGS_ONs<double> bfgs_rdmft;





};

}


#endif