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





  protected:








  private:


    RDMFT<TK, TR>* rdmft_solver = nullptr;

    rdmft::EBI ebi;

    rdmft::BFGS_ONs<double> bfgs_rdmft;





};

}


#endif