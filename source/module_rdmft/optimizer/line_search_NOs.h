//==========================================================
// Author: Jingang Han
// DATE : 2025-09-24
//==========================================================
#ifndef LINE_SEARCH_NOS_H
#define LINE_SEARCH_NOS_H

// temp
#include <torch/torch.h>

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/bfgs_opti.h"
#include  "module_rdmft/optimizer/line_search_method.h"


namespace rdmft
{


template<typename TK, typename TR>
class LineSearch_NOs
{

  public:

    LineSearch_NOs();
    ~LineSearch_NOs();

    void init(RDMFT<TK, TR>* rdmft_in);

    //! use an approximate line search method to find a suitable step size
    double do_line_search(const bool start_guess = false);

    void before_opti();



  protected:

    //! generate start guess
    virtual void get_start_guess();

  private:




};

}


#endif