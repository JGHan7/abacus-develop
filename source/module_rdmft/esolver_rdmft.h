//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"

namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ESolver_FP
{
  public:
    ESolver_RDMFT();
    ~ESolver_RDMFT();

    virtual void before_all_runners(const Input_para& inp, UnitCell& ucell) override;

	virtual void runner(const int istep, UnitCell& ucell) override;

	int maxniter;     // maximum iter steps for scf

  private:

    RDMFT rdmft_solver;





};









}

#endif

