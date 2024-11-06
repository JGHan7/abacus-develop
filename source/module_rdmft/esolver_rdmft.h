//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"
#include "module_esolver/esolver_fp.h"

namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ModuleESolver::ESolver_FP
{
  public:
    ESolver_RDMFT();
    ~ESolver_RDMFT();

    virtual void before_all_runners(const Input_para& inp, UnitCell& ucell) override;

	virtual void runner(const int istep, UnitCell& ucell) override;

    virtual double cal_energy() override;

    virtual void cal_force(ModuleBase::matrix& force) override;

    virtual void cal_stress(ModuleBase::matrix& stress) override;

	int maxniter;     // maximum iter steps for scf

  private:

    rdmft::RDMFT<TK, TR> rdmft_solver;





};









}

#endif

