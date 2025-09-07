//==========================================================
// Author: Jingang Han
// DATE : 2025-09-07
//==========================================================
#ifndef ESOLVER_RDMFT_TORCH_AD_H
#define ESOLVER_RDMFT_TORCH_AD_H


// #include <torch/torch.h>
// #include "module_rdmft/esolver_rdmft.h"
// #include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"

#include "module_rdmft/esolver_rdmft_torch.h"


namespace rdmft
{


template<typename TK, typename TR>
class ESolver_RDMFT_Torch_AD: public rdmft::ESolver_RDMFT_Torch<TK,TR>
{
  public:

    ESolver_RDMFT_Torch_AD();
    ~ESolver_RDMFT_Torch_AD();

    // virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    // virtual void runner(UnitCell& ucell, const int istep) override;


  protected:







  private:




    


};


}

#endif



