//==========================================================
// Author: Jingang Han
// DATE : 2025-05-23
//==========================================================
#ifndef FT_RDMFT_H
#define FT_RDMFT_H


#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/idmft.h"


namespace rdmft
{

//! @brief physical notation: natural orbitals = NOs

template<typename TK, typename TR>
class FT_RDMFT: public rdmft::IDMFT<TK,TR>
{

  public:
    void init(const int nk_total_in,
              const K_Vectors& kv_in,
              const Parallel_2D& para_Fij_in,
              const Parallel_Orbitals& ParaV_in,
              RDMFT<TK, TR>* rdmft_solver_in,
              const ModuleBase::matrix& ekb_in);

    virtual void get_Fock() override;

    
    double dE_dk = 0.0;

    void optimize_kappa();





};




}


#endif
