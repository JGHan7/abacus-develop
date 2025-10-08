//==========================================================
// Author: Jingang Han
// DATE : 2025-05-23
//==========================================================
#ifndef FT_RDMFT_H
#define FT_RDMFT_H


#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/idmft.h"
#include "module_rdmft/optimizer/bfgs_opti.h"


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

    
    double dE_dk_sum = 0.0;

    std::vector<double> dE_dk;

    std::vector<double> pk;

    std::vector<double> kappa_tensor;

    double average_k = 0.0;

    double max_diff_kappa = 0.0;

    rdmft::BFGS_method<double> bfgs_opti_k;



    void optimize_kappa();

    // temp parameter
    bool first_opti_k = true;


    virtual double cal_occ_num(const int is) override;


};




}


#endif
