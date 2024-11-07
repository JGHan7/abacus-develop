//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================
#ifndef ITER_DIAG_NOS_H
#define ITER_DIAG_NOS_H

#include "module_rdmft/rdmft.h"

namespace rdmft
{


template<typename TK, typename TR>
class IterDiag_NOs
{
  public:

    IterDiag_NOs();
    ~IterDiag_NOs();

    void init(int nk_total_in, const Parallel_2D* para_Fij_in, const Parallel_Orbitals* ParaV_in);

    // optimizing natural orbitals
    void optimize_orb(RDMFT<TK, TR>& rdmft_solver);

    int nk_total = 0;

    Parallel_2D* para_Fij = nullptr;

    Parallel_Orbitals* ParaV = nullptr;

  private:

    std::vector< std::vector<TK> > lambda;

    std::vector< std::vector<TK> > Fock_like_mat;

    void get_lambda(ModuleBase::matrix& wg,
                      ModuleBase::matrix& wk_fun_occNum,
                      std::vector< std::vector<TK> >& H_no_exx, 
                      std::vector< std::vector<TK> >& H_exx);


};
















}

#endif

