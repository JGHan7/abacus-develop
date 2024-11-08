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

    void init(const int nk_total_in, const Parallel_2D& para_Fij_in, const Parallel_Orbitals& ParaV_in);

    // optimizing natural orbitals
    void optimize_orb(RDMFT<TK, TR>& rdmft_solver);

    // use initial values ​​to form a first guess for iterative diagonalization
    void get_start_guess(RDMFT<TK, TR>& rdmft_solver);

    int nk_total = 0;

    const Parallel_2D* para_Fij = nullptr;

    const Parallel_Orbitals* ParaV = nullptr;

  private:

    std::vector< std::vector<TK> > lambda;

    std::vector< std::vector<TK> > Fock_like_mat;

    void get_lambda(const ModuleBase::matrix& wg,
                      const ModuleBase::matrix& wk_fun_occNum,
                      const std::vector< std::vector<TK> >& H_no_exx, 
                      const std::vector< std::vector<TK> >& H_exx);

    // symmetrize lambda
    void symmetr_lambda(const Parallel_2D* para_mat, 
                          const std::vector< std::vector<TK> >& lambda, 
                          std::vector< std::vector<TK> >& symm_lambda);

    // fill the diagonal elements of F
    void fill_diag_elem(const Parallel_2D* para_mat, 
                          const std::vector< std::vector<TK> >& mat_filling, 
                          std::vector< std::vector<TK> >& mat_filled);

    // temp, because there is no pzheev interface, only pzhegvx can be used
    std::vector<TK> identi_mat;
    std::vector<TK> get_identi_mat(const Parallel_2D* para_mat);


};
















}

#endif

