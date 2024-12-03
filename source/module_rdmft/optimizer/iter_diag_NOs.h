//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================
#ifndef ITER_DIAG_NOS_H
#define ITER_DIAG_NOS_H

#include "module_rdmft/rdmft.h"

namespace rdmft
{

//! @brief physical notation: natural orbitals = NOs

template<typename TK, typename TR>
class IterDiag_NOs
{
  public:

    IterDiag_NOs();
    ~IterDiag_NOs();

    void init(const int nk_total_in, const Parallel_2D& para_Fij_in, const Parallel_Orbitals& ParaV_in);

    void before_opti(int* scale_factor = nullptr);

    // optimizing natural orbitals
    double optimize_orb(RDMFT<TK, TR>& rdmft_solver);

    // use initial values ​​to form a first guess for iterative diagonalization
    void get_start_guess(RDMFT<TK, TR>& rdmft_solver);

    const Parallel_2D* para_Fij = nullptr;

    const Parallel_Orbitals* ParaV = nullptr;

    double scale_zeta;


  protected:

    int nk_total = 0;

    int nbands_total = 0;

  private:

    std::vector< std::vector<TK> > lambda;

    std::vector< std::vector<TK> > Fock_like_mat;

    // eigenvalues ​​of the Fock matrix, ascending order
    std::vector< std::vector<double> > diag_Fii;

    // wfc under the representation of natural orbitals
    std::vector< std::vector<TK> > nos_rep_wfc;
    // std::vector< std::vector<TK> > nos_rep_wfc0;

    // rotate the Fock-like matrix to the first step natural orbitals representation
    std::vector< std::vector<TK> > rotation_mat;

    // new wfc in NAOs
    psi::Psi<TK> new_wfc;

    // used to scaling the off-diagonal elements of Fock
    // double scale_zeta;
    std::vector<double> scale_zeta_vector;
    int energy_drop = 0;
    int energy_rise = 0;

    double etotal = 0.0, etotal_old = 0.0;

    void get_lambda(const ModuleBase::matrix& wg,
                      const ModuleBase::matrix& wk_fun_occNum,
                      const std::vector< std::vector<TK> >& H_no_exx, 
                      const std::vector< std::vector<TK> >& H_exx);

    void get_Fock();

    // scale the off-diagonal elements of Fock
    // ? the physical reasons still need to be considered
    void scale_Fock();

    // adjust scale_zeta used in scale_Fock()
    // any other better solutions?
    void adjust_scale();

    // rotate the Fock to the natural orbital representation of step 1
    void rotate_Fock();

    // temporary
    void check_hermi(std::vector< std::vector<TK> >& mat);

    // // fill the diagonal elements of F
    // void fill_diag_elem(const Parallel_2D* para_mat, 
    //                       const std::vector< std::vector<TK> >& mat_filling, 
    //                       std::vector< std::vector<TK> >& mat_filled);

    // symmetrize lambda, delete in the future?
    void symmetr_lambda(const Parallel_2D* para_mat, 
                          const std::vector< std::vector<TK> >& lambda, 
                          std::vector< std::vector<TK> >& symm_lambda);

    // // temp, because there is no pzheev interface, only pzhegvx can be used
    // std::vector<TK> identi_mat;
    // std::vector<TK> get_identi_mat(const Parallel_2D* para_mat);


};
















}

#endif

