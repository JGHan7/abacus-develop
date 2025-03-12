//==========================================================
// Author: Jingang Han
// DATE : 2025-03-11
//==========================================================
#ifndef IDMFT_H
#define IDMFT_H

#include "module_rdmft/rdmft.h"
#include <map>

namespace rdmft
{

//! @brief physical notation: natural orbitals = NOs

template<typename TK, typename TR>
class IDMFT
{
  public:
    IDMFT();
    ~IDMFT();

    void init(const int nk_total_in,
              const K_Vectors& kv_in,
              const Parallel_2D& para_Fij_in,
              const Parallel_Orbitals& ParaV_in,
              RDMFT<TK, TR>* rdmft_solver_in);

    // void before_opti(int* scale_factor = nullptr);

    //! optimizing natural orbitals
    double optimize_orb();

    //! the occupation number is obtained from the eigenvalues ​​of the Fock-like matrix and the Fermi-Dirac distribution
    void opti_occ_num();

    // void solve_zero_occ_num();

    const Parallel_2D* para_Fij = nullptr;

    const Parallel_Orbitals* ParaV = nullptr;

    double max_off_diag_F = 0.0;


  protected:

    const K_Vectors* kv;

    int nk_total = 0;

    int nk_nospin = 0;

    int nbands = 0;

    std::vector<double> wk_nospin;

    //! the number of symmetric k-points
    std::vector<double> num_symm_k;

    std::vector<double> sys_nelec_spin;

    RDMFT<TK, TR>* rdmft_solver = nullptr;

    double diff_Etotal = 0.0;

  private:

    std::vector< std::vector<TK> > Fock_like_mat;

    // eigenvalues ​​of the Fock matrix, ascending order
    std::vector< std::vector<double> > diag_Fii;

    // wfc under the representation of natural orbitals
    std::vector< std::vector<TK> > nos_rep_wfc;

    // NAOs_rep_wfc1 = nos_rep_wfc1 * NAOs_rep_wfc0
    psi::Psi<TK> naos_rep_wfc1;
    bool if_get_wfc1 = false;

    // new wfc in NAOs
    psi::Psi<TK> new_wfc;
  
    // rotate the Fock-like matrix to the first step natural orbitals representation
    std::vector< std::vector<TK> > rotation_mat;


    double etotal = 0.0, etotal_old = 0.0;

    void get_Fock();

    void solving_mu();

    double cal_occ_num(const int is);

    std::vector< std::vector<double> > occ_number;

    std::vector<double> mu;

    // rotate the Fock to the natural orbital representation of step 1
    void rotate_Fock();
    bool if_rotate_Fock = true;

};
















}

#endif

