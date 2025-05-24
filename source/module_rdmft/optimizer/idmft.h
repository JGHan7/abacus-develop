//==========================================================
// Author: Jingang Han
// DATE : 2025-03-11
//==========================================================
#ifndef IDMFT_H
#define IDMFT_H

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/mixing_dmk.h"
#include <map>
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h" // temp

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

    //! serves for mixing, can be deleted after refactoring
    void before_opti(const std::vector< std::vector<TK> >& DM_in, hamilt::Hamilt<TK>* p_hamilt_in);

    //! optimizing natural orbitals and occupation numbers
    double optimize();

    //! the occupation number is obtained from the eigenvalues ​​of the Fock-like matrix and the Fermi-Dirac distribution
    void opti_occ_num(ModuleBase::matrix& occ_num_pass);

    // void solve_zero_occ_num();

    const Parallel_2D* para_Fij = nullptr;

    const Parallel_Orbitals* ParaV = nullptr;

    double max_off_diag_F = 0.0;

    const std::vector< std::vector<double> >& get_energy_level() { return this->diag_Fii; }

    double get_diff_DM_max() { return this->diff_DM_max; }

    double get_mu(const int is) { return this->mu[is]; }

    // tempeture kappa
    double kappa = 0.0;

  // protected:

    virtual void get_Fock();

    const K_Vectors* kv;

    int nk_total = 0;

    int nk_nospin = 0;

    int nbands = 0;

    // std::vector<double> wk_nospin;

    //! the number of symmetric k-points
    std::vector<double> num_symm_k;

    std::vector<double> sys_nelec_spin;

    RDMFT<TK, TR>* rdmft_solver = nullptr;

    double diff_Etotal = 0.0;

    double diff_DM_max = 0.0;

    //! DMk in NAOs representation
    // TODO: DM and related convergence judgment conditions can be moved to esolver_rdmft
    std::vector< std::vector<TK> > DM;

    //! DMk in NOs representation
    std::vector< std::vector<TK> > DM_nos_rep;

    // TODO: it may be a better choice to put it in esolver_rdmft, so that the mixed code can be used by multiple optimization methods
    // then the optimizer of various methods will only give new occ_num and wfc, and will not implement the function of update_elec() in the RDMFT object.
    Mixing_DMk<TK> mixing_dmk;

    std::vector<TK> dmk_in;

    std::vector<TK> dmk_out;

    hamilt::HamiltLCAO<TK, TR>* p_hamilt_lcao = nullptr; // temp, for mixing

    int iter_step = 0;

    bool start_mixing = false;

    void do_mixing(std::vector< std::vector<TK> >& DM_new, ModuleBase::matrix& occ_num_pass);

  // private:

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

    void solving_mu();

    double cal_occ_num(const int is);

    std::vector< std::vector<double> > occ_number;

    std::vector<double> mu;

    // rotate the Fock to the natural orbital representation of step 1
    void rotate_Fock();

};
















}

#endif

