//==========================================================
// Author: Jingang Han
// DATE : 2024-11-01
//==========================================================
#ifndef ITER_DIAG_NOS_H
#define ITER_DIAG_NOS_H

#include "module_rdmft/rdmft.h"
#include "module_rdmft/optimizer/mixing_dmk.h"
#include <map>
#include "module_hamilt_lcao/hamilt_lcaodft/hamilt_lcao.h" // temp

namespace rdmft
{

//! @brief physical notation: natural orbitals = NOs

template<typename TK, typename TR>
class IterDiag_NOs
{
  public:

    IterDiag_NOs();
    ~IterDiag_NOs();

    void init(const int nk_total_in, const int nkstot_full_in, const Parallel_2D& para_Fij_in, const Parallel_Orbitals& ParaV_in, RDMFT<TK, TR>* rdmft_solver_in);

    //! use initial values ​​to form a first guess for iterative diagonalization
    void get_start_guess(RDMFT<TK, TR>& rdmft_solver_in, const bool conver_initial_value = false); // delete conver_initial_value in the future?

    void before_opti(hamilt::Hamilt<TK>* p_hamilt_in = nullptr, int* scale_factor = nullptr);

    //! optimizing natural orbitals
    double optimize_orb(RDMFT<TK, TR>& rdmft_solver_in);

    //! check the Hermitian property of lambda for all k points
    //! the mixing of Fock matrices broke this functionality
    //! TODO: modify it to a separate function implementation, not dependent on get_Fock()
    double check_hermi_lambda() { return this->max_off_diag_F; }

    double get_diff_DM_max() { return this->diff_DM_max; }

    const Parallel_2D* para_Fij = nullptr;

    const Parallel_Orbitals* ParaV = nullptr;

    double scale_zeta = 0.0;

    double max_off_diag_F = 0.0;

    bool init_orb_by_lambda = true;

    bool scale_F = false;


  protected:

    const K_Vectors* kv;

    int nk_total = 0;

    int nk_nospin = 0;

    int nbands_total = 0;

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

    // test: mixing the diag_Fii
    Mixing_DMk<double> mixing_Fii;

    std::vector<double> diag_Fii_in;

    std::vector<double> diag_Fii_out;

    hamilt::HamiltLCAO<TK, TR>* p_hamilt_lcao = nullptr; // temp, for mixing

    int iter_step = 0;

    bool start_mixing = false;

    void do_mixing(std::vector< std::vector<TK> >& DM_new);

    // void mixing();

    // bool mixing_rdmft = true;

    // // const int mixing_step = 3;

    // // //! the length of mixing_coef = mixing_step
    // // //! the index is in ascending order from step 0 to step k
    // // std::vector<double> mixing_coef = {0.15, 0.25, 0.6};

    // const int mixing_step = 3;

    // //! the length of mixing_coef = mixing_step
    // //! the index is in ascending order from step 0 to step k
    // std::vector<double> mixing_coef = {0.6, 0.25, 0.15};

    // bool start_mixing = false;

    // int iter_step = 0;

    // std::map<int, std::vector< std::vector<TK> > > Fock_record;



  private:

    std::vector< std::vector<TK> > lambda;

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

    // used to scaling the off-diagonal elements of Fock
    // double scale_zeta;
    std::vector<double> scale_zeta_vector;
    int energy_drop = 0;
    int energy_rise = 0;

    double etotal = 0.0, etotal_old = 0.0;

    // void get_lambda(const ModuleBase::matrix& wg,
    //                   const ModuleBase::matrix& wk_fun_occNum,
    //                   const std::vector< std::vector<TK> >& H_no_exx, 
    //                   const std::vector< std::vector<TK> >& H_exx);
    // temp
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
    bool if_rotate_Fock = true;


    // // temporary
    // void check_hermi(std::vector< std::vector<TK> >& mat);

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

