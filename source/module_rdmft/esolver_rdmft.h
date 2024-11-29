//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"
#include "module_esolver/esolver_fp.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/line_search_rdmft.h"
#include "module_rdmft/optimizer/ebi_constraint.h"
#include "module_rdmft/optimizer/bfgs_opti_ONs.h"


namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ModuleESolver::ESolver_FP
{
  public:
    ESolver_RDMFT();
    ~ESolver_RDMFT();

    virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    virtual void runner(UnitCell& ucell, const int istep) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    // temporary
    void update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver);
    bool dft_optimize = false;
    // std::vector< std::vector<TK> > Hamilt_rdmft;

    void opti_occ_num(bool first_time = false, bool dft_type = false);

	  int maxniter;     // maximum iter steps for scf

    int maxniter_occ_num;     // maximum iter steps for ONs

    int maxniter_orb;         // maximum iter steps for NOs

    double iter_diag_ethr;    // energy threshold, in iterDiag of NOs

    double lambda_thr;        // threshold for checking lambda Hermitianity, in iterDiag of NOs

    double occ_num_thr;       // occupation numbers threshold, in ONs optimization

    // Parallel_2D* para_H_ni_nj = nullptr;

  private:

    //! objective function E(orbs, occ_nums): provides Etotal_rdmft and first-order gradient
    rdmft::RDMFT<TK, TR> rdmft_solver;

    //! optimizing natural orbitals by iterative diagonalization
    rdmft::IterDiag_NOs<TK, TR> iter_diag_orb;

    //! optimizing natural occupation numbers by line search and quasi-Newton method BFGS combined with EBI method
    rdmft::LineSearch<TK, TR> ls_opti_occ_num;

    // rdmft::EBI ebi;

    // rdmft::BFGS_ONs<double> bfgs_rdmft;

    // std::vector< std::vector<TK> > lambda;

    // std::vector< std::vector<TK> > Fock_like_mat;





};









}

#endif

