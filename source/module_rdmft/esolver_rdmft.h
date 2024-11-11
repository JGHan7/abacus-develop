//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"
#include "module_esolver/esolver_fp.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"

namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ModuleESolver::ESolver_FP
{
  public:
    ESolver_RDMFT();
    ~ESolver_RDMFT();

    virtual void before_all_runners(const Input_para& inp, UnitCell& ucell) override;

    virtual void runner(const int istep, UnitCell& ucell) override;

    double cal_energy() override;

    void cal_force(ModuleBase::matrix& force) override;

    void cal_stress(ModuleBase::matrix& stress) override;

    // temporary
    void update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver);
    bool dft_optimize = false;
    // std::vector< std::vector<TK> > Hamilt_rdmft;

	  int maxniter;     // maximum iter steps for scf

    int maxniter_occ_num;     // maximum iter steps for ONs

    int maxniter_orb;         // maximum iter steps for NOs

    double iter_diag_ethr;    // energy threshold, in iterDiag of NOs

    double lambda_thr;        // threshold for checking lambda Hermitianity, in iterDiag of NOs

    double occ_num_thr;       // occupation numbers threshold, in ONs optimization

    // Parallel_2D* para_H_ni_nj = nullptr;

  private:

    rdmft::IterDiag_NOs<TK, TR> iter_diag_rdmft;

    rdmft::RDMFT<TK, TR> rdmft_solver;

    // std::vector< std::vector<TK> > lambda;

    // std::vector< std::vector<TK> > Fock_like_mat;





};









}

#endif

