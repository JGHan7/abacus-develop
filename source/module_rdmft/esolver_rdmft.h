//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"
// #include "module_esolver/esolver_fp.h"
#include "module_esolver/esolver_ks_lcao.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/line_search_rdmft.h"
#include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"
#include "module_rdmft/optimizer/bfgs_opti_ONs.h"
#include "module_rdmft/optimizer/idmft.h"

#include "module_rdmft/optimizer/ft_rdmft.h"


namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ModuleESolver::ESolver_KS_LCAO<TK,TR>
{
  public:
    ESolver_RDMFT();
    ~ESolver_RDMFT();

    virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    virtual void runner(UnitCell& ucell, const int istep) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    double opti_occ_num(bool dft_type = false, bool first_time = false);

    // int maxniter_occ_num;     // maximum iter steps for ONs

    // int maxniter_orb;         // maximum iter steps for NOs

    double iter_diag_ethr;    // energy threshold, in iterDiag of NOs

    double lambda_thr;        // threshold for checking lambda Hermitianity, in iterDiag of NOs

    double occ_num_thr;       // occupation numbers threshold, in ONs optimization

    bool dft_optimize = false;    // if use dft type to optimize occ_number

    bool conver_initial_value = false;

    std::vector< std::vector<TK> > DM;

    // Parallel_2D* para_H_ni_nj = nullptr;

  private:

    //! objective function E(orbs, occ_nums): provides Etotal_rdmft and first-order gradient
    rdmft::RDMFT<TK, TR> rdmft_solver;

    //! optimizing natural orbitals by iterative diagonalization
    rdmft::IterDiag_NOs<TK, TR> iter_diag_orb;

    //! optimizing natural occupation numbers by line search and quasi-Newton method BFGS combined with EBI method
    rdmft::LineSearch<TK, TR> ls_opti_occ_num;

    //! just for test
    rdmft::BFGS_ONs<double> bfgs_opti_x;
    int dim_x = 2;
    std::vector<double> data_x;
    std::vector<double> df_dx;
    std::vector<double> pk;
    // the only minimum point is (−3.2, 2.8), f0=13.8
    double fx(std::vector<double>& x)
    {
        double f = std::pow(x[0]-1, 2) + std::pow(x[1]+2, 2) + 3 * x[0] *x[1];
        return f;
    }
    std::vector<double> grad_f(std::vector<double>& x)
    {
        std::vector<double> df_dx(x.size(), 0.0);
        df_dx[0] = 2*(x[0]-1) + 3*x[1];
        df_dx[1] = 2*(x[1]+2) + 3*x[0];
        return df_dx;
    }
    double norm_grad(std::vector<double>& df_dx)
    {
        double num = 0.0;
        for(int i=0; i<df_dx.size(); ++i)
        {
          num += std::pow(df_dx[i], 2);
        }
        return std::sqrt(num);
    }

    //! 
    rdmft::IDMFT<TK, TR> idmft;

    //! 
    rdmft::FT_RDMFT<TK, TR> ft_rdmft;

    // rdmft::BFGS_ONs<double> bfgs_rdmft;

    // std::vector< std::vector<TK> > lambda;

    // std::vector< std::vector<TK> > Fock_like_mat;

    void get_start_guess();

    // use dft type to update occ_number
    double update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver);

    void print_info();

    /********* the following is used in rdmft with libTorch *********/

    rdmft::EBI ebi_torch;

    psi::Psi<TK> wfc_new;
    ModuleBase::matrix occ_number_new;

    //! in EBI or other methods, the occupation numbers is parameterized using x
    std::vector<double> var_x;
    std::vector< std::vector<TK> > orb_vec;

    int nk_total = 0;

    // if we need has_cal_E_by_ONs and has_cal_E_by_NOs !!!!!!!!!!!!
    bool has_cal_E = false;
    
    //!
    // 
    double cal_Etotal(const torch::Tensor* var_x_tensor = nullptr,
                        const std::vector<torch::Tensor>* orb_tensor = nullptr,
                        bool cal_by_occ_num = false,
                        bool cal_by_orb = false);

    void cal_E_grad();






};






}

#endif

