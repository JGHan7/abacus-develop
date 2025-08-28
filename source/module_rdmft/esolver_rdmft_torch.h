//==========================================================
// Author: Jingang Han
// DATE : 2025-07-16
//==========================================================
#ifndef ESOLVER_RDMFT_TORCH_H
#define ESOLVER_RDMFT_TORCH_H

// #include "module_rdmft/rdmft.h"
// // #include "module_esolver/esolver_fp.h"
// #include "module_esolver/esolver_ks_lcao.h"
// #include "module_rdmft/optimizer/iter_diag_NOs.h"
// #include "module_rdmft/optimizer/line_search_rdmft.h"
// #include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"
// #include "module_rdmft/optimizer/bfgs_opti_ONs.h"
// #include "module_rdmft/optimizer/idmft.h"

// #include "module_rdmft/optimizer/ft_rdmft.h"

#include <torch/torch.h>

#include "module_rdmft/esolver_rdmft.h"
#include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"


namespace rdmft
{


template<typename TK, typename TR>
class ESolver_RDMFT_Torch: public rdmft::ESolver_RDMFT<TK,TR>
{
  public:

    ESolver_RDMFT_Torch();
    ~ESolver_RDMFT_Torch();

    virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    virtual void runner(UnitCell& ucell, const int istep) override;

    // Be clear about whether we're optimizing for C_t+1=exp(R_t)C_t or C_new=exp(R_t)C_0.
    // The former means that each time R_t is a small deltaR, we can gradually rotate from C_0 to R_final;
    // the latter means rotating C_0 to C_final all at once through the final R_final.
    // The derivation of the explicit first-order gradient we can provide relies on R_t being close to zero.
    // This means that only for the former is the gradient we provide approximately correct, so we recommend the former.
    bool opti_deltaR = true;


  protected:
    
    virtual void get_start_guess(UnitCell& ucell, const int istep) override;

    void select_optimizer();

    void do_optimize();

  private:

    int nk_total = 0;

    Parallel_2D* para_Fij;

    bool cal_molecular = false;

    void couple_opti();

    void decouple_opti();

    void before_opti();

    //! determine whether the optimization of NOs and ONs has converged
    //! if dm_conv() or occ_num_conv() is called in an iteration, the obtained value must be passed in
    bool converge(const bool* occ_num_conv_in = nullptr, const bool* dm_conv_in = nullptr);

    //! determine whether the density matrix converges
    //! will update DM, can only be called once in one iteration
    bool dm_conv();
    //
    double diff_DM_max = 0.0;
    //! DMk in NAOs representation
    std::vector< std::vector<TK> > DM;

    // determine whether NOs converges
    bool orb_conv();
    // only used to judge convergence
    psi::Psi<TK> wfc_record;
    //
    double diff_wfc_norm = 0.0;
    //
    double diff_wfc_max = 0.0;

    //! determine whether the ONs converges
    //! will update ONs, can only be called once in one iteration
    bool occ_num_conv();
    //
    double diff_occ_num_max =0.0;

    //!
    //! the result of calculating dE_dR is used. If you re-optimize ONs, please call the function containing dE_dR first
    double check_hermi_lambda();
    //!
    double max_off_diag_Fock = 0.0;






    /********* the following is used to optimize ONs  *********/
    //
    rdmft::EBI ebi;
    //
    ModuleBase::matrix occ_number_new;
    //
    ModuleBase::matrix occ_number_old;
    //! in EBI or other methods, the occupation numbers is parameterized using x
    std::vector<double> var_x;
    // 
    std::vector<double> dE_dx;
    //
    torch::Tensor var_x_tensor;
    // 
    torch::Tensor dE_dx_tensor;
    //
    bool has_cal_E_occ_num = false;
    void cal_dE_dx(torch::Tensor& dE_dx_tensor, const torch::Tensor* var_x_tensor = nullptr);
    //
    torch::Tensor trial_Ex_Egrad(bool cal_grad = true);
    //
    double optimize_x();
    //
    bool x_need_ls = true;
    //
    torch::optim::LBFGSOptions x_options_lbfgs;
    torch::optim::AdamOptions x_options_adam;
    //
    std::unique_ptr<torch::optim::Optimizer> var_x_optimizer;



    /********* the following is used to optimize ONs  *********/
    // 
    psi::Psi<TK> wfc_new;
    // 
    psi::Psi<TK> wfc_old;
    // 
    std::vector< std::vector<TK> > R_vec_global;
    // use C' = C*exp(R) to optimize NOs, C is the expansion coefficient of NOs under NAOs
    std::vector< std::vector<TK> > dE_dR_global;
    // can be deleted? 
    std::vector<torch::Tensor> R_tensor;
    //
    std::vector<torch::Tensor> R_tensor_real;
    //
    std::vector<torch::Tensor> R_tensor_imag;
    // 
    std::vector<torch::Tensor> dE_dR_tensor;
    //
    bool has_cal_E_wfc= false;
    // in the future, it can be modified to distinguish k points
    // void cal_dE_dR(std::vector<torch::Tensor>& dE_dR_tensor, const std::vector<torch::Tensor>* R_tensor = nullptr);
    void cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik = nullptr);

    const double grad_factor = 2.0;
    //
    torch::Tensor trial_ER_Egrad(const int ik, bool cal_grad = true);
    //
    double optimize_R();
    //
    bool R_need_ls = true;
    //
    torch::optim::LBFGSOptions R_options_lbfgs;
    torch::optim::AdamOptions R_options_adam;
    //
    std::vector< std::unique_ptr<torch::optim::Optimizer> > R_optimizer;

    // //! 
    // double cal_Etotal(const torch::Tensor* var_x_tensor = nullptr,
    //                     const std::vector<torch::Tensor>* R_tensor = nullptr,
    //                     bool cal_by_occ_num = false,
    //                     bool cal_by_orb = false);
    //! 
    double cal_Etotal(const torch::Tensor* var_x_tensor = nullptr,
                        const torch::Tensor* R_tensor_ik = nullptr,
                        bool cal_by_occ_num = false,
                        bool cal_by_orb = false,
                        const int ik = 0);




    // void cal_E_grad(bool by_occ_num,
    //                   bool by_wfc,
    //                   const torch::Tensor* var_x_tensor = nullptr,
    //                   const std::vector<torch::Tensor>* R_tensor = nullptr);

    // bool has_cal_E_occ_num = false;
    // void cal_dE_dx(torch::Tensor& dE_dx_tensor, const torch::Tensor* var_x_tensor = nullptr);

    // bool has_cal_E_wfc= false;
    // void cal_dE_dR(std::vector<torch::Tensor>& dE_dR_tensor, const std::vector<torch::Tensor>* R_tensor = nullptr);



};


}

#endif
