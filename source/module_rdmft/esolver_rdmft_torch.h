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


  protected:
    
    virtual void get_start_guess(UnitCell& ucell, const int istep) override;


  private:

    int nk_total = 0;

    Parallel_2D* para_Fij;


    /********* the following is used to optimize ONs  *********/
    //
    rdmft::EBI ebi;
    //
    ModuleBase::matrix occ_number_new;
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
    torch::Tensor trial_Ex_Egrad();
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
    //
    torch::Tensor trial_ER_Egrad(const int ik);
    //
    double optimize_R();
    //
    bool R_need_ls = true;
    //
    torch::optim::LBFGSOptions R_options_lbfgs;
    torch::optim::AdamOptions R_options_adam;
    //
    std::vector< std::unique_ptr<torch::optim::Optimizer> > R_optimizer;





    // test 
    std::vector< std::vector<TK> > grad;

    std::vector< std::vector<TK> > moment_m;

    std::vector< std::vector<double> > moment_v;

    std::vector< std::vector<double> > vhat_max;

    std::vector<TK> m_hat;

    std::vector<double> v_hat;






    
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
