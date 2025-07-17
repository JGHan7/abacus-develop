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





    /********* the following is used in rdmft with libTorch *********/

  private:
    int nk_total = 0;
    Parallel_2D* para_Fij;

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
    double optimize_x();



    // 
    psi::Psi<TK> wfc_new;
    // 
    std::vector< std::vector<TK> > R_vec_global;
    // use C' = C*exp(R) to optimize NOs, C is the expansion coefficient of NOs under NAOs
    std::vector< std::vector<TK> > dE_dR_global;
    // 
    std::vector<torch::Tensor> R_tensor;
    // 
    std::vector<torch::Tensor> dE_dR_tensor;
    //
    double optimize_R();


    // std::unique_ptr<torch::optim::Adam> var_x_optimizer;
    // var_x_optimizer = std::make_unique<torch::optim::Adam>(params, torch::optim::AdamOptions(0.05));

    bool x_need_ls = true;
    std::unique_ptr<torch::optim::Optimizer> var_x_optimizer;
    torch::optim::LBFGSOptions x_options_lbfgs;
    torch::optim::AdamOptions x_options_adam;

    torch::Tensor trial_Ex_Egrad();



    bool R_need_ls = true;
    std::unique_ptr<torch::optim::Optimizer> R_optimizer;
    torch::optim::LBFGSOptions R_options_lbfgs;
    torch::optim::AdamOptions R_options_adam;
    // torch::optim::Adam* R_optimizer = nullptr;
    // torch::optim::LBFGS* R_optimizer = nullptr;
    torch::Tensor trial_ER_Egrad(const int ik);






    
    //!
    // 
    double cal_Etotal(const torch::Tensor* var_x_tensor = nullptr,
                        const std::vector<torch::Tensor>* R_tensor = nullptr,
                        bool cal_by_occ_num = false,
                        bool cal_by_orb = false);

    // void cal_E_grad(bool by_occ_num,
    //                   bool by_wfc,
    //                   const torch::Tensor* var_x_tensor = nullptr,
    //                   const std::vector<torch::Tensor>* R_tensor = nullptr);

    bool has_cal_E_occ_num = false;
    void cal_dE_dx(torch::Tensor& dE_dx_tensor, const torch::Tensor* var_x_tensor = nullptr);

    bool has_cal_E_wfc= false;
    void cal_dE_dR(std::vector<torch::Tensor>& dE_dR_tensor, const std::vector<torch::Tensor>* R_tensor = nullptr);


























};


}

#endif
