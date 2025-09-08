//==========================================================
// Author: Jingang Han
// DATE : 2025-09-07
//==========================================================
#ifndef ESOLVER_RDMFT_TORCH_AD_H
#define ESOLVER_RDMFT_TORCH_AD_H


// #include <torch/torch.h>
// #include "module_rdmft/esolver_rdmft.h"
// #include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"

#include "module_rdmft/esolver_rdmft_torch.h"


namespace rdmft
{


template<typename TK, typename TR>
class ESolver_RDMFT_Torch_AD: public rdmft::ESolver_RDMFT_Torch<TK,TR>
{
  public:

    ESolver_RDMFT_Torch_AD();
    ~ESolver_RDMFT_Torch_AD();

    // virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    // virtual void runner(UnitCell& ucell, const int istep) override;


  protected:

    virtual void init_opti_param() override;

    virtual void select_optimizer() override;

    virtual void get_start_guess(UnitCell& ucell, const int istep) override;

    // virtual void couple_opti() override;

    // virtual void decouple_opti() override;








  private:

    int64_t nbands64 = 0;
    int64_t nbasis64 = 0;

    std::vector<torch::Tensor> thetaR_real;
    std::vector<torch::Tensor> thetaR_imag;
    torch::Tensor rotation_R;

    std::vector<torch::Tensor> wfc_new_tensor;
    std::vector<torch::Tensor> wfc_0_tensor;
    std::vector<torch::Tensor> dE_dwfc_tensor;

    void update_R_tensor(const int ik = 0);

    virtual void cal_dE_dR(torch::Tensor& dE_dR_tensor_ik, const int ik, const torch::Tensor* R_tensor_ik = nullptr) override;

    virtual torch::Tensor trial_ER_Egrad(const int ik, bool cal_grad = true) override;

    virtual double cal_Etotal(const torch::Tensor* var_x_tensor = nullptr,
                                const torch::Tensor* R_tensor_ik = nullptr,
                                bool cal_by_occ_num = false,
                                bool cal_by_orb = false,
                                const int ik = 0) override;


    virtual double check_hermi_lambda() override;


    


};


}

#endif



