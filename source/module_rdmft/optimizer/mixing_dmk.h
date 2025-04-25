//==========================================================
// Author: Jingang Han
// DATE : 2025-04-01
//==========================================================
#ifndef MIXING_DMK_H
#define MIXING_DMK_H

#include "module_rdmft/rdmft.h"
#include "module_base/module_mixing/pulay_mixing.h"
// #include "module_base/module_mixing/broyden_mixing.h"

namespace rdmft
{

template<typename TK>
class Mixing_DMk
{
  public:

    Mixing_DMk();

    ~Mixing_DMk();

    void init(const std::string& mixing_mode_in,
                const double& mixing_beta_in,
                const int& mixing_ndim_in,
                const int& length_in);

    void push_data(const TK* dmk_in, const TK* dmk_out);

    void cal_coef();

    void mix_dmk(TK* dmk_mixed);

    std::vector<double>& get_coef()
    {
        return this->mixing->coef;
    }

    void reset()
    {
        this->mixing->reset();
        this->dmk_mdata.reset();
    }

    std::string mixing_mode = "pulay";
    double mixing_beta = 0.8;
    int mixing_ndim = 8;
    int length = 1;

    // mixing_data
    Base_Mixing::Mixing* mixing = nullptr;
    Base_Mixing::Mixing_Data dmk_mdata;





};


}



#endif
