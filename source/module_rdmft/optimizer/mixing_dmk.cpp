//==========================================================
// Author: Jingang Han
// DATE : 2025-04-01
//==========================================================

#include "module_rdmft/optimizer/mixing_dmk.h"
#include "module_rdmft/rdmft_tools.h"
#include "module_base/module_container/base/third_party/blas.h"

#include <type_traits>

namespace rdmft
{


template<typename TK>
Mixing_DMk<TK>::Mixing_DMk()
{
    ;
}


template<typename TK>
Mixing_DMk<TK>::~Mixing_DMk()
{
    delete mixing;
}


template<typename TK>
void Mixing_DMk<TK>::init(const std::string& mixing_mode_in,
                            const double& mixing_beta_in,
                            const int& mixing_ndim_in,
                            const int& length_in)
{

    this->mixing_mode = mixing_mode_in;
    this->mixing_beta = mixing_beta_in;
    this->mixing_ndim = mixing_ndim_in;
    this->length = length_in;

    if (this->mixing_mode == "pulay")
    {
        delete this->mixing;
        this->mixing = new Base_Mixing::Pulay_Mixing(this->mixing_ndim, this->mixing_beta);
    }
    else
    {
        ModuleBase::WARNING_QUIT("DMk_Mixing", "This Mixing mode is not implemended in rdmft.");
    }

    this->mixing->init_mixing_data(this->dmk_mdata, this->length, sizeof(TK));

}


template<typename TK>
void Mixing_DMk<TK>::push_data(const TK* dmk_in, const TK* dmk_out)
{
    auto mix_dmk = [this](TK* out, const TK* in, const TK* residual)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
        for(int i=0; i<this->length; ++i)
        {
            out[i] = in[i] + this->mixing_beta * residual[i];
        }
    };

    this->mixing->push_data(this->dmk_mdata, dmk_in, dmk_out, nullptr, mix_dmk, true);

}


template<typename TK>
void Mixing_DMk<TK>::cal_coef()
{
    // temporary, no process
    auto inner_product = [this](const TK* resi_i, const TK* resi_j)
    {   
        const char C_char = 'C';
        const char T_char = 'T';
        const int one_int = 1;
        const TK one_complex = 1.0;
        const TK zero_complex = 0.0;
        std::vector<TK> temp(this->length, 0.0);

        if constexpr (std::is_same<TK, double>::value)
        {
            BlasConnector::gemv(T_char, this->length, one_int, one_complex, resi_i, this->length, resi_j, one_int, zero_complex, temp.data(), one_int);
        }
        else if constexpr (std::is_same<TK, std::complex<double>>::value)
        {
            BlasConnector::gemv(C_char, this->length, one_int, one_complex, resi_i, this->length, resi_j, one_int, zero_complex, temp.data(), one_int);
        }

        return std::abs(temp[0]);
    };

    this->mixing->cal_coef(this->dmk_mdata, inner_product);
}


template<typename TK>
void Mixing_DMk<TK>::mix_dmk(TK* dmk_mixed)
{
    this->mixing->mix_data(this->dmk_mdata, dmk_mixed);
}








template class Mixing_DMk<double>;
template class Mixing_DMk<std::complex<double>>;



}