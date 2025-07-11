//==========================================================
// Author: Jingang Han
// DATE : 2025-07-10
//==========================================================


#include <cmath>
#include <algorithm>

#include "module_rdmft/optimizer/parameterize_ONs/softmax.h"
#include "module_parameter/parameter.h"
#include "module_base/parallel_common.h"


namespace rdmft
{


SOFTMAX::SOFTMAX()
{

}


SOFTMAX::~SOFTMAX()
{

}


void SOFTMAX::get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number_in)
{
    
}


void SOFTMAX::update_x_occ_num(std::vector<double>& x_in)
{
    // update member variable x from external x_in
    std::vector<double> max_x_in(PARAM.inp.nspin, 0.0);
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        max_x_in[is] = *std::max_element(x_in.begin() + is*(nk_nospin*nbands), x_in.begin() + (is+1)*(nk_nospin*nbands));

        for(int ik=0; ik<nk_nospin; ++ik)
        {
            for(int ib=0; ib<nbands; ++ib)
            {
                // preventing exp() overflow
                this->x[is][ik*nbands+ib] = x_in[ is*(nk_nospin*nbands) + ik*nbands +ib ] - max_x_in[is];
            }
        }

        double sum_exp = this->cal_occ_num(is);

        this->check_occ_num(is, sum_exp);
    }

    if( this->modify_x )
    {
        std::vector< std::vector<double> > x_temp = this->x;
        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            for(int j=0; j<x_temp[is].size(); ++j)
            {
                x_temp[is][j] += max_x_in[is];
            }
        }
        this->convert_x2vec(x_temp, x_in);
    }
    this->modify_x = false;

}


void SOFTMAX::get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx)
{
    
}


double SOFTMAX::cal_occ_num(const int is)
{
    // softmax
    double sum = 0.0;
    for(int ik=0; ik<nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            this->occ_number[is][ik*nbands + ib] = std::exp(this->x[is][ik*nbands + ib]);
            sum += this->occ_number[is][ik*nbands + ib] * this->num_symm_k[is*this->nk_nospin + ik];
        }
    }

    for(int i=0; i<this->occ_number[is].size(); ++i)
    {
        this->occ_number[is][i] *= this->sys_nelec_spin[is]/sum;
    }

    return sum;
}


void SOFTMAX::check_occ_num(const int is, const double sum_exp)
{
    // according to the minimum value of the ONs, the minimum value of x is approximately
    bool occur_min = false;
    double min_x = std::log(sum_exp * PARAM.inp.min_occ_num / this->sys_nelec_spin[is]);
    for(int ik=0; ik<nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            double num = this->occ_number[is][ik*this->nbands + ib];
            if( num < PARAM.inp.min_occ_num )
            {
                this->x[is][ik*nbands + ib] = min_x;
                occur_min = true;

                this->modify_x = true;
            }
        }
    }

    Parallel_Common::bcast_double(this->x[is].data(), nk_nospin*nbands);
    if( occur_min )
    {
        this->cal_occ_num(is);
    }
}












}









