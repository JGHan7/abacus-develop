//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================

#include <cmath>
#include <algorithm>
// #include <random>
// #include <limits>
// #include <algorithm>

#include "module_rdmft/optimizer/ebi_constraint.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_parameter/parameter.h"

namespace rdmft
{

EBI::EBI()
{

}


EBI::~EBI()
{

}


void EBI::init(int nk_total)
{
    this->solve_mu_thr = 1e-10; 
    this->nk_nospin = nk_total/PARAM.inp.nspin;
    this->nbands = PARAM.inp.nbands;
    mu.resize(PARAM.inp.nspin);
    nelec_spin.resize(PARAM.inp.nspin);
    x.resize(PARAM.inp.nspin);
    occ_number.resize(PARAM.inp.nspin);
    dmu_dx.resize(PARAM.inp.nspin);
    doccNum_dx.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        x[is].resize(nk_nospin*nbands);
        occ_number.resize(nk_nospin*nbands);
        dmu_dx[is].resize(nk_nospin*nbands);
        doccNum_dx.resize(nk_nospin*nbands * nk_nospin*nbands);
    }

    if( PARAM.inp.nspin == 1 )
    {
        this->nelec_spin[0] = PARAM.inp.nelec / 2.0;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->nelec_spin[0] = (PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0;
        this->nelec_spin[1] = (PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0;
    }

}


void EBI::get_inital_guess(std::vector<double>& x_in, std::vector<double>& dE_dx, ModuleBase::matrix* occ_number_in)
{
    if( x_in.size() == dE_dx.size() && x_in.size() == nk_nospin * PARAM.inp.nspin * nbands )
    {
        std::fill(x_in.begin(), x_in.end(), 0.0);
        std::fill(dE_dx.begin(), dE_dx.end(), 0.0);
    }
    else
    {
        x_in.resize(nk_nospin * PARAM.inp.nspin * nbands, 0.0);
        dE_dx.resize(nk_nospin * PARAM.inp.nspin * nbands, 0.0);
    }

    // use random numbers to generate initial values
    if(occ_number_in == nullptr)
    {
        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            int M = 0;
            std::vector<double> random_num(nk_nospin*nbands, 0.0);
            rdmft::random_descend(random_num, &nelec_spin[is], &M);

            // to avoid the low bands with large-k points getting too small values ​
            // ​and the high bands with small-k points getting too large values
            // we fill the bands with descending occ_number first
            for(int ik=0; ik<nk_nospin; ++ik)
            {
                for(int ib=0; ib<nbands; ++ib)
                {
                    if( ik*nbands+ib < M )
                    {
                        if( random_num[ik*nbands+ib] > (std::erf(2.0) + 1.0)/2.0 )  { this->x[is][ik*nbands+ib] = 2.0; }
                        else { this->x[is][ik*nbands+ib] = erf_inv_own( random_num[ik*nbands+ib] * 2.0 - 1.0 ); }
                    }
                    else
                    {
                        this->x[is][ik*nbands+ib] = -2.0;
                    }
                    x_in[is*(nk_nospin*nbands) + ik*nbands +ib ] = this->x[is][ik*nbands+ib];
                }
            }
        }
        this->solving_mu();
    }
    // use the externally passed occ_number_in as the initial value
    else
    {
        // distinguishing up and down spin, 0 < occNum < 1
        if( PARAM.inp.nspin == 1 ) { (*occ_number_in) *= 0.5; }

        // give an initial guess for mu, 0.0 or other value
        // also we can use the above approach, according to artificial rules, to get a set of x from a set of determined occ_num
        // then do solving_mu()
        mu.resize(PARAM.inp.nspin, 0.0);

        for(int ik=0; ik<occ_number_in->nr; ++ik)
        {
            for(int ib=0; ib<occ_number_in->nc; ++ib)
            {
                if( ik < occ_number_in->nr/PARAM.inp.nspin )
                {
                    x_in[ik*nbands + ib] = erf_inv_own( 2*(*occ_number_in)(ik, ib) - 1 ) - mu[0];
                    x[0][ik*nbands + ib] = x_in[ik*nbands + ib];
                    occ_number[0][ik*nbands + ib] = (*occ_number_in)(ik, ib);
                }
                else
                {
                    x_in[ik*nbands + ib] = erf_inv_own( 2*(*occ_number_in)(ik, ib) - 1 ) - mu[PARAM.inp.nspin-1];
                    x[PARAM.inp.nspin-1][ik*nbands + ib] = x_in[ik*nbands + ib];
                    occ_number[PARAM.inp.nspin-1][ik*nbands + ib] = (*occ_number_in)(ik, ib);
                }
            }
        }
    }

}


void EBI::get_dE_dx(const std::vector<double>& x_in, std::vector<double>& dE_dx)
{
    // // update member variable x from external x_in
    // for(int is=0; is<PARAM.inp.nspin; ++is)
    // {
    //     for(int ik=0; ik<nk_nospin; ++ik)
    //     {
    //         for(int ib=0; ib<nbands; ++ib)
    //         {
    //             this->x[is][ik*nbands+ib] = x_in[is*(nk_nospin*nbands) + ik*nbands +ib ];
    //         }
    //     }
    // }

    // this->solving_mu();



}

ModuleBase::matrix EBI::update_x_occ_num(const std::vector<double>& x_in)
{
    // update member variable x from external x_in
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        for(int ik=0; ik<nk_nospin; ++ik)
        {
            for(int ib=0; ib<nbands; ++ib)
            {
                this->x[is][ik*nbands+ib] = x_in[is*(nk_nospin*nbands) + ik*nbands +ib ];
            }
        }
    }

    // get the new mu and occ_number
    this->solving_mu();

    return this->get_occ_number();   
}





void EBI::solving_mu()
{
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        /********* is there an error in the paper formula? mu should be updated at each step *********/
        // double mu_temp = 0.0;
        // std::vector<double> f_der(2, 1.0);

        // while( f_der[0] > this->solve_mu_thr )
        // {
        //     f_der = this->cal_f_der(this->mu[is], is);
        //     double f1_divided_f2 = std::abs( f_der[0]/f_der[1] );
        //     double sign = 0.0;
        //     if( f_der[0]>0 ) { sign = 1.0; }
        //     else if ( f_der[0]<0 ) { sign = -1.0; }

        //     if( f1_divided_f2 > 1.0 )
        //     {
        //         mu_temp -= sign;
        //     }
        //     else
        //     {
        //         std::vector<double> f_der_temp = this->cal_f_der(mu_temp, is);
        //         mu_temp -= sign * std::abs( f_der_temp[0]/f_der_temp[1] );
        //     }
        // }
        // this->mu[is] = mu_temp;
        /********* is there an error in the paper formula? *********/

        this->mu[is] = 0.0;
        std::vector<double> f_der(2, 1.0);

        while( f_der[0] > this->solve_mu_thr )
        {
            f_der = this->cal_f_der(this->mu[is], is);
            double f1_divided_f2 = std::abs( f_der[0]/f_der[1] );

            double sign = 0.0;
            if( f_der[0]>0 ) { sign = 1.0; }
            else if ( f_der[0]<0 ) { sign = -1.0; }

            if( f1_divided_f2 > 1.0 )
            {
                this->mu[is] -= sign;
            }
            else
            {
                this->mu[is] -= sign * f1_divided_f2;
            }
        }
    }

    this->cal_occ_num();
}

ModuleBase::matrix EBI::get_occ_number()
{
    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);

    for(int ik=0; ik<occ_num_pass.nr; ++ik)
    {
        for(int ib=0; ib<occ_num_pass.nc; ++ib)
        {
            if( ik < occ_num_pass.nr/PARAM.inp.nspin )
            {
                occ_num_pass(ik, ib) = this->occ_number[0][ik*nbands + ib];
            }
            else
            {
                occ_num_pass(ik, ib) = this->occ_number[PARAM.inp.nspin-1][ik*nbands + ib];
            }
        }
    }

    if(PARAM.inp.nspin == 1) occ_num_pass *= 2;

    return occ_num_pass;
}


// std::vector< std::vector<double> > EBI::cal_occ_num()
// {
//     std::vector< std::vector<double> > sum(PARAM.inp.nspin, std::vector<double>(3, 0.0));
//     for(int is=0; is<PARAM.inp.nspin; ++is)
//     {
//         for(int i=0; i<this->x[is].size(); ++i)
//         {
//             this->occ_number[is][i] = ( std::erf(this->x[is][i] + this->mu[is]) + 1.0 )/2.0;

//             sum[is][0] += this->occ_number[is][i];
//             sum[is][1] += erf_der1(this->x[is][i] + this->mu[is]);
//             sum[is][2] += erf_der2(this->x[is][i] + this->mu[is]);
//         }
//     }
//     return sum;
// }

void EBI::cal_occ_num()
{
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        for(int i=0; i<this->x[is].size(); ++i)
        {
            this->occ_number[is][i] = ( std::erf(this->x[is][i] + this->mu[is]) + 1.0 )/2.0;
        }
    }
}


std::vector<double> EBI::cal_f_der(double mu_in, int is)
{
    std::vector<double> f_der(2, 0.0);
    std::vector<double> sum = this->cal_sum(mu_in, is);

    f_der[0] = ( sum[0] - this->nelec_spin[is] ) * sum[1];
    f_der[1] = 0.5*std::pow(sum[1], 2) + ( sum[0] - this->nelec_spin[is] ) * sum[2];

    return f_der;
}


std::vector<double> EBI::cal_sum(double mu_in, int is)
{
    std::vector<double> sum(3, 0.0);
    for(int i=0; i<this->x[is].size(); ++i)
    {
        sum[0] += ( std::erf(this->x[is][i] + mu_in) + 1.0 )/2.0;
        sum[1] += erf_der1(this->x[is][i] + mu_in);
        sum[2] += erf_der2(this->x[is][i] + mu_in);
    }
    return sum;
}



}

