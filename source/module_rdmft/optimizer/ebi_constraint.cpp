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


void EBI::init(const int nk_total, const int nkstot_full)
{
    // this->random_inital = PARAM.inp.;
    
    this->solve_mu_thr = 1e-10; 
    this->nk_nospin = nk_total/PARAM.inp.nspin;
    this->nbands = PARAM.inp.nbands;
    mu.resize(PARAM.inp.nspin);
    sys_nelec_spin.resize(PARAM.inp.nspin);
    x.resize(PARAM.inp.nspin);
    occ_number.resize(PARAM.inp.nspin);
    // dmu_dx.resize(PARAM.inp.nspin);
    // docc_num_dx.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        x[is].resize(nk_nospin*nbands);
        occ_number[is].resize(nk_nospin*nbands);
        // dmu_dx[is].resize(nk_nospin*nbands);
        // docc_num_dx.resize(nk_nospin*nbands * nk_nospin*nbands);
    }

    if( PARAM.inp.nspin == 1 )
    {
        this->sys_nelec_spin[0] = (PARAM.inp.nelec / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->sys_nelec_spin[0] = ((PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0) * nkstot_full;
        this->sys_nelec_spin[1] = ((PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
        std::cout << "\n******\n" << "this->sys_nelec_spin[1]: " << this->sys_nelec_spin[1] << "\n******\n" << std::endl;
    }

}


void EBI::get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number_in)
{
    // use random numbers to generate initial values
    if(occ_number_in == nullptr)
    {
        this->random_inital = true;
        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            int M = 0;
            std::vector<double> random_num(nk_nospin*nbands, 0.0);
            rdmft::random_descend(random_num, &sys_nelec_spin[is], &M);

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
                    x_pass[ is*(nk_nospin*nbands) + ik*nbands + ib ] = this->x[is][ik*nbands+ib];
                }
            }
        }
        this->solving_mu();
    }
    // use the externally passed occ_number_in as the initial value
    else
    {
        // distinguishing up and down spin, 0 < occNum < 1
        ModuleBase::matrix num_temp = (*occ_number_in);
        // if( PARAM.inp.nspin == 1 ) { num_temp *= 0.5; }

        // give an initial guess for mu, 0.0 or other value
        // also we can use the above approach, according to artificial rules, to get x from determined occ_num, then do solving_mu()
        mu.resize(PARAM.inp.nspin, 0.0);

        for(int ik=0; ik<num_temp.nr; ++ik)
        {
            for(int ib=0; ib<num_temp.nc; ++ib)
            {
                if( ik < num_temp.nr/PARAM.inp.nspin )
                {
                    x[0][ik*nbands + ib] = erf_inv_own( 2 * num_temp(ik, ib) - 1 ) - mu[0];
                    occ_number[0][ik*nbands + ib] = num_temp(ik, ib);
                    x_pass[ ik*nbands + ib ] = this->x[0][ik*nbands+ib];
                }
                else
                {
                    x[PARAM.inp.nspin-1][ik*nbands + ib] = erf_inv_own( 2 * num_temp(ik, ib) - 1 ) - mu[PARAM.inp.nspin-1];
                    occ_number[PARAM.inp.nspin-1][ik*nbands + ib] = num_temp(ik, ib);
                    x_pass[ (PARAM.inp.nspin-1)*(nk_nospin*nbands) + ik*nbands + ib ] = this->x[PARAM.inp.nspin-1][ik*nbands+ib];
                }
            }
        }
    }
    std::cout << "\n******\n" << "start_guess: ebi, 1.0" << "\n******\n" << std::endl;
}



void EBI::update_x_occ_num(const std::vector<double>& x_in)
{
    // update member variable x from external x_in
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        for(int ik=0; ik<nk_nospin; ++ik)
        {
            for(int ib=0; ib<nbands; ++ib)
            {
                this->x[is][ik*nbands+ib] = x_in[ is*(nk_nospin*nbands) + ik*nbands +ib ];
            }
        }
    }

    // get the new mu and occ_number
    this->solving_mu();
    // return this->get_occ_number();  
}


void EBI::get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx)
{
    const int N = nk_nospin*nbands;
    // const double factor =  PARAM.inp.nspin==1 ? 2.0 : 1.0;
    // const double factor =  PARAM.inp.nspin==1 ? 0.5 : 1.0;   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    const double factor = 1.0;
    std::vector< std::vector<double> > dE_deta(PARAM.inp.nspin, std::vector<double>(N, factor));

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        // convert format. consider spin up and spin down separately
        for(int j=0; j<N; ++j) { dE_deta[is][j] *= dE_docc_num[is*N + j]; }

        std::vector<double> dmu_dx(N, 0.0);
        std::vector<double> docc_num_dx(N * N, 0.0);

        this->cal_dmu_dx(dmu_dx, is);
        this->cal_docc_num_dx(dmu_dx, docc_num_dx, is);

        // rdmft::dgemm_lapack( docc_num_dx.data(), dE_deta[is].data(), (dE_dx.data() + is*N), N, 1, N );
        rdmft::dgemm_lapack( docc_num_dx.data(), dE_deta[is].data(), (dE_dx.data() + is*N), N, 1, N );
    }
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

    // if(PARAM.inp.nspin == 1) occ_num_pass *= 2;

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

    // std::vector<double> sum = this->cal_sum(mu_in, is);
    std::vector<double> sum(3, 0.0);
    for(int i=0; i<this->x[is].size(); ++i)
    {
        sum[0] += ( std::erf(this->x[is][i] + mu_in) + 1.0 )/2.0;
        sum[1] += erf_der1(this->x[is][i] + mu_in);
        sum[2] += erf_der2(this->x[is][i] + mu_in);
    }

    f_der[0] = ( sum[0] - this->sys_nelec_spin[is] ) * sum[1];
    f_der[1] = 0.5*std::pow(sum[1], 2) + ( sum[0] - this->sys_nelec_spin[is] ) * sum[2];

    return f_der;
}


// std::vector<double> EBI::cal_sum(double mu_in, int is)
// {
//     std::vector<double> sum(3, 0.0);
//     for(int i=0; i<this->x[is].size(); ++i)
//     {
//         sum[0] += ( std::erf(this->x[is][i] + mu_in) + 1.0 )/2.0;
//         sum[1] += erf_der1(this->x[is][i] + mu_in);
//         sum[2] += erf_der2(this->x[is][i] + mu_in);
//     }
//     return sum;
// }


void EBI::cal_dmu_dx(std::vector<double>& dmu_dx, int is)
{
    double sum_der1 = 0.0;
    for(int i=0; i<this->x[is].size(); ++i) { sum_der1 += erf_der1(this->x[is][i] + this->mu[is]); }

    for(int j=0; j<dmu_dx.size(); ++j) { dmu_dx[j] = - erf_der1(this->x[is][j] + this->mu[is]) / sum_der1; }
}


void EBI::cal_docc_num_dx(const std::vector<double>& dmu_dx, std::vector<double>& docc_num_dx, int is)
{
    const int N = nk_nospin*nbands;
    for(int i=0; i<N; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            if( i==j )
            {
                docc_num_dx[i*N+j] = 0.5 * erf_der1(this->x[is][i] + this->mu[is]) * (1 + dmu_dx[i]);
            }
            else
            {
                docc_num_dx[i*N+j] = 0.5 * erf_der1(this->x[is][i] + this->mu[is]) * dmu_dx[j];
            }
        }
    }
}






}

