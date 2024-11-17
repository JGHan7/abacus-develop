//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================

#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

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


std::vector<double> EBI::get_start_guess(ModuleBase::matrix* occ_number_in)
{
    std::vector<double> x_temp(nk_nospin * PARAM.inp.nspin * nbands , 0.0);

    // use random numbers to generate initial values
    if(occ_number_in == nullptr)
    {
        // Initialize the random number generator and distribution
        std::random_device rd;      // random seed, requires hardware support
        std::mt19937 gen(rd());     // mersenne Twister engine, or std::mt19937 gen(42);
        std::uniform_real_distribution<> dis( std::nextafter(0.0, 1.0), 1.0 ); // a uniform distribution in the range (0.0, 1.0)

        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            std::vector<double> random_num(nk_nospin*nbands, 0.0);
            for(int i=0; i<random_num.size(); ++i) { random_num[i] = dis(gen); }
            // sort descending
            std::sort(random_num.begin(), random_num.end(), std::greater<>());

            // for each spin, M being the smallest number satisfying \sum_{p=1}^{M} occNum_{p} >= N
            int M = 0;
            double sum_occ_number = 0.0;
            for(int i=0; i<random_num.size(); ++i)
            {
                sum_occ_number += random_num[i];
                if( sum_occ_number > nelec_spin[is] )
                {
                    M = i;
                    break;
                }
            }

            // to avoid the low bands with large-k points getting too small values ​
            // ​and the high bands with small-k points getting too large values
            // we fill the bands with descending occ_number first
            for(int ik=0; ik<nk_nospin; ++ik)
            {
                for(int ib=0; ib<nbands; ++ib)
                {
                    this->occ_number[is][ik*nbands+ib] = random_num[ik*nbands+ib];

                    if( ik*nbands+ib < M )
                    {
                        if( random_num[ik*nbands+ib] > (std::erf(2.0) + 1.0)/2.0 )  { this->x[is][ik*nbands+ib] = 2.0; }
                        else { this->x[is][ik*nbands+ib] = erf_inv_own( random_num[ik*nbands+ib] * 2.0 - 1.0 ); }
                    }
                    else
                    {
                        this->x[is][ik*nbands+ib] = -2.0;
                    }
                    x_temp[is*(nk_nospin*nbands) + ik*nbands +ib ] = this->x[is][ik*nbands+ib];
                }
            }

        }
        // this->solving_mu(); // need it ?
    }
    // use the externally passed occ_number_in as the initial value
    else
    {
        // distinguishing up and down spin, 0 < occNum < 1
        if( PARAM.inp.nspin == 1 ) { (*occ_number_in) *= 0.5; }

        // give an initial guess for mu, 0.0 or other value
        mu.resize(PARAM.inp.nspin, 0.0);

        for(int ik=0; ik<occ_number_in->nr; ++ik)
        {
            for(int ib=0; ib<occ_number_in->nc; ++ib)
            {
                if( ik < occ_number_in->nr/PARAM.inp.nspin )
                {
                    x_temp[ik*nbands + ib] = erf_inv_own( 2*(*occ_number_in)(ik, ib) - 1 ) - mu[0];
                    x[0][ik*nbands + ib] = x_temp[ik*nbands + ib];
                    occ_number[0][ik*nbands + ib] = (*occ_number_in)(ik, ib);
                }
                else
                {
                    x_temp[ik*nbands + ib] = erf_inv_own( 2*(*occ_number_in)(ik, ib) - 1 ) - mu[PARAM.inp.nspin-1];
                    x[PARAM.inp.nspin-1][ik*nbands + ib] = x_temp [ik*nbands + ib];
                    occ_number[PARAM.inp.nspin-1][ik*nbands + ib] = (*occ_number_in)(ik, ib);
                }
            }
        }
    }
    return x_temp;
}



void EBI::solving_mu()
{

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

    return occ_num_pass;
}




}

