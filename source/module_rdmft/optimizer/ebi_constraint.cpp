//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================

#include <cmath>
#include <random>
#include <limits>

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
    x.resize(PARAM.inp.nspin);
    dmu_dx.resize(PARAM.inp.nspin);
    doccNum_dx.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        x[is].resize(nk_nospin*nbands);
        dmu_dx[is].resize(nk_nospin*nbands);
        doccNum_dx.resize(nk_nospin*nbands * nk_nospin*nbands);
    }

}


std::vector<double> EBI::get_start_guess(ModuleBase::matrix* occ_number)
{
    std::vector<double> num(nk_nospin * PARAM.inp.nspin * nbands , 0.0);

    // use random numbers to generate initial values
    if(occ_number == nullptr)
    {
        // Initialize the random number generator and distribution
        // random seed, requires hardware support
        std::random_device rd;
        // mersenne Twister engine
        std::mt19937 gen(rd());     // std::mt19937 gen(42);
        // define a uniform distribution in the range (0.0, 1.0)
        std::uniform_real_distribution<> dis( std::nextafter(0.0, 1.0), 1.0 );

        // // Generate random numbers in a loop
        // for (int i = 0; i < 5; ++i) {
        //     double random_num = dis(gen); // Generate a random number
        //     std::cout << random_num << std::endl;
        // }

    }
    // use the externally passed occ_number as the initial value
    else
    {   
        // give an initial guess for mu, 0.0 or other value
        mu.resize(PARAM.inp.nspin, 0.0);

        for(int ik=0; ik<occ_number->nr; ++ik)
        {
            for(int ib=0; ib<occ_number->nc; ++ib)
            {
                if( ik < occ_number->nr/PARAM.inp.nspin )
                {
                    num[ik*nbands + ib] = erf_inv_own( 2*(*occ_number)(ik, ib) - 1 ) - mu[0];
                    x[0][ik*nbands + ib] = num[ik*nbands + ib];
                }
                else
                {
                    num[ik*nbands + ib] = erf_inv_own( 2*(*occ_number)(ik, ib) - 1 ) - mu[PARAM.inp.nspin-1];
                    x[PARAM.inp.nspin-1][ik*nbands + ib] = num [ik*nbands + ib];
                }
            }
        }
    }
    return num;
}






}

