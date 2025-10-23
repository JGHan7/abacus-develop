//==========================================================
// Author: Jingang Han
// DATE : 2025-07-10
//==========================================================


#include <cmath>
#include <algorithm>

#include "module_rdmft/optimizer/parameterize_ONs/param_ONs.h"
#include "module_parameter/parameter.h"

// temp
#include "module_rdmft/rdmft_tools.h"

namespace rdmft
{


PARAM_ONs::PARAM_ONs()
{

}


PARAM_ONs::~PARAM_ONs()
{

}


void PARAM_ONs::init(const int nk_total, const int nkstot_full, const std::vector<double> wk_in)
{
    this->nk_nospin = nk_total/PARAM.inp.nspin;
    this->nbands = PARAM.inp.nbands;
    this->num_symm_k = wk_in;
    this->sys_nelec_spin.resize(PARAM.inp.nspin, 0.0);
    this->x.resize(PARAM.inp.nspin);
    this->occ_number.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        this->x[is].resize(nk_nospin*nbands, 0.0);
        this->occ_number[is].resize(nk_nospin*nbands, 0.0);
    }

    if( PARAM.inp.nspin == 1 )
    {
        this->sys_nelec_spin[0] = (PARAM.inp.nelec / 2.0) * nkstot_full;
        // remove the weight of spin
        for(int ik=0; ik<this->num_symm_k.size(); ++ik)
        {
            this->num_symm_k[ik] /= 2.0;
        }
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->sys_nelec_spin[0] = ((PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0) * nkstot_full;
        this->sys_nelec_spin[1] = ((PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
        std::cout << "\n******\n" << "this->sys_nelec_spin[1]: " << this->sys_nelec_spin[1] << "\n******\n" << std::endl;
    }

    // get the number of symmetric k-points
    for(int iks=0; iks<this->num_symm_k.size(); ++iks)
    {
        this->num_symm_k[iks] *= nkstot_full;
    }
    std::cout << "\n******\n" << "this->num_symm_k.size(): " << this->num_symm_k.size() << "\n******\n" << std::endl;
    // rdmft::printMatrix_pointer(1, nk_total, this->num_symm_k.data(), "num_symm_k", 10);
}


ModuleBase::matrix PARAM_ONs::get_occ_number()
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


void PARAM_ONs::convert_x2vec(const std::vector< std::vector<double> > x_in, std::vector<double>& vec_x)
{
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        for(int ik=0; ik<this->nk_nospin; ++ik)
        {
            for(int ib=0; ib<PARAM.inp.nbands; ++ib)
            {
                vec_x[ is*(this->nk_nospin*nbands) + ik*nbands + ib ] = x_in[is][ik*nbands+ib];
            }
        }
    }
}



}
