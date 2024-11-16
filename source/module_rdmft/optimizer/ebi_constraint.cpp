//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================

#include <cmath>

#include "module_rdmft/optimizer/ebi_constraint.h"
// #include "module_rdmft/optimizer/optimizer_tools.h"
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
    mu.resize(PARAM.inp.nspin);
    dmu_dx.resize(PARAM.inp.nspin);
    doccNum_dx.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        dmu_dx[is].resize(nk_nospin*PARAM.inp.nbands);
        doccNum_dx.resize(nk_nospin*PARAM.inp.nbands * nk_nospin*PARAM.inp.nbands);
    }

}









}

