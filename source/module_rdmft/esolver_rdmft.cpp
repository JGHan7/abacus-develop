//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================

// #include "module_rdmft/rdmft.h"
#include "module_rdmft/esolver_rdmft.h"

namespace rdmft
{


template <typename TK, typename TR>
ESolver_RDMFT<TK, TR>::ESolver_RDMFT()
{
    this->classname = "ESolver_RDMFT";
}


template <typename TK, typename TR>
ESolver_RDMFT<TK, TR>::~ESolver_RDMFT()
{
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::before_all_runners(const Input_para& inp, UnitCell& ucell)
{
    rdmft_solver.before_all_runners(inp, ucell);
    this->maxniter = rdmft_solver.maxniter;

    //initialize rdmft
    rdmft_solver.init(ucell, PARAM.inp.dft_functional, PARAM.inp.rdmft_power_alpha);
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::runner(const int istep, UnitCell& ucell)
{
    rdmft_solver.before_scf(istep);
    rdmft_solver.update_ion(ucell);

    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        GlobalC::exx_info.info_global.hybrid_step = 1;
        rdmft_solver.runner(istep, ucell);
        GlobalC::exx_info.info_global.hybrid_step = PARAM.inp.exx_hybrid_step;
    }
    else
    {
        rdmft_solver.maxniter = 1;
        rdmft_solver.runner(istep, ucell);
    }
    ModuleBase::matrix occ_number_ks(rdmft_solver.pelec->wg);
    for(int ik=0; ik < occ_number_ks.nr; ++ik)
    {
        for(int inb=0; inb < occ_number_ks.nc; ++inb) occ_number_ks(ik, inb) /= rdmft_solver.kv.wk[ik];
    }
    this->rdmft_solver.update_elec(occ_number_ks, *(rdmft_solver.psi));


    for(int iter=1; iter <= this->maxniter; ++iter)
    {
        std::cout << "\n\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n" << std::endl;


    }
}








}

