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
    // rdmft_solver.before_scf(istep);
    rdmft_solver.update_ion(istep, ucell);

    // before the iterative electronic step, get initial value by one KS step
    if(GlobalC::exx_info.info_global.cal_exx)
    {
        // the command to stop the runner is in the exx_iter_finish() function of Exx_LRI_interface.hpp
        rdmft_solver.runner(istep, ucell);
    }
    else
    {
        rdmft_solver.maxniter = 1;
        rdmft_solver.runner(istep, ucell);
    }
    this->rdmft_solver.inital_wfc_occNum();


    for(int iter=1; iter <= this->maxniter; ++iter)
    {
        
    }

    std::cout << "\n******\n" << "maxniter of rdmft is: " << this->maxniter << "\n******\n" << std::endl;
    std::cout << "\n\n******\n" << "Optimization of 1-RDM is still under development" << "\n******\n" << std::endl;
}



template <typename TK, typename TR>
double ESolver_RDMFT<TK, TR>::cal_energy()
{
    return 0.0;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::cal_force(ModuleBase::matrix& force)
{
    ;
}


template <typename TK, typename TR>
void ESolver_RDMFT<TK, TR>::cal_stress(ModuleBase::matrix& stress)
{
    ;
}



template class ESolver_RDMFT<double, double>;
template class ESolver_RDMFT<std::complex<double>, double>;
template class ESolver_RDMFT<std::complex<double>, std::complex<double>>;

}

