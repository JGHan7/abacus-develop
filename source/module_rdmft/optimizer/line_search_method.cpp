//==========================================================
// Author: Jingang Han
// DATE : 2025-09-22
//==========================================================


#include  "module_rdmft/optimizer/line_search_method.h"


namespace rdmft
{



template<typename TX>
LineSearch<TX>::LineSearch()
{
    ;
}


template<typename TX>
LineSearch<TX>::~LineSearch()
{
    ;
}





































template class LineSearch<double>;
template class LineSearch<std::complex<double>>;

}

