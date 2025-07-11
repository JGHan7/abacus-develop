//==========================================================
// Author: Jingang Han
// DATE : 2025-07-10
//==========================================================
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>
#include "module_base/matrix.h"
#include "module_rdmft/optimizer/parameterize_ONs/param_ONs.h"

namespace rdmft
{


class SOFTMAX
{

  public:

    SOFTMAX();
    ~SOFTMAX();

    // void init(const int nk_total, const int nkstot_full, const std::vector<double> wk_in);

    void get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number = nullptr);

    void update_x_occ_num();

    void get_dE_dx();

    // //! pass the occ_number as a matrix object, spins and k-points share the same index
    // ModuleBase::matrix get_occ_number();







  private:

    
    int nk_nospin = 0;
    int nbands = 0;
    std::vector<double> sys_nelec_spin;

    //! the number of symmetric k-points
    std::vector<double> num_symm_k;

    std::vector< std::vector<double> > x;

    std::vector< std::vector<double> > occ_number;














    //! when x is determined, get occ_number
    double cal_occ_num(int is);


    //! ensure that the minimum occ_number is greater than min_occ_num, thus avoiding the problem of dE/docc_num divergence
    //! slight modification will be made to x
    void check_occ_num(std::vector<double>& x_pass);

















};





}


#endif
