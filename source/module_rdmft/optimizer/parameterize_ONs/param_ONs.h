//==========================================================
// Author: Jingang Han
// DATE : 2025-07-11
//==========================================================
#ifndef PARAM_ONS_H
#define PARAM_ONS_H

#include <vector>
#include "module_base/matrix.h"


namespace rdmft
{


class PARAM_ONs
{

  public:

    PARAM_ONs();
    virtual ~PARAM_ONs();

    virtual void init(const int nk_total, const int nkstot_full, const std::vector<double> wk_in);

    virtual void get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number_in = nullptr) = 0;

    virtual void update_x_occ_num(std::vector<double>& x_in) = 0;

    virtual void get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx) = 0;

    virtual void get_d2E_dx2(const std::vector<double>& dE_docc_num, std::vector<double>& d2E_dx2) = 0;

    //! pass the occ_number as a matrix object, spins and k-points share the same index
    ModuleBase::matrix get_occ_number();

    // temp 
    const std::vector<double>& get_nelec_spin() { return this->sys_nelec_spin; }

    //! pass the number of symmetric k-points
    const std::vector<double>& get_num_symm_k() { return this->num_symm_k; }


  protected:

    int nk_nospin = 0;
    int nbands = 0;
    std::vector<double> sys_nelec_spin;

    //! the number of symmetric k-points
    std::vector<double> num_symm_k;

    std::vector< std::vector<double> > x;

    std::vector< std::vector<double> > occ_number;

    // //! when x is determined, get occ_number
    // virtual double cal_occ_num(int is) = 0;

    // //! ensure that the minimum occ_number is greater than min_occ_num, thus avoiding the problem of dE/docc_num divergence
    // //! slight modification will be made to x
    // virtual void check_occ_num(std::vector<double>& x_pass) = 0;

    //! convert x to a vector
    void convert_x2vec(const std::vector< std::vector<double> > x_in, std::vector<double>& vec_x);

};

}

#endif
