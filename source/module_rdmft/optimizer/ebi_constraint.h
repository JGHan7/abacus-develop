//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef EBI_CONSTRAINT_H
#define EBI_CONSTRAINT_H


#include <vector>
#include "module_base/matrix.h"

namespace rdmft
{


class EBI
{
  
  public:
    
    EBI();
    ~EBI();

    void init(int nk_total);

    bool random_inital = false;

    void get_inital_guess(std::vector<double>& x_in, std::vector<double>& dE_dx, ModuleBase::matrix* occ_number = nullptr);

    //! update x and solve for mu and occ_number
    //! return occ_number to the outside
    ModuleBase::matrix update_x_occ_num(const std::vector<double>& x_in);

    //! when x and mu are determined, convert dE_docc_num to dE_dx
    void get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx);







    double solve_mu_thr;


    //! pass the occ_number as a matrix object, spins and k-points share the same index
    ModuleBase::matrix get_occ_number();

    //! pass the mu in EBI method
    std::vector<double> get_mu() { return this->mu; };



  protected:





  private:

    int nk_nospin = 0;
    int nbands = 0;
    std::vector<double> nelec_spin;

    std::vector<double> mu;

    std::vector< std::vector<double> > x;

    std::vector< std::vector<double> > occ_number;

    // std::vector< std::vector<double> > dmu_dx;

    // std::vector< std::vector<double> > docc_num_dx;

    //! when x is determined, get mu and occ_number
    void solving_mu();

    //! when x and mu are determined, get occ_number
    void cal_occ_num();

    // //! when spin = is, mu=mu_in, return the sum of occ_num, erf_der1, and erf_der2 respectively
    // std::vector<double> cal_sum(double mu_in, int is = 0);

    //! when spin = is, mu=mu_in, return F_der1, F_der2 respectively
    std::vector<double> cal_f_der(double mu_in, int is = 0);

    void cal_dmu_dx(std::vector<double>& dmu_dx, int is = 0);

    void cal_docc_num_dx(const std::vector<double>& dmu_dx, std::vector<double>& docc_num_dx, int is = 0);








};





}


#endif