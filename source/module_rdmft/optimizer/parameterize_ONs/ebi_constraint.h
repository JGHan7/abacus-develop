//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef EBI_CONSTRAINT_H
#define EBI_CONSTRAINT_H


#include <vector>
#include "module_base/matrix.h"
#include "module_rdmft/optimizer/parameterize_ONs/param_ONs.h"

namespace rdmft
{


class EBI: public rdmft::PARAM_ONs
{
  
  public:
    
    EBI();
    ~EBI();

    void init(const int nk_total, const int nkstot_full, const std::vector<double> wk_in) override;

    bool random_inital = false;

    void get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number_in = nullptr) override;

    //! use x_in to solve mu and update occ_number
    //! when the minimum occ_number is less than the lower limit(min_occ_num), x will be slightly modified
    void update_x_occ_num(std::vector<double>& x_in) override;

    //! when x and mu are determined, convert dE_docc_num to dE_dx
    void get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx) override;







    double solve_mu_thr = 0;

    double tot_nelec_thr = 0;

    // temp
    std::vector<double> wk_temp;


    // //! pass the occ_number as a matrix object, spins and k-points share the same index
    // ModuleBase::matrix get_occ_number();

    //! pass the mu in EBI method
    const std::vector<double>& get_mu() { return this->mu; }

    // // temp 
    // const std::vector<double>& get_nelec_spin() { return this->sys_nelec_spin; }

    // //! pass the number of symmetric k-points
    // const std::vector<double>& get_num_symm_k() { return this->num_symm_k; }


  protected:





  private:

    // //! the minimum value allowed for occ_num when obtaining the initial value (or during the entire optimization process)
    // const int min_occ_num = 1e-12; // 1e-16 now

    // //! find the occupancy number less than min_occ_num and modify it to min_occ_num
    // void check_occ_num();

    //! temp, for debug
    int solve_mu_times = 0;

    std::vector<double> mu;



    // int nk_nospin = 0;
    // int nbands = 0;
    // std::vector<double> sys_nelec_spin;

    // //! the number of symmetric k-points
    // std::vector<double> num_symm_k;

    // std::vector< std::vector<double> > x;

    // std::vector< std::vector<double> > occ_number;



    // std::vector< std::vector<double> > dmu_dx;

    // std::vector< std::vector<double> > docc_num_dx;

    //! when x is determined, get mu and occ_number
    void solving_mu();

    //! when x and mu are determined, get occ_number
    double cal_occ_num(int is) override;

    //! ensure that the minimum occ_number is greater than min_occ_num, thus avoiding the problem of dE/docc_num divergence
    //! slight modification will be made to x
    void check_occ_num(std::vector<double>& x_pass) override;

    //! convert x to a vector
    void convert_x2vec(std::vector<double>& vec_x);

    // //! when spin = is, mu=mu_in, return the sum of occ_num, erf_der1, and erf_der2 respectively
    // std::vector<double> cal_sum(double mu_in, int is = 0);

    //! when spin = is, mu=mu_in, return F_der1, F_der2 respectively
    std::vector<double> cal_f_der(double mu_in, int is = 0);

    void cal_dmu_dx(std::vector<double>& dmu_dx, int is = 0);

    void cal_docc_num_dx(const std::vector<double>& dmu_dx, std::vector<double>& docc_num_dx, int is = 0);







};





}


#endif