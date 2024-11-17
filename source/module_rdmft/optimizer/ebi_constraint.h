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

    std::vector<double> get_start_guess(ModuleBase::matrix* occ_number = nullptr);

    ModuleBase::matrix get_occ_number();

    //! when x is determined, get mu and occ_number
    void solving_mu();

    double solve_mu_thr;






  protected:


  std::vector<double> get_mu()
  {
    return this->mu;
  };







  private:

  int nk_nospin = 0;
  int nbands = 0;
  std::vector<double> nelec_spin;

  std::vector<double> mu;

  std::vector< std::vector<double> > x;

  std::vector< std::vector<double> > occ_number;

  std::vector< std::vector<double> > dmu_dx;

  std::vector< std::vector<double> > doccNum_dx;

  //! when x and mu are determined, get occ_number
  void cal_occ_num();

  //! when spin = is, mu=mu_in,  return the sum of occ_num, erf_der1, and erf_der2 respectively
  std::vector<double> cal_sum(double mu_in, int is = 0);

  std::vector<double> cal_f_der(double mu_in, int is = 0);








};





}


#endif