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

    void solving_mu();






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

  void cal_occ_num();










};





}


#endif