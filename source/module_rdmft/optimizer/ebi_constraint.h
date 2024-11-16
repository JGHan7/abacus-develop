//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================
#ifndef EBI_CONSTRAINT_H
#define EBI_CONSTRAINT_H


#include <vector>


namespace rdmft
{


class EBI
{
  
  public:
    
    EBI();
    ~EBI();

    void init(int nk_total);








  protected:


  std::vector<double> get_mu()
  {
    return this->mu;
  };







  private:

  int nk_nospin = 0;

  std::vector<double> mu;

  std::vector< std::vector<double> > dmu_dx;

  std::vector< std::vector<double> > doccNum_dx;












};





}


#endif