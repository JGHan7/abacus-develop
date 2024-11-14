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

    void init();








  protected:


  std::vector<double> get_mu()
  {
    return this->mu;
  };







  private:

  std::vector<double> mu;












};





}


#endif