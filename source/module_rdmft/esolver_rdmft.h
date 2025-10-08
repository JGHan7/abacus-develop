//==========================================================
// Author: Jingang Han
// DATE : 2024-11-05
//==========================================================
#ifndef ESOLVER_RDMFT_H
#define ESOLVER_RDMFT_H

#include "module_rdmft/rdmft.h"
// #include "module_esolver/esolver_fp.h"
#include "module_esolver/esolver_ks_lcao.h"
#include "module_rdmft/optimizer/iter_diag_NOs.h"
#include "module_rdmft/optimizer/line_search_ONs.h"
#include "module_rdmft/optimizer/line_search_NOs.h"
#include "module_rdmft/optimizer/parameterize_ONs/ebi_constraint.h"
#include "module_rdmft/optimizer/opti_method.h" // temporary
#include  "module_rdmft/optimizer/line_search_method.h" // temporary
#include "module_rdmft/optimizer/idmft.h"

#include "module_rdmft/optimizer/ft_rdmft.h"

#include <torch/torch.h>

namespace rdmft
{



template<typename TK, typename TR>
class ESolver_RDMFT: public ModuleESolver::ESolver_KS_LCAO<TK,TR>
{
  public:
    ESolver_RDMFT();
    virtual ~ESolver_RDMFT();

    virtual void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    virtual void runner(UnitCell& ucell, const int istep) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    double opti_occ_num(bool dft_type = false, bool first_time = false);

    // int maxniter_occ_num;     // maximum iter steps for ONs

    // int maxniter_orb;         // maximum iter steps for NOs

    double iter_diag_ethr;    // energy threshold, in iterDiag of NOs

    double lambda_thr;        // threshold for checking lambda Hermitianity, in iterDiag of NOs

    double occ_num_thr;       // occupation numbers threshold, in ONs optimization

    bool dft_optimize = false;    // if use dft type to optimize occ_number

    bool conver_initial_value = false;

    std::vector< std::vector<TK> > DM;

    // Parallel_2D* para_H_ni_nj = nullptr;

  protected:

    //! objective function E(orbs, occ_nums): provides Etotal_rdmft and first-order gradient
    rdmft::RDMFT<TK, TR> rdmft_solver;



  private:

    // //! optimizing natural orbitals by iterative diagonalization
    // rdmft::IterDiag_NOs<TK, TR> iter_diag_orb;

    // //! optimizing natural occupation numbers by line search and quasi-Newton method BFGS combined with EBI method
    // rdmft::LineSearch_ONs<TK, TR> ls_opti_occ_num;

    //! 
    rdmft::IDMFT<TK, TR> idmft;

    //! 
    rdmft::FT_RDMFT<TK, TR> ft_rdmft;

    //! just for test
    rdmft::LineSearch<double> ls;

    //! just for test
    std::unique_ptr< rdmft::Opti_method<double> > x_optimizer;
    int dim_x = 2;
    std::vector<double> data_x;
    std::vector<double> df_dx;
    std::vector<double> pk;
    // the only minimum point is (−3.2, 2.8), f0=13.8
    double fx(std::vector<double>& x)
    {
        double f = std::pow(x[0]-1, 2) + std::pow(x[1]+2, 2) + 3 * x[0] *x[1];
        return f;
    }
    std::vector<double> grad_f(std::vector<double>& x)
    {
        std::vector<double> df_dx(x.size(), 0.0);
        df_dx[0] = 2*(x[0]-1) + 3*x[1];
        df_dx[1] = 2*(x[1]+2) + 3*x[0];
        return df_dx;
    }
    double norm_grad(std::vector<double>& df_dx)
    {
        double num = 0.0;
        for(int i=0; i<df_dx.size(); ++i)
        {
          num += std::pow(df_dx[i], 2);
        }
        return std::sqrt(num);
    }

    // //! just for test (complex positive-definite quadratic)
    // std::unique_ptr< rdmft::Opti_method<std::complex<double>> > x_optimizer;
    // int dim_x = 2;
    // std::vector<std::complex<double>> data_x;
    // std::vector<std::complex<double>> df_dx;
    // std::vector<std::complex<double>> pk;

    // // Define H (Hermitian, positive-definite) and z_star:
    // // H = [ [3, 0.5+0.2i],
    // //       [0.5-0.2i, 4] ]
    // // z_star = (1+2i, 2+1i)
    // const std::complex<double> H00 = {3.0, 0.0};
    // const std::complex<double> H01 = {0.5, 0.2};
    // const std::complex<double> H10 = {0.5, -0.2};
    // const std::complex<double> H11 = {4.0, 0.0};

    // const std::complex<double> z0_0 = {1.0, 2.0};
    // const std::complex<double> z0_1 = {2.0, 1.0};

    // // f(z) = (z - z0)^H H (z - z0)
    // inline double fx(std::vector<std::complex<double>>& x)
    // {
    //     std::complex<double> dz0 = x[0] - z0_0;
    //     std::complex<double> dz1 = x[1] - z0_1;

    //     // w = H * dz
    //     std::complex<double> w0 = H00 * dz0 + H01 * dz1;
    //     std::complex<double> w1 = H10 * dz0 + H11 * dz1;

    //     // val = conj(dz0)*w0 + conj(dz1)*w1
    //     std::complex<double> val = std::conj(dz0) * w0 + std::conj(dz1) * w1;

    //     const double tol_imag = 1e-12;
    //     if (std::abs(val.imag()) > tol_imag) {
    //         std::cerr << "Warning: fx returned complex with non-negligible imag part = "
    //                   << val.imag() << "\n";
    //     }
    //     return val.real();
    // }

    // // gradient (Wirtinger): df/d(conj(z)) = H (z - z0)
    // inline std::vector<std::complex<double>> grad_f(std::vector<std::complex<double>>& x)
    // {
    //     std::vector<std::complex<double>> df_dx(x.size(), {0.0, 0.0});
    //     std::complex<double> dz0 = x[0] - z0_0;
    //     std::complex<double> dz1 = x[1] - z0_1;

    //     df_dx[0] = H00 * dz0 + H01 * dz1; // ∂f/∂conj(z1)
    //     df_dx[1] = H10 * dz0 + H11 * dz1; // ∂f/∂conj(z2)

    //     return df_dx;
    // }

    // inline double norm_grad(std::vector<std::complex<double>>& df_dx)
    // {
    //     double num = 0.0;
    //     for (size_t i = 0; i < df_dx.size(); ++i)
    //     {
    //         // std::norm returns |z|^2 for complex
    //         num += std::norm(df_dx[i]);
    //     }
    //     return std::sqrt(num);
    // }

  protected:

    //! optimizing natural orbitals by iterative diagonalization
    rdmft::IterDiag_NOs<TK, TR> iter_diag_orb;

    //! optimizing natural occupation numbers by line search and quasi-Newton method BFGS combined with EBI method
    rdmft::LineSearch_ONs<TK, TR> ls_opti_occ_num;

    //! 
    rdmft::LineSearch_NOs<TK, TR> ls_opti_orb;

    // rdmft::BFGS_method<double> bfgs_rdmft;

    // std::vector< std::vector<TK> > lambda;

    // std::vector< std::vector<TK> > Fock_like_mat;

    virtual void get_start_guess(UnitCell& ucell, const int istep);

    // use dft type to update occ_number
    double update_occ_num_dft(RDMFT<TK, TR>& rdmft_solver);

    virtual void print_info();



};






}

#endif

