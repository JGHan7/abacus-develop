#include "module_base/global_function.h"
#include "module_base/tool_quit.h"
#include "read_input.h"
#include "read_input_tool.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace ModuleIO
{
void ReadInput::item_others()
{
    // non-collinear spin-constrained
    {
        Input_Item item("sc_mag_switch");
        item.annotation = "switch to control spin-constrained DFT";
        read_sync_bool(input.sc_mag_switch);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.sc_mag_switch)
            {
                ModuleBase::WARNING_QUIT("ReadInput",
                                         "This feature is not stable yet and might lead to "
                                         "erroneous results.\n"
                                         " Please wait for the official release version.");
                // if (para.input.nspin != 4 && para.input.nspin != 2)
                // {
                //     ModuleBase::WARNING_QUIT("ReadInput", "nspin must be 2 or
                //     4 when sc_mag_switch > 0");
                // }
                // if (para.input.calculation != "scf")
                // {
                //     ModuleBase::WARNING_QUIT("ReadInput", "calculation must
                //     be scf when sc_mag_switch > 0");
                // }
                // if (para.input.nupdown > 0.0)
                // {
                //     ModuleBase::WARNING_QUIT("ReadInput", "nupdown should not
                //     be set when sc_mag_switch > 0");
                // }
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("decay_grad_switch");
        item.annotation = "switch to control gradient break condition";
        read_sync_bool(input.decay_grad_switch);
        this->add_item(item);
    }
    {
        Input_Item item("sc_thr");
        item.annotation = "Convergence criterion of spin-constrained iteration (RMS) in uB";
        read_sync_double(input.sc_thr);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.sc_thr < 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "sc_thr must >= 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("nsc");
        item.annotation = "Maximal number of spin-constrained iteration";
        read_sync_int(input.nsc);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.nsc <= 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "nsc must > 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("nsc_min");
        item.annotation = "Minimum number of spin-constrained iteration";
        read_sync_int(input.nsc_min);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.nsc_min <= 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "nsc_min must > 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("sc_scf_nmin");
        item.annotation = "Minimum number of outer scf loop before "
                          "initializing lambda loop";
        read_sync_int(input.sc_scf_nmin);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.sc_scf_nmin < 2)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "sc_scf_nmin must >= 2");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("alpha_trial");
        item.annotation = "Initial trial step size for lambda in eV/uB^2";
        read_sync_double(input.alpha_trial);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.alpha_trial <= 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "alpha_trial must > 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccut");
        item.annotation = "Maximal step size for lambda in eV/uB";
        read_sync_double(input.sccut);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.sccut <= 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "sccut must > 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("sc_drop_thr");
        item.annotation = "Convergence criterion ratio of lambda iteration in Spin-constrained DFT";
        read_sync_double(input.sc_drop_thr);
        this->add_item(item);
    }
    {
        Input_Item item("sc_scf_thr");
        item.annotation = "Density error threshold for inner loop of spin-constrained SCF";
        read_sync_double(input.sc_scf_thr);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.sc_scf_thr <= 0.0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "sc_scf_thr must > 0.0");
            }
        };
        this->add_item(item);
    }

    // Quasiatomic Orbital analysis
    {
        Input_Item item("qo_switch");
        item.annotation = "switch to control quasiatomic orbital analysis";
        read_sync_bool(input.qo_switch);
        this->add_item(item);
    }
    {
        Input_Item item("qo_basis");
        item.annotation = "type of QO basis function: hydrogen: hydrogen-like "
                          "basis, pswfc: read basis from pseudopotential";
        read_sync_string(input.qo_basis);
        this->add_item(item);
    }
    {
        Input_Item item("qo_thr");
        item.annotation = "accuracy for evaluating cutoff radius of QO basis function";
        read_sync_double(input.qo_thr);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.qo_thr > 1e-6)
            {
                ModuleBase::WARNING("ReadInput",
                                    "too high the convergence threshold might "
                                    "yield unacceptable result");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("qo_strategy");
        item.annotation = "strategy to generate generate radial orbitals";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            for (int i = 0; i < count; i++)
            {
                para.input.qo_strategy.push_back(item.str_values[i]);
            }
        };
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.qo_strategy.size() != para.input.ntype)
            {
                if (para.input.qo_strategy.size() == 1)
                {
                    para.input.qo_strategy.resize(para.input.ntype, para.input.qo_strategy[0]);
                }
                else
                {
                    std::string default_strategy;
                    if (para.input.qo_basis == "hydrogen")
                    {
                        default_strategy = "energy-valence";
                    }
                    else if ((para.input.qo_basis == "pswfc") || (para.input.qo_basis == "szv"))
                    {
                        default_strategy = "all";
                    }
                    else
                    {
                        ModuleBase::WARNING_QUIT("ReadInput",
                                                 "When setting default values for qo_strategy, "
                                                 "unexpected/unknown "
                                                 "qo_basis is found. Please check it.");
                    }
                    para.input.qo_strategy.resize(para.input.ntype, default_strategy);
                }
            }
        };
        sync_stringvec(input.qo_strategy, para.input.ntype, "all");
        this->add_item(item);
    }
    {
        Input_Item item("qo_screening_coeff");
        item.annotation = "rescale the shape of radial orbitals";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            for (int i = 0; i < count; i++)
            {
                para.input.qo_screening_coeff.push_back(std::stod(item.str_values[i]));
            }
        };
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (!item.is_read())
            {
                return;
            }
            if (para.input.qo_screening_coeff.size() != para.input.ntype)
            {
                if (para.input.qo_basis == "pswfc")
                {
                    double default_screening_coeff
                        = (para.input.qo_screening_coeff.size() == 1) ? para.input.qo_screening_coeff[0] : 0.1;
                    para.input.qo_screening_coeff.resize(para.input.ntype, default_screening_coeff);
                }
                else
                {
                    ModuleBase::WARNING_QUIT("ReadInput",
                                             "qo_screening_coeff should have the same number of "
                                             "elements as ntype");
                }
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            for (auto screen_coeff: para.input.qo_screening_coeff)
            {
                if (screen_coeff < 0)
                {
                    ModuleBase::WARNING_QUIT("ReadInput",
                                             "screening coefficient must >= 0 "
                                             "to tune the pswfc decay");
                }
                if (std::fabs(screen_coeff) < 1e-6)
                {
                    ModuleBase::WARNING_QUIT("ReadInput",
                                             "every low screening coefficient might yield very high "
                                             "computational cost");
                }
            }
        };
        sync_doublevec(input.qo_screening_coeff, para.input.ntype, 0.1);
        this->add_item(item);
    }

    // PEXSI
    {
        Input_Item item("pexsi_npole");
        item.annotation = "Number of poles in expansion";
        read_sync_int(input.pexsi_npole);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_inertia");
        item.annotation = "Whether inertia counting is used at the very "
                          "beginning of PEXSI process";
        read_sync_bool(input.pexsi_inertia);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_nmax");
        item.annotation = "Maximum number of PEXSI iterations after each "
                          "inertia counting procedure";
        read_sync_int(input.pexsi_nmax);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_comm");
        item.annotation = "Whether to construct PSelInv communication pattern";
        read_sync_bool(input.pexsi_comm);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_storage");
        item.annotation = "Storage space used by the Selected Inversion "
                          "algorithm for symmetric matrices";
        read_sync_bool(input.pexsi_storage);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_ordering");
        item.annotation = "Ordering strategy for factorization and selected inversion";
        read_sync_int(input.pexsi_ordering);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_row_ordering");
        item.annotation = "Row permutation strategy for factorization and "
                          "selected inversion, 0: NoRowPerm, 1: LargeDiag";
        read_sync_int(input.pexsi_row_ordering);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_nproc");
        item.annotation = "Number of processors for parmetis";
        read_sync_int(input.pexsi_nproc);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_symm");
        item.annotation = "Matrix symmetry";
        read_sync_bool(input.pexsi_symm);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_trans");
        item.annotation = "Whether to transpose";
        read_sync_bool(input.pexsi_trans);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_method");
        item.annotation = "pole expansion method, 1: Cauchy Contour Integral, "
                          "2: Moussa optimized method";
        read_sync_int(input.pexsi_method);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_nproc_pole");
        item.annotation = "Number of processes used by each pole";
        read_sync_int(input.pexsi_nproc_pole);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_temp");
        item.annotation = "Temperature, in the same unit as H";
        read_sync_double(input.pexsi_temp);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_gap");
        item.annotation = "Spectral gap";
        read_sync_double(input.pexsi_gap);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_delta_e");
        item.annotation = "An upper bound for the spectral radius of S^{-1} H";
        read_sync_double(input.pexsi_delta_e);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu_lower");
        item.annotation = "Initial guess of lower bound for mu";
        read_sync_double(input.pexsi_mu_lower);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu_upper");
        item.annotation = "Initial guess of upper bound for mu";
        read_sync_double(input.pexsi_mu_upper);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu");
        item.annotation = "Initial guess for mu (for the solver)";
        read_sync_double(input.pexsi_mu);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu_thr");
        item.annotation = "Stopping criterion in terms of the chemical "
                          "potential for the inertia counting procedure";
        read_sync_double(input.pexsi_mu_thr);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu_expand");
        item.annotation = "If the chemical potential is not in the initial "
                          "interval, the interval is expanded by "
                          "muInertiaExpansion";
        read_sync_double(input.pexsi_mu_expand);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_mu_guard");
        item.annotation = "Safe guard criterion in terms of the chemical potential to "
                          "reinvoke the inertia counting procedure";
        read_sync_double(input.pexsi_mu_guard);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_elec_thr");
        item.annotation = "Stopping criterion of the PEXSI iteration in terms "
                          "of the number of electrons compared to "
                          "numElectronExact";
        read_sync_double(input.pexsi_elec_thr);
        this->add_item(item);
    }
    {
        Input_Item item("pexsi_zero_thr");
        item.annotation = "if the absolute value of matrix element is less "
                          "than ZERO_Limit, it will be considered as 0";
        read_sync_double(input.pexsi_zero_thr);
        this->add_item(item);
    }

    // Only for Test
    {
        Input_Item item("out_alllog");
        item.annotation = "output information for each processor, when parallel";
        read_sync_bool(input.out_alllog);
        this->add_item(item);
    }
    {
        Input_Item item("nurse");
        item.annotation = "for coders";
        read_sync_int(input.nurse);
        this->add_item(item);
    }
    {
        Input_Item item("t_in_h");
        item.annotation = "calculate the kinetic energy or not";
        read_sync_bool(input.t_in_h);
        this->add_item(item);
    }
    {
        Input_Item item("vl_in_h");
        item.annotation = "calculate the local potential or not";
        read_sync_bool(input.vl_in_h);
        this->add_item(item);
    }
    {
        Input_Item item("vnl_in_h");
        item.annotation = "calculate the nonlocal potential or not";
        read_sync_bool(input.vnl_in_h);
        this->add_item(item);
    }
    {
        Input_Item item("vh_in_h");
        item.annotation = "calculate the hartree potential or not";
        read_sync_bool(input.vh_in_h);
        this->add_item(item);
    }
    {
        Input_Item item("vion_in_h");
        item.annotation = "calculate the local ionic potential or not";
        read_sync_bool(input.vion_in_h);
        this->add_item(item);
    }
    {
        Input_Item item("test_force");
        item.annotation = "test the force";
        read_sync_bool(input.test_force);
        this->add_item(item);
    }
    {
        Input_Item item("test_stress");
        item.annotation = "test the stress";
        read_sync_bool(input.test_stress);
        this->add_item(item);
    }
    {
        Input_Item item("test_skip_ewald");
        item.annotation = "whether to skip ewald";
        read_sync_bool(input.test_skip_ewald);
        this->add_item(item);
    }
    {
        Input_Item item("ri_hartree_benchmark");
        item.annotation = "whether to use the RI approximation for the Hartree term in LR-TDDFT for benchmark (with FHI-aims/ABACUS read-in style)";
        read_sync_string(input.ri_hartree_benchmark);
        this->add_item(item);
    }
    {
        Input_Item item("aims_nbasis");
        item.annotation = "the number of basis functions for each atom type used in FHI-aims (for benchmark)";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            for (int i = 0; i < count; i++)
            {
                para.input.aims_nbasis.push_back(std::stod(item.str_values[i]));
            }
            };
        sync_intvec(input.aims_nbasis, para.input.aims_nbasis.size(), 0);
        this->add_item(item);
    }

    // RDMFT, added by jghan, 2024-10-16
    {
        Input_Item item("rdmft_orb_opti");
        item.annotation = "optimization method of natural orbitals in rdmft";
        read_sync_string(input.rdmft_orb_opti);
        this->add_item(item);
    }
    {
        Input_Item item("occ_num_opti");
        item.annotation = "optimization method of natural occupation numbers in rdmft";
        read_sync_string(input.occ_num_opti);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_couple_opti");
        item.annotation = "whether to optimize both NOs and ONs in one iteration";
        read_sync_bool(input.rdmft_couple_opti);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft");
        item.annotation = "whether to perform rdmft calculation, default is false";
        read_sync_bool(input.rdmft);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_power_alpha");
        item.annotation = "the alpha parameter of power-functional, g(occ_number) = occ_number^alpha"
                          " used in exx-type functionals such as muller and power";
        read_sync_double(input.rdmft_power_alpha);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if( para.input.dft_functional == "hf" || para.input.dft_functional == "pbe0" || para.input.dft_functional == "hse" )
            {
                para.input.rdmft_power_alpha = 1.0;
            }
            else if( para.input.dft_functional == "muller" )
            {
                para.input.rdmft_power_alpha = 0.5;
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if( (para.input.rdmft_power_alpha < 0) || (para.input.rdmft_power_alpha > 1) )
            {
                ModuleBase::WARNING_QUIT("ReadInput", "rdmft_power_alpha should be greater than 0.0 and less than 1.0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("maxniter_orb");
        item.annotation = "maximum number of iterations to optimize the natural orbitals";
        read_sync_int(input.maxniter_orb);
        // item.reset_value = [](const Input_Item& item, Parameter& para) {
        //     if( para.input.rdmft_orb_opti == "iter_diag" || PARAM.inp.dft_opti )
        //     {
        //         para.input.maxniter_orb = para.input.scf_nmax;
        //     }
        // };
        this->add_item(item);
    }
    {
        Input_Item item("maxniter_occ_num");
        item.annotation = "maximum number of iterations to optimize the natural occupation numbers";
        read_sync_int(input.maxniter_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("dft_opti");
        item.annotation = "whether to perform DFT type optimization on natural occupation numbers";
        read_sync_bool(input.dft_opti);
        this->add_item(item);
    }
    {
        Input_Item item("conv_inital_value");
        item.annotation = "whether to use the converged DFT results as initial values";
        read_sync_bool(input.conv_inital_value);
        this->add_item(item);
    }
    {
        Input_Item item("init_orb_by_lambda");
        item.annotation = "whether the initial natural orbital is provided by the symmetrized lambda, or KS-DFT";
        read_sync_bool(input.init_orb_by_lambda);
        this->add_item(item);
    }
    {
        Input_Item item("scale_fock");
        item.annotation = "whether to scale the Fock-like matrix, in iterative diagonalization";
        read_sync_bool(input.scale_fock);
        this->add_item(item);
    }
    {
        Input_Item item("random_occ_num");
        item.annotation = "whether to use a random occupation numbers as the initial value";
        read_sync_bool(input.random_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("iter_diag_ethr");
        item.annotation = "energy convergence criterion for iterative diagonalization of orbitals";
        read_sync_double(input.iter_diag_ethr);
        this->add_item(item);
    }
    {
        Input_Item item("occ_num_thr");
        item.annotation = "convergence criterion for occupation numbers optimization";
        read_sync_double(input.occ_num_thr);
        this->add_item(item);
    }
    {
        Input_Item item("lambda_thr");
        item.annotation = "convergence criterion of the lambda matrix in iterative diagonalization";
        read_sync_double(input.lambda_thr);
        this->add_item(item);
    }
    {
        Input_Item item("solve_mu_thr");
        item.annotation = "convergence criterion of EBI method for solving mu";
        read_sync_double(input.solve_mu_thr);
        this->add_item(item);
    }
    {
        Input_Item item("tot_nelec_thr");
        item.annotation = "convergence criterion for the conservation of the total number of electrons";
        read_sync_double(input.tot_nelec_thr);
        this->add_item(item);
    }
    {
        Input_Item item("min_occ_num");
        item.annotation = "to avoid the problem of dE/docc_num divergence, set a minimum value for the natural occupation number";
        read_sync_double(input.min_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("occ_num_func");
        item.annotation = "function of parameterized occupation number. Currently, the softmax and error function(erf) can be used";
        read_sync_string(input.occ_num_func);
        this->add_item(item);
    }
    {
        Input_Item item("ls_condition");
        item.annotation = "condition type for inexact line search: swolfe, wolfe";
        read_sync_string(input.ls_condition);
        this->add_item(item);
    }
    {
        Input_Item item("ls_fixed_step");
        item.annotation = "fixed step when optimizie occupation numbers";
        read_sync_double(input.ls_fixed_step);
        this->add_item(item);
    }
    {
        Input_Item item("ls_wolfe_c1");
        item.annotation = "the first parameter of the wolfe condition";
        read_sync_double(input.ls_wolfe_c1);
        this->add_item(item);
    }
    {
        Input_Item item("ls_wolfe_c2");
        item.annotation = "the second parameter of the wolfe condition. Recommended: Quasi-Newton method 0.9(relaxed conditions: 0.999)";
        read_sync_double(input.ls_wolfe_c2);
        this->add_item(item);
    }
    {
        Input_Item item("ls_wolfe_c2_cg");
        item.annotation = "the second parameter of the wolfe condition. Recommended: Nonlinear Conjugate Gradient method 0.1 (relaxed conditions: 0.4)";
        read_sync_double(input.ls_wolfe_c2_cg);
        this->add_item(item);
    }
    {
        Input_Item item("ls_armijo_c1");
        item.annotation = "the first parameter of an Armijo condition";
        read_sync_double(input.ls_armijo_c1);
        this->add_item(item);
    }
    {
        Input_Item item("ls_armijo_c2");
        item.annotation = "the second parameter of an Armijo condition";
        read_sync_double(input.ls_armijo_c2);
        this->add_item(item);
    }
    {
        Input_Item item("scale_zeta");
        item.annotation = "scaling factor for off-diagonal elements of Fock-like matrix, in iterative diagonalization";
        read_sync_double(input.scale_zeta);
        this->add_item(item);
    }
    {
        Input_Item item("max_step_size");
        item.annotation = "maximum step size for line search";
        read_sync_double(input.max_step_size);
        this->add_item(item);
    }
    {
        Input_Item item("min_step_size");
        item.annotation = "minimum step size for line search";
        read_sync_double(input.min_step_size);
        this->add_item(item);
    }
    {
        Input_Item item("level_shifting");
        item.annotation = "perform opposite shifts on the occupied and unoccupied diagonal elements of a Fock-like matrix";
        read_sync_double(input.level_shifting);
        this->add_item(item);
    }
    {
        Input_Item item("print_fock");
        item.annotation = "print Fock and lambda matrix in iterative diagonalization";
        read_sync_bool(input.print_fock);
        this->add_item(item);
    }
    {
        Input_Item item("print_BFGS_Hk");
        item.annotation = "print BFGS-Hk matrix in occupation numbers optimization";
        read_sync_bool(input.print_BFGS_Hk);
        this->add_item(item);
    }
    {
        Input_Item item("rotate_fock");
        item.annotation = "rotate the Fock to the natural orbital representation of step 1";
        read_sync_bool(input.rotate_fock);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if( para.input.rdmft_orb_opti == "adam")
            {
                para.input.rotate_fock = false;
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("mixing_rdmft");
        item.annotation = "linear mixing of Fock matrix";
        read_sync_bool(input.mixing_rdmft);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if( para.input.rdmft_orb_opti == "adam")
            {
                para.input.mixing_rdmft = false;
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("idmft_kappa");
        item.annotation = "kappa parameter in i-DMFT";
        read_sync_double(input.idmft_kappa);
        this->add_item(item);
    }
    {
        Input_Item item("idmft_beta");
        item.annotation = "beta parameter in i-DMFT";
        read_sync_double(input.idmft_beta);
        this->add_item(item);
    }
    {
        Input_Item item("adam_learn_rate");
        item.annotation = "learning rate in adam which used by rdmft";
        read_sync_double(input.adam_learn_rate);
        this->add_item(item);
    }
    {
        Input_Item item("adam_lr_occ_num");
        item.annotation = "learning rate in adam which used by rdmft when optimizing occupation numbers";
        read_sync_double(input.adam_lr_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("adam_scaling_lr");
        item.annotation = "scaling factor of learning rate in adam which used by rdmft";
        read_sync_double(input.adam_scaling_lr);
        this->add_item(item);
    }
    {
        Input_Item item("adam_beta1");
        item.annotation = "parameter of first moment in adam which used by rdmft";
        read_sync_double(input.adam_beta1);
        this->add_item(item);
    }
    {
        Input_Item item("adam_beta2");
        item.annotation = "parameter of second moment in adam which used by rdmft";
        read_sync_double(input.adam_beta2);
        this->add_item(item);
    }
    {
        Input_Item item("opti_by_torch");
        item.annotation = "whether to use libTorch's optimizer to optimize rdmft's NOs and ONs";
        read_sync_bool(input.opti_by_torch);
        // item.reset_value = [](const Input_Item& item, Parameter& para) {
        //     if( para.input.rdmft_auto_diff == true)
        //     {
        //         para.input.opti_by_torch = true;
        //     }
        // };
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_auto_diff");
        item.annotation = "whether to use libTorch's automatic differentiation to obtain the first-order gradient";
        read_sync_bool(input.rdmft_auto_diff);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_auto_diff2");
        item.annotation = "whether to use libTorch's automatic differentiation to obtain second-order gradient";
        read_sync_bool(input.rdmft_auto_diff2);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_one_opti");
        item.annotation = "whether to use one optimizer to optimize both NOs and ONs";
        read_sync_bool(input.rdmft_one_opti);
        this->add_item(item);
    }
    {
        Input_Item item("rdmft_dm_conv");
        item.annotation = "whether to force density matrix convergence in rdmft optimization";
        read_sync_bool(input.rdmft_dm_conv);
        this->add_item(item);
    }
    {
        Input_Item item("scaling_rotation");
        item.annotation = "scaling_rotation";
        read_sync_double(input.scaling_rotation);
        this->add_item(item);
    }
    {
        Input_Item item("small_rotation");
        item.annotation = "whether to use small changes for the unitary transformation of the natural orbitals";
        read_sync_bool(input.small_rotation);
        this->add_item(item);
    }
    {
        Input_Item item("precond_occ_num");
        item.annotation = "whether to precondition the optimization of ONs";
        read_sync_bool(input.precond_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("precond_orb");
        item.annotation = "whether to precondition the optimization of NOs";
        read_sync_bool(input.precond_orb);
        this->add_item(item);
    }
    {
        Input_Item item("read_occ_num");
        item.annotation = "read the initial occupation numbers from the provided occ_num.txt file";
        read_sync_bool(input.read_occ_num);
        this->add_item(item);
    }
    {
        Input_Item item("precond_type");
        item.annotation = "the optimizer performs precond in two ways: 1 or 2";
        read_sync_int(input.precond_type);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if( para.input.rdmft_orb_opti == "cg" || para.input.occ_num_opti == "cg" )
            {
                para.input.precond_type = 1;
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("precond_g");
        item.annotation = "precond_g";
        read_sync_double(input.precond_g);
        this->add_item(item);
    }

    // EXX PW by rhx0820, 2025-03-10
    {
        Input_Item item("exxace");
        item.annotation = "whether to perform ace calculation in exxpw";
        read_sync_bool(input.exxace);
        this->add_item(item);
    }


}
} // namespace ModuleIO