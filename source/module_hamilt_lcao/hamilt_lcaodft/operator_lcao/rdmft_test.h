//==========================================================
// Author: Jingang Han
// DATE : 2023-12-22
//==========================================================
#ifndef RDMFT_TEST_H
#define RDMFT_TEST_H

#include "module_base/matrix.h"
//#include "module_elecstate/module_dm/density_matrix.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_matrix.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_lcao/module_gint/gint_gamma.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_elecstate/potentials/potential_new.h"
#include "module_base/blas_connector.h"
#include "module_base/scalapack_connector.h"
#include "module_basis/module_ao/parallel_2d.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_base/parallel_reduce.h"
#include "module_elecstate/module_dm/cal_dm_psi.h"

#include "module_hamilt_general/operator.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/operator_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/op_exx_lcao.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/ekinetic_new.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/nonlocal_new.h"
#include "module_hamilt_lcao/hamilt_lcaodft/operator_lcao/veff_lcao.h"

// used by Exx&LRI
#include "module_ri/RI_2D_Comm.h"
#include "module_ri/Exx_LRI.h"


// used by class Veff_rdmft
#include "module_base/timer.h"
#include "module_elecstate/potentials/potential_new.h"
#include "module_hamilt_lcao/hamilt_lcaodft/local_orbital_charge.h"
//#include "module_hamilt_lcao/module_gint/gint_gamma.h"
//#include "module_hamilt_lcao/module_gint/gint_k.h"
//#include "operator_lcao.h"
//#include "module_cell/module_neighbor/sltk_grid_driver.h"
//#include "module_cell/unitcell.h"
#include "module_elecstate/potentials/H_Hartree_pw.h"
#include "module_elecstate/potentials/pot_local.h"
#include "module_elecstate/potentials/pot_xc.h"
#include "module_hamilt_pw/hamilt_pwdft/structure_factor.h"


#include <iostream>
#include <type_traits>
#include <complex>
#include <vector>



namespace rdmft
{


// // for test use dgemm_
// void printResult_dgemm();

//for print matrix
template <typename TK>
void printMatrix_pointer(int M, int N, TK* matrixA, std::string nameA)
{
    std::cout << "\n" << nameA << ": \n";
    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            std::cout << *(matrixA+i*N+j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


template <typename TK>
void printMatrix_vector(int M, int N, std::vector<TK>& matrixA, std::string nameA)
{
    std::cout << "\n" << nameA << ": \n";
    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            std::cout << matrixA[i*N+j] << " ";
        }
        std::cout << "\n\n";
    }
    std::cout << "\n";
}



// this part of the code is copying from class Veff and do some modifications.
template <typename TK, typename TR>
class Veff_rdmft : public hamilt::OperatorLCAO<TK, TR>
{
  public:
    Veff_rdmft(Gint_k* GK_in,
                      Local_Orbital_Charge* loc_in,
                      LCAO_Matrix* LM_in,
                      const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                      const Charge* charge_in,
                      hamilt::HContainer<TR>* hR_in,
                      std::vector<TK>* hK_in,
                      const UnitCell* ucell_in,
                      Grid_Driver* GridD_in,
                      const Parallel_Orbitals* paraV,
                      const ModulePW::PW_Basis& rho_basis_in,
                      const ModuleBase::matrix& vloc_in,
                      const ModuleBase::ComplexMatrix& sf_in,
                      const std::string potential_in)
        : GK(GK_in),
          loc(loc_in),
          charge_(charge_in),
          ucell_(ucell_in),
          rho_basis_(rho_basis_in),
          vloc_(vloc_in),
          sf_(sf_in),
          potential_(potential_in),
          hamilt::OperatorLCAO<TK, TR>(LM_in, kvec_d_in, hR_in, hK_in)
    {
        this->cal_type = hamilt::lcao_gint;

        this->initialize_HR(ucell_in, GridD_in, paraV);
        GK_in->initialize_pvpR(*ucell_in, GridD_in);
    }
    Veff_rdmft(Gint_Gamma* GG_in,
                          Local_Orbital_Charge* loc_in,
                          LCAO_Matrix* LM_in,
                          const std::vector<ModuleBase::Vector3<double>>& kvec_d_in,
                          const Charge* charge_in,
                          hamilt::HContainer<TR>* hR_in,
                          std::vector<TK>* hK_in,
                          const UnitCell* ucell_in,
                          Grid_Driver* GridD_in,
                          const Parallel_Orbitals* paraV,
                          const ModulePW::PW_Basis& rho_basis_in,
                          const ModuleBase::matrix& vloc_in,
                          const ModuleBase::ComplexMatrix& sf_in,  
                          const std::string potential_in
                          )
        : GG(GG_in), 
          loc(loc_in), 
          charge_(charge_in),
          ucell_(ucell_in),
          rho_basis_(rho_basis_in),
          vloc_(vloc_in),
          sf_(sf_in),
          potential_(potential_in),
          hamilt::OperatorLCAO<TK, TR>(LM_in, kvec_d_in, hR_in, hK_in)
    {
        this->cal_type = hamilt::lcao_gint;

        this->initialize_HR(ucell_in, GridD_in, paraV);

        GG_in->initialize_pvpR(*ucell_in, GridD_in);
    }

    ~Veff_rdmft(){};

    virtual void contributeHR() override;


  private:
    // used for k-dependent grid integration.
    Gint_k* GK = nullptr;

    // used for gamma only algorithms.
    Gint_Gamma* GG = nullptr;

    // Charge calculating method in LCAO base and contained grid base calculation: DM_R, DM, pvpR_reduced
    Local_Orbital_Charge* loc = nullptr;

    elecstate::Potential* pot = nullptr;

    void initialize_HR(const UnitCell* ucell_in, Grid_Driver* GridD_in, const Parallel_Orbitals* paraV);

    // add by jghan
    const UnitCell* ucell_;

    const Charge* charge_;

    std::string potential_;

    const ModulePW::PW_Basis rho_basis_;

    const ModuleBase::matrix vloc_;

    const ModuleBase::ComplexMatrix sf_;

};

// now support XC_func_rdmft = "HF" or "power" 
double wg_func(double wg, int symbol = 0, const std::string XC_func_rdmft = "HF", const double alpha_power = 0.656);


template <typename TK>
void set_zero_vector(std::vector<TK>& HK)
{
    for(int i=0; i<HK.size(); ++i) HK[i] = 0.0;
}


template <typename TK>
void set_zero_psi(psi::Psi<TK>& wfc)
{
    TK* pwfc_in = wfc.get_pointer();

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static, 1024)
    #endif
    
    for(int i=0; i<wfc.size(); ++i) pwfc_in[i] = 0.0;
}


template <typename TK>
void conj_psi(psi::Psi<TK>& wfc)
{
    TK* pwfc = &wfc(0, 0, 0);
    for(int i=0; i<wfc.size(); ++i) pwfc[i] = std::conj( pwfc[i] );
}


template <>
void conj_psi<double>(psi::Psi<double>& wfc);


// wfc and H_wfc need to be k_firest and provide wfc(ik, 0, 0) and H_wfc(ik, 0, 0)
// psi::Psi<TK> psi's (), std::vector<TK> HK's [] operator overloading return TK
template <typename TK>
void HkPsi(const Parallel_Orbitals* ParaV, const TK& HK, const TK& wfc, TK& H_wfc)
{

    const int one_int = 1;
    //const double one_double = 1.0, zero_double = 0.0;
    const std::complex<double> one_complex = {1.0, 0.0};
    const std::complex<double> zero_complex = {0.0, 0.0};
    const char N_char = 'N';
    const char T_char = 'T';

    const int nbasis = ParaV->desc[2];
    const int nbands = ParaV->desc_wfc[3];

    //because wfc(bands, basis'), H(basis, basis'), we do wfc*H^T(in the perspective of cpp, not in fortran). And get H_wfc(bands, basis) is correct.
    pzgemm_( &T_char, &N_char, &nbasis, &nbands, &nbasis, &one_complex, &HK, &one_int, &one_int, ParaV->desc,
        &wfc, &one_int, &one_int, ParaV->desc_wfc, &zero_complex, &H_wfc, &one_int, &one_int, ParaV->desc_wfc );

    /*
    if ( std::is_same<TK, double>::value )
    {
        pdgemm_( &N_char, &T_char, &nbands, &nbasis, &nbasis, &one_double, &wfc, &one_int, &one_int, ParaV->desc_wfc,
                &HK, &one_int, &one_int, ParaV->desc, &zero_double, &H_wfc, &one_int, &one_int, ParaV->desc_wfc );
    }
    else if ( std::is_same<TK, std::complex<double>>::value )
    {
        pzgemm_( &N_char, &T_char, &nbands, &nbasis, &nbasis, &one_complex, &wfc, &one_int, &one_int, ParaV->desc_wfc,
                &HK, &one_int, &one_int, ParaV->desc, &zero_complex, &H_wfc, &one_int, &one_int, ParaV->desc_wfc );
    }
    else std::cout << "\n\n******\nthere maybe something wrong when calling rdmft_cal() and use pdemm_/pzgemm_\n******\n\n";
    */
}


template <>
void HkPsi<double>(const Parallel_Orbitals* ParaV, const double& HK, const double& wfc, double& H_wfc);


template <typename TK>
void psiDotPsi(const Parallel_Orbitals* ParaV, const Parallel_2D& para_Eij_in, const TK& wfc, const TK& H_wfc, std::vector<TK>& Dmn, double* wfcHwfc)
{
    const int one_int = 1;
    const std::complex<double> one_complex = {1.0, 0.0};
    const std::complex<double> zero_complex = {0.0, 0.0};
    const char N_char = 'N';
    const char C_char = 'C'; 

    const int nbasis = ParaV->desc[2];
    const int nbands = ParaV->desc_wfc[3];

    const int nrow_bands = para_Eij_in.get_row_size();
    const int ncol_bands = para_Eij_in.get_col_size();

    pzgemm_( &C_char, &N_char, &nbands, &nbands, &nbasis, &one_complex, &wfc, &one_int, &one_int, ParaV->desc_wfc,
            &H_wfc, &one_int, &one_int, ParaV->desc_wfc, &zero_complex, &Dmn[0], &one_int, &one_int, para_Eij_in.desc );

    for(int i=0; i<nrow_bands; ++i)
    {
        int i_global = para_Eij_in.local2global_row(i);
        for(int j=0; j<ncol_bands; ++j)
        {
            int j_global = para_Eij_in.local2global_col(j);
            if(i_global==j_global)
            {   
                // because the Dmn obtained from pzgemm_() is stored column-major
                wfcHwfc[j_global] = std::real( Dmn[i+j*nrow_bands] );
            }
        }
    }
}

template <>
void psiDotPsi<double>(const Parallel_Orbitals* ParaV, const Parallel_2D& para_wfc_in,
                        const double& wfc, const double& H_wfc, std::vector<double>& Dmn, double* wfcHwfc);


// realize wg_wfc = wg * wfc. Calling this function and we can get wfc = wg*wfc.
template <typename TK>
void wgMulPsi(const Parallel_Orbitals* ParaV, const ModuleBase::matrix& wg, psi::Psi<TK>& wfc, int symbol = 0,
                const std::string XC_func_rdmft = "HF", const double alpha = 0.656)
{
    const int nk_local = wfc.get_nk();
    const int nbands_local = wfc.get_nbands();
    const int nbasis_local = wfc.get_nbasis();

    const int nbasis = ParaV->desc[2];      // need to be deleted
    const int nbands = ParaV->desc_wfc[3];

    for (int ik = 0; ik < nk_local; ++ik)
    {
        for (int ib_local = 0; ib_local < nbands_local; ++ib_local)  // ib_local < nbands_local , some problem, ParaV->ncol_bands
        {
            const double wg_local = wg_func( wg(ik, ParaV->local2global_col(ib_local)), symbol, XC_func_rdmft, alpha);
            TK* wfc_pointer = &(wfc(ik, ib_local, 0));
            BlasConnector::scal(nbasis_local, wg_local, wfc_pointer, 1);
        }
    }
}


// add psi with eta and g(eta)
template <typename TK>
void add_psi(const Parallel_Orbitals* ParaV, const ModuleBase::matrix& wg, psi::Psi<TK>& psi_TV, psi::Psi<TK>& psi_hartree,
                psi::Psi<TK>& psi_XC, psi::Psi<TK>& wg_Hpsi, const std::string XC_func_rdmft = "HF", const double alpha = 0.656)
{
    const int nk = psi_TV.get_nk();
    const int nbn_local = psi_TV.get_nbands();
    const int nbs_local = psi_TV.get_nbasis();
    wgMulPsi(ParaV, wg, psi_TV);
    wgMulPsi(ParaV, wg, psi_hartree);
    wgMulPsi(ParaV, wg, psi_XC, 2, XC_func_rdmft, alpha);

    const int nbasis = ParaV->desc[2];
    const int nbands = ParaV->desc_wfc[3];

    for(int ik=0; ik<nk; ++ik)
    {
        for(int inbn=0; inbn<nbn_local; ++inbn)
        {
            TK* pwg_Hpsi = &( wg_Hpsi(ik, inbn, 0) );
            for(int inbs=0; inbs<nbs_local; ++inbs)
            {
                pwg_Hpsi[inbs] = psi_TV(ik, inbn, inbs) + psi_hartree(ik, inbn, inbs) + psi_XC(ik, inbn, inbs);
            }
        }
    }

}


// wg_wfcHwfc = wg*wfcHwfc + wg_wfcHwfc
// When symbol = 0, 1, 2, 3, 4, wg = wg, 0.5*wg, g(wg), 0.5*g(wg), d_g(wg)/d_ewg respectively. Default symbol=0.
void wgMul_wfcHwfc(const ModuleBase::matrix& wg, const ModuleBase::matrix& wfcHwfc, ModuleBase::matrix& wg_wfcHwfc,
                        int symbol = 0, const std::string XC_func_rdmft = "HF", const double alpha = 0.656);


// Default symbol = 0 for the gradient of Etotal with respect to occupancy
// symbol = 1 for the relevant calculation of Etotal
void add_wg(const ModuleBase::matrix& wg, const ModuleBase::matrix& wfcHwfc_TV_in, const ModuleBase::matrix& wfcHwfc_hartree_in,
            const ModuleBase::matrix& wfcHwfc_XC_in, ModuleBase::matrix& wg_wfcHwfc, const std::string XC_func_rdmft = "HF", const double alpha = 0.656, int symbol = 0);


//give certain wg_wfcHwfc, get the corresponding energy
double sumWg_getEnergy(const ModuleBase::matrix& wg_wfcHwfc);


// realization of energy and energy gradient in RDMFT
template <typename TK, typename TR, typename T_Gint>
double rdmft_cal(LCAO_Matrix* LM_in,
                        Parallel_Orbitals* ParaV,
                        const ModuleBase::matrix& wg,
                        const psi::Psi<TK>& wfc,
                        ModuleBase::matrix& wg_wfcHamiltWfc,
                        psi::Psi<TK>& wg_HamiltWfc,
                        const K_Vectors& kv_in,
                        T_Gint& G_in, //Gint_k& GK_in
                        Local_Orbital_Charge& loc_in,
                        const Charge& charge_in,
                        const ModulePW::PW_Basis& rho_basis_in,
                        const ModuleBase::matrix& vloc_in,
                        const ModuleBase::ComplexMatrix& sf_in,
                        const std::string XC_func_rdmft = "HF",
                        const double alpha_power = 0.656)   // 0.656 for soilds, 0.525 for dissociation of H2, 0.55~0.58 for HEG
{
    ModuleBase::TITLE("hamilt_lcao", "RDMFT_E&Egradient");
    ModuleBase::timer::tick("hamilt_lcao", "RDMFT_E&Egradient");

    std::ofstream ofs_running;
    std::ofstream ofs_warning;
    const std::vector<ModuleBase::Vector3<double>> kvec_d_in = kv_in.kvec_d;

    // get local index and global k-points of wfc
    const int nk_total = wfc.get_nk();
    const int nbands_local = wfc.get_nbands();
    const int nbasis_local = wfc.get_nbasis();
    
    // create desc[] and something about MPI to Eij(nbands*nbands)
    Parallel_2D para_Eij;
    para_Eij.set_block_size(GlobalV::NB2D);
    para_Eij.set_proc_dim(GlobalV::DSIZE);
    para_Eij.comm_2D = ParaV->comm_2D;
    para_Eij.blacs_ctxt = ParaV->blacs_ctxt;
    para_Eij.set_local2global( GlobalV::NBANDS, GlobalV::NBANDS, ofs_running, ofs_warning );
    para_Eij.set_desc( GlobalV::NBANDS, GlobalV::NBANDS, para_Eij.get_row_size(), false );

    // global HR or HK is nk*nbasis*nbasis matrix. nbasis is stored by 2d-block, nk or nR is always global
    hamilt::HContainer<TR> HR_TV(GlobalC::ucell, ParaV);
    hamilt::HContainer<TR> HR_hartree(GlobalC::ucell, ParaV);
    hamilt::HContainer<TR> HR_XC(GlobalC::ucell, ParaV);
    std::vector<TK> HK_TV(ParaV->get_row_size()*ParaV->get_col_size());
    std::vector<TK> HK_hartree(ParaV->get_row_size()*ParaV->get_col_size());
    std::vector<TK> HK_XC(ParaV->get_row_size()*ParaV->get_col_size());
    
    // set zero ( std::vector will automatically be set to zero )
    HR_TV.set_zero();
    HR_hartree.set_zero();
    HR_XC.set_zero();

    if( GlobalV::GAMMA_ONLY_LOCAL )
    {
        HR_TV.fix_gamma();
        HR_hartree.fix_gamma();
        HR_XC.fix_gamma();
    }


    /****** get every Hamiltion matrix ******/

    hamilt::OperatorLCAO<TK, TR>* V_ekinetic_potential = new hamilt::EkineticNew<hamilt::OperatorLCAO<TK, TR>>(
        LM_in,
        kvec_d_in,
        &HR_TV,
        &HK_TV,
        &GlobalC::ucell,
        &GlobalC::GridD,
        ParaV
    );
    
    hamilt::OperatorLCAO<TK, TR>* V_nonlocal = new hamilt::NonlocalNew<hamilt::OperatorLCAO<TK, TR>>(
        LM_in,
        kvec_d_in,
        &HR_TV,
        &HK_TV,
        &GlobalC::ucell,
        &GlobalC::GridD,
        ParaV
    );

    hamilt::OperatorLCAO<TK, TR>* V_local = new rdmft::Veff_rdmft<TK,TR>(
        &G_in,
        &loc_in,
        LM_in,
        kvec_d_in,
        &charge_in,
        &HR_TV,
        &HK_TV,
        &GlobalC::ucell,
        &GlobalC::GridD,
        ParaV,
        rho_basis_in,
        vloc_in,
        sf_in,
        "local"
    );

    hamilt::OperatorLCAO<TK, TR>* V_hartree = new rdmft::Veff_rdmft<TK,TR>(
        &G_in,
        &loc_in,
        LM_in,
        kvec_d_in,
        &charge_in,
        &HR_hartree,
        &HK_hartree,
        &GlobalC::ucell,
        &GlobalC::GridD,
        ParaV,
        rho_basis_in,
        vloc_in,
        sf_in,
        "hartree"
    );

    // construct V_XC based on different XC_functional
    hamilt::OperatorLCAO<TK, TR>* V_XC;
    if( XC_func_rdmft == "HF" )
    {
        V_XC = new hamilt::OperatorEXX<hamilt::OperatorLCAO<TK, TR>>(
            LM_in,
            &HR_XC,
            &HK_XC,
            kv_in
        );
    }
    else if( XC_func_rdmft == "power" || XC_func_rdmft == "Muller" )
    {
        // prepare for the special density matrix DM_XC(nk*nbasis_local*nbasis_local)
        std::vector< std::vector<TK> > DM_XC(nk_total, std::vector<TK>(ParaV->nloc));  // ParaV->nloc
        std::vector< const std::vector<TK>* > DM_XC_pointer(nk_total);
        for(int ik=0; ik<nk_total; ++ik) DM_XC_pointer[ik] = &DM_XC[ik];

        std::cout << "\n\n\n******\n" << "before conj_psi()" << "\n******\n\n\n";

        // get wg_wfc = g(wg)*conj(wfc), different XC_functional has different g(wg)
        psi::Psi<TK> wg_wfc(wfc);
        conj_psi(wg_wfc);

        std::cout << "\n\n\n******\n" << "before wgMulPsi()" << "\n******\n\n\n";

        wgMulPsi(ParaV, wg, wg_wfc, 2, XC_func_rdmft, alpha_power);

        std::cout << "\n\n\n******\n" << "before psiMulPsiMpi()" << "\n******\n\n\n";

        // get the special DM_XC used in constructing V_XC
        for(int ik=0; ik<wfc.get_nk(); ++ik)
        {
            TK* DM_Kpointer = DM_XC[ik].data();  // why &(DM_XC[ik][0]) is error
#ifdef __MPI
            elecstate::psiMulPsiMpi(wg_wfc, wfc, DM_Kpointer, ParaV->desc_wfc, ParaV->desc);
#else
            elecstate::psiMulPsi(wg_wfc, wfc, DM_Kpointer);
#endif            
        }

        std::cout << "\n\n\n******\n" << "before split_m2D_ktoR()" << "\n******\n\n\n";

        // transfer the DM_XC to appropriate format
        std::vector<std::map<int,std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>>> Ds_XC_d = 
            RI_2D_Comm::split_m2D_ktoR<double>(kv_in, DM_XC_pointer, *ParaV);
        std::vector<std::map<int,std::map<std::pair<int,std::array<int,3>>,RI::Tensor<std::complex<double>>>>> Ds_XC_c = 
            RI_2D_Comm::split_m2D_ktoR<std::complex<double>>(kv_in, DM_XC_pointer, *ParaV);


        std::cout << "\n\n\n******\n" << "before Exx_LRI Vxc_fromRI" << "\n******\n\n\n";

        // provide the Ds_XC to V_XC
        // when we doing V_XC.contributeHk(ik), we get HK_XC constructed by the special DM_XC
        if (GlobalC::exx_info.info_ri.real_number)
        {
            Exx_LRI<double> Vxc_fromRI_d(GlobalC::exx_info.info_ri);
            Vxc_fromRI_d.init(MPI_COMM_WORLD, kv_in);
            Vxc_fromRI_d.cal_exx_ions();
            Vxc_fromRI_d.cal_exx_elec(Ds_XC_d, *ParaV);

            std::cout << "\n\n\n******\n" << "before new OperatorEXX with Vxc_fromRI_d" << "\n******\n\n\n";

            V_XC = new hamilt::OperatorEXX<hamilt::OperatorLCAO<TK, TR>>(
                LM_in,
                &HR_XC,
                &HK_XC,
                kv_in,
                &Vxc_fromRI_d.Hexxs
            );
        }
        else
        {
            Exx_LRI<std::complex<double>> Vxc_fromRI_c(GlobalC::exx_info.info_ri);
            Vxc_fromRI_c.init(MPI_COMM_WORLD, kv_in);
            Vxc_fromRI_c.cal_exx_ions();
            Vxc_fromRI_c.cal_exx_elec(Ds_XC_c, *ParaV);

            std::cout << "\n\n\n******\n" << "before new OperatorEXX with Vxc_fromRI_c" << "\n******\n\n\n";

            V_XC = new hamilt::OperatorEXX<hamilt::OperatorLCAO<TK, TR>>(
                LM_in,
                &HR_XC,
                &HK_XC,
                kv_in,
                nullptr,
                &Vxc_fromRI_c.Hexxs,
                1
            );

        std::cout << "\n\n\n\n\n\n******\n" << "print Vxc_fromRI_c.Hexxs" << "\n******\n\n\n\n\n\n";
        for(const auto& outerMap : Vxc_fromRI_c.Hexxs)
        {
            std::cout << "\nVxc_fromRI_c.Hexxs Outer Map Size: " << outerMap.size() << std::endl;
            for (const auto& middleMap : outerMap)
            {
                std::cout << "\nVxc_fromRI_c.Hexxs Middle Map Size: " << middleMap.second.size() << std::endl;
                for (const auto& innerMap : middleMap.second)
                {
                    const RI::Tensor<std::complex<double>>& tensor_XC = innerMap.second;
                    //const std::array<int, 3> & tensor_shape = tensor_XC.shape;
                    const std::valarray<std::complex<double>>& tensor_data = *tensor_XC.data;

                    std::cout << "\nthe length of tensor_XC_data: " << tensor_data.size() << "\n";

                    std::cout << "\ntensor_XC shape: \n";
                    for(int ix=0; ix<tensor_XC.shape.size(); ++ix)
                    {
                        std::cout << tensor_XC.shape[ix] << " ";
                    }
                    // std::cout << "\ntensor_XC data: \n";
                    // for(size_t i = 0; i < tensor_data.size(); ++i)
                    // {
                    //     if(i%5==0) std::cout << "\n";
                    //     std::cout <<  tensor_data[i] << " ";
                    // }
                    std::cout << "\n\n\n";
                }
            }
        }


        std::cout << "\n\n\n\n\n\n******\n" << "print LM_in->Hexxc" << "\n******\n\n\n\n\n\n";
        for(const auto& outerMap : *LM_in->Hexxc)
        {
            std::cout << "\nHexx Outer Map Size: " << outerMap.size() << std::endl;
            for (const auto& middleMap : outerMap)
            {
                std::cout << "\nHexx Middle Map Size: " << middleMap.second.size() << std::endl;
                for (const auto& innerMap : middleMap.second)
                {
                    const RI::Tensor<std::complex<double>>& tensor_Hexx = innerMap.second;
                    //const std::array<int, 3> & tensor_shape = tensor_XC.shape;
                    const std::valarray<std::complex<double>>& tensor_Hexx_data = *tensor_Hexx.data;

                    std::cout << "\nthe length of tensor_Hexx_data: " << tensor_Hexx_data.size() << "\n";

                    std::cout << "\ntensor_Hexx shape: \n";
                    for(int ix=0; ix<tensor_Hexx.shape.size(); ++ix)
                    {
                        std::cout << tensor_Hexx.shape[ix] << " ";
                    }
                    // std::cout << "\ntensor_Hexx data: \n";
                    // for(size_t i = 0; i < tensor_Hexx_data.size(); ++i)
                    // {
                    //     if(i%5==0) std::cout << "\n";
                    //     std::cout <<  tensor_Hexx_data[i] << " ";
                    // }
                    std::cout << "\n\n\n";
                }
            }
        }


            std::cout << "\n\n\n******\n" << "after new OperatorEXX with Vxc_fromRI_c" << "\n******\n\n\n";
        }
    }

    /****** get every Hamiltion matrix ******/


    // in gamma only, must calculate HR_hartree before HR_local
    // HR_hartree has the HR of V_hartree. HR_XC get from another way, so don't need to do this 
    V_hartree->contributeHR();

    // now HR_TV has the HR of V_ekinetic + V_nonlcao + V_local, 
    V_ekinetic_potential->contributeHR();
    V_nonlocal->contributeHR();
    V_local->contributeHR();

    std::cout << "\n\n\n******\n" << "after contributeHR()" << "\n******\n\n\n";

    //prepare for actual calculation
    //wg is global matrix, wg.nr = nk_total, wg.nc = GlobalV::NBANDS
    ModuleBase::matrix wg_forEtotal(wg.nr, wg.nc, true);
    ModuleBase::matrix wfcHwfc_TV(wg.nr, wg.nc, true);
    ModuleBase::matrix wfcHwfc_hartree(wg.nr, wg.nc, true);
    ModuleBase::matrix wfcHwfc_XC(wg.nr, wg.nc, true);

    // let the 2d-block of H_wfc is same to wfc, so we can use desc_wfc and 2d-block messages of wfc to describe H_wfc
    psi::Psi<TK> H_wfc_TV(nk_total, nbands_local, nbasis_local);
    psi::Psi<TK> H_wfc_hartree(nk_total, nbands_local, nbasis_local);
    psi::Psi<TK> H_wfc_XC(nk_total, nbands_local, nbasis_local);

    // set zero
    set_zero_psi(H_wfc_TV);
    set_zero_psi(H_wfc_hartree);
    set_zero_psi(H_wfc_XC);

    // just for temperate. in the future when realize psiDotPsi() without pzgemm_/pdgemm_,we don't need it
    const int nrow_bands = para_Eij.get_row_size();
    const int ncol_bands = para_Eij.get_col_size();
    std::vector<TK> Eij_TV(nrow_bands*ncol_bands);
    std::vector<TK> Eij_hartree(nrow_bands*ncol_bands);
    std::vector<TK> Eij_XC(nrow_bands*ncol_bands);


    /****** get wg_wfcHamiltWfc, wg_HamiltWfc and Etotal ******/

    //calculate Hwfc, wfcHwfc for each potential
    for(int ik=0; ik<nk_total; ++ik)
    {
        // get the HK with ik-th k vector, the result is stored in HK_TV, HK_hartree and HK_XC respectively
        V_ekinetic_potential->contributeHk(ik);
        V_hartree->contributeHk(ik);
        std::cout << "\n\n\n******\n" << "after V_TV&V_hartree.contribute(ik)" << "\n******\n\n\n";
        V_XC->contributeHk(ik);
        std::cout << "\n\n\n******\n" << "after V_XC.contribute(ik)" << "\n******\n\n\n";

        // get H(k) * wfc
        HkPsi( ParaV, HK_TV[0], wfc(ik, 0, 0), H_wfc_TV(ik, 0, 0));
        HkPsi( ParaV, HK_hartree[0], wfc(ik, 0, 0), H_wfc_hartree(ik, 0, 0));
        HkPsi( ParaV, HK_XC[0], wfc(ik, 0, 0), H_wfc_XC(ik, 0, 0));
        
        // get wfc * H(k)_wfc
        psiDotPsi( ParaV, para_Eij, wfc(ik, 0, 0), H_wfc_TV(ik, 0, 0), Eij_TV, &(wfcHwfc_TV(ik, 0)) );
        psiDotPsi( ParaV, para_Eij, wfc(ik, 0, 0), H_wfc_hartree(ik, 0, 0), Eij_hartree, &(wfcHwfc_hartree(ik, 0)) );
        psiDotPsi( ParaV, para_Eij, wfc(ik, 0, 0), H_wfc_XC(ik, 0, 0), Eij_XC, &(wfcHwfc_XC(ik, 0)) );
        
        // let H(k)=0 to storing next one, H(k+1)
        set_zero_vector(HK_TV);
        set_zero_vector(HK_hartree);
        set_zero_vector(HK_XC);
    }

    // !this would transfer the value of H_wfc_TV, H_wfc_hartree, H_wfc_XC
    // get the gradient of energy with respect to the wfc, i.e., wg_HamiltWfc
    add_psi(ParaV, wg, H_wfc_TV, H_wfc_hartree, H_wfc_XC, wg_HamiltWfc, XC_func_rdmft, alpha_power);

    // get the gradient of energy with respect to the occupation numbers, i.e., wg_wfcHamiltWfc
    add_wg(wg, wfcHwfc_TV, wfcHwfc_hartree, wfcHwfc_XC, wg_wfcHamiltWfc, XC_func_rdmft, alpha_power);

    // get the total energy
    add_wg(wg, wfcHwfc_TV, wfcHwfc_hartree, wfcHwfc_XC, wg_forEtotal, XC_func_rdmft, alpha_power, 1);
    double Etotal_RDMFT = sumWg_getEnergy(wg_forEtotal);

    /****** get wg_wfcHamiltWfc, wg_HamiltWfc and Etotal ******/


    // for E_TV
    ModuleBase::matrix wg_forETV(wg.nr, wg.nc, true);
    wgMul_wfcHwfc(wg, wfcHwfc_TV, wg_forETV, 0);
    double ETV_RDMFT = sumWg_getEnergy(wg_forETV);

    // for Ehartree
    ModuleBase::matrix wg_forEhartree(wg.nr, wg.nc, true);
    wgMul_wfcHwfc(wg, wfcHwfc_hartree, wg_forEhartree, 1);
    double Ehartree_RDMFT = sumWg_getEnergy(wg_forEhartree);

    // for Exc
    ModuleBase::matrix wg_forExc(wg.nr, wg.nc, true);
    wgMul_wfcHwfc(wg, wfcHwfc_XC, wg_forExc, 3, XC_func_rdmft, alpha_power);
    double Exc_RDMFT = sumWg_getEnergy(wg_forExc);

    // add up the results obtained by all processors, or we can do reduce_all(wfcHwfc_) before add_wg() used for Etotal to replace it
    Parallel_Reduce::reduce_all(Etotal_RDMFT);
    Parallel_Reduce::reduce_all(ETV_RDMFT);
    Parallel_Reduce::reduce_all(Ehartree_RDMFT);
    Parallel_Reduce::reduce_all(Exc_RDMFT);

    // print results
    std::cout << "\n\n\n******\nEtotal_RDMFT:   " << Etotal_RDMFT << "\nETV_RDMFT: " << ETV_RDMFT << "\nEhartree_RDMFT: " 
                << Ehartree_RDMFT << "\nExc_RDMFT:      " << Exc_RDMFT << "\n******\n\n\n";

    ModuleBase::timer::tick("hamilt_lcao", "RDMFT_E&Egradient");
    
    return Etotal_RDMFT;

}



}

#endif
