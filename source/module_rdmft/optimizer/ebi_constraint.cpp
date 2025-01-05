//==========================================================
// Author: Jingang Han
// DATE : 2024-11-12
//==========================================================

#include <cmath>
#include <algorithm>
// #include <random>
// #include <limits>
// #include <algorithm>

#include "module_rdmft/optimizer/ebi_constraint.h"
#include "module_rdmft/optimizer/optimizer_tools.h"
#include "module_parameter/parameter.h"

#include "module_rdmft/rdmft_tools.h" // temp

namespace rdmft
{

EBI::EBI()
{

}


EBI::~EBI()
{

}


void EBI::init(const int nk_total, const int nkstot_full, const std::vector<double> wk_in)
{
    // this->random_inital = PARAM.inp.;
    
    this->solve_mu_thr = 1e-10;
    this->tot_nelec_thr = 1e-10;
    this->nk_nospin = nk_total/PARAM.inp.nspin;
    this->nbands = PARAM.inp.nbands;
    this->num_symm_k = wk_in;
    mu.resize(PARAM.inp.nspin);
    sys_nelec_spin.resize(PARAM.inp.nspin);
    x.resize(PARAM.inp.nspin);
    occ_number.resize(PARAM.inp.nspin);
    // dmu_dx.resize(PARAM.inp.nspin);
    // docc_num_dx.resize(PARAM.inp.nspin);

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        x[is].resize(nk_nospin*nbands);
        occ_number[is].resize(nk_nospin*nbands);
        // dmu_dx[is].resize(nk_nospin*nbands);
        // docc_num_dx.resize(nk_nospin*nbands * nk_nospin*nbands);
    }

    if( PARAM.inp.nspin == 1 )
    {
        this->sys_nelec_spin[0] = (PARAM.inp.nelec / 2.0) * nkstot_full;
        // remove the weight of spin
        for(int ik; ik<this->num_symm_k.size(); ++ik)
        {
            this->num_symm_k[ik] /= 2.0;
        }
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
    }
    else if( PARAM.inp.nspin == 2 )
    {
        this->sys_nelec_spin[0] = ((PARAM.inp.nelec + PARAM.inp.nupdown) / 2.0) * nkstot_full;
        this->sys_nelec_spin[1] = ((PARAM.inp.nelec - PARAM.inp.nupdown) / 2.0) * nkstot_full;
        std::cout << "\n******\n" << "this->sys_nelec_spin[0]: " << this->sys_nelec_spin[0] << "\n******\n" << std::endl;
        std::cout << "\n******\n" << "this->sys_nelec_spin[1]: " << this->sys_nelec_spin[1] << "\n******\n" << std::endl;
    }

    // get the number of symmetric k-points
    for(int ik; ik<this->num_symm_k.size(); ++ik)
    {
        this->num_symm_k[ik] *= nkstot_full;
    }

}


void EBI::get_inital_guess(std::vector<double>& x_pass, const ModuleBase::matrix* occ_number_in)
{
    // use random numbers to generate initial values
    if(occ_number_in == nullptr)
    {
        this->random_inital = true;
        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            std::vector<double> random_num(nk_nospin*nbands, 0.0);
            rdmft::random_descend(random_num);
            int M = rdmft::smallest_loc_big_value(this->nk_nospin, PARAM.inp.nbands, this->sys_nelec_spin[is], random_num, this->num_symm_k);
            rdmft::printMatrix_pointer(nk_nospin, PARAM.inp.nbands, random_num.data(), "random_num used in EBI", 10);

            // to avoid the low bands with large-k points getting too small values ​
            // ​and the high bands with small-k points getting too large values
            // we fill the bands with descending occ_number first
            for(int ik=0; ik<nk_nospin; ++ik)
            {
                for(int ib=0; ib<nbands; ++ib)
                {
                    // if( ik*nbands+ib < M )
                    // {
                    //     if( random_num[ik*nbands+ib] > (std::erf(2.0) + 1.0)/2.0 ) 
                    //     {
                    //         this->x[is][ik*nbands+ib] = 2.0;
                    //     }
                    //     else
                    //     {
                    //         this->x[is][ik*nbands+ib] = erf_inv_own( random_num[ik*nbands+ib] * 2.0 - 1.0 );
                    //     }
                    // }
                    // else
                    // {
                    //     this->x[is][ik*nbands+ib] = -2.0;
                    // }

                    if( ib*nk_nospin + ik < M )
                    {
                        if( random_num[ib*nk_nospin + ik] > (std::erf(2.0) + 1.0)/2.0 ) 
                        {
                            this->x[is][ik*nbands+ib] = 2.0;
                        }
                        else
                        {
                            this->x[is][ik*nbands+ib] = erf_inv_own( random_num[ib*nk_nospin + ik] * 2.0 - 1.0 );
                        }
                    }
                    else
                    {
                        this->x[is][ik*nbands+ib] = -2.0;
                    }



                    x_pass[ is*(nk_nospin*nbands) + ik*nbands + ib ] = this->x[is][ik*nbands+ib];
                }
            }
        }
        rdmft::printMatrix_pointer(nk_nospin, PARAM.inp.nbands, this->x[0].data(), "random inital var_x", 10);
        this->solving_mu();
        rdmft::printMatrix_pointer(nk_nospin, PARAM.inp.nbands, this->occ_number[0].data(), "random inital occ_number", 10);
    }
    // use the externally passed occ_number_in as the initial value
    else
    {
        // distinguishing up and down spin, 0 < occNum < 1
        ModuleBase::matrix num_temp( *occ_number_in );
        // if( PARAM.inp.nspin == 1 ) { num_temp *= 0.5; }

        // give an initial guess for mu, 0.0 or other value
        std::fill(mu.begin(), mu.end(), 0.01);

        // std::cout << "\n******\n" << "erf_inv_own(1.0 - 1e-12) = " << erf_inv_own(1.0 - 1e-12) << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "erf_inv_own(0.0 + 1e-6) = " << erf_inv_own(0.0 + 1e-6) << "\n******\n\n" << std::endl;
        // std::cout << "\n******\n" << "erf_inv_own(1.0 - 1e-16) = " << erf_inv_own(1.0 - 1e-16) << "\n******\n\n" << std::endl;

        // std::cout << "\n******\n" << "erf( erf_inv_own(1.0 - 1e-12) ) = " << std::erf( erf_inv_own(1.0 - 1e-12) ) << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "erf( erf_inv_own(1.0 - 1e-16) ) = " << std::erf( erf_inv_own(1.0 - 1e-16) ) << "\n******\n\n" << std::endl;

        // std::cout << "\n******\n" << "erf( 10 ) = " << std::erf( 10 ) << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "erf( -10 ) = " << std::erf( -10 ) << "\n******\n" << std::endl;
        // std::cout << "\n******\n" << "erf( -9 ) = " << std::erf( -9 ) << "\n******\n" << std::endl;


        // std::cout << "\n******\n" << "check: \n\n";
        // std::cout << "erf( erf_inv_own(0.9999) ) = " << std::erf( erf_inv_own(0.9999) ) << ", diff: " << std::erf( erf_inv_own(0.9999) ) - 0.9999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.8999) ) = " << std::erf( erf_inv_own(0.8999) ) << ", diff: " << std::erf( erf_inv_own(0.8999) ) - 0.8999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.7999) ) = " << std::erf( erf_inv_own(0.7999) ) << ", diff: " << std::erf( erf_inv_own(0.7999) ) - 0.7999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.6999) ) = " << std::erf( erf_inv_own(0.6999) ) << ", diff: " << std::erf( erf_inv_own(0.6999) ) - 0.6999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.5999) ) = " << std::erf( erf_inv_own(0.5999) ) << ", diff: " << std::erf( erf_inv_own(0.5999) ) - 0.5999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.6999) ) = " << std::erf( erf_inv_own(0.6372) ) << ", diff: " << std::erf( erf_inv_own(0.6372) ) - 0.6372 << std::endl;
        // std::cout << "erf( erf_inv_own(0.5999) ) = " << std::erf( erf_inv_own(0.5369) ) << ", diff: " << std::erf( erf_inv_own(0.5369) ) - 0.5369 << std::endl;
        // std::cout << "erf( erf_inv_own(0.4999) ) = " << std::erf( erf_inv_own(0.4999) ) << ", diff: " << std::erf( erf_inv_own(0.4999) ) - 0.4999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.3999) ) = " << std::erf( erf_inv_own(0.3999) ) << ", diff: " << std::erf( erf_inv_own(0.3999) ) - 0.3999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.2999) ) = " << std::erf( erf_inv_own(0.2999) ) << ", diff: " << std::erf( erf_inv_own(0.2999) ) - 0.2999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.1999) ) = " << std::erf( erf_inv_own(0.1999) ) << ", diff: " << std::erf( erf_inv_own(0.1999) ) - 0.1999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.0999) ) = " << std::erf( erf_inv_own(0.0999) ) << ", diff: " << std::erf( erf_inv_own(0.0999) ) - 0.0999 << std::endl;
        // std::cout << "erf( erf_inv_own(0.0099) ) = " << std::erf( erf_inv_own(0.0099) ) << ", diff: " << std::erf( erf_inv_own(0.0099) ) - 0.0099 << std::endl;
        // std::cout << "erf( erf_inv_own(0.0001) ) = " << std::erf( erf_inv_own(0.0001) ) << ", diff: " << std::erf( erf_inv_own(0.0001) ) - 0.0001 << std::endl;
        // std::cout << "erf( erf_inv_own(0.00001) ) = " << std::erf( erf_inv_own(0.00001) ) << ", diff: " << std::erf( erf_inv_own(0.00001) ) - 0.00001 << std::endl;
        // std::cout << "erf( erf_inv_own(1e-8) ) = " << std::erf( erf_inv_own(1e-8) ) << ", diff: " << std::erf( erf_inv_own(1e-8) ) - 1e-8 << "\n\n\n" << std::endl;

        // std::cout << "erf( erf_inv_own(-1.0+1e-8) ) = " << std::erf( erf_inv_own(-1.0+1e-8) ) << ", diff: " << std::erf( erf_inv_own(-1.0+1e-8) ) - (-1.0+1e-8) << std::endl;
        // std::cout << "erf( erf_inv_own(1.0+1e-18) ) = " << std::erf( erf_inv_own(1.0+1e-18) ) << ", diff: " << std::erf( erf_inv_own(1.0+1e-18) ) - (1.0+1e-18) << std::endl;
        // std::cout << "erf( erf_inv_own(1.0+1e-20) ) = " << std::erf( erf_inv_own(1.0+1e-20) ) << ", diff: " << std::erf( erf_inv_own(1.0+1e-20) ) - (1.0+1e-20) << std::endl;
        
        for(int is=0; is<PARAM.inp.nspin; ++is)
        {
            for(int ik=0; ik<nk_nospin; ++ik)
            {
                for(int ib=0; ib<PARAM.inp.nbands; ++ib)
                {
                    double num = num_temp( is*nk_nospin + ik, ib );

                    // find the correct and finite x to ensure that erf( x + mu[is] ) = 1.0 or 0.0
                    if( std::abs(num - 1.0) < 1e-16 )
                    {   
                        // min_occ_num = 1e-16, this->x[is][ik*nbands + ib] = 5.8;
                        num_temp( is*nk_nospin + ik, ib ) = 1.0 - 1e-16;
                    }
                    else if( std::abs(num - 0.0) < 1e-16 )
                    {
                        // min_occ_num = 1e-16, this->x[is][ik*nbands + ib] = -5.8;
                        num_temp( is*nk_nospin + ik, ib ) = 0.0 + 1e-16;
                    }

                    this->x[is][ik*nbands + ib] = erf_inv_own( 2 * num_temp( is*nk_nospin + ik, ib ) - 1 ) - this->mu[is];
                    this->occ_number[is][ik*nbands + ib] = num_temp( is*nk_nospin + ik, ib );
                    x_pass[ is*(nk_nospin*nbands) + ik*nbands + ib ] = this->x[is][ik*nbands+ib];    
                }
            }
        }

        double tot_occ_num_mow = this->cal_occ_num(0);
        std::cout << std::fixed << std::setprecision(16) << "mu by ks_occ_num: " << this->mu[0] <<", tot_occ_num_mow: " << tot_occ_num_mow <<  std::endl;

        // the strictness of the above method for the conservation of occupation number depends on the precision of own_erf_inv() (mu can be given arbitrarily)
        // after obtaining the appropriate x, solving_mu() can be used to make a small move of mu
        // and the strictness of the conservation of occupation number is consistent with that in solving_mu()
        this->solving_mu();
        tot_occ_num_mow = this->cal_occ_num(0);
        std::cout << "mu by ks_occ_num: " << this->mu[0] << ", tot_occ_num_mow: " << tot_occ_num_mow <<  std::endl << std::defaultfloat;

        // for(int ik=0; ik<num_temp.nr; ++ik)
        // {
        //     for(int ib=0; ib<num_temp.nc; ++ib)
        //     {
        //         if( ik < num_temp.nr/PARAM.inp.nspin )
        //         {
        //             x[0][ik*nbands + ib] = erf_inv_own( 2 * num_temp(ik, ib) - 1 ) - mu[0];
        //             occ_number[0][ik*nbands + ib] = num_temp(ik, ib);
        //             x_pass[ ik*nbands + ib ] = this->x[0][ik*nbands+ib];
        //         }
        //         else
        //         {
        //             x[PARAM.inp.nspin-1][ik*nbands + ib] = erf_inv_own( 2 * num_temp(ik, ib) - 1 ) - mu[PARAM.inp.nspin-1];
        //             occ_number[PARAM.inp.nspin-1][ik*nbands + ib] = num_temp(ik, ib);
        //             x_pass[ (PARAM.inp.nspin-1)*(nk_nospin*nbands) + ik*nbands + ib ] = this->x[PARAM.inp.nspin-1][ik*nbands+ib];
        //         }
        //     }
        // }
        rdmft::printMatrix_pointer(num_temp.nr, num_temp.nc, occ_number_in->c, "occ_number_from_ks", 10);
        rdmft::printMatrix_pointer(num_temp.nr, num_temp.nc, this->x[0].data(), "var_x from ks_occ_num", 10);
    }
    std::cout << "\n******\n" << "start_guess: ebi, 1.0" << "\n******\n" << std::endl;
}



void EBI::update_x_occ_num(const std::vector<double>& x_in)
{
    std::cout << "\n" << "Enter ebi.update_x_occ_num()" << "\n" << std::endl;
    // update member variable x from external x_in
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        for(int ik=0; ik<nk_nospin; ++ik)
        {
            for(int ib=0; ib<nbands; ++ib)
            {
                this->x[is][ik*nbands+ib] = x_in[ is*(nk_nospin*nbands) + ik*nbands +ib ];
            }
        }
    }

    std::cout << "\n" << "before solving_mu()" << "\n" << std::endl;
    // get the new mu and occ_number
    this->solving_mu();
    // return this->get_occ_number();  

    rdmft::printMatrix_pointer(nk_nospin*PARAM.inp.nspin, nbands, x_in.data(), "trial_x", 10);
}


void EBI::get_dE_dx(const std::vector<double>& dE_docc_num, std::vector<double>& dE_dx)
{
    const int N = nk_nospin*nbands;
    // const double factor =  PARAM.inp.nspin==1 ? 2.0 : 1.0;
    // const double factor =  PARAM.inp.nspin==1 ? 0.5 : 1.0;   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    const double factor = 1.0;
    std::vector< std::vector<double> > dE_deta(PARAM.inp.nspin, std::vector<double>(N, factor));

    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        // convert format. consider spin up and spin down separately
        for(int j=0; j<N; ++j) { dE_deta[is][j] *= dE_docc_num[is*N + j]; }

        std::vector<double> dmu_dx(N, 0.0);
        std::vector<double> docc_num_dx(N * N, 0.0);

        this->cal_dmu_dx(dmu_dx, is);
        this->cal_docc_num_dx(dmu_dx, docc_num_dx, is);

        // rdmft::dgemm_lapack( docc_num_dx.data(), dE_deta[is].data(), (dE_dx.data() + is*N), N, 1, N );
        rdmft::dgemm_lapack( docc_num_dx.data(), dE_deta[is].data(), (dE_dx.data() + is*N), N, 1, N );
    }
}




void EBI::solving_mu()
{
    std::cout << "\n" << "Enter solving_mu()" << "\n" << std::endl;
    for(int is=0; is<PARAM.inp.nspin; ++is)
    {
        /********* is there an error in the paper formula? mu should be updated at each step *********/
        // double mu_temp = 0.0;
        // std::vector<double> f_der(2, 1.0);

        // while( f_der[0] > this->solve_mu_thr )
        // {
        //     f_der = this->cal_f_der(this->mu[is], is);
        //     double f1_divided_f2 = std::abs( f_der[0]/f_der[1] );
        //     double sign = 0.0;
        //     if( f_der[0]>0 ) { sign = 1.0; }
        //     else if ( f_der[0]<0 ) { sign = -1.0; }

        //     if( f1_divided_f2 > 1.0 )
        //     {
        //         mu_temp -= sign;
        //     }
        //     else
        //     {
        //         std::vector<double> f_der_temp = this->cal_f_der(mu_temp, is);
        //         mu_temp -= sign * std::abs( f_der_temp[0]/f_der_temp[1] );
        //     }
        // }
        // this->mu[is] = mu_temp;
        /********* is there an error in the paper formula? *********/

        // this->mu[is] = 0.0; // Is it possible to consider using the last result of mu instead of 0.0 as the initial value?
        std::vector<double> f_der(2, 1.0);
        double occ_num_error = 1.0;

        // while( f_der[0] > this->solve_mu_thr )
        int solve_mu_times = 0; 
        while( std::abs(f_der[0]) > this->solve_mu_thr || std::abs(occ_num_error) > this->tot_nelec_thr )
        {
            ++solve_mu_times;
            if( solve_mu_times > 100 )
            {
                std::cout << "\n" << "solve_mu_times is too big: " << solve_mu_times << "\n" << std::endl;
                std::cout << "\n" << "electron number is not conserved !!!!!!!!!!! " << "\n" << std::endl;
                assert(solve_mu_times <= 100);
                break;
            }
            
            std::cout << "\n" << "in solving_mu(), while()" << ", solve_mu_times: " << solve_mu_times << "\n" << std::endl;
            f_der = this->cal_f_der(this->mu[is], is);
            double f1_divided_f2 = std::abs( f_der[0]/f_der[1] );

            std::cout << "\nis: " << is << ", mu: " << this->mu[is] << ", f_der1: " << f_der[0] << ", f_der2: " << f_der[1] 
                        << ", f1_divided_f2: " << f1_divided_f2 << "\n" << std::endl;

            double sign = 0.0;
            if( f_der[0]>0 )
            {
                sign = 1.0;
            }
            else if( f_der[0]<0 )
            {
                sign = -1.0;
            }

            if( f1_divided_f2 > 1.0 )
            {
                this->mu[is] -= sign;
            }
            else
            {
                this->mu[is] -= sign * f1_divided_f2;
            }

            if( std::abs(f_der[0]) < this->solve_mu_thr )
            {
                std::cout << "\n" << "before cal_occ_num()" << "\n" << std::endl;
                occ_num_error = this->cal_occ_num(is) - this->sys_nelec_spin[is];

                // determine whether it converges to a local minimum
                if( std::abs(occ_num_error) > this->tot_nelec_thr )
                {
                    // solve_mu_times = 0;
                    // double sum_x = std::accumulate(this->x[is].begin(), this->x[is].end(), 0.0);
                    // this->mu[is] = ( rdmft::erf_inv_own( 2*this->sys_nelec_spin[is] - nk_nospin*nbands ) - sum_x )/nk_nospin*nbands;
                    std::cout << "******\n" << "local minimum mu: " << this->mu[is] << ", occ_num_error: " << occ_num_error << std::endl;

                    rdmft::printMatrix_pointer(nk_nospin, nbands, this->x[is].data(), "now var_x", 10);

                    rdmft::printMatrix_pointer(nk_nospin, nbands, this->occ_number[is].data(), "now occ_number", 10);

                    std::cout << "******\n" << std::endl;
                    
                    // a more appropriate step size should be used for mu
                    // if(occ_num_error > 0)
                    // {
                    //     this->mu[is] -= 1.0;
                    // }
                    // else
                    // {
                    //     this->mu[is] += 1.0;
                    // }
                    if( std::abs(occ_num_error) >= 0.5 && f1_divided_f2 < 0.5 )
                    {
                        double step = (occ_num_error > 0) ? 1.0 : -1.0;
                        this->mu[is] -= step;
                    }
                    else
                    {
                        // use the step length and direction of the previous step to calculate one more step
                        // this->mu[is] -= sign * f1_divided_f2;
                        
                        // do the calculation again
                        continue;
                    }

                }
            }
            // if( std::abs(f1_divided_f2) - 1 > 0 )
            // {
            //     std::cout << "\n" << "solving_mu, f1_divided_f2: " << f1_divided_f2 << "\n" << std::endl;
            // }
        }

        std::cout << "******\n" << "solving_mu, mu[" << is << "]: " << this->mu[is] << "\n******" << std::endl;
    }

    // this->cal_occ_num();

    //test
    ModuleBase::matrix print_occ( this->get_occ_number() );
    rdmft::printMatrix_pointer(print_occ.nr, print_occ.nc, print_occ.c, "occ_number_after_ebi", 10);

    double trial_occ_num = 0.0;
    for(int i=0; i<print_occ.nr*print_occ.nc; ++i)
    {
        trial_occ_num += print_occ.c[i];
    }
    std::cout << "\n total_trial_occ_num: " <<  trial_occ_num  << "\n" << std::endl;
}

ModuleBase::matrix EBI::get_occ_number()
{
    ModuleBase::matrix occ_num_pass(nk_nospin*PARAM.inp.nspin, nbands);

    for(int ik=0; ik<occ_num_pass.nr; ++ik)
    {
        for(int ib=0; ib<occ_num_pass.nc; ++ib)
        {
            if( ik < occ_num_pass.nr/PARAM.inp.nspin )
            {
                occ_num_pass(ik, ib) = this->occ_number[0][ik*nbands + ib];
            }
            else
            {
                occ_num_pass(ik, ib) = this->occ_number[PARAM.inp.nspin-1][ik*nbands + ib];
            }
        }
    }

    // if(PARAM.inp.nspin == 1) occ_num_pass *= 2;

    return occ_num_pass;
}


// std::vector< std::vector<double> > EBI::cal_occ_num()
// {
//     std::vector< std::vector<double> > sum(PARAM.inp.nspin, std::vector<double>(3, 0.0));
//     for(int is=0; is<PARAM.inp.nspin; ++is)
//     {
//         for(int i=0; i<this->x[is].size(); ++i)
//         {
//             this->occ_number[is][i] = ( std::erf(this->x[is][i] + this->mu[is]) + 1.0 )/2.0;

//             sum[is][0] += this->occ_number[is][i];
//             sum[is][1] += erf_der1(this->x[is][i] + this->mu[is]);
//             sum[is][2] += erf_der2(this->x[is][i] + this->mu[is]);
//         }
//     }
//     return sum;
// }

double EBI::cal_occ_num(int is)
{
    double tot_occ_num = 0.0;
    // for(int i=0; i<this->x[is].size(); ++i)
    // {
    //     this->occ_number[is][i] = ( std::erf(this->x[is][i] + this->mu[is]) + 1.0 )/2.0;
    //     occ_num_now += this->occ_number[is][i];
    // }

    for(int ik=0; ik<nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            
            this->occ_number[is][ik*nbands + ib] = ( std::erf(this->x[is][ik*nbands + ib] + this->mu[is]) + 1.0 )/2.0;
            tot_occ_num += this->occ_number[is][ik*nbands + ib] * this->num_symm_k[ik];
        }
    }

    return tot_occ_num;
}


std::vector<double> EBI::cal_f_der(double mu_in, int is)
{
    std::vector<double> f_der(2, 0.0);

    // std::vector<double> sum = this->cal_sum(mu_in, is);
    std::vector<double> sum(3, 0.0);
    // for(int i=0; i<this->x[is].size(); ++i)
    // {
    //     sum[0] += ( std::erf(this->x[is][i] + mu_in) + 1.0 )/2.0;
    //     sum[1] += erf_der1(this->x[is][i] + mu_in);
    //     sum[2] += erf_der2(this->x[is][i] + mu_in);
    // }

    for(int ik=0; ik<nk_nospin; ++ik)
    {
        for(int ib=0; ib<PARAM.inp.nbands; ++ib)
        {
            sum[0] += ( std::erf(this->x[is][ik*nbands + ib] + mu_in) + 1.0 ) * 0.5 * this->num_symm_k[ik];
            sum[1] += erf_der1(this->x[is][ik*nbands + ib] + mu_in) * this->num_symm_k[ik];
            sum[2] += erf_der2(this->x[is][ik*nbands + ib] + mu_in) * this->num_symm_k[ik];
        }
    }



    std::cout << "\n" << "in cal_f_der(): " << std::endl;
    std::cout << "sum[0]: " << sum[0] << ", sum[1]: " << sum[1] << ", sum[2]: " << sum[2] << "\n" << std::endl;

    f_der[0] = ( sum[0] - this->sys_nelec_spin[is] ) * sum[1];
    f_der[1] = 0.5*std::pow(sum[1], 2) + ( sum[0] - this->sys_nelec_spin[is] ) * sum[2];

    return f_der;
}


// std::vector<double> EBI::cal_sum(double mu_in, int is)
// {
//     std::vector<double> sum(3, 0.0);
//     for(int i=0; i<this->x[is].size(); ++i)
//     {
//         sum[0] += ( std::erf(this->x[is][i] + mu_in) + 1.0 )/2.0;
//         sum[1] += erf_der1(this->x[is][i] + mu_in);
//         sum[2] += erf_der2(this->x[is][i] + mu_in);
//     }
//     return sum;
// }


void EBI::cal_dmu_dx(std::vector<double>& dmu_dx, int is)
{
    double sum_der1 = 0.0;
    for(int i=0; i<this->x[is].size(); ++i) { sum_der1 += erf_der1(this->x[is][i] + this->mu[is]); }

    for(int j=0; j<dmu_dx.size(); ++j) { dmu_dx[j] = - erf_der1(this->x[is][j] + this->mu[is]) / sum_der1; }
}


void EBI::cal_docc_num_dx(const std::vector<double>& dmu_dx, std::vector<double>& docc_num_dx, int is)
{
    const int N = nk_nospin*nbands;
    for(int i=0; i<N; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            if( i==j )
            {
                docc_num_dx[i*N+j] = 0.5 * erf_der1(this->x[is][i] + this->mu[is]) * (1 + dmu_dx[i]);
            }
            else
            {
                docc_num_dx[i*N+j] = 0.5 * erf_der1(this->x[is][i] + this->mu[is]) * dmu_dx[j];
            }
        }
    }
}






}

