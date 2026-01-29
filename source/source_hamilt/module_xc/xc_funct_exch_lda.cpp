// This file contains realization of LDA exchange functionals
// Spin unpolarized ones:
//  1. slater: ordinary Slater exchange with alpha=2/3
//  2. slater1: Slater exchange with alpha=1
//  3. slater_rxc : Slater exchange with alpha=2/3 and Relativistic exchange
// And their spin polarized counterparts:
//  1. slater_spin
//  2. slater1_spin
//  3. slater_rxc_spin

#include "xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/constants.h"

//Slater exchange with alpha=2/3
void XC_Functional::slater(const double &rs, double &ex, double &vx)
{
	// f = -9/8*(3/2pi)^(2/3)
	const double f = -0.687247939924714e0;
	const double alpha = 2.00 / 3.00;
	ex = f * alpha / rs;
	vx = 4.0 / 3.0 * f * alpha / rs;
	return;
}

//Slater exchange with alpha=1, corresponding to -1.374/r_s Ry
//used to recover old results
void XC_Functional::slater1(const double &rs, double &ex, double &vx)
{
	const double f = -0.687247939924714e0;
	const double alpha = 1.0;
	ex = f * alpha / rs;
	vx = 4.0 / 3.0 * f * alpha / rs;
	return;
}

// Slater exchange with alpha=2/3 and Relativistic exchange
void XC_Functional::slater_rxc(const double &rs, double &ex, double &vx)
{
    const double trd = 1.0 / 3.0;
    //const double ftrd = 4.0 / 3.0;
    //const double tftm = pow(2.0, ftrd) - 2.0;
    const double a0 = pow((4.0 / (9.0 * ModuleBase::PI)), trd);
    // X-alpha parameter:
    const double alp = 2 * trd;

    double vxp = -3 * alp / (2 * ModuleBase::PI * a0 * rs);
    double exp = 3 * vxp / 4;
    const double beta = 0.014 / rs;
    const double sb = sqrt(1 + beta * beta);
    const double alb = log(beta + sb);
    vxp = vxp * (-0.5 + 1.5 * alb / (beta * sb));
    double x = (beta * sb - alb) / (beta * beta);
    exp = exp * (1.0 - 1.5 * x * x);
    vx = vxp;
    ex = exp;
    return;
}

// Slater exchange with alpha=2/3, spin-polarized case
void XC_Functional::slater_spin( const double &rho, const double &zeta, 
		double &ex, double &vxup, double &vxdw)
{
    const double f = - 1.107838149573033610;
    const double alpha = 2.00 / 3.00;
    // f = -9/8*(3/pi)^(1/3)
    const double third = 1.0 / 3.0;
    const double p43 = 4.0 / 3.0;

    double rho13 = pow(((1.0 + zeta) * rho) , third);
    double exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow(((1.0 - zeta) * rho) , third);
    double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.50 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);

    return;
}

// Slater exchange with alpha=2/3, spin-polarized case
void XC_Functional::slater1_spin( const double &rho, const double &zeta, double &ex, double &vxup, double &vxdw)
{
    const double f = - 1.107838149573033610;
	const double alpha = 1.00;
	const double third = 1.0 / 3.0;
	const double p43 = 4.0 / 3.0;
    // f = -9/8*(3/pi)^(1/3)

    double rho13 = pow(((1.0 + zeta) * rho) , third);
    double exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow(((1.0 - zeta) * rho) , third);
    double exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.50 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);

    return;
} // end subroutine slater1_spin

// Slater exchange with alpha=2/3, relativistic exchange case
void XC_Functional::slater_rxc_spin( const double &rho, const double &z, 
		double &ex, double &vxup, double &vxdw)
{
    if (rho <= 0.0)
    {
        ex = vxup = vxdw = 0.0;
        return;
    }

    const double trd = 1.0 / 3.0;
	const double ftrd = 4.0 / 3.0;
	double tftm = pow(2.0, ftrd) - 2;
    double a0 = pow((4 / (9 * ModuleBase::PI)), trd);

    double alp = 2 * trd;

	double fz = (pow((1 + z), ftrd) + pow((1 - z), ftrd) - 2) / tftm;
	double fzp = ftrd * (pow((1 + z), trd) - pow((1 - z), trd)) / tftm;

    double rs = pow((3 / (4 * ModuleBase::PI * rho)), trd);
    double vxp = -3 * alp / (2 * ModuleBase::PI * a0 * rs);
    double exp = 3 * vxp / 4;

    double beta = 0.014 / rs;
    double sb = sqrt(1 + beta * beta);
    double alb = log(beta + sb);
    vxp = vxp * (-0.5 + 1.5 * alb / (beta * sb));
    exp = exp * (1.0- 1.5*( (beta*sb-alb) / (beta*beta) )
			        * 1.5*( (beta*sb-alb) / (beta*beta) ));

    double x = (beta * sb - alb) / (beta * beta);

    exp = exp * (1.0 - 1.5 * x * x);

    double vxf = pow(2.0, trd) * vxp;

    double exf = pow(2.0, trd) * exp;

    vxup  = vxp + fz * (vxf - vxp) + (1 - z) * fzp * (exf - exp);

    vxdw  = vxp + fz * (vxf - vxp) - (1 + z) * fzp * (exf - exp);

    ex    = exp + fz * (exf - exp);

    return;
}


// E_theta functionals of TAO-DFT at LDA level
void XC_Functional::lda_theta(const double &rho, double &ex, double &vx)
{
    // theta (atomic units)
    const double theta = PARAM.inp.smearing_sigma;

    // constants
    const double CF   = 0.3 * std::pow(3.0 * ModuleBase::PI * ModuleBase::PI, 2.0 / 3.0);
    const double pref = (ModuleBase::PI * ModuleBase::PI) / std::sqrt(2.0);
    const double y0   = 3.0 * ModuleBase::PI / (4.0 * std::sqrt(2.0));

    // y and u
    const double y = pref * rho * std::pow(theta, -1.5);
    const double u = std::pow(y, 2.0 / 3.0);

    double f    = 0.0;
    double dfdy = 0.0;

    if (y <= y0)
    {
        // -------- small-y branch --------
        const std::vector<double> coef1 = {
            -0.8791880215,        // y^0
             0.1989718742,        // y^1
             0.001068697043,      // y^2
            -0.008812685726,      // y^3
             0.01272183027,       // y^4
            -0.009772758583,      // y^5
             0.003820630477,      // y^6
            -0.0005971217041      // y^7
        };

        // f(y)
        f = std::log(y);
        for (int n = 0; n <= 7; ++n)
        {
            f += coef1[n] * std::pow(y, n);
        }

        // f'(y)
        dfdy = 1.0 / y;
        for (int n = 1; n <= 7; ++n)
        {
            dfdy += n * coef1[n] * std::pow(y, n - 1);
        }

    }
    else
    {
        // -------- large-y branch --------
        const std::vector<double> coef2 = {
             0.7862224183,    // u^1
            -1.882979454,     // u^{-1}
             0.5321952681,    // u^{-3}
             2.304457955,     // u^{-5}
           -16.14280772,      // u^{-7}
            52.28431386,      // u^{-9}
           -95.92645619,      // u^{-11}
            94.62230172,      // u^{-13}
           -38.93753937       // u^{-15}
        };

        const std::vector<int> powers = {
             1, -1, -3, -5, -7, -9, -11, -13, -15
        };

        // f(y)
        for (size_t i = 0; i < coef2.size(); ++i)
        {
            f += coef2[i] * std::pow(u, powers[i]);
        }

        // df/du
        double dfdu = 0.0;
        for (size_t i = 0; i < coef2.size(); ++i)
        {
            dfdu += coef2[i] * powers[i] * std::pow(u, powers[i] - 1);
        }

        // df/dy = (df/du)*(du/dy), du/dy = (2/3) y^{-1/3}
        dfdy = dfdu * (2.0 / 3.0) * std::pow(y, -1.0 / 3.0);
    }

    // per-particle energy density
    ex = CF * std::pow(rho, 2.0 / 3.0) - theta * f;

    // functional derivative
    vx = (5.0 / 3.0) * CF * std::pow(rho, 2.0 / 3.0)
         - theta * (f + y * dfdy);

    return;
}


// LDA-theta, spin-polarized case
// rho : total density = rho_up + rho_down
// zeta = (rho_up - rho_down) / rho   (仅用于构造 rho_up / rho_down)
// ex   : per-particle energy
// vxup, vxdw : spin-dependent potentials

void XC_Functional::lda_theta_spin(const double &rho,
                                   const double &zeta,
                                   double &ex,
                                   double &vxup,
                                   double &vxdw)
{
    // spin densities
    const double rho_up = 0.5 * rho * (1.0 + zeta);
    const double rho_dw = 0.5 * rho * (1.0 - zeta);

    double ex_up = 0.0;
    double ex_dw = 0.0;

    // unpolarized functional evaluated at 2*rho_sigma
    if (rho_up > 0.0)
    {
        lda_theta(2.0 * rho_up, ex_up, vxup);
    }

    if (rho_dw > 0.0)
    {
        lda_theta(2.0 * rho_dw, ex_dw, vxdw);
    }

    // total per-particle energy
    ex = 0.5 * ( ex_up + ex_dw );

    return;
}

