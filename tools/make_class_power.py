"""This module creates a matter power spectrum using classy,
a python interface to the CLASS Boltzmann code.

See:
http://class-code.net/
It parses an MP-GenIC parameter file and generates a matter power spectrum
and transfer function at the initial redshift. It generates a second transfer
function at a slightly lower redshift to allow computing the growth function.
Files are saved where MP-GenIC expects to read them. Existing files will not be over-written.

Script should be compatible with python 2.7 and 3.

Cite CLASS paper:
 D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933 [astro-ph.CO], JCAP 1107 (2011) 034
Call with:
    python make_class_power.py <MP-GenIC parameter file> <external power spectrum file>
    where the second external power spectrum file is optional and is a primordial power spectrum for CLASS."""

from __future__ import print_function
import math
import os.path
import argparse
import numpy as np
from classy import Class
import configobj
import validate

GenICconfigspec = """
FileWithInputSpectrum = string(default='')
FileWithTransferFunction = string(default='')
Ngrid = integer(min=0)
BoxSize = float(min=0)
Omega0 = float(0,1)
OmegaLambda = float(0,1)
OmegaBaryon = float(0,1,default=0.0486)
HubbleParam = float(0,2)
Redshift = float(0,1100)
Sigma8 = float(default=-1)
InputPowerRedshift = float(default=-1)
DifferentTransferFunctions = integer(0,1, default=1)
UnitLength_in_cm  = float(default=3.085678e21)
Omega_fld = float(0,1,default=0)
w0_fld = float(default=-1)
wa_fld = float(default=0)
Omega_ur = float(default=0)
MNue = float(min=0, default=0)
MNum = float(min=0, default=0)
MNut = float(min=0, default=0)
MWDM_Therm = float(min=0, default=0)
PrimordialIndex = float(default=0.971)
PrimordialAmp = float(default=2.215e-9)
PrimordialRunning = float(default=0)
CMBTemperature = float(default=2.7255)""".split('\n')

# note: Omega_ur should be set to a non-zero value if both massive and massless neutrinos are present, e.g., if N_nu_massive  = 3 (N_ur_massive = 3.0396) and the default value N_eff is desired, then N_ur_massless = 3.044 - 3.0396 = 0.0044 (Omega_ur = Omega_g * 0.22710731766023898 * (3.044 - 3.0396))

def _check_genic_config(config):
    """Check that the MP-GenIC config file is sensible for running CLASS on."""
    vtor = validate.Validator()
    config.validate(vtor)
    filekeys = ['FileWithInputSpectrum', ]
    if config['DifferentTransferFunctions'] == 1.:
        filekeys += ['FileWithTransferFunction',]
    for ff in filekeys:
        if config[ff] == '':
            raise IOError("No savefile specified for ",ff)
        if os.path.exists(config[ff]):
            raise IOError("Refusing to write to existing file: ",config[ff])

    #Check unsupported configurations
    if config['MWDM_Therm'] > 0:
        raise ValueError("Warm dark matter power spectrum cutoff not yet supported.")
    if config['DifferentTransferFunctions'] == 1.:
        if config['InputPowerRedshift'] >= 0:
            raise ValueError("Rescaling with different transfer functions not supported.")

def _build_cosmology_params(config, convention='CLASS'):
    """Build a correctly-named-for-class set of cosmology parameters from the MP-GenIC config file."""
    #Class takes omega_m h^2 as parameters
    h0 = config['HubbleParam']
    omeganu = (config['MNue'] + config['MNum'] + config['MNut'])/93.14/h0**2
    if config['OmegaBaryon'] < 0.001:
        config['OmegaBaryon'] = 0.0486
    ocdm = config['Omega0'] - config['OmegaBaryon'] - omeganu

    T_CMB = config["CMBTemperature"]  # default cmb temperature
    Omega_g = 4.480075654158969e-07 * T_CMB**4 / config['HubbleParam']**2

    if convention == 'CLASS':
        omegak = 1-config['OmegaLambda']-config['Omega_fld']-config['Omega0'] - Omega_g - config['Omega_ur']
    elif convention == 'CAMB':
        omegak = 1-config['OmegaLambda']-config['Omega_fld']-config['Omega0']

    if np.abs(omegak) > 1e-5:
        print("Curvature present: Omega_K = %g" % omegak)
    # avoid numerical issue due to very small OmegaK
    if np.abs(omegak) < 1e-9:
        omegak = 0

    gparams = {'h':config['HubbleParam'], 'Omega_cdm':ocdm,'Omega_b':config['OmegaBaryon'], 'Omega_k':omegak, 'n_s': config['PrimordialIndex'], 'alpha_s': config['PrimordialRunning'],'T_cmb':config["CMBTemperature"]}
    #One may specify either OmegaLambda or Omega_fld,
    #and the other is worked out by summing all matter to unity.
    #Specify Omega_fld even if we have Lambda, to avoid floating point.
    gparams['Omega_fld'] = config['Omega_fld']
    if config['Omega_fld'] > 0:
        gparams['w0_fld'] = config['w0_fld']
        gparams['wa_fld'] = config['wa_fld']
    #Set up massive neutrinos
    N_ncdm = 0
    m_ncdm = ''
    if omeganu > 0:
        # gparams['m_ncdm'] = '%.8f,%.8f,%.8f' % (config['MNue'], config['MNum'], config['MNut'])
        # gparams['N_ncdm'] = 3  # this is not consistent with CAMB when some of the neutrinos are massless
        # loop over neutrinos to avoid the above inconsistency
        for mnu in [config['MNue'], config['MNum'], config['MNut']]:
            if mnu == 0:
                continue
            if N_ncdm == 0:
                m_ncdm += '%.8f' % mnu
            else:
                m_ncdm += ',%.8f' % mnu 
            N_ncdm += 1 
        gparams['m_ncdm'] = m_ncdm
        gparams['N_ncdm'] = N_ncdm 

        N_ur = config['Omega_ur'] / Omega_g / 0.22710731766023898 + 1.013198221453432* (3 - N_ncdm)
        gparams['N_ur'] = N_ur # the part beyond massive neutrinos
        #Neutrino accuracy: Default pk_ref.pre has tol_ncdm_* = 1e-10,
        #which takes 45 minutes (!) on my laptop.
        #tol_ncdm_* = 1e-8 takes 20 minutes and is machine-accurate.
        #Default parameters are fast but off by 2%.
        #I chose 1e-4, which takes 20 minutes and is accurate to 1e-4
        gparams['tol_ncdm_newtonian'] = 1e-4
        gparams['tol_ncdm_synchronous'] = 1e-4
        gparams['tol_ncdm_bg'] = 1e-10
        gparams['l_max_ncdm'] = 50
        #For accurate, but very slow, P_nu, set ncdm_fluid_approximation = 3
        #CAMB does this better.
        gparams['ncdm_fluid_approximation'] = 2
        gparams['ncdm_fluid_trigger_tau_over_tau_k'] = 10000.
    else:
        gparams['N_ur'] = 3.044
    #Power spectrum amplitude: sigma8 is ignored by classy.
    if config['Sigma8'] > 0:
        print("Warning: classy does not read sigma8. GenIC must rescale P(k).")
    gparams['A_s'] = config["PrimordialAmp"]
    return gparams

def make_class_power(paramfile, external_pk = None, extraz=None, verbose=False, convention='CLASS'):
    """Main routine: parses a parameter file and makes a matter power spectrum.
    Will not over-write power spectra if already present.
    Options are loaded from the MP-GenIC parameter file.
    Supported:
        - Omega_fld and DE parameters.
        - Massive neutrinos.
        - Using Sigma8 to set the power spectrum scale.
        - Different transfer functions.

    We use class velocity transfer functions to have accurate initial conditions
    even on superhorizon scales, and to properly support multiple species.
    The alternative is to use rescaling.

    Not supported:
        - Warm dark matter power spectra.
        - Rescaling with different transfer functions."""
    config = configobj.ConfigObj(infile=paramfile, configspec=GenICconfigspec, file_error=True)
    #Input sanitisation
    _check_genic_config(config)

    #Precision
    pre_params = {'k_per_decade_for_pk': 50, 'k_bao_width': 8, 'k_per_decade_for_bao':  200, 'neglect_CMB_sources_below_visibility' : 1.e-30, 'transfer_neglect_late_source': 3000., 'l_max_g' : 50, 'l_max_ur':150}

    #Important! Densities are in synchronous gauge!
    pre_params['gauge'] = 'synchronous'

    gparams = _build_cosmology_params(config, convention)
    pre_params.update(gparams)
    redshift = config['Redshift']
    if config['InputPowerRedshift'] >= 0:
        redshift = config['InputPowerRedshift']
    outputs = redshift
    if extraz is not None:
        outputs = [outputs,]+ extraz
        strout = ", ".join([str(o) for o in outputs])
    else:
        strout = str(outputs)
    #Pass options for the power spectrum
    MPC_in_cm = 3.085678e24
    boxmpc = config['BoxSize'] / MPC_in_cm * config['UnitLength_in_cm']
    maxk = max(10, 2*math.pi/boxmpc*config['Ngrid']*4)
    #CLASS needs the first redshift to be relatively high for some internal interpolation reasons
    maxz = max(1 + np.max(outputs), 99)
    powerparams = {'output': 'dTk vTk mPk', 'P_k_max_h/Mpc' : maxk, "z_max_pk" : maxz,'z_pk': strout, 'extra metric transfer functions': 'y'}
    pre_params.update(powerparams)

    if verbose:
        verb_params = {'input_verbose': 1, 'background_verbose': 1, 'thermodynamics_verbose': 1, 'perturbations_verbose': 1, 'transfer_verbose': 1, 'primordial_verbose': 1, 'lensing_verbose': 1, 'output_verbose': 1}
        pre_params.update(verb_params)

    #Specify an external primordial power spectrum
    if external_pk is not None:
        pre_params['P_k_ini'] = "external_pk"
        pre_params["command"] = "cat ",external_pk
    #Print the class parameters to terminal in a format
    #readable by the command line class.
    if verbose:
        for k in pre_params:
            print(k, '=', pre_params[k])

    if 'ncdm_fluid_approximation' in pre_params:
        print('Starting CLASS power spectrum with accurate P(k) for massive neutrinos.')
        print('Computation may take several minutes')
    #Save directory
    sdir = os.path.split(paramfile)[0]
    if config['DifferentTransferFunctions'] == 1.:
        tfile = os.path.join(sdir, config['FileWithTransferFunction'])
        if os.path.exists(tfile):
            raise IOError("Refusing to write to existing file: ",tfile)
    pkfile = os.path.join(sdir, config['FileWithInputSpectrum'])
    if os.path.exists(pkfile):
        raise IOError("Refusing to write to existing file: ",pkfile)

    #Make the power spectra module
    powspec = Class()
    powspec.set(pre_params)
    powspec.compute()
    print("sigma_8(z=0) = ", powspec.sigma8(), "A_s = ",pre_params["A_s"])
    #Get and save the transfer functions if needed
    trans = powspec.get_transfer(z=redshift)
    if config['DifferentTransferFunctions'] == 1.:
        tfile = os.path.join(sdir, config['FileWithTransferFunction'])
        if os.path.exists(tfile):
            raise IOError("Refusing to write to existing file: ",tfile)
        save_transfer(trans, tfile)
    #fp-roundoff
    khmpc = trans['k (h/Mpc)']
    khmpc[-1] *= 0.9999
    #Note pk lin has no h unit! But the file we want to save should have it.
    kmpc = khmpc * pre_params['h']
    #Get and save the matter power spectrum. We want (Mpc/h)^3 units but the default is Mpc^3.
    pk_lin = np.array([powspec.pk_lin(k=kk, z=redshift) for kk in kmpc])*pre_params['h']**3
    pkfile = os.path.join(sdir, config['FileWithInputSpectrum'])
    if os.path.exists(pkfile):
        raise IOError("Refusing to write to existing file: ",pkfile)
    np.savetxt(pkfile, np.vstack([khmpc, pk_lin]).T)
    if extraz is not None:
        for red in extraz:
            trans = powspec.get_transfer(z=red)
            tfile = os.path.join(sdir, config['FileWithTransferFunction']+"-"+str(red))
            if os.path.exists(tfile):
                raise IOError("Refusing to write to existing file: ",tfile)
            save_transfer(trans, tfile)
            #Get and save the matter power spectrum
            pk_lin_z = np.array([powspec.pk_lin(k=kk, z=red) for kk in kmpc])*pre_params['h']**3
            pkfile_z = os.path.join(sdir, config['FileWithInputSpectrum']+"-"+str(red))
            if os.path.exists(pkfile_z):
                raise IOError("Refusing to write to existing file: ",pkfile_z)
            np.savetxt(pkfile_z, np.vstack([khmpc, pk_lin_z]).T)

def save_transfer(transfer, transferfile):
    """Save a transfer function. Note we save the CLASS FORMATTED transfer functions.
    The transfer functions differ from CAMB by:
        T_CAMB(k) = -T_CLASS(k)/k^2 """
    header="""Transfer functions T_i(k) for adiabatic (AD) mode (normalized to initial curvature=1)
d_i   stands for (delta rho_i/rho_i)(k,z) with above normalization
d_tot stands for (delta rho_tot/rho_tot)(k,z) with rho_Lambda NOT included in rho_tot
(note that this differs from the transfer function output from CAMB/CMBFAST, which gives the same
 quantities divided by -k^2 with k in Mpc^-1; use format=camb to match CAMB)
t_i   stands for theta_i(k,z) with above normalization
t_tot stands for (sum_i [rho_i+p_i] theta_i)/(sum_i [rho_i+p_i]))(k,z)
If some neutrino species are massless, or degenerate, the d_ncdm and t_ncdm columns may be missing below.
1:k (h/Mpc)              2:d_g                    3:d_b                    4:d_cdm                  5:d_ur        6:d_ncdm[0]              7:d_ncdm[1]              8:d_ncdm[2]              9:d_tot                 10:phi     11:psi                   12:h                     13:h_prime               14:eta                   15:eta_prime     16:t_g                   17:t_b                   18:t_ur        19:t_ncdm[0]             20:t_ncdm[1]             21:t_ncdm[2]             22:t_tot"""
    #This format matches the default output by CLASS command line.
    if "d_ncdm[0]" in transfer.keys():
        wanted_trans_keys = ['k (h/Mpc)', 'd_g', 'd_b', 'd_cdm', 'd_ur', "d_ncdm[0]", "d_ncdm[1]", "d_ncdm[2]", 'd_tot', 'phi', 'psi', 'h', 'h_prime', 'eta', 'eta_prime', 't_g', 't_b', 't_ur', 't_ncdm[0]', 't_ncdm[1]', 't_ncdm[2]', 't_tot']
    else:
        wanted_trans_keys = ['k (h/Mpc)', 'd_g', 'd_b', 'd_cdm', 'd_ur', 'd_tot', 'phi', 'psi', 'h', 'h_prime', 'eta', 'eta_prime', 't_g', 't_b', 't_ur', 't_tot']
    transferarr = np.vstack([transfer[kk] for kk in wanted_trans_keys]).T
    np.savetxt(transferfile, transferarr, header=header)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile', type=str, help='genic paramfile')
    parser.add_argument('--extpk', type=str, help='optional external primordial power spectrum',required=False)
    parser.add_argument('--extraz', type=float,nargs='*', help='Space separated list of other redshifts at which to output power spectra',required=False)
    parser.add_argument('--verbose', action='store_true', help='print class runtime information',required=False)
    parser.add_argument('--convention', type=str, default='CLASS', help='CLASS or CAMB convention of densities (Friedmann)',required=False)
    args = parser.parse_args()
    make_class_power(args.paramfile, args.extpk, args.extraz,args.verbose, args.convention)
