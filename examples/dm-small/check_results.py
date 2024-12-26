"""Checks output of DM simulation with halo catalogue and power"""
import os
import os.path
import numpy as np
from numpy.testing import assert_allclose
import scipy.interpolate
import bigfile

def check_hmf(pig):
    """Check we have the mass functions."""
    bff = bigfile.BigFile(pig)
    hh = bff["Header"].attrs["HubbleParam"]
    fofmasses = bff['FOFGroups/Mass'][:]*10**10/hh
    assert np.max(fofmasses > 9e12)
    #It is 33 at the moment, but some of the smaller halos are flaky.
    assert np.size(fofmasses > 30)
    savedfof = np.array([9.93470260e+12, 8.22182375e+12, 7.53667146e+12, 7.19409494e+12,
                  6.39475122e+12, 6.28055942e+12, 6.16636762e+12, 5.93798365e+12,
                  5.82379148e+12, 5.70959968e+12, 5.48121571e+12, 5.48121571e+12,
                  5.36702353e+12, 5.36702353e+12, 5.13863956e+12, 4.68187162e+12,
                  4.68187162e+12, 4.56767982e+12, 4.45348765e+12, 4.33929585e+12,
                  4.33929585e+12, 4.22510367e+12, 4.11091187e+12, 4.11091187e+12,
                  3.99671970e+12, 3.88252790e+12, 3.88252790e+12, 3.88252790e+12,
                  3.88252790e+12, 3.76833573e+12, 3.76833573e+12, 3.65414356e+12,
                  3.65414356e+12])
    assert_allclose(fofmasses, savedfof, rtol=0.05, atol=0)

def modecount_rebin(kk, pk, modes, minmodes=2, ndesired=20):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    k_list=[kk[0]]
    pk_list=[pk[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    k_list = np.array(k_list)
    pk_list = np.array(pk_list)
    return (k_list, pk_list)

def get_power(matpow, rebin=True):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        modes = data[:,2][ii]
        return modecount_rebin(kk, pk, modes)
    return (kk,pk)

def check_power(scalefactor, rtol=0.18):
    """Check power spectrum is right"""
    matpow = "output/powerspectrum-%.4f.txt" % scalefactor
    kk_sim, pk_sim = get_power(matpow)
    zz = 1/scalefactor - 1
    if zz > 8.5:
        pk_camb = np.loadtxt("class_pk_9.dat")
    else:
        pk_camb = np.loadtxt("class_pk_9.dat-%.1f" % zz)
    pk_camb_int = scipy.interpolate.interp1d(pk_camb[:,0], pk_camb[:,1])
    assert_allclose(pk_sim[:6], pk_camb_int(kk_sim)[:6], rtol=rtol, atol=0.)

# Check that the power spectrum output is sensible and that the mass functions are right.
# asserting the initial power spectrum is 7% accurate
check_power(0.1, rtol=0.07)
check_power(0.2)
check_power(0.25)
check_hmf('output/PIG_002')
