/*Tests for the cosmology module, ported from N-GenIC.*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "stub.h"
#include <libgadget/config.h>
#include <libgenic/power.h>
#include <libgadget/cosmology.h>

struct test_state
{
    struct power_params PowerP;
    Cosmology CP;
};

/*stub*/
void _bigfile_utils_create_block_from_c_array(BigFile * bf, void * baseptr, char * name, char * dtype, size_t dims[], ptrdiff_t elsize, int64_t TotNumPart, MPI_Comm comm)
{
    return;
}

/*Simple test without rescaling*/
static void
test_read_no_rescale(void ** state)
{
    /*Do setup*/
    struct power_params PowerP = ((struct test_state *) (*state))->PowerP;
    Cosmology CP = ((struct test_state *) (*state))->CP;
    /*Test without rescaling*/
    PowerP.InputPowerRedshift = -1;
    PowerP.DifferentTransferFunctions = 1;
    /*Test without rescaling*/
    int nentry = init_powerspectrum(0, 0.01, 3.085678e21, &CP, &PowerP);
    assert_int_equal(nentry, 347);
    /*Check that the tabulated power spectrum gives the right answer
     * Should be the same k as in the file (but /10^3 for Mpc -> kpc)
     * Note that our PowerSpec function is 2pi^3 larger than that in S-GenIC.*/
    //Now check total power: k divided by 10^3,
    //Conversion for P(k) is 10^9/(2pi)^3
    assert_true(fabs(pow(DeltaSpec(1.124995061548053968e-02/1e3, DELTA_TOT),2) / 4.745074933325402533/1e9 - 1) < 1e-5);
    assert_true(fabs(pow(DeltaSpec(1.010157135208153312e+00/1e3, DELTA_TOT),2) / 1.15292e-02/1e9 - 1) < 1e-5);
    //Check that it gives reasonable results when interpolating
    int k;
    for (k = 1; k < 100; k++) {
        double newk = 0.10022E+01/1e3+ k*(0.10362E+01-0.10022E+01)/1e3/100;
        assert_true(DeltaSpec(newk,DELTA_TOT) < DeltaSpec(0.10022E+01/1e3,DELTA_TOT));
        assert_true(DeltaSpec(newk,DELTA_TOT) > DeltaSpec(0.10362E+01/1e3,DELTA_TOT));
        assert_true(DeltaSpec(newk,DELTA_BAR)/DeltaSpec(0.10362E+01/1e3,DELTA_CDM) < 1);
        /*Check that the CDM + baryon power is the same as the total power for massless neutrinos*/
        assert_true(fabs(DeltaSpec(newk,DELTA_CB)/DeltaSpec(newk,DELTA_TOT)-1) < 1e-5);
    }
    //Now check transfer functions: ratio of total to species should be ratio of T_s to T_tot squared: large scales where T~ 1
    //CDM
    assert_true(fabs(DeltaSpec(2.005305808001081169e-03/1e3,DELTA_CDM)/DeltaSpec(2.005305808001081169e-03/1e3,DELTA_TOT)- 1.193460280018762132e+05/1.193185119820504624e+05) < 1e-5);
    //Small scales where there are differences
    //T_tot=0.255697E+06
    //Baryons
    assert_true(fabs(DeltaSpec(1.079260830861467901e-01/1e3,DELTA_BAR)/DeltaSpec(1.079260830861467901e-01/1e3,DELTA_CB)- 9.735695830700024089e+03/1.394199788775037632e+04) < 1e-4);
    //CDM
    assert_true(fabs(DeltaSpec(1.079260830861467901e-01/1e3,DELTA_CDM)/DeltaSpec(1.079260830861467901e-01/1e3,DELTA_CB)- 1.477251880454670209e+04/1.394199788775037632e+04) < 1e-4);
}

static void
test_growth_numerical(void ** state)
{
    /*Do setup*/
    struct power_params PowerP = ((struct test_state *) (*state))->PowerP;
    Cosmology CP = ((struct test_state *) (*state))->CP;
    /*Test without rescaling*/
    PowerP.InputPowerRedshift = -1;
    PowerP.DifferentTransferFunctions = 1;
    int nentry = init_powerspectrum(0, 0.01, 3.085678e21, &CP, &PowerP);
    assert_int_equal(nentry, 347);
    //Test sub-horizon scales
    int k, nk = 100;
    //Smaller scales than BAO
    double lowk = 0.4;
    double highk = 10;
    for (k = 1; k < nk; k++) {
        double newk = exp(log(lowk) + k*(log(highk) - log(lowk))/nk);
        newk/=1e3;
/*         message(1,"k=%g G = %g F = %g G0 = %g\n",newk*1e3,dlogGrowth(newk, DELTA_TOT), F_Omega(0.01),dlogGrowth(newk, 1)); */
        //Total growth should be very close to F_Omega.
        assert_true(fabs(dlogGrowth(newk,DELTA_TOT) / (F_Omega(&CP, 0.01) * DeltaSpec(newk, DELTA_TOT)) -1) < 0.01);
        //Growth of CDM should be lower, growth of baryons should be higher.
        assert_true(dlogGrowth(newk,DELTA_CDM) < F_Omega(&CP, 0.01) * DeltaSpec(newk, DELTA_CDM));
        assert_true(fabs(dlogGrowth(newk,DELTA_CDM) / DeltaSpec(newk, DELTA_CDM) - 0.9389) < 0.01);
        assert_true(dlogGrowth(newk,DELTA_BAR) > 1.25 * DeltaSpec(newk, DELTA_BAR));
        assert_true(dlogGrowth(newk,DELTA_BAR) < 1.3 * DeltaSpec(newk, DELTA_BAR));
    }
    //Test super-horizon scales
    lowk = 1e-3;
    highk = 5e-3;
    for (k = 1; k < nk; k++) {
        double newk = exp(log(lowk) + k*(log(highk) - log(lowk))/nk);
        newk/=1e3;
/*         message(1,"k=%g G = %g F = %g\n",newk*1e3,dlogGrowth(newk, 7), dlogGrowth(newk, 1)); */
        //Total growth should be around 1.05
        assert_true(dlogGrowth(newk,DELTA_TOT) < 1.055 * DeltaSpec(newk, DELTA_TOT));
        assert_true(dlogGrowth(newk,DELTA_TOT) > 1. * DeltaSpec(newk, DELTA_TOT));
        //CDM and baryons should match total
        assert_true(fabs(dlogGrowth(newk,DELTA_BAR)/dlogGrowth(newk,DELTA_TOT) -1)  < 0.008);
        assert_true(fabs(dlogGrowth(newk,DELTA_CDM)/dlogGrowth(newk,DELTA_TOT) -1)  < 0.008);
    }
}

/*Check normalising to a different sigma8 and redshift*/
static void
test_read_rescale_sigma8(void ** state)
{
    /*Do setup*/
    struct power_params PowerP = ((struct test_state *) (*state))->PowerP;
    Cosmology CP = ((struct test_state *) (*state))->CP;
    /* Test rescaling to an earlier time
     * (we still use the same z=99 power which should not be rescaled in a real simulation)*/
    PowerP.InputPowerRedshift = 9;
    PowerP.DifferentTransferFunctions = 0;
    int nentry = init_powerspectrum(0, 0.05, 3.085678e21, &CP, &PowerP);
    assert_int_equal(nentry, 347);
    assert_true(fabs(pow(DeltaSpec(1.124995061548053968e-02/1e3, DELTA_TOT),2)* 4 /4.745074933325402533/1e9 - 1) < 1e-2);
}


static int setup(void ** state)
{
    static struct test_state st = {0};
    st.PowerP.InputPowerRedshift = -1;
    st.PowerP.DifferentTransferFunctions = 1;
    st.PowerP.Sigma8 = -1;
    st.PowerP.FileWithInputSpectrum = GADGET_TESTDATA_ROOT "/examples/class_pk_99.dat";
    st.PowerP.FileWithTransferFunction = GADGET_TESTDATA_ROOT "/examples/class_tk_99.dat";
    st.PowerP.WhichSpectrum = 2;
    st.PowerP.PrimordialIndex = 1.0;
    st.CP.Omega0 = 0.2814;
    st.CP.OmegaLambda = 1 - st.CP.Omega0;
    st.CP.OmegaBaryon = 0.0464;
    st.CP.HubbleParam = 0.697;
    st.CP.Omega_fld = 0;
    st.CP.w0_fld = -1;
    st.CP.wa_fld = 0;
    st.CP.CMBTemperature = 2.7255;
    st.CP.RadiationOn = 1;
    st.CP.MNu[0] = 0;
    st.CP.MNu[1] = 0;
    st.CP.MNu[2] = 0;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&st.CP, 0.01, units);
    *state = &st;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_no_rescale),
        cmocka_unit_test(test_read_rescale_sigma8),
        cmocka_unit_test(test_growth_numerical)
    };
    return cmocka_run_group_tests_mpi(tests, setup, NULL);
}
