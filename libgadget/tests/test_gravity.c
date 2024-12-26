/*Simple test for gravitational force accuracy.*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <omp.h>

#include "stub.h"

#include <libgadget/utils/mymalloc.h>
#include <libgadget/utils/system.h>
#include <libgadget/utils/endrun.h>
#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/gravity.h>
#include <libgadget/petapm.h>
#include <libgadget/timestep.h>
#include <libgadget/physconst.h>

static struct ClockTable CT;
/* The true struct for the state variable*/
struct forcetree_testdata
{
    gsl_rng * r;
};
static const double G = 43.0071;

static void
grav_force(const int this, const int other, const double * offset, double * accns)
{

    double r2 = 0;
    int d;
    double dist[3];
    for(d = 0; d < 3; d ++) {
        /* the distance vector points to 'other' */
        dist[d] = offset[d] + P[this].Pos[d] - P[other].Pos[d];
        r2 += dist[d] * dist[d];
    }

    const double r = sqrt(r2);

    const double h = FORCE_SOFTENING();

    double fac = 1 / (r2 * r);
    if(r < h) {
        double h_inv = 1.0 / h;
        double h3_inv = h_inv * h_inv * h_inv;
        double u = r * h_inv;
        if(u < 0.5)
            fac = 1. * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
        else
            fac =
                1. * h3_inv * (21.333333333333 - 48.0 * u +
                        38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
    }

    for(d = 0; d < 3; d ++) {
        accns[3*this + d] += - dist[d] * fac * G * P[other].Mass;
        accns[3*other + d] += dist[d] * fac * G * P[this].Mass;
    }
}

void check_accns(double * meanerr_tot, double * maxerr_tot, double *PairAccn, double meanacc)
{
    double meanerr=0, maxerr=-1;
    int64_t i;
    /* This checks that the short-range force accuracy is being correctly estimated.*/
    #pragma omp parallel for reduction(+: meanerr) reduction(max: maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = fabs((PairAccn[3*i+k] - (P[i].GravPM[k] + P[i].FullTreeGravAccel[k]))/meanacc);
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(&meanerr, meanerr_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxerr, maxerr_tot, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    *meanerr_tot/= (tot_npart*3.);
}

static void find_means(double * meangrav, double * suppmean, double * suppaccns)
{
    int i;
    double meanacc = 0, meanforce = 0;
    #pragma omp parallel for reduction(+: meanacc) reduction(+: meanforce)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            if(suppaccns)
                meanacc += fabs(suppaccns[3*i+k]);
            meanforce += fabs(P[i].GravPM[k] + P[i].FullTreeGravAccel[k]);
        }
    }
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(suppaccns) {
        MPI_Allreduce(&meanacc, suppmean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        *suppmean/= (tot_npart*3.);
    }
    MPI_Allreduce(&meanforce, meangrav, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *meangrav/= (tot_npart*3.);
}


/* This checks the force on each particle using a direct summation:
 * very slow, but accurate.
 * Periodic boundary conditions are included by mirroring the box.*/
static void force_direct(double * accn)
{
    memset(accn, 0, 3 * sizeof(double) * PartManager->NumPart);
    int xx, yy, zz;
    /* Checked that increasing this has no visible effect on the computed force accuracy*/
    int repeat = 1;
    /* (slowly) compute gravitational force, accounting for periodicity by just inventing extra boxes on either side.*/
    for(xx=-repeat; xx <= repeat; xx++)
        for(yy=-repeat; yy <= repeat; yy++)
            for(zz=-repeat; zz <= repeat; zz++)
            {
                int i;
                double offset[3] = {PartManager->BoxSize * xx, PartManager->BoxSize * yy, PartManager->BoxSize * zz};
                for(i = 0; i < PartManager->NumPart; i++) {
                    int j;
                    for(j = i+1; j < PartManager->NumPart; j++)
                        grav_force(i, j, offset, accn);
                }
            }
}

static int check_against_force_direct(double ErrTolForceAcc)
{
    double * accn = (double *) mymalloc("accelerations", 3*sizeof(double) * PartManager->NumPart);
    force_direct(accn);
    double meanerr=0, maxerr=-1, meanacc=0, meanforce=0;
    find_means(&meanacc, &meanforce, accn);
    check_accns(&meanerr, &maxerr, accn, meanacc);
    myfree(accn);
    message(0, "Mean rel err is: %g max rel err is %g, meanacc %g mean grav force %g\n", meanerr, maxerr, meanacc, meanforce);
    /*Make some statements about the force error*/
    assert_true(maxerr < 3*ErrTolForceAcc);
    assert_true(meanerr < 0.8*ErrTolForceAcc);

    return 0;
}

static void do_force_test(int Nmesh, double Asmth, double ErrTolForceAcc, int direct)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    /*Sort by peano key so this is more realistic*/
    int i;
    #pragma omp parallel for
    for(i=0; i<PartManager->NumPart; i++) {
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].ID = i + (ThisTask<<20L);
        P[i].TimeBinHydro = 0;
        P[i].TimeBinGravity = 0;
        P[i].IsGarbage = 0;
    }

    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp);

    PetaPM pm = {0};
    gravpm_init_periodic(&pm, PartManager->BoxSize, Asmth, Nmesh, G);
    gravshort_fill_ntab(SHORTRANGE_FORCE_WINDOW_TYPE_EXACT, Asmth);
    /* Setup cosmology*/
    Cosmology CP ={0};
    CP.MNu[0] = CP.MNu[1] = CP.MNu[2] = 0;
    CP.OmegaCDM = 0.3;
    CP.CMBTemperature = 2.72;
    CP.HubbleParam = 0.7;
    CP.Omega0 = 0.3;
    CP.OmegaBaryon = 0.045;
    CP.OmegaLambda = 0.7;
    struct UnitSystem units = get_unitsystem(3.085678e21, 1.989e43, 1e5);
    init_cosmology(&CP, 0.01, units);

    gravpm_force(&pm, &ddecomp, &CP, 0.1, CM_PER_MPC/1000., ".", 0.01);
    ForceTree Tree = {0};
    force_tree_full(&Tree, &ddecomp, 1, NULL);
    const double rho0 = CP.Omega0 * 3 * CP.Hubble * CP.Hubble / (8 * M_PI * G);

    /* Barnes-Hut on first iteration*/
    struct gravshort_tree_params treeacc = {0};
    treeacc.BHOpeningAngle = 0.175;
    treeacc.TreeUseBH = 1;
    treeacc.Rcut = 7;
    treeacc.ErrTolForceAcc = ErrTolForceAcc;
    treeacc.FractionalGravitySoftening = 1./30.;

    set_gravshort_treepar(treeacc);
    gravshort_set_softenings(PartManager->BoxSize / cbrt(PartManager->NumPart));

    /* Twice so the opening angle is consistent*/
    ActiveParticles act = init_empty_active_particles(PartManager);
    grav_short_tree(&act, &pm, &Tree, NULL, rho0, 0);
    grav_short_tree(&act, &pm, &Tree, NULL, rho0, 0);

    force_tree_free(&Tree);
    petapm_destroy(&pm);
    domain_free(&ddecomp);
    if(direct)
        check_against_force_direct(ErrTolForceAcc);
}

static void test_force_flat(void ** state) {
    /*Set up the particle data*/
    int numpart = PartManager->NumPart, ThisTask, NTask;
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int64_t startpart = ThisTask * tot_npart / NTask;
    int ncbrt = cbrt(tot_npart);
    message(1, "np: %ld sp: %ld tot %ld\n", PartManager->NumPart, startpart, tot_npart);
    particle_alloc_memory(PartManager, 8, numpart*1.1);
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Pos[0] = (PartManager->BoxSize/ncbrt) * ((i + startpart)/ncbrt/ncbrt);
        P[i].Pos[1] = (PartManager->BoxSize/ncbrt) * (((i+startpart)/ncbrt) % ncbrt);
        P[i].Pos[2] = (PartManager->BoxSize/ncbrt) * ((i + startpart) % ncbrt);
    }
    PartManager->NumPart = numpart;
    do_force_test(48, 1.5, 0.002, 0);
    /* For a homogeneous mass distribution, the force should be zero*/
    double meanerr=0, maxerr=-1;
    #pragma omp parallel for reduction(+: meanerr) reduction(max: maxerr)
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int k;
        for(k=0; k<3; k++) {
            double err = fabs((P[i].GravPM[k] + P[i].FullTreeGravAccel[k]));
            meanerr += err;
            if(maxerr < err)
                maxerr = err;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &meanerr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    meanerr/= (tot_npart*3.);

    message(0, "Max force %g, mean grav force %g\n", maxerr, meanerr);
    /*Make some statements about the force error*/
    assert_true(maxerr < 0.015);
    assert_true(meanerr < 0.005);
    myfree(P);
}

static void test_force_close(void ** state) {
    /*Set up the particle data*/
    int ThisTask, NTask;
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int64_t startpart = ThisTask * tot_npart / NTask;
    message(1, "np: %ld sp: %ld tot %ld\n", PartManager->NumPart, startpart, tot_npart);
    int ncbrt = cbrt(tot_npart);

    int numpart = PartManager->NumPart;
    double close = 5000;
    particle_alloc_memory(PartManager, 8, numpart*1.2);
    /* Create particles clustered in one place, all of type 1.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Pos[0] = 4. + ((i+startpart)/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4. + (((i+startpart)/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4. + ((i+startpart) % ncbrt)/close;
    }
    PartManager->NumPart = numpart;
    do_force_test(48, 1.5, 0.002, 1);
    myfree(P);
}

void do_random_test(gsl_rng * r, const int numpart)
{
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    for(i=0; i<numpart/4; i++) {
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize * gsl_rng_uniform(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize/2 + PartManager->BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = PartManager->BoxSize*0.1 + PartManager->BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    PartManager->NumPart = numpart;
    do_force_test(48, 1.5, 0.002, 1);
}

static void test_force_random(void ** state) {
    /*Set up the particle data*/
    int numpart = PartManager->NumPart;
    struct forcetree_testdata * data = * (struct forcetree_testdata **) state;
    gsl_rng * r = data->r;
    particle_alloc_memory(PartManager, 8, numpart*1.1);
    int i;
    for(i=0; i<2; i++) {
        do_random_test(r, numpart);
    }
    myfree(P);
}

static int setup_tree(void **state) {
    walltime_init(&CT);
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    /*Set up the important parts of the All structure.*/
    /*Particles should not be outside this*/
    PartManager->BoxSize = 8;
    PartManager->NumPart = 16*16*16 / NTask;
    if(ThisTask == NTask -1)
        PartManager->NumPart += 16*16*16 % NTask;
    int64_t tot_npart;
    MPI_Allreduce(&PartManager->NumPart, &tot_npart, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    assert_true(tot_npart == 16*16*16);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 2;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    petapm_module_init(omp_get_max_threads());
    init_forcetree_params(0.7);
    /*Set up the top-level domain grid*/
    struct forcetree_testdata *data = malloc(sizeof(struct forcetree_testdata));
    data->r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(data->r, 0);
    *state = (void *) data;
    return 0;
}

static int teardown_tree(void **state) {
    struct forcetree_testdata * data = (struct forcetree_testdata * ) *state;
    free(data->r);
    free(data);
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_force_flat),
        cmocka_unit_test(test_force_close),
        cmocka_unit_test(test_force_random),
    };
    return cmocka_run_group_tests_mpi(tests, setup_tree, teardown_tree);
}
