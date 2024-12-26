#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "init.h"
#include "utils.h"

#include "cooling.h"
#include "forcetree.h"
#include "density.h"

#include "timefac.h"
#include "petaio.h"
#include "domain.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "exchange.h"
#include "fof.h"
#include "timestep.h"
#include "timebinmgr.h"
#include "cosmology.h"
#include "gravity.h"
#include "physconst.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */

static struct init_params
{
    /* Gas temperature in the ICs*/
    double InitGasTemp;
    /*!< in order to maintain work-load balance, the particle load will usually
        NOT be balanced.  Each processor allocates memory for PartAllocFactor times
        the average number of particles to allow for that */
    double PartAllocFactor;

    int ExcursionSetReionOn;
    double ExcursionSetZStart;
} InitParams;

/*Set the global parameters*/
void
set_init_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        InitParams.InitGasTemp = param_get_double(ps, "InitGasTemp");
        InitParams.PartAllocFactor = param_get_double(ps, "PartAllocFactor");

        InitParams.ExcursionSetReionOn = param_get_int(ps,"ExcursionSetReionOn");
        InitParams.ExcursionSetZStart = param_get_int(ps,"ExcursionSetZStart");
    }
    MPI_Bcast(&InitParams, sizeof(InitParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Setup a list of sync points until the end of the simulation.*/
void init_timeline(Cosmology * CP, int RestartSnapNum, double TimeMax, const struct header_data * header, const int SnapshotWithFOF)
{
    /*Add TimeInit and TimeMax to the output list*/
    if (RestartSnapNum < 0) {
        /* allow a first snapshot at IC time; */
        setup_sync_points(CP,header->TimeIC, TimeMax, 0.0, SnapshotWithFOF);
    } else {
        /* skip dumping the exactly same snapshot */
        setup_sync_points(CP, header->TimeIC, TimeMax, header->TimeSnapshot, SnapshotWithFOF);
        /* If TimeInit is not in a sensible place on the integer timeline
         * (can happen if the outputs changed since it was written)
         * start the integer timeline anew from TimeInit */
        inttime_t ti_init = ti_from_loga(log(header->TimeSnapshot)) % TIMEBASE;
        if(round_down_power_of_two(ti_init) != ti_init) {
            message(0,"Resetting integer timeline (as %lx != %lx) to current snapshot\n",ti_init, round_down_power_of_two(ti_init));
            setup_sync_points(CP, header->TimeSnapshot, TimeMax, header->TimeSnapshot, SnapshotWithFOF);
        }
    }
}


static void get_mean_separation(double * MeanSeparation, const double BoxSize, const int64_t * NTotalInit);
static void check_omega(struct part_manager_type * PartManager, Cosmology * CP, int generations, double * MassTable);
static void check_positions(struct part_manager_type * PartManager);
static void check_smoothing_length(struct part_manager_type * PartManager, double * MeanSpacing);
static void init_alloc_particle_slot_memory(struct part_manager_type * PartManager, struct slots_manager_type * SlotsManager, const double PartAllocFactor, struct header_data * header, MPI_Comm Comm);

/*! This function reads the initial conditions, allocates storage for the
 *  particle data, validates and initialises the particle data.
 */
inttime_t init(int RestartSnapNum, const char * OutputDir, struct header_data * header, Cosmology * CP)
{
    int i;

    init_alloc_particle_slot_memory(PartManager, SlotsManager, InitParams.PartAllocFactor, header, MPI_COMM_WORLD);

    /*Read the snapshot*/
    petaio_read_snapshot(RestartSnapNum, OutputDir, CP, header, PartManager, SlotsManager, MPI_COMM_WORLD);

    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);

    check_omega(PartManager, CP, get_generations(), header->MassTable);

    check_positions(PartManager);

    double MeanSeparation[6] = {0};

    get_mean_separation(MeanSeparation, PartManager->BoxSize, header->NTotalInit);

    if(RestartSnapNum >= 0)
        check_smoothing_length(PartManager, MeanSeparation);

    /* As the above will mostly take place
     * on Task 0, there will be a lot of imbalance*/
    MPIU_Barrier(MPI_COMM_WORLD);

    gravshort_set_softenings(MeanSeparation[1]);
    fof_init(MeanSeparation[1]);

    inttime_t Ti_Current = init_timebins(header->TimeSnapshot);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)	/* initialize sph_properties */
    {
        int j;
        P[i].Ti_drift = Ti_Current;
#ifdef DEBUG
        P[i].Ti_kick_grav = Ti_Current;
        P[i].Ti_kick_hydro = Ti_Current;
#endif
        if(RestartSnapNum == -1 && P[i].Type == 5 )
        {
            /* Note: Gadget-3 sets this to the seed black hole mass.*/
            BHP(i).Mass = P[i].Mass;
        }

        if(P[i].Type == 4 )
        {
            /* Touch up zero star smoothing lengths, not saved in the snapshots.*/
            if(P[i].Hsml == 0)
                P[i].Hsml = 0.1 * MeanSeparation[0];
        }

        if(P[i].Type == 5)
        {
            for(j = 0; j < 3; j++) {
                BHP(i).DFAccel[j] = 0;
                BHP(i).DragAccel[j] = 0;
            }
        }

        if(P[i].Type != 0)
            continue;

        for(j = 0; j < 3; j++)
        {
            SPHP(i).HydroAccel[j] = 0;
        }

        if(!isfinite(SPHP(i).DelayTime ))
            endrun(6, "Bad DelayTime %g for part %d id %ld\n", SPHP(i).DelayTime, i, P[i].ID);
        SPHP(i).DtEntropy = 0;

        if(RestartSnapNum == -1)
        {
            SPHP(i).Density = -1;
            SPHP(i).EgyWtDensity = -1;
            SPHP(i).DhsmlEgyDensityFactor = -1;
            SPHP(i).Entropy = -1;
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
            SPHP(i).CurlVel = 0;
            SPHP(i).DelayTime = 0;
            SPHP(i).Metallicity = 0;
            memset(SPHP(i).Metals, 0, NMETALS*sizeof(float));
            /* Initialise to primordial abundances for H and He*/
            SPHP(i).Metals[0] = HYDROGEN_MASSFRAC;
            SPHP(i).Metals[1] = 1- HYDROGEN_MASSFRAC;
            SPHP(i).Sfr = 0;
            SPHP(i).MaxSignalVel = 0;
        }
#ifdef EXCUR_REION
        /* If we are starting before reionisation, initialise reion properties
         * this allows us to restart from runs without excursion set
         * these properties aren't used without the ES so its fine to init them here*/
        if(InitParams.ExcursionSetReionOn && header->TimeSnapshot < 1./(1. + InitParams.ExcursionSetZStart)){
            SPHP(i).local_J21 = 0;
            SPHP(i).zreion = -1;
        }
#endif
    }
    walltime_measure("/Init");
    return Ti_Current;
}


/*! This routine computes the mass content of the box and compares it to the
 * specified value of Omega-matter.  If discrepant, the run is terminated.
 */
void check_omega(struct part_manager_type * PartManager, Cosmology * CP, int generations, double * MassTable)
{
    double mass = 0, masstot, omega;
    int64_t i, totbad, badmass = 0;
    double omegas[6] = {0};

    #pragma omp parallel for reduction(+: mass) reduction(+: badmass) reduction(+: omegas[:6])
    for(i = 0; i < PartManager->NumPart; i++) {
        /* In case zeros have been written to the saved mass array,
         * recover the true masses*/
        if(P[i].Mass == 0) {
            P[i].Mass = MassTable[P[i].Type] * ( 1. - (double)P[i].Generation/generations);
            badmass++;
        }
        if(P[i].Type < 6)
            omegas[P[i].Type] += P[i].Mass;
        mass += P[i].Mass;
    }

    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &omegas, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&badmass, &totbad, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    if(totbad)
        message(0, "Warning: recovering from %ld Mass entries corrupted on disc\n",totbad);

    double massnorm = (PartManager->BoxSize * PartManager->BoxSize * PartManager->BoxSize) * CP->RhoCrit;
    omega = masstot / massnorm;
    for(i = 0; i<6; i++)
        omegas[i] /= massnorm;

    double omeganu = get_omega_nu_nopart(&CP->ONu, 1);
    /*Add the density for analytically follows massive neutrinos*/
    if(CP->MassiveNuLinRespOn)
        omega += omeganu;
    message(0, "Matter content: OmegaB = %g OmegaCDM = %g OmegaNu (LRA) = %g OmegaNu (particle) = %g Omega* = %g OmegaBH = %g\n",
            omegas[0], omegas[1], omeganu, omegas[2], omegas[4], omegas[5]);

    if(fabs(omega - CP->Omega0) > 1.0e-3)
        endrun(0, "The mass content is Omega0 = %g,but you specified Omega0 = %g in the parameterfile.\n", omega, CP->Omega0);
}

/* Allocate the memory for particles and slots. First the total amount of particles are counted, then allocations are made*/
static void
init_alloc_particle_slot_memory(struct part_manager_type * PartManager, struct slots_manager_type * SlotsManager, const double PartAllocFactor, struct header_data * header, MPI_Comm Comm)
{
    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    int ptype;

    int64_t TotNumPart = 0;
    int64_t TotNumPartInit= 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        TotNumPart += header->NTotal[ptype];
        TotNumPartInit += header->NTotalInit[ptype];
    }

    message(0, "Total number of particles: %018ld\n", TotNumPart);

    const char * PARTICLE_TYPE_NAMES [] = {"Gas", "DarkMatter", "Neutrino", "Unknown", "Star", "BlackHole"};

    for(ptype = 0; ptype < 6; ptype ++) {
        double MeanSeparation = header->BoxSize / pow(header->NTotalInit[ptype], 1.0 / 3);
        message(0, "%11s: Total: %018ld Init: %018ld Mean-Sep %g \n",
                PARTICLE_TYPE_NAMES[ptype], header->NTotal[ptype], header->NTotalInit[ptype], MeanSeparation);
    }

    /* sets the maximum number of particles that may reside on a processor */
    int MaxPart = (int) (PartAllocFactor * TotNumPartInit / NTask);

    /*Allocate the particle memory*/
    particle_alloc_memory(PartManager, header->BoxSize, MaxPart);

    for(ptype = 0; ptype < 6; ptype ++) {
        int64_t start = ThisTask * header->NTotal[ptype] / NTask;
        int64_t end = (ThisTask + 1) * header->NTotal[ptype] / NTask;
        header->NLocal[ptype] = end - start;
        PartManager->NumPart += header->NLocal[ptype];
    }

    /* Allocate enough memory for stars and black holes.
     * This will be dynamically increased as needed.*/

    if(PartManager->NumPart >= PartManager->MaxPart) {
        endrun(1, "Overwhelmed by part: %ld > %ld\n", PartManager->NumPart, PartManager->MaxPart);
    }

    /* Now allocate memory for the secondary particle data arrays.
     * This may be dynamically resized later!*/

    /*Ensure all processors have initially the same number of particle slots*/
    int64_t newSlots[6] = {0};

    /* Can't use MPI_IN_PLACE, which is broken for arrays and MPI_MAX at least on intel mpi 19.0.5*/
    MPI_Allreduce(header->NLocal, newSlots, 6, MPI_INT64, MPI_MAX, Comm);

    for(ptype = 0; ptype < 6; ptype ++) {
            newSlots[ptype] *= PartAllocFactor;
    }
    /* Boost initial amount of stars allocated, as it is often uneven.
     * The total number of stars is usually small so this doesn't
     * waste that much memory*/
    newSlots[4] *= 2;

    slots_reserve(0, newSlots, SlotsManager);

    /* so we can set up the memory topology of secondary slots */
    slots_setup_topology(PartManager, header->NLocal, SlotsManager);
}

/*! This routine checks that the initial positions of the particles are within the box.
 * If not, there is likely a bug in the IC generator and we abort.
 * It also checks for multiple zeros in the positions, guarding against a common fs bug.
 */
void check_positions(struct part_manager_type * PartManager)
{
    int i;
    int numzero = 0;
    int lastzero = -1;

    #pragma omp parallel for reduction(+: numzero) reduction(max:lastzero)
    for(i=0; i< PartManager->NumPart; i++){
        int j;
        double * Pos = PartManager->Base[i].Pos;
        for(j=0; j<3; j++) {
            if(Pos[j] < 0 || Pos[j] > PartManager->BoxSize || !isfinite(Pos[j]))
                endrun(0,"Particle %d is outside the box (L=%g) at (%g %g %g)\n",i, PartManager->BoxSize, Pos[0], Pos[1], Pos[2]);
        }
        if((Pos[0] < 1e-35) && (Pos[1] < 1e-35) && (Pos[2] < 1e-35)) {
            numzero++;
            lastzero = i;
        }
    }
    if(numzero > 1)
        endrun(5, "Particle positions contain %d zeros at particle %d. Pos %g %g %g. Likely write corruption!\n",
                numzero, lastzero, PartManager->Base[lastzero].Pos[0], PartManager->Base[lastzero].Pos[1], PartManager->Base[lastzero].Pos[2]);
}

/*! This routine checks that the initial smoothing lengths of the particles
 *  are sensible and resets them to mean interparticle spacing if not.
 *  Guards against a problem writing the snapshot. Matters because
 *  a very large initial smoothing length will cause density() to go crazy.
 */
void check_smoothing_length(struct part_manager_type * PartManager, double * MeanSpacing)
{
    int i;
    int numprob = 0;
    int lastprob = -1;
    #pragma omp parallel for reduction(+: numprob) reduction(max:lastprob)
    for(i=0; i< PartManager->NumPart; i++){
        if(P[i].Type != 5 && P[i].Type != 0)
            continue;
        if(P[i].Hsml > PartManager->BoxSize || P[i].Hsml <= 0) {
            P[i].Hsml = MeanSpacing[P[i].Type];
            numprob++;
            lastprob = i;
        }
    }
    if(numprob > 0)
        message(5, "Bad smoothing lengths %d last bad %d hsml %g id %ld\n", numprob, lastprob, P[lastprob].Hsml, P[lastprob].ID);
}

/* When we restart, validate the SPH properties of the particles.
 * This also allows us to increase MinEgySpec on a restart if we choose.*/
void check_density_entropy(Cosmology * CP, const double MinEgySpec, const double atime)
{
    const double a3 = pow(atime, 3);
    int i;
    int bad = 0;
    double meanbar = CP->OmegaBaryon * 3 * HUBBLE * CP->HubbleParam * HUBBLE * CP->HubbleParam/ (8 * M_PI * GRAVITY);
    #pragma omp parallel for reduction(+: bad)
    for(i = 0; i < SlotsManager->info[0].size; i++) {
        /* This allows us to continue gracefully if
            * there was some kind of bug in the run that output the snapshot.
            * density() below will fix this up.*/
        if(SphP[i].Density <= 0 || !isfinite(SphP[i].Density)) {
            SphP[i].Density = meanbar;
            bad++;
        }
        if(SphP[i].EgyWtDensity <= 0 || !isfinite(SphP[i].EgyWtDensity)) {
            SphP[i].EgyWtDensity = SphP[i].Density;
        }
        double minent = GAMMA_MINUS1 * MinEgySpec / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
        if(SphP[i].Entropy < minent)
            SphP[i].Entropy = minent;
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(bad > 0)
        message(0, "Detected bad densities in %d particles on disc\n",bad);
}

void get_mean_separation(double * MeanSeparation, const double BoxSize, const int64_t * NTotalInit)
{
    int i;
    for(i = 0; i < 6; i++) {
        if(NTotalInit[i] > 0)
            MeanSeparation[i] = BoxSize / pow(NTotalInit[i], 1.0 / 3);
    }
}

/* Initialize the entropy variable in Pressure-Entropy Sph.
 * Initialization of the entropy variable is a little trickier in this version of SPH,
 * since we need to make sure it 'talks to' the density appropriately */
static void
setup_density_indep_entropy(const ActiveParticles * act, ForceTree * Tree, Cosmology * CP, struct sph_pred_data * sph_pred, double u_init, double a3, int BlackHoleOn, const inttime_t Ti_Current)
{
    int j;
    int stop = 0;

    message(0, "Converting u -> entropy, with density split sph\n");

    /* This gives better convergence than initializing EgyWtDensity before Density is known*/
    #pragma omp parallel for
    for(j = 0; j < SlotsManager->info[0].size; j++)
        SphP[j].EgyWtDensity = SphP[j].Density;

    MyFloat * olddensity = (MyFloat *)mymalloc("olddensity ", SlotsManager->info[0].size * sizeof(MyFloat));
    for(j = 0; j < 100; j++)
    {
        int i;
        /* since ICs give energies, not entropies, need to iterate get this initialized correctly */
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].EgyWtDensity / a3 , GAMMA_MINUS1);
            olddensity[i] = SphP[i].EgyWtDensity;
        }
        /* Empty kick factors as we do not move*/
        DriftKickTimes times = init_driftkicktime(Ti_Current);
        /* Update the EgyWtDensity*/
        density(act, 0, DensityIndependentSphOn(), BlackHoleOn, times, CP, sph_pred, NULL, Tree);
        slots_free_sph_pred_data(sph_pred);
        if(stop)
            break;

        double maxdiff = 0;
        #pragma omp parallel for reduction(max: maxdiff)
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            double value = fabs(SphP[i].EgyWtDensity - olddensity[i]) / SphP[i].EgyWtDensity;
            maxdiff = DMAX(maxdiff,value);
        }
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        message(0, "iteration %d, max relative change in EgyWtDensity = %g \n", j, maxdiff);

        /* If maxdiff is small, do one more iteration and stop*/
        if(maxdiff < 1e-3)
            stop = 1;

    }
    myfree(olddensity);
}

/*! This function is used to find an initial smoothing length and initial entropy
 *  for each SPH particle. Entropies are set using the initial gas temperature.
 *  It guarantees that the number of neighbours will be between
 *  desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 *  of the smoothing length is provided to the function density(), which will
 *  then iterate if needed to find the right smoothing length.
 */
void
setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, Cosmology * CP, int BlackHoleOn, double MinEgySpec, double uu_in_cgs, const inttime_t Ti_Current, const double atime, const int64_t NTotGasInit)
{
    int i;
    const double a3 = pow(atime, 3);

    if(RestartSnapNum >= 0)
        return;

    if(InitParams.InitGasTemp < 0)
        InitParams.InitGasTemp = CP->CMBTemperature / atime;

    const double MeanGasSeparation = PartManager->BoxSize / pow(NTotGasInit, 1.0 / 3);

    int64_t tot_sph, tot_bh;
    MPI_Allreduce(&SlotsManager->info[0].size, &tot_sph, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&SlotsManager->info[5].size, &tot_bh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Do nothing if we are a pure DM run*/
    if(tot_sph + tot_bh == 0)
        return;

    ForceTree Tree = {0};
    /* Finds fathers for each gas and BH particle, so need BH*/
    force_tree_rebuild_mask(&Tree, ddecomp, GASMASK+BHMASK, NULL);
    /* Set the initial smoothing length for gas and DM, compute tree moments.*/
    set_init_hsml(&Tree, ddecomp, MeanGasSeparation);

    /* for clean IC with U input only, we need to iterate to find entropy */
    double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * InitParams.InitGasTemp;
    u_init /= uu_in_cgs; /* unit conversion */

    double molecular_weight;
    if(InitParams.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
        molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    else				/* assuming NEUTRAL GAS */
        molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);
    u_init /= molecular_weight;

    if(u_init < MinEgySpec)
        u_init = MinEgySpec;
    /* snapshot already has EgyWtDensity; hope it is read in correctly.
        * (need a test on this!) */
    /*Allocate the extra SPH data for transient SPH particle properties.*/
    struct sph_pred_data sph_pred = {0};

    /* Empty kick factors as we do not move*/
    DriftKickTimes times = init_driftkicktime(Ti_Current);
    /*At the first time step all particles should be active*/
    ActiveParticles act = init_empty_active_particles(PartManager);
    density(&act, 1, 0, BlackHoleOn, times, CP, &sph_pred, NULL, &Tree);
    slots_free_sph_pred_data(&sph_pred);

    if(DensityIndependentSphOn()) {
        setup_density_indep_entropy(&act, &Tree, CP, &sph_pred, u_init, a3, BlackHoleOn, Ti_Current);
    }
    else {
        /*Initialize to initial energy*/
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++)
            SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
    }
    force_tree_free(&Tree);
}
