#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "physconst.h"
#include "walltime.h"
#include "treewalk.h"
#include "sidm.h"
#include "utils.h"

/*! \file sidm.c
 *  \brief Computation of forces between self-interactive dark matter via pairwise interactions.
 */

/* This structure stores fixed parameters for the SIDM module. The idea is that these are
 * specified in the parameter file and the values never change over the course of a simulation.*/
static struct sidm_params
{
    /* Enables SIDM */
    int SIDMOn;
} SIDMParams;

/*Set the parameters of the SIDM module*/
void
set_hydro_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        SIDMParams.SIDMOn = param_get_int(ps, "SIDMOn");
    }
    MPI_Bcast(&SIDMParams, sizeof(struct sidm_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

struct SIDMPriv {
    /* This stores the 'global variables' associated with the SIDM module.
     * Cosmological time could be here, as an example.*/
    double atime;
};

/* This macro is used to access SIDMPriv inside the sidm functions*/
#define SIDM_GET_PRIV(tw) ((struct SIDMPriv*) ((tw)->priv))

/* This struct stores bits of a particle which are used as *inputs* to the SIDM functions.
 * sidm_copy is supposed to copy the fields from the particle structure P into this struct,
 * and is run for each particle. */
typedef struct {
    TreeWalkQueryBase base;
    double Vel[3];
    double Hsml;
    double Mass;
} TreeWalkQuerySIDM;

/* This struct stores bits of a particle which are used as *outputs* to the SIDM functions.
 * sidm_reduce is supposed to copy the fields from this struct back into the particle structure P,
 * and is run for each particle. */
typedef struct {
    TreeWalkResultBase base;
    MyFloat Acc[3];
} TreeWalkResultSIDM;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterSIDM;

/* Returns 1 if a particle needs an SIDM force computed, 0 otherwise.*/
static int
sidm_haswork(int n, TreeWalk * tw);

/* Postprocessing: usually used to change units. May not be necessary,
 * but runs after everything else is done.*/
static void
sidm_postprocess(int i, TreeWalk * tw);

static void
sidm_ngbiter(
    TreeWalkQuerySIDM * I,
    TreeWalkResultSIDM * O,
    TreeWalkNgbIterSIDM * iter,
    LocalTreeWalk * lv
   );

static void
sidm_copy(int place, TreeWalkQuerySIDM * input, TreeWalk * tw);

static void
sidm_reduce(int place, TreeWalkResultSIDM * result, enum TreeWalkReduceMode mode, TreeWalk * tw);

/*! This function is the driver routine for the calculation of hydrodynamical
 *  force and rate of change of entropy due to shock heating for all active
 *  particles .
 */
void
sidm_force(const ActiveParticles * act, const double atime, Cosmology * CP, const ForceTree * const tree)
{
    int i;
    TreeWalk tw[1] = {{0}};

    struct SIDMPriv priv[1];

    tw->ev_label = "SIDM";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) sidm_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterSIDM);
    tw->haswork = sidm_haswork;
    tw->fill = (TreeWalkFillQueryFunction) sidm_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sidm_reduce;
    tw->postprocess = (TreeWalkProcessFunction) sidm_postprocess;
    tw->query_type_elsize = sizeof(TreeWalkQuerySIDM);
    tw->result_type_elsize = sizeof(TreeWalkResultSIDM);
    tw->tree = tree;
    tw->priv = priv;


    /* Initialise the HSML for each particle: this is used
     * to find which particles are close in a pairwise fashion.
     * Only need this on the first timestep? */
    double global_closeness = 1;
    #pragma omp parallel for
    for(i=0; i < act->NumActiveParticle; i++) {
        int p = act->ActiveParticle ? act->ActiveParticle[i] : i;
        if(sidm_haswork(p, tw))
            P[p].Hsml = global_closeness;
    }

//    if(!tree->hmax_computed_flag)
  //      endrun(5, "Hydro called before hmax computed\n");

    /* Example of how to set a private member*/
    SIDM_GET_PRIV(tw)->atime = atime;
    treewalk_run(tw, act->ActiveParticle, act->NumActiveParticle);

    walltime_measure("/SPH/SIDM");
}

/* Copy fields from the particle table to the communication struct*/
static void
sidm_copy(int place, TreeWalkQuerySIDM * input, TreeWalk * tw)
{
    /*Compute predicted velocity*/
    input->Vel[0] = P[place].Vel[0];
    input->Vel[1] = P[place].Vel[1];
    input->Vel[2] = P[place].Vel[2];
    input->Hsml = P[place].Hsml;
    input->Mass = P[place].Mass;
}

/* Copy fields from the communication output struct back to the particle table.*/
static void
sidm_reduce(int place, TreeWalkResultSIDM * result, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
#ifdef SIDM
    int k;
    for(k = 0; k < 3; k++)
    {
        TREEWALK_REDUCE(P[place].SIDMAccel[k], result->Acc[k]);
    }
#endif
}


/*! This function is the 'core' of the force computation. A target
 *  particle is specified which may either be local, or reside in the
 *  communication buffer. The communication is managed transparently.
 */
static void
sidm_ngbiter(
    TreeWalkQuerySIDM * I,
    TreeWalkResultSIDM * O,
    TreeWalkNgbIterSIDM * iter,
    LocalTreeWalk * lv
   )
{
    /* This clause initialises all member variables of the TreeWalkNgbIterSIDM struct.
     * It is important!*/
    if(iter->base.other == -1) {
        iter->base.Hsml = I->Hsml;
        iter->base.mask = 1;
        /* Note: the specification of ASYMMETRIC is allowed because all SIDM particles
         * here have the same interaction radius. If that changes then you should
         * specify NGB_TREEFIND_SYMMETRIC and be more careful when building the force tree*/
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }

    /* Some useful properties precomputed*/
    /* Second particle in the pair interaction*/
    int other = iter->base.other;
    double rsq = iter->base.r2;
    /* 3D vector with distances between particles*/
    double * dist = iter->base.dist;
    /* Scalar euclidean distance*/
    double r = iter->base.r;

    /* At this point you have two particles which are interacting and thus are closer than I->Hsml.
     * The first particle is accessed through the member properties of I->. You have to use I-> because
     * the original particle may not be on this processor.
     * The second particle is accessed through the fields of P[other] and is on this processor.
     * Results should be copied into O-> and will them be copied back to the processor of origin.*/
    //doscatt();
    /* Example: compute the (physical) relative x-velocity between the two particles*/
    double rel_vel = (I->Vel[0] - P[other].Vel[0])/sqrt(SIDM_GET_PRIV(lv->tw)->atime);
}

/* For example, this selects all DM particles*/
static int
sidm_haswork(int i, TreeWalk * tw)
{
    return P[i].Type == 1;
}

static void
sidm_postprocess(int i, TreeWalk * tw)
{
    /* Only for dark matter particles*/
    if(P[i].Type == 1)
    {
        // Do something if needed
    }
}
