#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include "allvars.h"
#include "cooling.h"
#include "forcetree.h"
#include "densitykernel.h"
#include "proto.h"
#include "treewalk.h"
#include "mymalloc.h"
#include "endrun.h"

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
*/
typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel kernel;
    double kernel_volume;
#ifdef BLACK_HOLES
    DensityKernel bh_feedback_kernel;
#endif
} TreeWalkNgbIterDensity;

typedef struct
{
    TreeWalkQueryBase base;
    MyFloat Vel[3];
    MyFloat Hsml;
#ifdef VOLUME_CORRECTION
    MyFloat DensityOld;
#endif
#ifdef WINDS
    MyFloat DelayTime;
#endif
    int Type;
} TreeWalkQueryDensity;

typedef struct {
    TreeWalkResultBase base;
#ifdef DENSITY_INDEPENDENT_SPH
    MyFloat EgyRho;
    MyFloat DhsmlEgyDensity;
#endif
    MyDouble Rho;
    MyDouble DhsmlDensity;
    MyDouble Ngb;
    MyDouble Div, Rot[3];

#ifdef BLACK_HOLES
    MyDouble SmoothedEntropy;
    MyDouble SmoothedPressure;
    MyDouble FeedbackWeightSum;
    MyDouble GasVel[3];
#endif

#ifdef HYDRO_COST_FACTOR
    int Ninteractions;
#endif

#ifdef VOLUME_CORRECTION
    MyFloat DensityStd;
#endif

#ifdef SPH_GRAD_RHO
    MyFloat GradRho[3];
#endif
} TreeWalkResultDensity;

static void
density_ngbiter(
        TreeWalkQueryDensity * I,
        TreeWalkResultDensity * O,
        TreeWalkNgbIterDensity * iter,
        LocalTreeWalk * lv);

static int density_isactive(int n);
static void density_post_process(int i);
static void density_check_neighbours(int i, MyFloat * Left, MyFloat * Right);


static void density_reduce(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode);
static void density_copy(int place, TreeWalkQueryDensity * I);

/*! \file density.c
 *  \brief SPH density computation and smoothing length determination
 *
 *  This file contains the "first SPH loop", where the SPH densities and some
 *  auxiliary quantities are computed.  There is also functionality that
 *  corrects the smoothing length if needed.
 */


/*! This function computes the local density for each active SPH particle, the
 * number of neighbours in the current smoothing radius, and the divergence
 * and rotation of the velocity field.  The pressure is updated as well.  If a
 * particle with its smoothing region is fully inside the local domain, it is
 * not exported to the other processors. The function also detects particles
 * that have a number of neighbours outside the allowed tolerance range. For
 * these particles, the smoothing length is adjusted accordingly, and the
 * density() computation is called again.  Note that the smoothing length is
 * not allowed to fall below the lower bound set by MinGasHsml (this may mean
 * that one has to deal with substantially more than normal number of
 * neighbours.)
 */

void density(void)
{
    MyFloat *Left, *Right;

    TreeWalk tw = {0};

    tw.ev_label = "DENSITY";
    tw.visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw.ngbiter_type_elsize = sizeof(TreeWalkNgbIterDensity);
    tw.ngbiter = (TreeWalkNgbIterFunction) density_ngbiter;

    tw.isactive = density_isactive;
    tw.fill = (TreeWalkFillQueryFunction) density_copy;
    tw.reduce = (TreeWalkReduceResultFunction) density_reduce;
    tw.UseNodeList = 1;
    tw.query_type_elsize = sizeof(TreeWalkQueryDensity);
    tw.result_type_elsize = sizeof(TreeWalkResultDensity);

    int i, iter = 0;

    int64_t ntot = 0;

    double timeall = 0;
    double timecomp, timecomp3 = 0, timecomm, timewait;

    double tstart, tend;

    walltime_measure("/Misc");

    Left = (MyFloat *) mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *) mymalloc("Right", NumPart * sizeof(MyFloat));

    int Nactive;
    int * queue;

    /* this has to be done before get_queue so that
     * all particles are return for the first loop over all active particles.
     * */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        P[i].DensityIterationDone = 0;
    }

    /* the queue has every particle. Later on after some iterations are done
     * Nactive will decrease -- the queue would be shorter.*/
    queue = treewalk_get_queue(&tw, &Nactive);
#pragma omp parallel for if(Nactive > 32)
    for(i = 0; i < Nactive; i ++) {
        int p = queue[i];
        Left[p] = 0;
        Right[p] = 0;
#ifdef BLACK_HOLES
        P[p].SwallowID = 0;
#endif
    }
    myfree(queue);

    /* allocate buffers to arrange communication */

    walltime_measure("/SPH/Density/Init");

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {

        treewalk_run(&tw);

        /* do final operations on results */
        tstart = second();

        queue = treewalk_get_queue(&tw, &Nactive);

        int npleft = 0;
#pragma omp parallel for if(Nactive > 32)
        for(i = 0; i < Nactive; i++) {
            int p = queue[i];
            density_post_process(p);
            /* will notify by setting DensityIterationDone */
            density_check_neighbours(p, Left, Right);
            if(iter >= MAXITER - 10)
            {
                 message(1, "i=%d task=%d ID=%lu Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                     queue[p], ThisTask, P[p].ID, P[p].Hsml, Left[p], Right[p],
                     (float) P[p].n.NumNgb, Right[p] - Left[p], P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
            }

            if(!P[p].DensityIterationDone) {
#pragma omp atomic
                npleft ++;
            }
        }

        myfree(queue);
        tend = second();
        timecomp3 += timediff(tstart, tend);

        sumup_large_ints(1, &npleft, &ntot);

        if(ntot > 0)
        {
            iter++;

            /*
            if(ntot < 1 ) {
                for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
                {
                    if(density_isactive(i) && !P[i].DensityIterationDone) {
                        message
                            (1, "i=%d task=%d ID=%llu type=%d, Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                             i, ThisTask, P[i].ID, P[i].Type, P[i].Hsml, Left[i], Right[i],
                             (float) P[i].n.NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                    }
                }

            }
            */

            if(iter > 0) {
                message(0, "ngb iteration %d: need to repeat for %ld particles.\n", iter, ntot);
            }

            if(iter > MAXITER)
            {
                endrun(1155, "failed to converge in neighbour iteration in density()\n");
            }
        }
    }
    while(ntot > 0);

    myfree(Right);
    myfree(Left);


    /* collect some timing information */

    timeall = walltime_measure(WALLTIME_IGNORE);

    timecomp = timecomp3 + tw.timecomp1 + tw.timecomp2;
    timewait = tw.timewait1 + tw.timewait2;
    timecomm = tw.timecommsumm1 + tw.timecommsumm2;

    walltime_add("/SPH/Density/Compute", timecomp);
    walltime_add("/SPH/Density/Wait", timewait);
    walltime_add("/SPH/Density/Comm", timecomm);
    walltime_add("/SPH/Density/Misc", timeall - (timecomp + timewait + timecomm));
}

double density_decide_hsearch(int targettype, double h) {
#ifdef BLACK_HOLES
    if(targettype == 5 && All.BlackHoleFeedbackRadius > 0) {
        /* BlackHoleFeedbackRadius is in comoving.
         * The Phys radius is capped by BlackHoleFeedbackRadiusMaxPhys
         * just like how it was done for grav smoothing.
         * */
        double rds;
        rds = All.BlackHoleFeedbackRadiusMaxPhys / All.cf.a;

        if(rds > All.BlackHoleFeedbackRadius) {
            rds = All.BlackHoleFeedbackRadius;
        }
        return rds;
    } else {
        return h;
    }
#else
    return h;
#endif

}

static void density_copy(int place, TreeWalkQueryDensity * I) {
    I->Hsml = P[place].Hsml;

    I->Type = P[place].Type;

#ifdef BLACK_HOLES
    if(P[place].Type != 0)
    {
        I->Vel[0] = 0;
        I->Vel[1] = 0;
        I->Vel[2] = 0;
    }
    else
#endif
    {
        I->Vel[0] = SPHP(place).VelPred[0];
        I->Vel[1] = SPHP(place).VelPred[1];
        I->Vel[2] = SPHP(place).VelPred[2];
    }
#ifdef VOLUME_CORRECTION
    I->DensityOld = SPHP(place).DensityOld;
#endif

#ifdef WINDS
    I->DelayTime = SPHP(place).DelayTime;
#endif

}

static void density_reduce(int place, TreeWalkResultDensity * remote, enum TreeWalkReduceMode mode) {
    TREEWALK_REDUCE(P[place].n.dNumNgb, remote->Ngb);

#ifdef HYDRO_COST_FACTOR
    /* these will be added */
    P[place].GravCost += HYDRO_COST_FACTOR * All.cf.a * remote->Ninteractions;
#endif

    if(P[place].Type == 0)
    {
        TREEWALK_REDUCE(SPHP(place).Density, remote->Rho);
        TREEWALK_REDUCE(SPHP(place).DhsmlDensityFactor, remote->DhsmlDensity);
#ifdef DENSITY_INDEPENDENT_SPH
        TREEWALK_REDUCE(SPHP(place).EgyWtDensity, remote->EgyRho);
        TREEWALK_REDUCE(SPHP(place).DhsmlEgyDensityFactor, remote->DhsmlEgyDensity);
#endif

        TREEWALK_REDUCE(SPHP(place).DivVel, remote->Div);
        TREEWALK_REDUCE(SPHP(place).Rot[0], remote->Rot[0]);
        TREEWALK_REDUCE(SPHP(place).Rot[1], remote->Rot[1]);
        TREEWALK_REDUCE(SPHP(place).Rot[2], remote->Rot[2]);

#ifdef VOLUME_CORRECTION
        TREEWALK_REDUCE(SPHP(place).DensityStd, remote->DensityStd);
#endif

#ifdef SPH_GRAD_RHO
        TREEWALK_REDUCE(SPHP(place).GradRho[0], remote->GradRho[0]);
        TREEWALK_REDUCE(SPHP(place).GradRho[1], remote->GradRho[1]);
        TREEWALK_REDUCE(SPHP(place).GradRho[2], remote->GradRho[2]);
#endif

    }

#ifdef BLACK_HOLES
    if(P[place].Type == 5)
    {
        TREEWALK_REDUCE(BHP(place).Density, remote->Rho);
        TREEWALK_REDUCE(BHP(place).FeedbackWeightSum, remote->FeedbackWeightSum);
        TREEWALK_REDUCE(BHP(place).Entropy, remote->SmoothedEntropy);
        TREEWALK_REDUCE(BHP(place).Pressure, remote->SmoothedPressure);

        TREEWALK_REDUCE(BHP(place).SurroundingGasVel[0], remote->GasVel[0]);
        TREEWALK_REDUCE(BHP(place).SurroundingGasVel[1], remote->GasVel[1]);
        TREEWALK_REDUCE(BHP(place).SurroundingGasVel[2], remote->GasVel[2]);
    }
#endif
}
/*! This function represents the core of the SPH density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */

static void
density_ngbiter(
        TreeWalkQueryDensity * I,
        TreeWalkResultDensity * O,
        TreeWalkNgbIterDensity * iter,
        LocalTreeWalk * lv)
{
    if(O == NULL) {
        double h;
        double hsearch;
        h = I->Hsml;
        hsearch = density_decide_hsearch(I->Type, h);

        density_kernel_init(&iter->kernel, h);
        iter->kernel_volume = density_kernel_volume(&iter->kernel);
    #ifdef BLACK_HOLES
        density_kernel_init(&iter->bh_feedback_kernel, hsearch);
    #endif

        iter->base.Hsml = hsearch;
        iter->base.mask = 1; /* gas only */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;
    double * dist = iter->base.dist;

#ifdef WINDS
    if(HAS(All.WindModel, WINDS_DECOUPLE_SPH)) {
        if(SPHP(other).DelayTime > 0)	/* partner is a wind particle */
            if(!(I->DelayTime > 0))	/* if I'm not wind, then ignore the wind particle */
                return;
    }
#endif
#ifdef BLACK_HOLES
    if(P[other].Mass == 0)
        return;
#ifdef WINDS
        /* blackhole doesn't accrete from wind, regardlies coupled or
         * not */
    if(I->Type == 5 && SPHP(other).DelayTime > 0)	/* partner is a wind particle */
        return;
#endif
#endif

    if(r2 < iter->kernel.HH)
    {

        double u = r * iter->kernel.Hinv;
        double wk = density_kernel_wk(&iter->kernel, u);
        double dwk = density_kernel_dwk(&iter->kernel, u);

        double mass_j = P[other].Mass;

#ifdef VOLUME_CORRECTION
        O->Rho += (mass_j * wk * pow(I->DensityOld / SPHP(other).DensityOld, VOLUME_CORRECTION));
        O->DensityStd += (mass_j * wk);
#else
        O->Rho += (mass_j * wk);
#endif
        O->Ngb += wk * iter->kernel_volume;

        /* Hinv is here becuase O->DhsmlDensity is drho / dH.
         * nothing to worry here */
        O->DhsmlDensity += mass_j * density_kernel_dW(&iter->kernel, u, wk, dwk);

#ifdef DENSITY_INDEPENDENT_SPH
        O->EgyRho += mass_j * SPHP(other).EntVarPred * wk;
        O->DhsmlEgyDensity += mass_j * SPHP(other).EntVarPred * density_kernel_dW(&iter->kernel, u, wk, dwk);
#endif


#ifdef BLACK_HOLES
        O->SmoothedPressure += (mass_j * wk * SPHP(other).Pressure);
        O->SmoothedEntropy += (mass_j * wk * SPHP(other).Entropy);
        O->GasVel[0] += (mass_j * wk * SPHP(other).VelPred[0]);
        O->GasVel[1] += (mass_j * wk * SPHP(other).VelPred[1]);
        O->GasVel[2] += (mass_j * wk * SPHP(other).VelPred[2]);
#endif

#ifdef SPH_GRAD_RHO
        if(r > 0)
        {
            int d;
            for (d = 0; d < 3; d ++) {
                O->GradRho[d] += mass_j * dwk * dist[d] / r;
            }
        }
#endif

        if(r > 0)
        {
            double fac = mass_j * dwk / r;
            double dv[3];
            double rot[3];
            int d;
            for(d = 0; d < 3; d ++) {
                dv[d] = I->Vel[d] - SPHP(other).VelPred[d];
            }
            O->Div += -fac * dotproduct(dist, dv);

            crossproduct(dv, dist, rot);
            for(d = 0; d < 3; d ++) {
                O->Rot[d] += fac * rot[d];
            }
        }
    }
#ifdef BLACK_HOLES
    if(I->Type == 5 && r2 < iter->bh_feedback_kernel.HH)
    {
#ifdef WINDS
        /* blackhole doesn't accrete from wind, regardlies coupled or
         * not */
        if(SPHP(other).DelayTime > 0)	/* partner is a wind particle */
            return;
#endif
        double mass_j;
        if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_OPTTHIN)) {
            double nh0 = 1.0;
            double nHeII = 0;
            double ne = SPHP(other).Ne;
            struct UVBG uvbg;
            GetParticleUVBG(other, &uvbg);
            AbundanceRatios(DMAX(All.MinEgySpec,
                        SPHP(other).Entropy / GAMMA_MINUS1
                        * pow(SPHP(other).EOMDensity * All.cf.a3inv,
                            GAMMA_MINUS1)),
                    SPHP(other).Density * All.cf.a3inv, &uvbg, &ne, &nh0, &nHeII);
            if(r2 > 0)
                O->FeedbackWeightSum += (P[other].Mass * nh0) / r2;
        } else {
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_MASS)) {
                mass_j = P[other].Mass;
            } else {
                mass_j = P[other].Hsml * P[other].Hsml * P[other].Hsml;
            }
            if(HAS(All.BlackHoleFeedbackMethod, BH_FEEDBACK_SPLINE)) {
                double u = r * iter->bh_feedback_kernel.Hinv;
                O->FeedbackWeightSum += (mass_j *
                      density_kernel_wk(&iter->bh_feedback_kernel, u)
                       );
            } else {
                O->FeedbackWeightSum += (mass_j);
            }
        }
    }
#endif

    /* some performance measures not currently used */
#ifdef HYDRO_COST_FACTOR
    O->Ninteractions ++;
#endif
}

static int density_isactive(int n)
{
    if(P[n].DensityIterationDone) return 0;

    if(P[n].TimeBin < 0) {
        endrun(9999, "TimeBin negative!\n use DensityIterationDone flag");
    }
#ifdef BLACK_HOLES
    if(P[n].Type == 5)
        return 1;
#endif

    if(P[n].Type == 0)
        return 1;

    return 0;
}

static void density_post_process(int i) {
    if(P[i].Type == 0)
    {
        if(SPHP(i).Density > 0)
        {
#ifdef VOLUME_CORRECTION
            SPHP(i).DensityOld = SPHP(i).DensityStd;
#endif
            SPHP(i).DhsmlDensityFactor *= P[i].Hsml / (NUMDIMS * SPHP(i).Density);
            if(SPHP(i).DhsmlDensityFactor > -0.9)	/* note: this would be -1 if only a single particle at zero lag is found */
                SPHP(i).DhsmlDensityFactor = 1 / (1 + SPHP(i).DhsmlDensityFactor);
            else
                SPHP(i).DhsmlDensityFactor = 1;

#ifdef DENSITY_INDEPENDENT_SPH
            if((SPHP(i).EntVarPred>0)&&(SPHP(i).EgyWtDensity>0))
            {
                SPHP(i).DhsmlEgyDensityFactor *= P[i].Hsml/ (NUMDIMS * SPHP(i).EgyWtDensity);
                SPHP(i).DhsmlEgyDensityFactor *= -SPHP(i).DhsmlDensityFactor;
                SPHP(i).EgyWtDensity /= SPHP(i).EntVarPred;
            } else {
                SPHP(i).DhsmlEgyDensityFactor=0;
                SPHP(i).EntVarPred=0;
                SPHP(i).EgyWtDensity=0;
            }
#endif

            SPHP(i).CurlVel = sqrt(SPHP(i).Rot[0] * SPHP(i).Rot[0] +
                    SPHP(i).Rot[1] * SPHP(i).Rot[1] +
                    SPHP(i).Rot[2] * SPHP(i).Rot[2]) / SPHP(i).Density;

            SPHP(i).DivVel /= SPHP(i).Density;

        }

#ifdef DENSITY_INDEPENDENT_SPH
        SPHP(i).Pressure = pow(SPHP(i).EntVarPred*SPHP(i).EgyWtDensity,GAMMA);
#else
        int dt_step = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0);
        int dt_entr = (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;
        SPHP(i).Pressure = (SPHP(i).Entropy + SPHP(i).DtEntropy * dt_entr) * pow(SPHP(i).Density, GAMMA);
#endif // DENSITY_INDEPENDENT_SPH

#ifdef SOFTEREQS
        /* use an intermediate EQS, between isothermal and the full multiphase model */
        if(SPHP(i).Density * All.cf.a3inv >= All.PhysDensThresh) {
            SPHP(i).Pressure = All.FactorForSofterEQS * SPHP(i).Pressure +
                (1 - All.FactorForSofterEQS) * All.cf.fac_egy * GAMMA_MINUS1 * SPHP(i).Density * All.InitGasU;
        }
#endif //SOFTEREQS
    }

#ifdef BLACK_HOLES
    if(P[i].Type == 5)
    {
        if(BHP(i).Density > 0)
        {
            BHP(i).Entropy /= BHP(i).Density;
            BHP(i).Pressure /= BHP(i).Density;

            BHP(i).SurroundingGasVel[0] /= BHP(i).Density;
            BHP(i).SurroundingGasVel[1] /= BHP(i).Density;
            BHP(i).SurroundingGasVel[2] /= BHP(i).Density;
        }
    }
#endif
}

void density_check_neighbours (int i, MyFloat * Left, MyFloat * Right) {
    /* now check whether we had enough neighbours */

    double desnumngb = All.DesNumNgb;

#ifdef BLACK_HOLES
    if(P[i].Type == 5)
        desnumngb = All.DesNumNgb * All.BlackHoleNgbFactor;
#endif

    if(P[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation) ||
            (P[i].n.NumNgb > (desnumngb + All.MaxNumNgbDeviation)
             && P[i].Hsml > (1.01 * All.MinGasHsml)))
    {
        /* need to redo this particle */
        if(P[i].DensityIterationDone) {
            /* should have been 0*/
            endrun(999993, "Already has DensityIterationDone set, bad memory intialization.");
        }

        if(Left[i] > 0 && Right[i] > 0)
            if((Right[i] - Left[i]) < 1.0e-3 * Left[i])
            {
                /* this one should be ok */
                P[i].DensityIterationDone = 1;
                return;
            }

        if(P[i].n.NumNgb < (desnumngb - All.MaxNumNgbDeviation))
            Left[i] = DMAX(P[i].Hsml, Left[i]);
        else
        {
            if(Right[i] != 0)
            {
                if(P[i].Hsml < Right[i])
                    Right[i] = P[i].Hsml;
            }
            else
                Right[i] = P[i].Hsml;
        }

        if(Right[i] > 0 && Left[i] > 0)
            P[i].Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
        else
        {
            if(Right[i] == 0 && Left[i] == 0)
                endrun(8188, "Cannot occur. Check for memory corruption.");	/* can't occur */

            if(Right[i] == 0 && Left[i] > 0)
            {
                if(P[i].Type == 0 && fabs(P[i].n.NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].n.NumNgb -
                            desnumngb) / (NUMDIMS * P[i].n.NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

                    if(fac < 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml *= 1.26;
                }
                else
                    P[i].Hsml *= 1.26;
            }

            if(Right[i] > 0 && Left[i] == 0)
            {
                if(P[i].Type == 0 && fabs(P[i].n.NumNgb - desnumngb) < 0.5 * desnumngb)
                {
                    double fac = 1 - (P[i].n.NumNgb -
                            desnumngb) / (NUMDIMS * P[i].n.NumNgb) *
                        SPHP(i).DhsmlDensityFactor;

                    if(fac > 1 / 1.26)
                        P[i].Hsml *= fac;
                    else
                        P[i].Hsml /= 1.26;
                }
                else
                    P[i].Hsml /= 1.26;
            }
        }

        if(P[i].Hsml < All.MinGasHsml)
            P[i].Hsml = All.MinGasHsml;

#ifdef BLACK_HOLES
        if(P[i].Type == 5)
            if(Left[i] > All.BlackHoleMaxAccretionRadius)
            {
                /* this will stop the search for a new BH smoothing length in the next iteration */
                P[i].Hsml = Left[i] = Right[i] = All.BlackHoleMaxAccretionRadius;
            }
#endif

    }
    else {
        P[i].DensityIterationDone = 1;
    }

}

