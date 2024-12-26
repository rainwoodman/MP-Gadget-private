#include <math.h>
#include "treewalk.h"
#include "densitykernel.h"
#include "bhdynfric.h"
#include "density.h"
#include "walltime.h"

#define BHPOTVALUEINIT 1.0e29

static struct BlackholeDynFricParams
{
    int BH_DynFrictionMethod;/*0 for off; 1 for Star Only; 2 for DM+Star; 3 for DM+Star+Gas */
    int BH_DFBoostFactor; /*Optional boost factor for DF */
    double BH_DFbmax; /* the maximum impact range, in physical unit of kpc. */
    int BlackHoleRepositionEnabled; /* If true, enable repositioning the BH to the potential minimum. If false, do dynamic friction.*/
} blackhole_dynfric_params;

/*Set the parameters of the BH module*/
void set_blackhole_dynfric_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        blackhole_dynfric_params.BH_DynFrictionMethod = param_get_int(ps, "BH_DynFrictionMethod");
        blackhole_dynfric_params.BH_DFBoostFactor = param_get_int(ps, "BH_DFBoostFactor");
        blackhole_dynfric_params.BH_DFbmax = param_get_double(ps, "BH_DFbmax");
        blackhole_dynfric_params.BlackHoleRepositionEnabled = param_get_int(ps, "BlackHoleRepositionEnabled");
    }
    MPI_Bcast(&blackhole_dynfric_params, sizeof(struct BlackholeDynFricParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int
BHGetRepositionEnabled(void)
{
    return blackhole_dynfric_params.BlackHoleRepositionEnabled;
}

#define BHDYN_GET_PRIV(tw) ((struct BHDynFricPriv *) (tw->priv))

/*****************************************************************************/
typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
} TreeWalkQueryBHDynfric;

typedef struct {
    TreeWalkResultBase base;

    MyFloat SurroundingVel[3];
    MyFloat SurroundingDensity;
    MyFloat SurroundingRmsVel;
    /* Minimum potential for diagnostics*/
    MyFloat BH_MinPotPos[3];
    MyFloat BH_MinPotVel[3];
    MyFloat BH_MinPot;
} TreeWalkResultBHDynfric;

typedef struct {
    TreeWalkNgbIterBase base;
    DensityKernel dynfric_kernel;
} TreeWalkNgbIterBHDynfric;

int
blackhole_dynfric_treemask(void)
{
    /* dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1, gas if BH_DynFrictionMethod  == 3.
     * The BH do not contribute dynamic friction but are here so the potential minimum is updated. */
    int treemask = STARMASK + BHMASK;
    /* Don't necessarily need dark matter */
    if(blackhole_dynfric_params.BH_DynFrictionMethod > 1)
        treemask += DMMASK;
    if(blackhole_dynfric_params.BH_DynFrictionMethod > 2)
        treemask += GASMASK;
    /* For repositioning we want all particles*/
    if(blackhole_dynfric_params.BlackHoleRepositionEnabled)
        treemask = ALLMASK;
    return treemask;
}

/*************************************************************************************/
/* Compute the DF acceleration in the BH from stored quantities*/
static void
blackhole_compute_dfaccel(const int n, const double atime, const double Grav)
{
    int j;
    for(j = 0; j < 3; j++)
        BHP(n).DFAccel[j] = 0;
    /***********************************************************************************/
    /* This is Gizmo's implementation of dynamic friction                              */
    /* c.f. section 3.1 in http://www.tapir.caltech.edu/~phopkins/public/notes_blackholes.pdf */
    /* Compute dynamic friction accel when DF turned on                                */
    /* averaged value for colomb logarithm and integral over the distribution function */
    /* acc_friction = -4*pi*G^2 * Mbh * log(lambda) * rho * f_of_x * bhvel / |bhvel^3| */
    /*       f_of_x = [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                                */
    /*       lambda = b_max * v^2 / G / (M+m)                                          */
    /*        b_max = Size of system (e.g. Rvir)                                       */
    /*            v = Relative velocity of BH with respect to the environment          */
    /*            M = Mass of BH                                                       */
    /*            m = individual mass elements composing the large system (e.g. m<<M)  */
    /*            x = v/sqrt(2)/sigma                                                  */
    /*        sigma = width of the max. distr. of the host system                      */
    /*                (e.g. sigma = v_disp / 3                                         */
    if(BHP(n).DF_SurroundingDensity > 0){
        /* Calculate Coulumb Logarithm */
        double bhvel = 0;
        for(j = 0; j < 3; j++)
            bhvel += pow(P[n].Vel[j] - BHP(n).DF_SurroundingVel[j], 2);
        bhvel = sqrt(bhvel);

        if(!isfinite(bhvel)) {
            endrun(6, "Bad bhvel %g vel %g %g %g DF vel %g %g %g id %ld n %d mass %g\n",
                bhvel, P[n].Vel[0], P[n].Vel[1], P[n].Vel[2],
                BHP(n).DF_SurroundingVel[0], BHP(n).DF_SurroundingVel[1], BHP(n).DF_SurroundingVel[2],P[n].ID, n, P[n].Mass);
        }
        /* There is no parameter in physical unit, so I kept everything in code unit */

        double x = bhvel / sqrt(2) / (BHP(n).DF_SurroundingRmsVel / 3);
        /* First term is approximation of the error function */
        const double a_erf = 8 * (M_PI - 3) / (3 * M_PI * (4. - M_PI));
        double f_of_x = x / fabs(x) * sqrt(1 - exp(-x * x * (4 / M_PI + a_erf * x * x)
            / (1 + a_erf * x * x))) - 2 * x / sqrt(M_PI) * exp(-x * x);
        /* Floor at zero */
        if (f_of_x < 0)
            f_of_x = 0;

        double lambda = 1. + blackhole_dynfric_params.BH_DFbmax * pow((bhvel/atime),2) / Grav / P[n].Mass;

        for(j = 0; j < 3; j++)
        {
            /* prevent DFAccel from exploding */
            if(bhvel > 0){
                BHP(n).DFAccel[j] = - 4. * M_PI * Grav * Grav * P[n].Mass * BHP(n).DF_SurroundingDensity * log(lambda) * f_of_x * (P[n].Vel[j] - BHP(n).DF_SurroundingVel[j]) / pow(bhvel, 3);
                BHP(n).DFAccel[j] *= atime;  // convert to code unit of acceleration
                BHP(n).DFAccel[j] *= blackhole_dynfric_params.BH_DFBoostFactor; // Add a boost factor
            }
        }
// #ifdef DEBUG
        // message(2,"x=%e, log(lambda)=%e, fof_x=%e, Mbh=%e, ratio=%e \n",
           // x,log(lambda),f_of_x,P[n].Mass,BHP(n).DFAccel[0]/P[n].FullTreeGravAccel[0]);
// #endif
    }
}

static void
blackhole_dynfric_postprocess(int n, TreeWalk * tw)
{
    if(BHP(n).DF_SurroundingDensity > 0){
        /* normalize velocity/dispersion */
        BHP(n).DF_SurroundingRmsVel /= BHP(n).DF_SurroundingDensity;
        BHP(n).DF_SurroundingRmsVel = sqrt(BHP(n).DF_SurroundingRmsVel);
        int j;
        for(j = 0; j < 3; j++)
            BHP(n).DF_SurroundingVel[j] /= BHP(n).DF_SurroundingDensity;
    }
    else {
        #pragma omp atomic update
        BHDYN_GET_PRIV(tw)->ZeroDF++;
        #pragma omp atomic update
        BHDYN_GET_PRIV(tw)->ZeroDFMass += BHP(n).Mass;
    }
}

static void
blackhole_repos_reduce(int place, TreeWalkResultBHDynfric * remote, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    int k;
    if(BHP(place).MinPot > remote->BH_MinPot)
    {
        BHP(place).JumpToMinPot = blackhole_dynfric_params.BlackHoleRepositionEnabled;
        BHP(place).MinPot = remote->BH_MinPot;
        for(k = 0; k < 3; k++) {
            /* Movement occurs in drift.c */
            BHP(place).MinPotPos[k] = remote->BH_MinPotPos[k];
            BHP(place).MinPotVel[k] = remote->BH_MinPotVel[k];
        }
    }
}

static void
blackhole_dynfric_reduce(int place, TreeWalkResultBHDynfric * remote, enum TreeWalkReduceMode mode, TreeWalk * tw){
    TREEWALK_REDUCE(BHP(place).DF_SurroundingDensity, remote->SurroundingDensity);
    int j;
    for(j = 0; j < 3; j++)
        TREEWALK_REDUCE(BHP(place).DF_SurroundingVel[j], remote->SurroundingVel[j]);
    TREEWALK_REDUCE(BHP(place).DF_SurroundingRmsVel, remote->SurroundingRmsVel);
    /* Find minimum potential*/
    blackhole_repos_reduce(place, remote, mode, tw);
}

static void
blackhole_dynfric_copy(int place, TreeWalkQueryBHDynfric * I, TreeWalk * tw){
    /* SPH kernel width should be the only thing needed */
    I->Hsml = P[place].Hsml;
}

static void
blackhole_minpot_ngbiter(TreeWalkQueryBHDynfric * I,
        TreeWalkResultBHDynfric * O,
        TreeWalkNgbIterBHDynfric * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        O->BH_MinPot = BHPOTVALUEINIT;
        int d;
        for(d = 0; d < 3; d++) {
            O->BH_MinPotPos[d] = I->base.Pos[d];
        }
        iter->base.mask = blackhole_dynfric_treemask();
        iter->base.Hsml = I->Hsml;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        density_kernel_init(&iter->dynfric_kernel, I->Hsml, GetDensityKernelType());
        return;
    }

    int other = iter->base.other;
    double r2 = iter->base.r2;

    /* Find the black hole potential minimum. */
    if(r2 < iter->dynfric_kernel.HH && P[other].Potential < O->BH_MinPot)
    {
        int d;
        O->BH_MinPot = P[other].Potential;
        for(d = 0; d < 3; d++) {
            O->BH_MinPotPos[d] = P[other].Pos[d];
            O->BH_MinPotVel[d] = P[other].Vel[d];
        }
    }
}

static void
blackhole_dynfric_ngbiter(TreeWalkQueryBHDynfric * I,
        TreeWalkResultBHDynfric * O,
        TreeWalkNgbIterBHDynfric * iter,
        LocalTreeWalk * lv) {

    /* Update potential minimum*/
    blackhole_minpot_ngbiter(I, O, iter, lv);

   if(iter->base.other == -1)
       return;

    int other = iter->base.other;
    double r = iter->base.r;
    double r2 = iter->base.r2;

    /* Collect Star/+DM/+Gas density/velocity for DF computation */
    if(P[other].Type == 4 || (P[other].Type == 1 && blackhole_dynfric_params.BH_DynFrictionMethod > 1) ||
        (P[other].Type == 0 && blackhole_dynfric_params.BH_DynFrictionMethod == 3) ){
        if(r2 < iter->dynfric_kernel.HH) {
            double u = r * iter->dynfric_kernel.Hinv;
            double wk = density_kernel_wk(&iter->dynfric_kernel, u);
            float mass_j = P[other].Mass;
            int k;
            O->SurroundingDensity += (mass_j * wk);
            MyFloat VelPred[3];
            if(P[other].Type == 0)
                SPH_VelPred(other, VelPred, BHDYN_GET_PRIV(lv->tw)->kf);
            else {
                DM_VelPred(other, VelPred, BHDYN_GET_PRIV(lv->tw)->kf);
            }
            for (k = 0; k < 3; k++){
                O->SurroundingVel[k] += (mass_j * wk * VelPred[k]);
                O->SurroundingRmsVel += (mass_j * wk * pow(VelPred[k], 2));
            }
        }
    }
}

/* Initialise the minimum potential*/
static void
blackhole_minpot_preprocess(int n, TreeWalk * tw)
{
    int j;
    /* Note that the potential is only updated when it is from all particles.
     * In particular this means that it is not updated for hierarchical gravity
     * when the number of active particles is less than the total number of particles
     * (because then the tree does not contain all forces). */
    BHP(n).MinPot = P[n].Potential;

    for(j = 0; j < 3; j++) {
        BHP(n).MinPotPos[j] = P[n].Pos[j];
    }
}

/* Simple treewalk that just finds the local potential minimum for BH repositioning.*/
void
blackhole_minpot(int * ActiveBlackHoles, const int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, struct BHDynFricPriv * priv)
{
    /* Repositioning uses all particles: in practice it will usually be stars, gas or BH.*/
    ForceTree tree[1] = {0};
    message(0, "Building tree with all particles for repositioning\n");
    force_tree_rebuild_mask(tree, ddecomp, blackhole_dynfric_treemask(), NULL);
    walltime_measure("/BH/BuildRepos");


    /*************************************************************************/
    TreeWalk tw_repos[1] = {{0}};

    tw_repos->ev_label = "BH_REPOS";
    tw_repos->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_repos->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHDynfric);
    tw_repos->ngbiter = (TreeWalkNgbIterFunction) blackhole_minpot_ngbiter;
    tw_repos->haswork = NULL;
    tw_repos->postprocess = NULL;
    tw_repos->preprocess = blackhole_minpot_preprocess;
    tw_repos->fill = (TreeWalkFillQueryFunction) blackhole_dynfric_copy;
    tw_repos->reduce = (TreeWalkReduceResultFunction) blackhole_repos_reduce;
    tw_repos->query_type_elsize = sizeof(TreeWalkQueryBHDynfric);
    tw_repos->result_type_elsize = sizeof(TreeWalkResultBHDynfric);
    tw_repos->tree = tree;
    tw_repos->priv = priv;

    treewalk_run(tw_repos, ActiveBlackHoles, NumActiveBlackHoles);

    force_tree_free(tree);
    /*************************************************************************/
    walltime_measure("/BH/Repos");
}

static int
blackhole_dynfric_haswork(int n, TreeWalk * tw){
    /*Black hole not being swallowed*/
    return (P[n].Type == 5) && (!P[n].Swallowed) &&
    is_timebin_active(BHP(n).TimeBinDynFric, BHDYN_GET_PRIV(tw)->Ti_Current);
}

/* Returns total number of dynamic-friction active particles over all processors*/
int
blackhole_dynfric_num_active(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, inttime_t Ti_Current)
{
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0)
        return 0;

    int64_t i, nactive = 0;
    TreeWalk tw_dynfric[1] = {{0}};
    struct BHDynFricPriv priv[1] = {{0}};
    priv->Ti_Current = Ti_Current;
    tw_dynfric->priv = priv;
    #pragma omp parallel for reduction(+: nactive)
    for(i = 0; i < NumActiveBlackHoles; i++)
    {
        int n = ActiveBlackHoles[i];
        nactive += blackhole_dynfric_haswork(n, tw_dynfric);
    }
    int64_t totactive;
    MPI_Allreduce(&nactive, &totactive, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    return totactive;
}

void
blackhole_dynfric(int * ActiveBlackHoles, int64_t NumActiveBlackHoles, DomainDecomp * ddecomp, struct BHDynFricPriv * priv)
{
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0) {
        /* If there is no dynamic friction, do repositioning, and
         * run a special walk to find the potential minimum.*/
        if(blackhole_dynfric_params.BlackHoleRepositionEnabled)
            blackhole_minpot(ActiveBlackHoles, NumActiveBlackHoles, ddecomp, priv);
        return;
    }
    int64_t totdynfric = blackhole_dynfric_num_active(ActiveBlackHoles, NumActiveBlackHoles, priv->Ti_Current);
    if(!totdynfric)
        return;

    /* dynamical friction uses: stars, DM if BH_DynFrictionMethod > 1 gas if BH_DynFrictionMethod  == 3.
     * The DM in dynamic friction and accretion doesn't really do anything, so could perhaps be removed from the treebuild later.*/
    ForceTree tree[1] = {0};
    int treemask = blackhole_dynfric_treemask();
    message(0, "Building dynamic friction tree with types %d\n", treemask);
    force_tree_rebuild_mask(tree, ddecomp, treemask, NULL);
    walltime_measure("/BH/BuildDF");

    TreeWalk tw_dynfric[1] = {{0}};
    tw_dynfric->ev_label = "BH_DYNFRIC";
    tw_dynfric->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw_dynfric->ngbiter_type_elsize = sizeof(TreeWalkNgbIterBHDynfric);
    tw_dynfric->ngbiter = (TreeWalkNgbIterFunction) blackhole_dynfric_ngbiter;
    tw_dynfric->postprocess = (TreeWalkProcessFunction) blackhole_dynfric_postprocess;
    tw_dynfric->preprocess = blackhole_minpot_preprocess;
    tw_dynfric->fill = (TreeWalkFillQueryFunction) blackhole_dynfric_copy;
    tw_dynfric->reduce = (TreeWalkReduceResultFunction) blackhole_dynfric_reduce;
    tw_dynfric->query_type_elsize = sizeof(TreeWalkQueryBHDynfric);
    tw_dynfric->result_type_elsize = sizeof(TreeWalkResultBHDynfric);
    tw_dynfric->tree = tree;
    tw_dynfric->priv = priv;
    tw_dynfric->haswork = blackhole_dynfric_haswork;

    treewalk_run(tw_dynfric, ActiveBlackHoles, NumActiveBlackHoles);
    force_tree_free(tree);
    size_t totalzerodf;
    double totalzeromass;
    MPI_Reduce(&priv->ZeroDF, &totalzerodf, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&priv->ZeroDFMass, &totalzeromass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(totalzerodf > 0)
        message(0, "Dynamic Friction density is zero for %ld BHs avg mass %g.\n", totalzerodf, totalzeromass/totalzerodf);
}

/* Compute the DF acceleration for all active black holes*/
void
blackhole_dfaccel(int * ActiveBlackHoles, size_t NumActiveBlackHoles, const double atime, const double GravInternal)
{
    if (blackhole_dynfric_params.BH_DynFrictionMethod == 0)
        return;

    size_t i;
    #pragma omp parallel for
    for(i = 0; i < NumActiveBlackHoles; i++) {
        size_t n = ActiveBlackHoles ? ActiveBlackHoles[i] : i;
        blackhole_compute_dfaccel(n, atime, GravInternal);
    }
}
