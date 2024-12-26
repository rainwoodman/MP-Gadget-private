#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#include "utils.h"

#include "domain.h"
#include "forcetree.h"
#include "timestep.h"
#include "timebinmgr.h"
#include "exchange.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "walltime.h"
#include "bhdynfric.h"
#include "utils/paramset.h"
#include "utils/peano.h"
#include "utils/mpsort.h"

#define TAG_GRAV_A        18
#define TAG_GRAV_B        19

/*! \file domain.c
 *  \brief code for domain decomposition
 *
 *  This file contains the code for the domain decomposition of the
 *  simulation volume.  The domains are constructed from disjoint subsets
 *  of the leaves of a fiducial top-level tree that covers the full
 *  simulation volume. Domain boundaries hence run along tree-node
 *  divisions of a fiducial global BH tree. As a result of this method, the
 *  tree force are in principle strictly independent of the way the domains
 *  are cut. The domain decomposition can be carried out for an arbitrary
 *  number of CPUs. Individual domains are not cubical, but spatially
 *  coherent since the leaves are traversed in a Peano-Hilbert order and
 *  individual domains form segments along this order.  This also ensures
 *  that each domain has a small surface to volume ratio, which minimizes
 *  communication.
 */

static DomainParams domain_params;
/**
 * Policy for domain decomposition.
 *
 * Several policies will be attempted before we give up and die.
 * */
typedef struct {
    int SubSampleDistance; /** Frequency of subsampling */
    int PreSort; /** PreSort the local particles before subsampling, creating a fair subsample */
    int NTopLeaves; /** Number of Peano-Hilbert segments to create before balancing. Should be DomainOverDecompositionFactor * NTask*/
} DomainDecompositionPolicy;

/* It is important for the stability of the code that this struct is 64-bit aligned!*/
struct local_topnode_data
{
    /*These members are copied into topnode_data*/
    peano_t StartKey;		/*!< first Peano-Hilbert key in top-level node */
    int Shift;		/*!< log2 of number of Peano-Hilbert mesh-cells represented by top-level node */
    int Daughter;			/*!< index of first daughter cell (out of 8) of top-level node */
    /*Below members are only used in this file*/
    int Parent;
    int64_t Count;      /* the number of 'subsample' particles in this top-level node */
    int64_t Cost;       /* the cost of 'subsample' particle in this top-level node */
};

struct local_particle_data
{
    peano_t Key;
    int64_t Cost;
};

/*This is a helper for the tests*/
void set_domain_par(DomainParams dp)
{
    domain_params = dp;
}

/*Set the parameters of the domain module*/
void set_domain_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        domain_params.DomainOverDecompositionFactor = param_get_int(ps, "DomainOverDecompositionFactor");
        /* Create one domain per thread. This helps the balance and makes the treebuild merge faster*/
        if(domain_params.DomainOverDecompositionFactor < 0)
            domain_params.DomainOverDecompositionFactor = omp_get_max_threads();
        if(domain_params.DomainOverDecompositionFactor == 0)
            domain_params.DomainOverDecompositionFactor = floor(omp_get_max_threads()/2);                
        if(domain_params.DomainOverDecompositionFactor < 4)
            domain_params.DomainOverDecompositionFactor = 4;
        domain_params.TopNodeAllocFactor = param_get_double(ps, "TopNodeAllocFactor");
        domain_params.DomainUseGlobalSorting = param_get_int(ps, "DomainUseGlobalSorting");
        domain_params.SetAsideFactor = 1.;
    }
    MPI_Bcast(&domain_params, sizeof(DomainParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void
domain_assign_balanced(DomainDecomp * ddecomp, int64_t * cost, const int NsegmentPerTask);

static int domain_allocate(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy);

static int
domain_check_memory_bound(const DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount);

static int domain_attempt_decompose(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy, const int MaxTopNodes);

static int
domain_balance(DomainDecomp * ddecomp);

static int domain_determine_global_toptree(DomainDecompositionPolicy * policy, struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes, MPI_Comm DomainComm);

static void
domain_compute_costs(DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount);

static void
domain_toptree_merge(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB, int * treeASize, const int MaxTopNodes);

static int domain_check_for_local_refine_subsample(
    DomainDecompositionPolicy * policy,
    struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes, const MPI_Comm DomainComm
    );

static int
domain_global_refine(struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes, const int64_t countlimit, const int64_t costlimit);

static void
domain_create_topleaves(DomainDecomp * ddecomp, int no, int * next);

static int
domain_layoutfunc(int n, const void * userdata);

static int
domain_policies_init(DomainDecompositionPolicy policies[], const int Npolicies);

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
#define NPOLICY 16
void domain_decompose_full(DomainDecomp * ddecomp)
{
    static DomainDecompositionPolicy policies[NPOLICY];
    static int Npolicies = 0;

    /* start from last successful policy to avoid retries */
    static int LastSuccessfulPolicy = 0;

    if (Npolicies == 0) {
        Npolicies = domain_policies_init(policies, NPOLICY);
    }

    message(0, "domain decomposition... (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    int decompose_failed = 1;
    int i;
    for(i = LastSuccessfulPolicy; i < Npolicies; i ++)
    {
        domain_free(ddecomp);

#ifdef DEBUG
        domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);
#endif
        message(0, "Attempting new domain decomposition policy: Topleaves=%d GlobalSort=%d, SubSampleDistance=%d PreSort=%d\n", policies[i].NTopLeaves, domain_params.DomainUseGlobalSorting, policies[i].SubSampleDistance, policies[i].PreSort);

        /* Keep going with the same policy until we have enough topnodes to make it work.*/
        do {
            int MaxTopNodes = domain_allocate(ddecomp, &policies[i]);

            decompose_failed = domain_attempt_decompose(ddecomp, &policies[i], MaxTopNodes);
            decompose_failed = MPIU_Any(decompose_failed, ddecomp->DomainComm);

            if(decompose_failed) {
                domain_free(ddecomp);
                /* We have not enough topnodes, get more.*/
                domain_params.TopNodeAllocFactor *= 1.2;
                message(0, "Increasing topnodes from %d to %d.\n", MaxTopNodes, (int) (MaxTopNodes * 1.2));
                if(domain_params.TopNodeAllocFactor > 10)
                    endrun(5, "TopNodeAllocFactor = %g, unreasonably large!\n", domain_params.TopNodeAllocFactor);
            }
        } while(decompose_failed);

        /* Still try an exchange if this is the last policy.*/
        if(domain_balance(ddecomp) && (i < Npolicies-1))
            continue;

        /* copy the used nodes from temp to the true. */
        struct topleaf_data * OldTopLeaves = ddecomp->TopLeaves;
        struct topnode_data * OldTopNodes = ddecomp->TopNodes;

        ddecomp->TopNodes  = (struct topnode_data *) mymalloc2("TopNodes", sizeof(ddecomp->TopNodes[0]) * ddecomp->NTopNodes);
        /* add 1 extra to mark the end of TopLeaves; see assign */
        ddecomp->TopLeaves = (struct topleaf_data *) mymalloc2("TopLeaves", sizeof(ddecomp->TopLeaves[0]) * (ddecomp->NTopLeaves + 1));

        memcpy(ddecomp->TopLeaves, OldTopLeaves, ddecomp->NTopLeaves* sizeof(ddecomp->TopLeaves[0]));
        memcpy(ddecomp->TopNodes, OldTopNodes, ddecomp->NTopNodes * sizeof(ddecomp->TopNodes[0]));

        /* no longer useful */
        myfree(OldTopLeaves);
        myfree(OldTopNodes);

        if(domain_exchange(domain_layoutfunc, ddecomp, NULL, PartManager, SlotsManager, ddecomp->DomainComm)) {
            message(0,"Could not exchange particles\n");
            if(i == Npolicies - 1)
                endrun(5, "Ran out of policies!\n");
            continue;
        }
        else {
            LastSuccessfulPolicy = i;
            break;
        }
    }

    if(decompose_failed) {
        endrun(0, "No suitable domain decomposition policy worked for this particle distribution\n");
    }


    /*Do a garbage collection so that the slots are ordered
     *the same as the particles, garbage is at the end and all particles are in peano order.*/
    slots_gc_sorted(PartManager, SlotsManager);

    /*Ensure collective*/
    MPIU_Barrier(ddecomp->DomainComm);
    message(0, "Domain decomposition done.\n");

    report_memory_usage("DOMAIN");

    walltime_measure("/Domain/PeanoSort");
}

/*Check whether a particle is inside the volume covered by a node,
 * by checking whether each dimension is close enough to center (L1 metric).*/
static inline int inside_topleaf(const int topleaf, const double Pos[3], const ForceTree * const tree)
{
    /* During fof particle exchange topleaf is over-written with the target task.
     * This is usually fine: if the index of the target top leaf happens to be the
     * same as the target task we just have nothing to do. But make sure we don't have a bad index,
     * just in case. Usually we should have at least one topleaf per task so this should never happen. */
    if(topleaf >= tree->NTopLeaves || topleaf < 0)
        return 0;
    /* Find treenode corresponding to this topleaf*/
    const struct NODE * const node = &tree->Nodes[tree->TopLeaves[topleaf].treenode];

    /*One can also use a loop, but the compiler unrolls it only at -O3,
     *so this is a little faster*/
    int inside =
        (fabs(2*(Pos[0] - node->center[0])) <= node->len) *
        (fabs(2*(Pos[1] - node->center[1])) <= node->len) *
        (fabs(2*(Pos[2] - node->center[2])) <= node->len);
    return inside;
}

/* This is a cut-down version of the domain decomposition that leaves the
 * domain grid intact, but exchanges the particles and rebuilds the tree */
int domain_maintain(DomainDecomp * ddecomp, struct DriftData * drift)
{
    message(0, "Attempting a domain exchange\n");

    /* Find drift factor*/
    int i;
    /* Can't update the random shift without re-decomposing domain*/
    const double rel_random_shift[3] = {0};
    double ddrift = 0;
    if(drift)
        ddrift = get_exact_drift_factor(drift->CP, drift->ti0, drift->ti1);

    /*Garbage particles are counted so we have an accurate memory estimate*/
    int ngarbage = 0;
    gadget_thread_arrays gthread = gadget_setup_thread_arrays("exchangelist", 1, PartManager->NumPart);

    ForceTree tree = force_tree_top_build(ddecomp, 1);
    /* flag the particles that need to be exported */
#pragma omp parallel
    {
        size_t nexthr_local = 0;
        const int tid = omp_get_thread_num();
        int * threx_local = gthread.srcs[tid];
    #pragma omp for schedule(static, gthread.schedsz) reduction(+: ngarbage)
    for(i=0; i < PartManager->NumPart; i++) {
        if(drift) {
            real_drift_particle(&PartManager->Base[i], SlotsManager, ddrift, PartManager->BoxSize, rel_random_shift);
            PartManager->Base[i].Ti_drift = drift->ti1;
        }
        /* Garbage is not in the tree*/
        if(PartManager->Base[i].IsGarbage) {
            ngarbage++;
            threx_local[nexthr_local] = i;
            nexthr_local++;
            continue;
        }
        /* If we aren't using DM for the dynamic friction, we don't need to build a tree with inactive DM particles.
         * Velocity dispersions are computed on a PM step only.
         * In this case, keep the particles on this processor.*/
        if(!(blackhole_dynfric_treemask() & DMMASK))
            if(PartManager->Base[i].Type == 1 && !is_timebin_active(PartManager->Base[i].TimeBinGravity, PartManager->Base[i].Ti_drift))
                continue;
        if(!inside_topleaf(PartManager->Base[i].TopLeaf, PartManager->Base[i].Pos, &tree)) {
            const int no = domain_get_topleaf(PEANO(PartManager->Base[i].Pos, PartManager->BoxSize), ddecomp);
            /* Set the topleaf for layoutfunc.*/
            PartManager->Base[i].TopLeaf = no;
        }
        int target = domain_layoutfunc(i, ddecomp);
        if(target != tree.ThisTask) {
            threx_local[nexthr_local] = i;
            nexthr_local++;
        }
    }
    gthread.sizes[tid] = nexthr_local;
    }
    force_tree_free(&tree);
    PreExchangeList ExchangeData[1] = {0};
    ExchangeData->ngarbage = ngarbage;
    /*Merge step for the queue.*/
    ExchangeData->nexchange = gadget_compact_thread_arrays(&ExchangeData->ExchangeList, &gthread);
    /*Shrink memory*/
    ExchangeData->ExchangeList = (int *) myrealloc(ExchangeData->ExchangeList, sizeof(int) * ExchangeData->nexchange);
    walltime_measure("/Domain/drift");

    /* Try a domain exchange. Note ExchangeList is freed inside.*/
    int errno = domain_exchange(domain_layoutfunc, ddecomp, ExchangeData, PartManager, SlotsManager, ddecomp->DomainComm);
    return errno;
}

/* this function generates several domain decomposition policies for attempting
 * creating the domain. */
static int
domain_policies_init(DomainDecompositionPolicy policies[],
        const int NPolicy)
{
    int NTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    int i;
    for(i = 0; i < NPolicy; i ++) {
        /* Sort the local particle distribution before subsampling, and get rid of garbage.
         * Does almost nothing to the balance in practice.*/
        policies[i].PreSort = 0;
        if(i >= 2)
            policies[i].PreSort = 1;
        /* Changing the subsample distance is not generally very effective, because we have sorted already
         * and we are balancing by particle load anyway. Better to increase the number of topnodes.
         * But we can try this as a last resort.*/
        policies[i].SubSampleDistance = 256;
        if(i > 4 && policies[i-1].SubSampleDistance > 2)
            policies[i].SubSampleDistance = policies[i-1].SubSampleDistance / 2;
        /* Desired number of TopLeaves should scale like the total number of processors. If we don't get a good balance domain decomposition, we need more topnodes.
         * Need to scale evenly with processors so the round robin balances.*/
        policies[i].NTopLeaves = domain_params.DomainOverDecompositionFactor * NTask * (i+1);
    }

    return NPolicy;
}

/*! This function allocates all the stuff that will be required for the tree-construction/walk later on */
static int
domain_allocate(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy)
{
    size_t bytes, all_bytes = 0;

    /* Number of local topnodes and local topleaves allowed.*/
    const int MaxTopNodes = domain_params.TopNodeAllocFactor * (PartManager->NumPart + 1);

    /* Build the domain over the global all-processors communicator.
     * We use a symbol in case we want to do fancy things in the future.*/
    ddecomp->DomainComm = MPI_COMM_WORLD;

    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /* Add a tail item to avoid special treatments */
    ddecomp->Tasks = (struct task_data *) mymalloc2("Tasks", bytes = ((NTask + 1)* sizeof(ddecomp->Tasks[0])));

    all_bytes += bytes;

    ddecomp->TopNodes = (struct topnode_data *) mymalloc("TopNodes",
        bytes = (MaxTopNodes * (sizeof(ddecomp->TopNodes[0]))));

    all_bytes += bytes;

    ddecomp->TopLeaves = (struct topleaf_data *) mymalloc("TopLeaves",
        bytes = (MaxTopNodes * sizeof(ddecomp->TopLeaves[0])));

    all_bytes += bytes;

    message(0, "Allocated %g MByte for top-level domain structure\n", all_bytes / (1024.0 * 1024.0));

    ddecomp->domain_allocated_flag = 1;

    return MaxTopNodes;
}

void domain_free(DomainDecomp * ddecomp)
{
    if(ddecomp->domain_allocated_flag)
    {
        myfree(ddecomp->TopLeaves);
        myfree(ddecomp->TopNodes);
        myfree(ddecomp->Tasks);
        ddecomp->domain_allocated_flag = 0;
    }
}

/*! This function carries out the actual domain decomposition for all
 *  particle types. It will try to balance the work-load for each ddecomp,
 *  as estimated based on the P[i]-GravCost values.  The decomposition will
 *  respect the maximum allowed memory-imbalance given by the value of
 *  PartAllocFactor.
 */
static int
domain_attempt_decompose(DomainDecomp * ddecomp, DomainDecompositionPolicy * policy, const int MaxTopNodes)
{

    /* points to the root node of the top-level tree */
    struct local_topnode_data *topTree = (struct local_topnode_data *) mymalloc("LocaltopTree",  MaxTopNodes * sizeof(struct local_topnode_data));
    memset(topTree, 0, sizeof(topTree[0]) * MaxTopNodes);

    report_memory_usage("DOMAIN");

    if(domain_determine_global_toptree(policy, topTree, &ddecomp->NTopNodes, MaxTopNodes, ddecomp->DomainComm)) {
        myfree(topTree);
        return 1;
    }

    int i;
    /* copy what we need for the topnodes */
    #pragma omp parallel for
    for(i = 0; i < ddecomp->NTopNodes; i++)
    {
        ddecomp->TopNodes[i].StartKey = topTree[i].StartKey;
        ddecomp->TopNodes[i].Shift = topTree[i].Shift;
        ddecomp->TopNodes[i].Daughter = topTree[i].Daughter;
        ddecomp->TopNodes[i].Leaf = -1; /* will be assigned by create_topleaves*/
    }

    myfree(topTree);

    ddecomp->NTopLeaves = 0;
    domain_create_topleaves(ddecomp, 0, &ddecomp->NTopLeaves);

    message(0, "NTopLeaves= %d  NTopNodes=%d (space for %d)\n", ddecomp->NTopLeaves, ddecomp->NTopNodes, MaxTopNodes);

    walltime_measure("/Domain/DetermineTopTree/CreateLeaves");

    if(ddecomp->NTopLeaves < policy->NTopLeaves) {
        message(0, "Number of Topleaves is less than required over decomposition");
    }

    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /* this is fatal */
    if(ddecomp->NTopLeaves < NTask) {
        endrun(0, "Number of Topleaves is less than NTask");
    }

    return 0;
}

/**
 * attempt to assign segments to tasks such that the load or work is balanced.
 *
 * */
static int
domain_balance(DomainDecomp * ddecomp)
{
    /*!< a table that gives the total number of particles held by each processor */
    int64_t * TopLeafCount = (int64_t *) mymalloc("TopLeafCount",  ddecomp->NTopLeaves * sizeof(TopLeafCount[0]));

    domain_compute_costs(ddecomp, NULL, TopLeafCount);

    /* first try work balance */
    domain_assign_balanced(ddecomp, TopLeafCount, 1);

    int status = domain_check_memory_bound(ddecomp, NULL, TopLeafCount);
    if(status != 0)
        message(0, "Domain decomposition is outside memory bounds.\n");

    walltime_measure("/Domain/Decompose");

    myfree(TopLeafCount);

    return status;
}

static int
domain_check_memory_bound(const DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    /*Only used if the memory bound is not met */
    int64_t * list_load = ta_malloc("list_load",int64_t, NTask);
    int64_t * list_work = ta_malloc("list_work",int64_t, NTask);

    int64_t max_work = 0, max_load = 0, sumload = 0, sumwork = 0;
    int ta;

    #pragma omp parallel for reduction(+: sumwork) reduction(+: sumload) reduction(max: max_load) reduction(max:max_work)
    for(ta = 0; ta < NTask; ta++)
    {
        int64_t load = 0;
        int64_t work = 0;
        int i;
        for(i = ddecomp->Tasks[ta].StartLeaf; i < ddecomp->Tasks[ta].EndLeaf; i ++)
        {
            load += TopLeafCount[i];
            if(TopLeafWork)
                work += TopLeafWork[i];
        }

        list_load[ta] = load;
        list_work[ta] = work;

        sumwork += work;
        sumload += load;

        if(load > max_load)
            max_load = load;
        if(work > max_work)
            max_work = work;
    }

    if(TopLeafWork)
        message(0, "Largest load: work=%g particle=%g\n",
            max_work / ((double)sumwork / NTask), max_load / (((double) sumload) / NTask));
    else
        message(0, "Largest particle load=%g\n", max_load / (((double) sumload) / NTask));

    /*Leave a small number of particles for star formation */
    if(max_load > PartManager->MaxPart * domain_params.SetAsideFactor)
    {
        message(0, "desired memory imbalance=%g  (limit=%g, needed=%ld)\n",
                    (max_load * ((double) sumload ) / NTask ) / PartManager->MaxPart, domain_params.SetAsideFactor * PartManager->MaxPart, max_load);
        message(0, "Balance breakdown:\n");
        int i;
        for(i = 0; i < NTask; i++)
        {
            message(0, "Task: [%3d]  work=%8.4f  particle load=%8.4f\n", i,
               list_work[i] / ((double) sumwork / NTask), list_load[i] / (((double) sumload) / NTask));
        }
        return 1;
    }
    ta_free(list_work);
    ta_free(list_load);
    return 0;
}


/* Data of TopTree leaf nodes that are used for assignment to tasks */
struct topleaf_extdata {
    peano_t Key;
    int Task;        /** The task that receives the node */
    int topnode;     /** The node */
    int64_t cost;    /** cost value, can be either number of calculations or number of particles. */
};

static int
topleaf_ext_order_by_task_and_key(const void * c1, const void * c2)
{
    const struct topleaf_extdata * p1 = (const struct topleaf_extdata *) c1;
    const struct topleaf_extdata * p2 = (const struct topleaf_extdata *) c2;
    if(p1->Task < p2->Task) return -1;
    if(p1->Task > p2->Task) return 1;
    if(p1->Key < p2->Key) return -1;
    if(p1->Key > p2->Key) return 1;
    return 0;
}

static int
topleaf_ext_order_by_key(const void * c1, const void * c2)
{
    const struct topleaf_extdata * p1 = (const struct topleaf_extdata *) c1;
    const struct topleaf_extdata * p2 = (const struct topleaf_extdata *) c2;
    if(p1->Key < p2->Key) return -1;
    if(p1->Key > p2->Key) return 1;
    return 0;
}


/**
 * This function assigns TopLeaves to Segments, trying to ensure uniform cost
 * by assigning TopLeaves (contiguously) until a Segment has a desired size.
 * Then each Segment is assigned to a Task in a round-robin fashion starting from the largest Segment
 * until all Segments are assigned.
 * At the moment the Segment/Task distinction is not used: we call this with NSegmentPerTask = 1, because
 * we are able to create Segments of roughly equal cost from the TopLeaves.
 *
 * This creates the index in Tasks[Task].StartLeaf and Tasks[Task].EndLeaf
 * cost is the cost per TopLeaves
 *
 * */
static void
domain_assign_balanced(DomainDecomp * ddecomp, int64_t * cost, const int NsegmentPerTask)
{
    int NTask;
    MPI_Comm_size(ddecomp->DomainComm, &NTask);

    int Nsegment = NTask * NsegmentPerTask;
    /* we work with TopLeafExt then replace TopLeaves*/

    struct topleaf_extdata * TopLeafExt;

    /* A Segment is a subset of the TopLeaf nodes */

    TopLeafExt = (struct topleaf_extdata *) mymalloc("TopLeafExt", ddecomp->NTopLeaves * sizeof(TopLeafExt[0]));

    /* copy the data over */
    int i;
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        TopLeafExt[i].topnode = ddecomp->TopLeaves[i].topnode;
        TopLeafExt[i].Key = ddecomp->TopNodes[ddecomp->TopLeaves[i].topnode].StartKey;
        TopLeafExt[i].Task = -1;
        TopLeafExt[i].cost = cost[i];
    }

    /* make sure TopLeaves are sorted by Key for locality of segments -
     * likely not necessary be cause when this function
     * is called it is already true */
    qsort_openmp(TopLeafExt, ddecomp->NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_key);

    int64_t totalcost = 0;
    #pragma omp parallel for reduction(+ : totalcost)
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        totalcost += TopLeafExt[i].cost;
    }
    int64_t totalcostLeft = totalcost;

    /* start the assignment; We try to create segments that are of the
     * mean_expected cost, then assign them to Tasks in a round-robin fashion.
     * The default is that there is one segment per task, which makes the assignment trivial.
     */

    double mean_expected = 1.0 * totalcost / Nsegment;
    double mean_task = 1.0 * totalcost /NTask;
    int curleaf = 0;
    int curseg = 0;
    int curtask = 0; /* between 0 and NTask - 1*/
    int nrounds = 0; /*Number of times we looped*/
    int64_t curload = 0; /* cumulative load for the current segment */
    int64_t curtaskload = 0; /* cumulative load for current task */
    int64_t maxleafcost = 0;

    /* we maintain that after the loop curleaf is the number of leaves scanned,
     * curseg is number of segments created.
     * */
    while(nrounds < ddecomp->NTopLeaves) {
        int append = 0;
        int advance = 0;
        if(curleaf == ddecomp->NTopLeaves) {
            /* to maintain the invariance */
            advance = 1;
        } else if(ddecomp->NTopLeaves - curleaf == Nsegment - curseg) {
            /* just enough for one segment per leaf; this line ensures
             * at least Nsegment segments are created. */
            append = 1;
            advance = 1;
        } else {
            /* append a leaf to the segment if there is room left.
             * Calculate room left based on a rolling average of the total so far..*/
            int64_t totalassigned = (totalcost - totalcostLeft) + curload;
            if((mean_expected * (curseg +1) - totalassigned > 0.5 * TopLeafExt[curleaf].cost)
            || curload == 0 /* but at least add one leaf */
                ) {
                append = 1;
            } else {
                /* will be too big of a segment, cut it */
                advance = 1;
            }
        }
        if(append) {
            /* assign the leaf to the task */
            curload += TopLeafExt[curleaf].cost;
            TopLeafExt[curleaf].Task = curtask;
            curleaf ++;
        }

        if(advance) {
            /*Add this segment to the current task*/
            curtaskload += curload;
            /* If we have allocated enough segments to fill this processor,
             * or if we have one segment per task left, proceed to the next task.
             * We do not use round robin so that neighbouring (by key)
             * topTree are on the same processor.*/
            if((mean_task - curtaskload < 0.5 * mean_expected) || (Nsegment - curseg <= NTask - curtask)
               ){
                curtaskload = 0;
                curtask ++;
            }
            /* move on to the next segment.*/
            totalcostLeft -= curload;
            curload = 0;
            curseg ++;

            /* finished a round for all tasks */
            if(curtask == NTask) {
                /*Back to task zero*/
                curtask = 0;
                /* Need a new mean_expected value: we want to
                 * divide the remaining segments evenly between the processors.*/
                mean_expected = 1.0 * totalcostLeft / Nsegment;
                mean_task = 1.0 * totalcostLeft / NTask;
                nrounds++;
            }
            if(curleaf == ddecomp->NTopLeaves)
                break;
            if(TopLeafExt[curleaf].cost > maxleafcost)
                maxleafcost = TopLeafExt[curleaf].cost;
        }
        //message(0, "curleaf = %d advance = %d append = %d, curload = %d cost=%ld left=%ld\n", curleaf, advance, append, curload, TopLeafExt[curleaf].cost, totalcostLeft);
    }

    /*In most 'normal' cases nrounds == 1 here*/
    message(0, "Created %d segments in %d rounds. Max leaf cost/expected: %g expected: %g \n", curseg, nrounds, (1.0*maxleafcost)/(totalcost) * Nsegment, (1.0*totalcost)/Nsegment);

    if(curseg < Nsegment) {
        endrun(0, "Not enough segments were created (%d instead of %d). This should not happen.\n", curseg, Nsegment);
    }

    if(totalcostLeft != 0) {
        endrun(0, "Assertion failed. Total cost is not fully assigned to all ranks\n");
    }

    /* lets rearrange the TopLeafExt by task, such that we can build the Tasks table */
    qsort_openmp(TopLeafExt, ddecomp->NTopLeaves, sizeof(TopLeafExt[0]), topleaf_ext_order_by_task_and_key);
    for(i = 0; i < ddecomp->NTopLeaves; i ++) {
        ddecomp->TopNodes[TopLeafExt[i].topnode].Leaf = i;
        ddecomp->TopLeaves[i].Task = TopLeafExt[i].Task;
        ddecomp->TopLeaves[i].topnode = TopLeafExt[i].topnode;
    }

    myfree(TopLeafExt);
    /* here we reduce the number of code branches by adding an item to the end. */
    ddecomp->TopLeaves[ddecomp->NTopLeaves].Task = NTask;
    ddecomp->TopLeaves[ddecomp->NTopLeaves].topnode = -1;

    int ta = 0;
    ddecomp->Tasks[ta].StartLeaf = 0;
    for(i = 0; i <= ddecomp->NTopLeaves; i ++) {

        if(ddecomp->TopLeaves[i].Task == ta) continue;

        ddecomp->Tasks[ta].EndLeaf = i;
        ta ++;
        while(ta < ddecomp->TopLeaves[i].Task) {
            ddecomp->Tasks[ta].EndLeaf = i;
            ddecomp->Tasks[ta].StartLeaf = i;
            ta ++;
        }
        /* the last item will set Tasks[NTask], but we allocated memory for it already */
        ddecomp->Tasks[ta].StartLeaf = i;
    }
    if(ta != NTask) {
        endrun(0, "Assertion failed: not all tasks are assigned. This indicates a bug.\n");
    }
}

/*! This function determines which particles that are currently stored
 *  on the local CPU have to be moved off according to the domain
 *  decomposition.
 *
 *  layoutfunc decides the target Task of particle p (used by
 *  subfind_distribute).
 *  Uses the toptree, instead of the Peano key*/
static int
domain_layoutfunc(int n, const void * userdata) {
    const DomainDecomp * ddecomp = (DomainDecomp *) userdata;
    const int topleaf = PartManager->Base[n].TopLeaf;
    if(topleaf < 0 || topleaf >= ddecomp->NTopLeaves)
        endrun(6, "Invalid topleaf %d (ntop %d) for particle %d id %ld x-pos %g garbage %d\n",
               topleaf, ddecomp->NTopLeaves, n, PartManager->Base[n].ID, PartManager->Base[n].Pos[0], PartManager->Base[n].IsGarbage);
    return ddecomp->TopLeaves[topleaf].Task;
}

/*! This function walks the global top tree in order to establish the
 *  number of leaves it has. These leaves are then distributed to different
 *  processors.
 *
 *  the pointer next points to the next free item on TopLeaves array.
 */
static void
domain_create_topleaves(DomainDecomp * ddecomp, int no, int * next)
{
    int i;
    if(ddecomp->TopNodes[no].Daughter == -1)
    {
        ddecomp->TopNodes[no].Leaf = *next;
        ddecomp->TopLeaves[*next].topnode = no;
        (*next)++;
    }
    else
    {
        for(i = 0; i < 8; i++)
            domain_create_topleaves(ddecomp, ddecomp->TopNodes[no].Daughter + i, next);
    }
}

static int
domain_toptree_get_subnode(struct local_topnode_data * topTree,
        peano_t key)
{
    int no = 0;
    while(topTree[no].Daughter >= 0) {
        no = topTree[no].Daughter + ((key - topTree[no].StartKey) >> (topTree[no].Shift - 3));
    }
    return no;
}

static int
domain_toptree_insert(struct local_topnode_data * topTree,
        peano_t Key, int64_t cost)
{
    /* insert */
    int leaf = domain_toptree_get_subnode(topTree, Key);
    topTree[leaf].Count ++;
    topTree[leaf].Cost += cost;
    return leaf;
}

static int
domain_toptree_split(struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes,
    int i)
{
    int j;
    /* we ran out of top nodes and must get more.*/
    if((*topTreeSize + 8) > MaxTopNodes)
        return 1;

    if(topTree[i].Shift < 3) {
        endrun(1, "Failed to build a TopTree -- particles overly clustered.\n");
    }

    /*Make a new topnode section attached to this node*/
    topTree[i].Daughter = *topTreeSize;
    (*topTreeSize) += 8;

    /* Initialise this topnode with new sub nodes*/
    for(j = 0; j < 8; j++)
    {
        const int sub = topTree[i].Daughter + j;
        /* The new sub nodes have this node as parent
         * and no daughters.*/
        topTree[sub].Daughter = -1;
        topTree[sub].Parent = i;
        /* Shorten the peano key by a factor 8, reflecting the oct-tree level.*/
        topTree[sub].Shift = topTree[i].Shift - 3;
        /* This is the region of peanospace covered by this node.*/
        topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
        /* We will compute the cost in the node below.*/
        topTree[sub].Count = 0;
        topTree[sub].Cost = 0;
    }
    return 0;
}

static void
domain_toptree_update_cost(struct local_topnode_data * topTree, int start)
{
    if(topTree[start].Daughter == -1) return;

    int j = 0;
    for(j = 0; j < 8; j ++) {
        int sub = topTree[start].Daughter + j;
        domain_toptree_update_cost(topTree, sub);
        topTree[start].Count += topTree[sub].Count;
        topTree[start].Cost += topTree[sub].Cost;
    }
}

/* This function recursively identifies and terminate tree branches that are cheap.*/
static void
domain_toptree_truncate_r(struct local_topnode_data * topTree, int start, int64_t countlimit, int64_t costlimit)
{
    if(topTree[start].Daughter == -1) return;

    if(topTree[start].Count < countlimit &&
       topTree[start].Cost < costlimit) {
        /* truncate here */
        topTree[start].Daughter = -1;
        return;
    }

    int j;
    for(j = 0; j < 8; j ++) {
        int sub = topTree[start].Daughter + j;
        domain_toptree_truncate_r(topTree, sub, countlimit, costlimit);
    }
}

/* remove the nodes that are no longer useful after the truncation.
 *
 * We walk the topTree top-down to collect useful nodes, and move them to
 * the head of the topTree list.
 *
 * This works because any child node is stored after the parent in the list
 * -- we are destroying the old tree just slow enough
 * */

static void
domain_toptree_garbage_collection(struct local_topnode_data * topTree, int start, int * last_free)
{

    if(topTree[start].Daughter == -1) return;
    int j;

    int oldd = topTree[start].Daughter;
    int newd = *last_free;

    topTree[start].Daughter = newd;

    (*last_free) += 8;

    /* copy first in case oldd + j is overwritten by the recursed gc
     * if last_free is less than oldd */
    for(j = 0; j < 8; j ++) {
        topTree[newd + j] = topTree[oldd + j];
        topTree[newd + j].Parent = start;
    }

    for(j = 0; j < 8; j ++) {
        domain_toptree_garbage_collection(topTree, newd + j, last_free);
    }
}

static void
domain_toptree_truncate(
    struct local_topnode_data * topTree, int * topTreeSize,
    int64_t countlimit, int64_t costlimit)
{

    /* first terminate the tree.*/
    domain_toptree_truncate_r(topTree, 0, countlimit, costlimit);

    /* then remove the unused nodes from the topTree storage. This is important
     * for efficient global merge */
    *topTreeSize = 1; /* put in the root node -- it's never a garbage . */
    domain_toptree_garbage_collection(topTree, 0, topTreeSize);
}


static int
order_by_key(const void *a, const void *b)
{
    const struct local_particle_data * pa  = (const struct local_particle_data *) a;
    const struct local_particle_data * pb  = (const struct local_particle_data *) b;
    if(pa->Key < pb->Key)
        return -1;

    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

static void
mp_order_by_key(const void * data, void * radix, void * arg)
{
    const struct local_particle_data * pa  = (const struct local_particle_data *) data;
    ((uint64_t *) radix)[0] = pa->Key;
}

/**
 * This function performs local refinement of the topTree.
 *
 * It creates the local refinement by quickly going through
 * a skeleton tree of a fraction (subsample) of all local particles.
 *
 * Next, we add flesh to the skeleton: all particles are deposited
 * to the tree.
 *
 * Finally, in _truncation(), we chop off the cheap branches from the skeleton,
 * terminating the branches when the cost and count are both sufficiently small.
 *
 * We do not use the full particle data. Sufficient to use LP, which contains
 * the peano key and cost of each particle, and sorted by the peano key.
 * */
static int
domain_check_for_local_refine_subsample(
    DomainDecompositionPolicy * policy,
    struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes,
    const MPI_Comm DomainComm)
{

    int i;

    /* Watchout : Peano/Morton ordering is required by the tree
     * building algorithm in local_refine.
     *
     * We can either use a global or a local sorting here; the code will run
     * without crashing.
     *
     * A global sorting is chosen to ensure the local topTrees are really local
     * and the leaves almost disjoint. This makes the merged topTree a more accurate
     * representation of the true cost / load distribution, for merging
     * and secondary refinement are approximated.
     *
     * A local sorting may be faster but makes the tree less accurate due to
     * more likely running into overlapped local topTrees.
     * */

    int Nsample = PartManager->NumPart / policy->SubSampleDistance;

    if(Nsample == 0 && PartManager->NumPart != 0) Nsample = 1;
    struct local_particle_data * LP = (struct local_particle_data*) mymalloc("LocalParticleData", Nsample * sizeof(LP[0]));

    if(policy->PreSort) {
        struct local_particle_data * LPfull = (struct local_particle_data*) mymalloc2("LocalParticleData", PartManager->NumPart * sizeof(LP[0]));
        int64_t garbage = 0;
        #pragma omp parallel for reduction(+: garbage)
        for(i = 0; i < PartManager->NumPart; i ++)
        {
            if(P[i].IsGarbage) {
                LPfull[i].Key = PEANOCELLS;
                LPfull[i].Cost = 0;
                garbage++;
                continue;
            }
            LPfull[i].Key = PEANO(P[i].Pos, PartManager->BoxSize);
            LPfull[i].Cost = 1;
        }

        /* First sort to ensure spatially 'even' subsamples and remove garbage.*/
        qsort_openmp(LPfull, PartManager->NumPart, sizeof(struct local_particle_data), order_by_key);
        Nsample = (PartManager->NumPart - garbage) / policy->SubSampleDistance;
        if(Nsample == 0 && PartManager->NumPart > garbage) Nsample = 1;

        /* now subsample */
        #pragma omp parallel for
        for(i = 0; i < Nsample; i ++)
        {
            LP[i].Key = LPfull[i * policy->SubSampleDistance].Key;
            LP[i].Cost = LPfull[i * policy->SubSampleDistance].Cost;
        }
        myfree(LPfull);
    }
    else {
        /* Subsample, computing keys*/
        #pragma omp parallel for
        for(i = 0; i < Nsample; i ++)
        {
            int j = i * policy->SubSampleDistance;
            LP[i].Key = PEANO(P[j].Pos, PartManager->BoxSize);
            LP[i].Cost = 1;
        }
    }

    if(domain_params.DomainUseGlobalSorting) {
        mpsort_mpi(LP, Nsample, sizeof(struct local_particle_data), mp_order_by_key, 8, NULL, DomainComm);
    } else {
        qsort_openmp(LP, Nsample, sizeof(struct local_particle_data), order_by_key);
    }

    walltime_measure("/Domain/DetermineTopTree/Sort");

    *topTreeSize = 1;
    topTree[0].Daughter = -1;
    topTree[0].Parent = -1;
    topTree[0].Shift = BITS_PER_DIMENSION * 3;
    topTree[0].StartKey = 0;
    topTree[0].Count = 0;
    topTree[0].Cost = 0;

    /* The tree building algorithm here requires the LP structure to
     * be sorted by key, in which case we either refine
     * the node the last particle lives in, or jump into a new empty node,
     * as the particles are scanned through.
     *
     * I unfortunately cannot find the direct reference of this with
     * the proof. I found it around 2012~2013 when writing psphray2
     * and didn't cite it properly then!
     *
     * Here is a blog link that is related:
     *
     * https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
     *
     * A few interesting papers that may make this faster and cheaper.
     *
     * http://sc07.supercomputing.org/schedule/pdf/pap117.pdf
     *
     * */

    /* 1. create a skeleton of the toptree.
     * We do not use all particles to avoid excessive memory consumption.
     *
     * */


    /* During the loop, topTree[?].Count is a flag to indicate if
     * the node has been finished. We either refine the last leaf node
     * or create a new leaf because of the peano/morton sorting.
     * */
    peano_t last_key = PEANOT_MAX;
    int last_leaf = -1;
    i = 0;
    while(i < Nsample) {

        int leaf = domain_toptree_get_subnode(topTree, LP[i].Key);

        if (leaf == last_leaf && topTree[leaf].Shift >= 3) {
            /* two particles in a node? need refinement if possible. */
            if(0 != domain_toptree_split(topTree, topTreeSize, MaxTopNodes, leaf)) {
                /* out of memory, retry */
                myfree(LP);
                return 1;
            }
            /* pop the last particle and reinsert it */
            topTree[leaf].Count = 0;

            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            /* retry the current particle. */
            continue;
        } else {
            if(topTree[leaf].Count != 0 && leaf != last_leaf) {
                /* meeting a node that's already been finished ?
                 * This shall not happen when key is already sorted;
                 * due to the sorting.
                 *
                 * If the Key hasn't changed sufficiently with this new particle we will see the last leaf again.
                 * Normally we would refine, but if Shift == 0 we don't have space.
                 * In this case we just add the current particle sample to the last toptree node.
                 */
                endrun(10, "toptree[%d].Count=%ld, shift %d, last_leaf=%d key = %ld i= %d Nsample = %d\n",
                        leaf, topTree[leaf].Count, topTree[leaf].Shift, last_leaf,LP[i].Key, i, Nsample);
            }
            /* this will create a new node. */
            last_key = LP[i].Key;
            last_leaf = domain_toptree_insert(topTree, last_key, 0);
            i++;
            continue;
        }
    }
    /* Alternativly we could have kept last_cost in the above loop and avoid
     * the second and third step: */

    /* 2. Remove the skelton particles from the tree to make it a skeleton;
     * note that we never bothered to add Cost when the skeleton was built.
     * otherwise we shall clean it here too.*/
    for(i = 0; i < *topTreeSize; i ++ ) {
        topTree[i].Count = 0;
    }

    /* 3. insert all particles to the skeleton tree; Count will be correct because we cleaned them.*/
    for(i = 0; i < Nsample; i ++ ) {
        domain_toptree_insert(topTree, LP[i].Key, LP[i].Cost);
    }

    myfree(LP);

    /* then compute the costs of the internal nodes. */
    domain_toptree_update_cost(topTree, 0);

    walltime_measure("/Domain/DetermineTopTree/LocalRefine");

    /* we leave truncation in another function, for costlimit and countlimit must be
     * used in secondary refinement*/
    return 0;
}

/* Combine the toptree. Returns a (collective) error code which is non-zero if an error occured*/
static int
domain_nonrecursively_combine_topTree(struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes, MPI_Comm DomainComm)
{
    /*
     * combine topTree non recursively, this uses MPI_Bcast within a group.
     * it shall be quite a bit faster (~ x2) than the old recursive scheme.
     *
     * it takes less time at higher sep.
     *
     * The communication should have been done with MPI Inter communicator.
     * but I couldn't figure out how to do it that way.
     * */
    int sep = 1;
    int errorflag = 0;
    /*Number of tasks to decompose to*/
    int NTask;
    MPI_Comm_size(DomainComm, &NTask);
    /*Which task is this?*/
    int ThisTask;
    MPI_Comm_rank(DomainComm, &ThisTask);

    for(sep = 1; sep < NTask; sep *=2) {
        /* build the subcommunicators for broadcasting */
        int Color = ThisTask / sep;
        int Key = ThisTask % sep;

        /* non leaders will skip exchanges */
        if(Key != 0)
            continue;

        /* leaders of even color will combine nodes from next odd color,
         * so that when sep is increased eventually rank 0 will have all
         * nodes */
        if(Color % 2 == 0) {
            /* even guys recv */
            int recvTask = ThisTask + sep;
            int ntopnodes_import = 0;
            if(recvTask < NTask) {
                MPI_Recv(&ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A,
                        DomainComm, MPI_STATUS_IGNORE);
                if(ntopnodes_import < 0) {
                    endrun(1, "severe domain error using a unintended rank \n");
                }
                int mergesize = ntopnodes_import;
                if(ntopnodes_import < *topTreeSize)
                    mergesize = *topTreeSize;
                struct local_topnode_data * topTree_import = (struct local_topnode_data *) mymalloc("topTree_import",
                            mergesize * sizeof(struct local_topnode_data));

                MPI_Recv(topTree_import,
                        ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE,
                        recvTask, TAG_GRAV_B,
                        DomainComm, MPI_STATUS_IGNORE);

                if((*topTreeSize + ntopnodes_import) > MaxTopNodes) {
                    errorflag = 1;
                } else {
                    if(ntopnodes_import > 0 ) {
                        domain_toptree_merge(topTree, topTree_import, 0, 0, topTreeSize, MaxTopNodes);
                    }
                }
                myfree(topTree_import);
            }
        } else {
            /* odd guys send */
            int recvTask = ThisTask - sep;
            if(recvTask >= 0) {
                MPI_Send(topTreeSize, 1, MPI_INT, recvTask, TAG_GRAV_A, DomainComm);
                MPI_Send(topTree, (*topTreeSize) * sizeof(struct local_topnode_data), MPI_BYTE,
                         recvTask, TAG_GRAV_B, DomainComm);
            }
            *topTreeSize = -1;
        }
    }

    MPI_Bcast(topTreeSize, 1, MPI_INT, 0, DomainComm);
    /* Check that the merge succeeded*/
    if(*topTreeSize < 0 || *topTreeSize >= MaxTopNodes) {
        errorflag = 1;
    }
    int errorflagall = MPIU_Any(errorflag, DomainComm);
    if(errorflagall == 0)
        MPI_Bcast(topTree, (*topTreeSize) * sizeof(struct local_topnode_data), MPI_BYTE, 0, DomainComm);
    return errorflagall;
}

/*! This function constructs the global top-level tree node that is used
 *  for the domain decomposition. This is done by considering the string of
 *  Peano-Hilbert keys for all particles, which is recursively chopped off
 *  in pieces of eight segments until each segment holds at most a certain
 *  number of particles.
 */
int domain_determine_global_toptree(DomainDecompositionPolicy * policy,
        struct local_topnode_data * topTree, int * topTreeSize, int MaxTopNodes, MPI_Comm DomainComm)
{
    /*
     * Build local refinement with a subsample of particles
     * 1/16 is used because each local topTree node takes about 32 bytes.
     **/

    int local_refine_failed = domain_check_for_local_refine_subsample(policy, topTree, topTreeSize, MaxTopNodes, DomainComm);

    if(MPIU_Any(local_refine_failed, DomainComm)) {
        message(0, "We are out of Topnodes: have %d.\n", MaxTopNodes);
        return 1;
    }

    int64_t TotCost = 0, TotCount = 0;
    int64_t costlimit, countlimit;

    MPI_Allreduce(&topTree[0].Cost, &TotCost, 1, MPI_INT64, MPI_SUM, DomainComm);
    MPI_Allreduce(&topTree[0].Count, &TotCount, 1, MPI_INT64, MPI_SUM, DomainComm);

    costlimit = TotCost / (policy->NTopLeaves);
    countlimit = TotCount / (policy->NTopLeaves);

    domain_toptree_truncate(topTree, topTreeSize, countlimit, costlimit);

    if(*topTreeSize > 10 * policy->NTopLeaves) {
        message(1, "local TopTree Size =%d >> expected = %d; Usually this indicates very bad imbalance, due to a giant density peak.\n",
            *topTreeSize, policy->NTopLeaves);
    }

#if 0
    char buf[1000];
    sprintf(buf, "topnodes.bin.%d", ThisTask);
    FILE * fd = fopen(buf, "w");

    fwrite(topTree, sizeof(struct local_topnode_data), *topTreeSize, fd);
    fclose(fd);

    //MPI_Barrier(DomainComm);
    //MPI_Abort(DomainComm, 0);
#endif

    /* we now need to exchange tree parts and combine them as needed */
    int combine_failed = domain_nonrecursively_combine_topTree(topTree, topTreeSize, MaxTopNodes, DomainComm);

    if(combine_failed) {
        message(0, "can't combine trees due to lack of storage. Will try again.\n");
        return 1;
    }
    /* now let's see whether we should still more refinements, based on the estimated cumulative cost/count in each cell */
    int prerefinetoptree = *topTreeSize;
    int global_refine_failed = domain_global_refine(topTree, topTreeSize, MaxTopNodes, countlimit, costlimit);

    if(MPIU_Any(global_refine_failed, DomainComm)) {
        message(0, "Global refine failed: toptreeSize = %d, MaxTopNodes = %d\n", *topTreeSize, MaxTopNodes);
        return 1;
    }

    message(0, "Final topTree size = %d ntopleaves %d ratio to desired ntopleaves = %g (before refine %d).\n", *topTreeSize, policy->NTopLeaves, 1.0 * (*topTreeSize) / (policy->NTopLeaves), prerefinetoptree);
    return 0;
}

static int
domain_global_refine(
    struct local_topnode_data * topTree, int * topTreeSize, const int MaxTopNodes,
    const int64_t countlimit, const int64_t costlimit)
{
    int i;

    /* At this point we have refined the local particle tree so that each
     * topNode contains a Cost and Count below the cost threshold. We have then
     * done a global merge of the particle tree. Some of our topTree may now contain
     * more particles than the Cost threshold, but repeating refinement using the local
     * algorithm is complicated - particles inside any particular topNode may be
     * on another processor. So we do a local volume based refinement here. This
     * just cuts each topNode above the threshold into 8 equal-sized portions by
     * subdividing the peano key.
     *
     * NOTE: Just like the merge, this does not correctly preserve costs!
     * Costs are just divided by 8, because recomputing them for the daughter nodes
     * will be expensive.
     * In practice this seems to work fine, probably because the cost distribution
     * is not that unbalanced. */

    /*Note that *topTreeSize will change inside the loop*/
    for(i = 0; i < *topTreeSize; i++)
    {
        /*If this node has no children and non-zero size*/
        if(topTree[i].Daughter >= 0 || topTree[i].Shift <= 0) continue;

        /*If this node is also more costly than the limit*/
        if(topTree[i].Count < countlimit && topTree[i].Cost < costlimit) continue;

        /*If we have no space for another 8 topTree, exit */
        if((*topTreeSize + 8) > MaxTopNodes) {
            return 1;
        }

        topTree[i].Daughter = *topTreeSize;
        int j;
        for(j = 0; j < 8; j++)
        {
            int sub = topTree[i].Daughter + j;
            topTree[sub].Shift = topTree[i].Shift - 3;
            topTree[sub].Count = topTree[i].Count / 8;
            topTree[sub].Cost = topTree[i].Cost / 8;
            topTree[sub].Daughter = -1;
            topTree[sub].Parent = i;
            topTree[sub].StartKey = topTree[i].StartKey + j * (1L << topTree[sub].Shift);
        }
        (*topTreeSize) += 8;
    }
    return 0;
}


static void
domain_compute_costs(DomainDecomp * ddecomp, int64_t *TopLeafWork, int64_t *TopLeafCount)
{
    int i;
    int NumThreads = omp_get_max_threads();
    int64_t * local_TopLeafWork = NULL;
    if(TopLeafWork) {
        local_TopLeafWork = (int64_t *) mymalloc("local_TopLeafWork", NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafWork[0]));
        memset(local_TopLeafWork, 0, NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafWork[0]));
    }
    int64_t * local_TopLeafCount = (int64_t *) mymalloc("local_TopLeafCount", NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafCount[0]));
    memset(local_TopLeafCount, 0, NumThreads * ddecomp->NTopLeaves * sizeof(local_TopLeafCount[0]));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int n;

        #pragma omp for
        for(n = 0; n < PartManager->NumPart; n++)
        {
            /* Skip garbage particles: they have zero work
             * and can be removed by exchange if under memory pressure.*/
            if(P[n].IsGarbage)
                continue;

            const int leaf = domain_get_topleaf(PEANO(P[n].Pos, PartManager->BoxSize), ddecomp);
            /* Set the topleaf so we can use it for exchange*/
            P[n].TopLeaf = leaf;

            if(local_TopLeafWork)
                local_TopLeafWork[leaf + tid * ddecomp->NTopLeaves] += 1;

            local_TopLeafCount[leaf + tid * ddecomp->NTopLeaves] += 1;
        }
    }


#pragma omp parallel for
    for(i = 0; i < ddecomp->NTopLeaves; i++)
    {
        int tid;
        if(local_TopLeafWork)
            for(tid = 1; tid < NumThreads; tid++) {
                local_TopLeafWork[i] += local_TopLeafWork[i + tid * ddecomp->NTopLeaves];
            }

        for(tid = 1; tid < NumThreads; tid++) {
            local_TopLeafCount[i] += local_TopLeafCount[i + tid * ddecomp->NTopLeaves];
        }
    }

    if(local_TopLeafWork) {
        MPI_Allreduce(local_TopLeafWork, TopLeafWork, ddecomp->NTopLeaves, MPI_INT64, MPI_SUM, ddecomp->DomainComm);
        myfree(local_TopLeafWork);
    }

    MPI_Allreduce(local_TopLeafCount, TopLeafCount, ddecomp->NTopLeaves, MPI_INT64, MPI_SUM, ddecomp->DomainComm);
    myfree(local_TopLeafCount);
}

/**
 * Merge treeB into treeA.
 *
 * The function recursively merge the cost and refinement of treeB into treeA.
 *
 * At the initial call, noA and noB must both be the root node.
 *
 * When the structure is mismatched, e.g. a leaf in A meets a branch in B or
 * a leaf in B meets a branch in A, the cost of the leaf is splitted evenly.
 * This is only an approximation then the leaf is not empty.
 *
 * We therefore have an incentive to minimize overlaps between treeB and treeA.
 * local_refinement does a global sorting of keys to help that.
 *
 * */

static void
domain_toptree_merge(struct local_topnode_data *treeA,
                     struct local_topnode_data *treeB,
                     int noA, int noB, int * treeASize, const int MaxTopNodes)
{
    int j, sub;
    int64_t count;
    int64_t cost;

    if(treeB[noB].Shift < treeA[noA].Shift)
    {
        /* Add B to A */
        /* Create a daughter to a, since we will merge B to A's daughter*/
        if(treeA[noA].Daughter < 0)
        {
            if((*treeASize + 8) >= MaxTopNodes) {
                endrun(88, "Too many Topnodes; this shall not happen because we ensure there is enough and bailed earlier than this\n");
            }
            /* noB must have a parent if we are here, since noB is lower than noA;
             * noA must have already merged in the cost of noB's parent.
             * This is the first time we create these children in A, thus,
             * We shall evenly divide the non-B part of the cost in these children;
             * */

            count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
            cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;

            treeA[noA].Daughter = *treeASize;
            for(j = 0; j < 8; j++)
            {

                sub = treeA[noA].Daughter + j;
                treeA[sub].Shift = treeA[noA].Shift - 3;
                treeA[sub].Count = (j + 1) * count / 8 - j * count / 8;
                treeA[sub].Cost  = (j + 1) * cost / 8 - j * cost / 8;
                treeA[sub].Daughter = -1;
                treeA[sub].Parent = noA;
                treeA[sub].StartKey = treeA[noA].StartKey + j * (1L << treeA[sub].Shift);
            }
            (*treeASize) += 8;
        }

        /* find the sub node in A for me and merge, this would bring noB and sub on the same shift, drop to next case */
        sub = treeA[noA].Daughter + ((treeB[noB].StartKey - treeA[noA].StartKey) >> (treeA[noA].Shift - 3));
        domain_toptree_merge(treeA, treeB, sub, noB, treeASize, MaxTopNodes);
    }
    else if(treeB[noB].Shift == treeA[noA].Shift)
    {
        treeA[noA].Count += treeB[noB].Count;
        treeA[noA].Cost += treeB[noB].Cost;

        /* Prefer to go down B; this would trigger the previous case  */
        if(treeB[noB].Daughter >= 0)
        {
            for(j = 0; j < 8; j++)
            {
                sub = treeB[noB].Daughter + j;
                if(treeB[sub].Shift >= treeB[noB].Shift) {
                    endrun(1, "Child node %d has shift %d, parent %d has shift %d. treeB is corrupt. \n",
                        sub, treeB[sub].Shift, noB, treeB[noB].Shift);
                }
                domain_toptree_merge(treeA, treeB, noA, sub, treeASize, MaxTopNodes);
            }
        }
        else
        {
            /* We can't divide by B so we do it for A, this may trigger the next branch, since
             * we are lowering A */
            if(treeA[noA].Daughter >= 0) {
                for(j = 0; j < 8; j++) {
                    sub = treeA[noA].Daughter + j;
                    domain_toptree_merge(treeA, treeB, sub, noB, treeASize, MaxTopNodes);
                }
            }
        }
    }
    else if(treeB[noB].Shift > treeA[noA].Shift)
    {
        /* Since we only know how to split A, here we simply add a spatial average to A */
        uint64_t n = 1L << (treeB[noB].Shift - treeA[noA].Shift);

        if(treeB[noB].Shift - treeA[noA].Shift > 60) {
            message(1, "Warning: Refusing to merge two tree nodes of wildly different depth: %d %d;\n ", treeB[noB].Shift, treeA[noA].Shift);
            n = 0;
        }

        count = treeB[noB].Count;
        cost = treeB[noB].Cost;

        if (n > 0) {
            /* this is no longer conserving total cost but it should be fine .. */
            treeA[noA].Count += count / n;
            treeA[noA].Cost += cost / n;

            // message(1, "adding cost to %d %td, %td\n", noA, count / n, cost / n);
            if(treeA[noA].Daughter >= 0) {
                for(j = 0; j < 8; j++) {
                    sub = treeA[noA].Daughter + j;
                    domain_toptree_merge(treeA, treeB, sub, noB, treeASize, MaxTopNodes);
                }
            }
        }
    }
}
