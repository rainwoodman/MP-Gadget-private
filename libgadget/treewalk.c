#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <alloca.h>
#include "utils.h"

#include "treewalk.h"
#include "partmanager.h"
#include "domain.h"
#include "forcetree.h"

#include <signal.h>
#define BREAKPOINT raise(SIGTRAP)

#define FACT1 0.366025403785    /* FACT1 = 0.5 * (sqrt(3)-1) */

/*!< Memory factor to leave for (N imported particles) > (N exported particles). */
static double ImportBufferBoost;
/* 7/9/24: The code segfaults if the send/recv buffer is larger than 4GB in size.
 * Likely a 32-bit variable is overflowing but it is hard to debug. Easier to enforce a maximum buffer size.*/
static size_t MaxExportBufferBytes = 3584*1024*1024L;

/*Initialise global treewalk parameters*/
void set_treewalk_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
        ImportBufferBoost = param_get_double(ps, "ImportBufferBoost");
    MPI_Bcast(&ImportBufferBoost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* This function is to allow a test which fills up the exchange buffer*/
void treewalk_set_max_export_buffer(const size_t maxbuf)
{
    MaxExportBufferBytes = maxbuf;
}

static void ev_primary(TreeWalk * tw);
static int ev_ndone(TreeWalk * tw, MPI_Comm comm);

static int
ngb_treefind_threads(TreeWalkQueryBase * I,
        TreeWalkNgbIterBase * iter,
        int startnode,
        LocalTreeWalk * lv);

#ifdef DEBUG
/*
 * for debugging
 */
#define WATCH { \
        printf("tw->WorkSet[0] = %d (%d) %s:%d\n", tw->WorkSet ? tw->WorkSet[0] : 0, tw->WorkSetSize, __FILE__, __LINE__); \
    }
static TreeWalk * GDB_current_ev = NULL;
#endif

static void
ev_init_thread(TreeWalk * const tw, LocalTreeWalk * lv)
{
    const size_t thread_id = omp_get_thread_num();
    lv->tw = tw;
    lv->maxNinteractions = 0;
    lv->minNinteractions = 1L<<45;
    lv->Ninteractions = 0;
    lv->Nexport = 0;
    lv->NThisParticleExport = 0;
    lv->nodelistindex = 0;
    if(tw->ExportTable_thread)
        lv->DataIndexTable = tw->ExportTable_thread[thread_id];
    else
        lv->DataIndexTable = NULL;
    if(tw->Ngblist)
        lv->ngblist = tw->Ngblist + thread_id * tw->tree->NumParticles;
}

static void
ev_begin(TreeWalk * tw, int * active_set, const size_t size)
{
    /* Needs to be 64-bit so that the multiplication in Ngblist malloc doesn't overflow*/
    const size_t NumThreads = omp_get_max_threads();
    MPI_Comm_size(MPI_COMM_WORLD, &tw->NTask);
    /* The last argument is may_have_garbage: in practice the only
     * trivial haswork is the gravtree. This has no (active) garbage because
     * the active list was just rebuilt, but on a PM step the active list is NULL
     * and we may still have swallowed BHs around. So in practice this avoids
     * computing gravtree for swallowed BHs on a PM step.*/
    int may_have_garbage = 0;
    /* Note this is not collective, but that should not matter.*/
    if(!active_set && SlotsManager->info[5].size > 0)
        may_have_garbage = 1;
    treewalk_build_queue(tw, active_set, size, may_have_garbage);

    /* Start first iteration at the beginning*/
    tw->WorkSetStart = 0;

    if(!tw->NoNgblist)
        tw->Ngblist = (int*) mymalloc("Ngblist", tw->tree->NumParticles * NumThreads * sizeof(int));
    else
        tw->Ngblist = NULL;

    /* Assert that the query and result structures are aligned to  64-bit boundary,
     * so that our MPI Send/Recv's happen from aligned memory.*/
    if(tw->query_type_elsize % 8 != 0)
        endrun(0, "Query structure has size %ld, not aligned to 64-bit boundary.\n", tw->query_type_elsize);
    if(tw->result_type_elsize % 8 != 0)
        endrun(0, "Result structure has size %ld, not aligned to 64-bit boundary.\n", tw->result_type_elsize);

    /*The amount of memory eventually allocated per tree buffer*/
    size_t bytesperbuffer = sizeof(struct data_index) + tw->query_type_elsize + tw->result_type_elsize;
    /*This memory scales like the number of imports. In principle this could be much larger than Nexport
     * if the tree is very imbalanced and many processors all need to export to this one. In practice I have
     * not seen this happen, but provide a parameter to boost the memory for Nimport just in case.*/
    bytesperbuffer += ceil(ImportBufferBoost * (tw->query_type_elsize + tw->result_type_elsize));
    /*Use all free bytes for the tree buffer, as in exchange. Leave some free memory for array overhead.*/
    size_t freebytes = (size_t) mymalloc_freebytes();
    freebytes -= 4096 * 10 * bytesperbuffer;

    tw->BunchSize = (size_t) floor(((double)freebytes)/ bytesperbuffer);
    if(tw->BunchSize * tw->query_type_elsize > MaxExportBufferBytes)
        tw->BunchSize = MaxExportBufferBytes / tw->query_type_elsize;
    /* Per thread*/
    tw->BunchSize /= omp_get_max_threads();

    if(freebytes <= 4096 * bytesperbuffer || tw->BunchSize < 100) {
        endrun(1231245, "Not enough free memory in %s to export particles: needed %ld bytes have %ld. can export %ld \n", tw->ev_label, bytesperbuffer, freebytes, tw->BunchSize);
    }

    /* Print some balance numbers*/
    int64_t nmin, nmax, total;
    MPI_Reduce(&tw->WorkSetSize, &nmin, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->WorkSetSize, &nmax, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->WorkSetSize, &total, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "Treewalk %s iter %ld: total part %ld max/MPI: %ld min/MPI: %ld balance: %g query %ld result %ld BunchSize %ld.\n",
            tw->ev_label, tw->Niteration, total, nmax, nmin, (double)nmax/((total+0.001)/tw->NTask), tw->query_type_elsize, tw->result_type_elsize, tw->BunchSize);

    report_memory_usage(tw->ev_label);
}

static void ev_finish(TreeWalk * tw)
{
    if(tw->Ngblist)
        myfree(tw->Ngblist);
    if(!tw->work_set_stolen_from_active)
        myfree(tw->WorkSet);
}

static void
treewalk_init_query(TreeWalk * tw, TreeWalkQueryBase * query, int i, const int * const NodeList)
{
#ifdef DEBUG
    query->ID = P[i].ID;
#endif

    int d;
    for(d = 0; d < 3; d ++) {
        query->Pos[d] = P[i].Pos[d];
    }

    if(NodeList) {
        memcpy(query->NodeList, NodeList, sizeof(query->NodeList[0]) * NODELISTLENGTH);
    } else {
        query->NodeList[0] = tw->tree->firstnode; /* root node */
        query->NodeList[1] = -1; /* terminate immediately */
    }

    tw->fill(i, query, tw);
}

static void
treewalk_init_result(TreeWalk * tw, TreeWalkResultBase * result, TreeWalkQueryBase * query)
{
    memset(result, 0, tw->result_type_elsize);
#ifdef DEBUG
    result->ID = query->ID;
#endif
}

static void
treewalk_reduce_result(TreeWalk * tw, TreeWalkResultBase * result, int i, enum TreeWalkReduceMode mode)
{
    if(tw->reduce != NULL)
        tw->reduce(i, result, mode, tw);
#ifdef DEBUG
    if(P[i].ID != result->ID)
        endrun(2, "Mismatched ID (%ld != %ld) for particle %d in treewalk reduction, mode %d\n", P[i].ID, result->ID, i, mode);
#endif
}

void
treewalk_build_queue(TreeWalk * tw, int * active_set, const size_t size, int may_have_garbage)
{
    tw->NThread = omp_get_max_threads();

    if(!tw->haswork && !may_have_garbage)
    {
        tw->WorkSetSize = size;
        tw->WorkSet = active_set;
        tw->work_set_stolen_from_active = 1;
        return;
    }

    tw->work_set_stolen_from_active = 0;
    /* Explicitly deal with the case where the queue is zero and there is nothing to do.
     * Some OpenMP compilers (nvcc) seem to still execute the below loop in that case*/
    if(size == 0) {
        tw->WorkSet = (int *) mymalloc("ActiveQueue", sizeof(int));
        tw->WorkSetSize = size;
        return;
    }

    /*We want a lockless algorithm which preserves the ordering of the particle list.*/
    gadget_thread_arrays gthread = gadget_setup_thread_arrays("ActiveQueue", 0, size);
    /* We enforce schedule static to ensure that each thread executes on contiguous particles.
     * Note static enforces the monotonic modifier but on OpenMP 5.0 nonmonotonic is the default.
     * static also ensures that no single thread gets more than tsize elements.*/
    #pragma omp parallel
    {
        size_t i;
        const int tid = omp_get_thread_num();
        size_t nqthrlocal = 0;
        int *thrqlocal = gthread.srcs[tid];
        #pragma omp for schedule(static, gthread.schedsz)
        for(i=0; i < size; i++)
        {
            /*Use raw particle number if active_set is null, otherwise use active_set*/
            const int p_i = active_set ? active_set[i] : (int) i;

            /* Skip the garbage /swallowed particles */
            if(P[p_i].IsGarbage || P[p_i].Swallowed)
                continue;

            if(tw->haswork && !tw->haswork(p_i, tw))
                continue;
    #ifdef DEBUG
            if(nqthrlocal >= gthread.total_size)
                endrun(5, "tid = %d nqthr = %ld, tsize = %ld size = %ld, tw->Nthread = %ld i = %ld\n", tid, nqthrlocal, gthread.total_size, size, tw->NThread, i);
    #endif
            thrqlocal[nqthrlocal] = p_i;
            nqthrlocal++;
        }
        gthread.sizes[tid] = nqthrlocal;
    }
    /*Merge step for the queue.*/
    size_t nqueue = gadget_compact_thread_arrays(&tw->WorkSet, &gthread);
    /*Shrink memory*/
    tw->WorkSet = (int *) myrealloc(tw->WorkSet, sizeof(int) * nqueue);

#if 0
    /* check the uniqueness of the active_set list. This is very slow. */
    qsort_openmp(tw->WorkSet, nqueue, sizeof(int), cmpint);
    for(i = 0; i < nqueue - 1; i ++) {
        if(tw->WorkSet[i] == tw->WorkSet[i+1]) {
            endrun(8829, "A few particles are twicely active.\n");
        }
    }
#endif
    tw->WorkSetSize = nqueue;
}

/* returns struct containing export counts */
static void
ev_primary(TreeWalk * tw)
{
    int64_t maxNinteractions = 0, minNinteractions = 1L << 45, Ninteractions=0;
#pragma omp parallel reduction(min:minNinteractions) reduction(max:maxNinteractions) reduction(+: Ninteractions)
    {
        LocalTreeWalk lv[1];
        /* Note: exportflag is local to each thread */
        ev_init_thread(tw, lv);
        lv->mode = TREEWALK_PRIMARY;

        /* use old index to recover from a buffer overflow*/;
        TreeWalkQueryBase * input = (TreeWalkQueryBase *) alloca(tw->query_type_elsize);
        TreeWalkResultBase * output = (TreeWalkResultBase *) alloca(tw->result_type_elsize);
        /* We must schedule dynamically so that we have reduced imbalance.
        * We do not need to worry about the export buffer filling up.*/
        /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
        * FoF treewalk needs a larger chnksz to avoid contention.*/
        int64_t chnksz = tw->WorkSetSize / (4*tw->NThread);
        if(chnksz < 1)
            chnksz = 1;
        if(chnksz > 100)
            chnksz = 100;
        int k;
        #pragma omp for schedule(dynamic, chnksz)
        for(k = 0; k < tw->WorkSetSize; k++) {
            const int i = tw->WorkSet ? tw->WorkSet[k] : k;
            /* Primary never uses node list */
            treewalk_init_query(tw, input, i, NULL);
            treewalk_init_result(tw, output, input);
            lv->target = i;
            tw->visit(input, output, lv);
            treewalk_reduce_result(tw, output, i, TREEWALK_PRIMARY);
        }
        if(maxNinteractions < lv->maxNinteractions)
            maxNinteractions = lv->maxNinteractions;
        if(minNinteractions > lv->maxNinteractions)
            minNinteractions = lv->minNinteractions;
        Ninteractions = lv->Ninteractions;
    }
    tw->maxNinteractions = maxNinteractions;
    tw->minNinteractions = minNinteractions;
    tw->Ninteractions += Ninteractions;
    tw->Nlistprimary += tw->WorkSetSize;
}

static int ev_ndone(TreeWalk * tw, MPI_Comm comm)
{
    int ndone;
    int done = !(tw->BufferFullFlag);
    MPI_Allreduce(&done, &ndone, 1, MPI_INT, MPI_SUM, comm);
    return ndone;
}

/* export a particle at target and no, thread safely
 *
 * This can also be called from a nonthreaded code
 *
 * */
int treewalk_export_particle(LocalTreeWalk * lv, int no)
{
    if(lv->mode != TREEWALK_TOPTREE || no < lv->tw->tree->lastnode) {
        endrun(1, "Called export not from a toptree.\n");
    }
    if(!lv->DataIndexTable)
        endrun(1, "DataIndexTable not allocated, treewalk_export_particle called in the wrong way\n");
    if(no - lv->tw->tree->lastnode > lv->tw->tree->NTopLeaves)
        endrun(1, "Bad export leaf: no = %d lastnode %ld ntop %d target %d\n", no, lv->tw->tree->lastnode, lv->tw->tree->NTopLeaves, lv->target);
    const int target = lv->target;
    TreeWalk * tw = lv->tw;
    const int task = tw->tree->TopLeaves[no - tw->tree->lastnode].Task;
    /* This index is a unique entry in the global DataIndexTable.*/
    size_t nexp = lv->Nexport;
    /* If the last export was to this task, we can perhaps just add this export to the existing NodeList. We can
     * be sure that all exports of this particle are contiguous.*/
    if(lv->NThisParticleExport >= 1 && lv->DataIndexTable[nexp-1].Task == task) {
#ifdef DEBUG
        /* This is just to be safe: only happens if our indices are off.*/
        if(lv->DataIndexTable[nexp - 1].Index != target)
            endrun(1, "Previous of %ld exports is target %d not current %d\n", lv->NThisParticleExport, lv->DataIndexTable[nexp-1].Index, target);
#endif
        if(lv->nodelistindex < NODELISTLENGTH) {
#ifdef DEBUG
            if(lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex] != -1)
                endrun(1, "Current nodelist %ld entry (%d) not empty!\n", lv->nodelistindex, lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex]);
#endif
            lv->DataIndexTable[nexp-1].NodeList[lv->nodelistindex] = tw->tree->TopLeaves[no - tw->tree->lastnode].treenode;
            lv->nodelistindex++;
            return 0;
        }
    }
    /* out of buffer space. Need to interrupt. */
    if(lv->Nexport >= tw->BunchSize) {
        return -1;
    }
    lv->DataIndexTable[nexp].Task = task;
    lv->DataIndexTable[nexp].Index = target;
    lv->DataIndexTable[nexp].NodeList[0] = tw->tree->TopLeaves[no - tw->tree->lastnode].treenode;
    int i;
    for(i = 1; i < NODELISTLENGTH; i++)
        lv->DataIndexTable[nexp].NodeList[i] = -1;
    lv->Nexport++;
    lv->nodelistindex = 1;
    lv->NThisParticleExport++;
    return 0;
}

void
alloc_export_memory(TreeWalk * tw)
{
    tw->Nexport_thread = ta_malloc2("localexports", size_t, tw->NThread);
    tw->ExportTable_thread = ta_malloc2("localexports", data_index *, tw->NThread);
    int i;
    for(i = 0; i < tw->NThread; i++)
        tw->ExportTable_thread[i] = (data_index*) mymalloc("DataIndexTable", sizeof(data_index) * tw->BunchSize);
    tw->QueueChunkEnd = ta_malloc2("queueend", int64_t, tw->NThread);
    for(i = 0; i < tw->NThread; i++)
        tw->QueueChunkEnd[i] = -1;
    tw->QueueChunkRestart = ta_malloc2("queuerestart", int, tw->NThread);
}

void
free_export_memory(TreeWalk * tw)
{
    myfree(tw->QueueChunkRestart);
    myfree(tw->QueueChunkEnd);
    int i;
    for(i = tw->NThread - 1; i >= 0; i--)
        myfree(tw->ExportTable_thread[i]);
    myfree(tw->ExportTable_thread);
    myfree(tw->Nexport_thread);
}

int
ev_toptree(TreeWalk * tw)
{
    tw->BufferFullFlag = 0;
    int64_t currentIndex = tw->WorkSetStart;
    int BufferFullFlag = 0;

    if(tw->Nexportfull > 0)
        message(0, "Toptree %s, iter %ld. First particle %ld size %ld.\n", tw->ev_label, tw->Nexportfull, tw->WorkSetStart, tw->WorkSetSize);

#pragma omp parallel reduction(+: BufferFullFlag)
    {
        LocalTreeWalk lv[1];
        /* Note: exportflag is local to each thread */
        ev_init_thread(tw, lv);
        lv->mode = TREEWALK_TOPTREE;
        /* Signals a full export buffer on this thread*/
        int BufferFull_thread = 0;
        const int tid = omp_get_thread_num();

        TreeWalkQueryBase * input = (TreeWalkQueryBase *) alloca(tw->query_type_elsize);
        TreeWalkResultBase * output = (TreeWalkResultBase *) alloca(tw->result_type_elsize);

        /* We schedule dynamically so that we have reduced imbalance.
         * We do not use the openmp dynamic scheduling, but roll our own
         * so that we can break from the loop if needed.*/
        int64_t chnk = 0;
        /* chunk size: 1 and 1000 were slightly (3 percent) slower than 8.
         * FoF treewalk needs a larger chnksz to avoid contention.*/
        int64_t chnksz = tw->WorkSetSize / (4*tw->NThread);
        if(chnksz < 1)
            chnksz = 1;
        if(chnksz > 1000)
            chnksz = 1000;
        do {
            int64_t end;
            /* Restart a previously partially evaluated chunk if there is one*/
            if(tw->Nexportfull > 0 && tw->QueueChunkEnd[tid] > 0) {
                chnk = tw->QueueChunkRestart[tid];
                end = tw->QueueChunkEnd[tid];
                tw->QueueChunkEnd[tid] = -1;
                //message(1, "T%d Restarting chunk %ld -> %ld\n", tid, chnk, end);
            }
            else {
                /* Get another chunk from the global queue*/
                chnk = atomic_fetch_and_add_64(&currentIndex, chnksz);
                /* This is a hand-rolled version of what openmp dynamic scheduling is doing.*/
                end = chnk + chnksz;
                /* Make sure we do not overflow the loop*/
                if(end > tw->WorkSetSize)
                    end = tw->WorkSetSize;
            }
            /* Reduce the chunk size towards the end of the walk*/
            if((tw->WorkSetSize  < end + chnksz * tw->NThread) && chnksz >= 2)
                chnksz /= 2;
            int k;
            for(k = chnk; k < end; k++) {
                const int i = tw->WorkSet ? tw->WorkSet[k] : k;
                /* Toptree never uses node list */
                treewalk_init_query(tw, input, i, NULL);
                lv->target = i;
                /* Reset the number of exported particles.*/
                lv->NThisParticleExport = 0;
                const int rt = tw->visit(input, output, lv);
                if(lv->NThisParticleExport > 1000)
                    message(5, "%ld exports for particle %d! Odd.\n", lv->NThisParticleExport, i);
                /* If we filled up, we need to remove the partially evaluated last particle from the export list,
                 * save the partially evaluated chunk, and leave this loop.*/
                if(rt < 0) {
                    //message(5, "Export buffer full for particle %d chnk: %ld -> %ld on thread %d with %ld exports\n", i, chnk, end, tid, lv->NThisParticleExport);
                    /* export buffer has filled up, can't do more work.*/
                    BufferFull_thread = 1;
                    /* Drop partial exports on the current particle, whose toptree will be re-evaluated*/
                    lv->Nexport -= lv->NThisParticleExport;
                    /* Check that the final export in the list is indeed from a different particle*/
                    if(lv->NThisParticleExport > 0 && lv->DataIndexTable[lv->Nexport-1].Index >= i)
                        endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d < index %d\n", lv->Nexport,
                            lv->NThisParticleExport, i, lv->DataIndexTable[lv->Nexport-1].Index);
                    /* Check that the earliest dropped export in the list is from the same particle*/
                    if(lv->NThisParticleExport > 0 && lv->DataIndexTable[lv->Nexport].Index != i)
                        endrun(5, "Something screwed up in export queue: nexp %ld (local %ld) last %d != index %d\n", lv->Nexport,
                            lv->NThisParticleExport, i, lv->DataIndexTable[lv->Nexport].Index);
                    /* Store information for the current chunk, so we can resume successfully exactly where we left off.
                        Each thread stores chunk information */
                    tw->QueueChunkRestart[tid] = k;
                    tw->QueueChunkEnd[tid] = end;
                    break;
                }
            }
        } while(chnk < tw->WorkSetSize && BufferFull_thread == 0);
        tw->Nexport_thread[tid] = lv->Nexport;
        BufferFullFlag += BufferFull_thread;
    }

    if(BufferFullFlag > 0) {
        size_t Nexport = 0;
        int i;
        for(i = 0; i < tw->NThread; i++)
            Nexport += tw->Nexport_thread[i];
        message(1, "Tree export buffer full on %d of %ld threads with %lu exports (%lu Mbytes). First particle %ld new start: %ld size %ld.\n",
                        BufferFullFlag, tw->NThread, Nexport, Nexport*tw->query_type_elsize/1024/1024, tw->WorkSetStart, currentIndex, tw->WorkSetSize);
        if(currentIndex == tw->WorkSetStart)
            endrun(5, "Not enough export space to make progress! lastsuc %ld Bunchsize: %ld \n", currentIndex, tw->BunchSize);
    }
    // else
        // message(1, "Finished toptree on %d threads. First particle %ld next start: %ld size %ld.\n", BufferFullFlag, tw->WorkSetStart, currentIndex, tw->WorkSetSize);
    /* Start again with the next chunk not yet evaluated*/
    tw->WorkSetStart = currentIndex;
    tw->BufferFullFlag = BufferFullFlag;
    return tw->BufferFullFlag;
}

struct ImpExpCounts
{
    int64_t * Export_count;
    int64_t * Import_count;
    int64_t * Export_offset;
    int64_t * Import_offset;
    MPI_Comm comm;
    int NTask;
    /* Number of particles exported to this processor*/
    size_t Nimport;
    /* Number of particles exported from this processor*/
    size_t Nexport;
};

/* Allocate/free a commbuffer with space for N tasks.*/
void alloc_commbuffer(struct CommBuffer * buffer, int NTask, int alloc_high)
{
    if(alloc_high) {
        buffer->rdata_all = ta_malloc2("requests", MPI_Request, NTask);
        buffer->rqst_task = ta_malloc2("rqst", int, NTask);
        buffer->displs = ta_malloc2("displs", int, NTask);
    }
    else {
        buffer->rdata_all = ta_malloc("requests", MPI_Request, NTask);
        buffer->rqst_task = ta_malloc("rqst", int, NTask);
        buffer->displs = ta_malloc("displs", int, NTask);
    }
    buffer->nrequest_all = 0;
    buffer->databuf = NULL;
}

void free_commbuffer(struct CommBuffer * buffer)
{
    if(buffer->databuf) {
        myfree(buffer->databuf);
        buffer->databuf = NULL;
    }
    if(buffer->displs) {
        ta_free(buffer->displs);
        ta_free(buffer->rqst_task);
        ta_free(buffer->rdata_all);
    }
}

void free_impexpcount(struct ImpExpCounts * count)
{
    ta_free(count->Export_count);
}

#define COMM_RECV 1
#define COMM_SEND 0

/* Routine to send data to all tasks async. If receive is set, the routine receives data. The structure stores the requests.
 Empty tasks are skipped. Must call alloc_commbuffer on the buffer first and buffer->databuf must be set.*/
void
MPI_fill_commbuffer(struct CommBuffer * buffer, int64_t *cnts, int64_t *displs, MPI_Datatype type, int receive, int tag, MPI_Comm comm)
{
    int ThisTask;
    int NTask;
    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);
    ptrdiff_t lb, elsize;
    MPI_Type_get_extent(type, &lb, &elsize);
    int nrequests = 0;

    int i;
    /* Loop over all tasks, starting with the one just past this one*/
    for(i = 1; i < NTask; i++)
    {
        int target = (ThisTask + i) % NTask;
        if(cnts[target] == 0) continue;
        buffer->rqst_task[nrequests] = target;
        if(receive == COMM_RECV) {
            MPI_Irecv(((char*) buffer->databuf) + elsize * displs[target], cnts[target],
                type, target, tag, comm, &buffer->rdata_all[nrequests++]);
        }
        else {
            MPI_Isend(((char*) buffer->databuf) + elsize * displs[target], cnts[target],
                type, target, tag, comm, &buffer->rdata_all[nrequests++]);
        }
    }
    buffer->nrequest_all = nrequests;
}

/* Waits for all the requests in the bufferbuffer to be complete*/
void wait_commbuffer(struct CommBuffer * buffer)
{
    MPI_Waitall(buffer->nrequest_all, buffer->rdata_all, MPI_STATUSES_IGNORE);
}

static struct CommBuffer ev_secondary(struct CommBuffer * imports, struct ImpExpCounts* counts, TreeWalk * tw)
{
    struct CommBuffer res_imports = {0};
    alloc_commbuffer(&res_imports, counts->NTask, 1);
    res_imports.databuf = (char *) mymalloc2("ImportResult", counts->Nimport * tw->result_type_elsize);

    MPI_Datatype type;
    MPI_Type_contiguous(tw->result_type_elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);
    int * complete_array = ta_malloc("completes", int, imports->nrequest_all);

    int tot_completed = 0;
    /* Test each request in turn until it completes*/
    while(tot_completed < imports->nrequest_all) {
        int complete_cnt = MPI_UNDEFINED;
        /* Check for some completed requests: note that cleanup is performed if the requests are complete.
         * There may be only 1 completed request, and we need to wait again until we have more.*/
        MPI_Waitsome(imports->nrequest_all, imports->rdata_all, &complete_cnt, complete_array, MPI_STATUSES_IGNORE);
        /* This happens if all requests are MPI_REQUEST_NULL. It should never be hit*/
        if (complete_cnt == MPI_UNDEFINED)
            break;
        int j;
        for(j = 0; j < complete_cnt; j++) {
            const int i = complete_array[j];
            /* Note the task number index is not the index in the request array (some tasks were skipped because we have zero exports)! */
            const int task = imports->rqst_task[i];
            const int64_t nimports_task = counts->Import_count[task];
            // message(1, "starting at %d with %d for iport %d task %d\n", counts->Import_offset[task], counts->Import_count[task], i, task);
            char * databufstart = imports->databuf + counts->Import_offset[task] * tw->query_type_elsize;
            char * dataresultstart = res_imports.databuf + counts->Import_offset[task] * tw->result_type_elsize;
            /* This sends each set of imports to a parallel for loop. This may lead to suboptimal resource allocation if only a small number of imports come from a processor.
            * If there are a large number of importing ranks each with a small number of imports, a better scheme could be to send each chunk to a separate openmp task.
            * However, each openmp task by default only uses 1 thread. One may explicitly enable openmp nested parallelism, but I think that is not safe,
            * or it would be enabled by default.*/
            #pragma omp parallel
                {
                    int64_t j;
                    LocalTreeWalk lv[1];

                    ev_init_thread(tw, lv);
                    lv->mode = TREEWALK_GHOSTS;
                    #pragma omp for
                    for(j = 0; j < nimports_task; j++) {
                        TreeWalkQueryBase * input = (TreeWalkQueryBase *) (databufstart + j * tw->query_type_elsize);
                        TreeWalkResultBase * output = (TreeWalkResultBase *) (dataresultstart + j * tw->result_type_elsize);
                        treewalk_init_result(tw, output, input);
                        lv->target = -1;
                        tw->visit(input, output, lv);
                    }
                }
            /* Send the completed data back*/
            res_imports.rqst_task[res_imports.nrequest_all] = task;
            MPI_Isend(dataresultstart, nimports_task, type, task, 101923, counts->comm, &res_imports.rdata_all[res_imports.nrequest_all++]);
            tot_completed++;
        }
    };
    myfree(complete_array);
    MPI_Type_free(&type);
    return res_imports;
}

static struct ImpExpCounts
ev_export_import_counts(TreeWalk * tw, MPI_Comm comm)
{
    int NTask;
    struct ImpExpCounts counts = {0};
    MPI_Comm_size(comm, &NTask);
    counts.NTask = NTask;
    counts.comm = comm;
    counts.Export_count = ta_malloc("Tree_counts", int64_t, 4*NTask);
    counts.Export_offset = counts.Export_count + NTask;
    counts.Import_count = counts.Export_offset + NTask;
    counts.Import_offset = counts.Import_count + NTask;
    memset(counts.Export_count, 0, sizeof(int64_t)*4*NTask);

    int64_t i;
    counts.Nexport=0;
    /* Calculate the amount of data to send. */
    for(i = 0; i < tw->NThread; i++)
    {
        int64_t * exportcount = counts.Export_count;
        size_t k;
        #pragma omp parallel for reduction(+: exportcount[:NTask])
        for(k = 0; k < tw->Nexport_thread[i]; k++)
            exportcount[tw->ExportTable_thread[i][k].Task]++;
        /* This is over all full buffers.*/
        tw->Nexport_sum += tw->Nexport_thread[i];
        /* This is the export count*/
        counts.Nexport += tw->Nexport_thread[i];
    }
    /* Exchange the counts. Note this is synchronous so we need to ensure the toptree walk, which happens before this, is balanced.*/
    MPI_Alltoall(counts.Export_count, 1, MPI_INT64, counts.Import_count, 1, MPI_INT64, counts.comm);
    // message(1, "Exporting %ld particles. Thread 0 is %ld\n", counts.Nexport, tw->Nexport_thread[0]);

    counts.Nimport = counts.Import_count[0];
    tw->NExportTargets = (counts.Export_count[0] > 0);
    for(i = 1; i < NTask; i++)
    {
        counts.Nimport += counts.Import_count[i];
        counts.Export_offset[i] = counts.Export_offset[i - 1] + counts.Export_count[i - 1];
        counts.Import_offset[i] = counts.Import_offset[i - 1] + counts.Import_count[i - 1];
        tw->NExportTargets += (counts.Export_count[i] > 0);
    }
    return counts;
}

/* Builds the list of exported particles and async sends the export queries. */
static void ev_send_recv_export_import(struct ImpExpCounts * counts, TreeWalk * tw, struct CommBuffer * exports, struct CommBuffer * imports)
{
    alloc_commbuffer(exports, counts->NTask, 0);
    exports->databuf = (char *) mymalloc("ExportQuery", counts->Nexport * tw->query_type_elsize);

    alloc_commbuffer(imports, counts->NTask, 0);
    imports->databuf = (char *) mymalloc("ImportQuery", counts->Nimport * tw->query_type_elsize);

    MPI_Datatype type;
    MPI_Type_contiguous(tw->query_type_elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);

    /* Post recvs before sends. This sometimes allows for a fastpath.*/
    MPI_fill_commbuffer(imports, counts->Import_count, counts->Import_offset, type, COMM_RECV, 101922, counts->comm);

    /* prepare particle data for export */
    int64_t * real_send_count = ta_malloc("tmp_send_count", int64_t, tw->NTask);
    memset(real_send_count, 0, sizeof(int64_t)*tw->NTask);
    int64_t i;
    for(i = 0; i < tw->NThread; i++)
    {
        size_t k;
        for(k = 0; k < tw->Nexport_thread[i]; k++) {
            const int place = tw->ExportTable_thread[i][k].Index;
            const int task = tw->ExportTable_thread[i][k].Task;
            const int64_t bufpos = real_send_count[task] + counts->Export_offset[task];
            TreeWalkQueryBase * input = (TreeWalkQueryBase*) (exports->databuf + bufpos * tw->query_type_elsize);
            real_send_count[task]++;
            treewalk_init_query(tw, input, place, tw->ExportTable_thread[i][k].NodeList);
        }
    }
#ifdef DEBUG
/* Checks!*/
    for(i = 0; i < tw->NTask; i++)
        if(real_send_count[i] != counts->Export_count[i])
            endrun(6, "Inconsistent export to task %ld of %d: %ld expected %ld\n", i, tw->NTask, real_send_count[i], counts->Export_count[i]);
#endif
    myfree(real_send_count);
    MPI_fill_commbuffer(exports, counts->Export_count, counts->Export_offset, type, COMM_SEND, 101922, counts->comm);
    MPI_Type_free(&type);
    return;
}

static void ev_recv_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts, TreeWalk * tw)
{
    alloc_commbuffer(exportbuf, counts->NTask, 1);
    MPI_Datatype type;
    MPI_Type_contiguous(tw->result_type_elsize, MPI_BYTE, &type);
    MPI_Type_commit(&type);
    exportbuf->databuf = (char*) mymalloc2("ExportResult", counts->Nexport * tw->result_type_elsize);
    /* Post the receives first so we can hit a zero-copy fastpath.*/
    MPI_fill_commbuffer(exportbuf, counts->Export_count, counts->Export_offset, type, COMM_RECV, 101923, counts->comm);
    // alloc_commbuffer(&res_imports, counts.NTask, 0);
    // MPI_fill_commbuffer(import, counts->Import_count, counts->Import_offset, type, COMM_SEND, 101923, counts->comm);
    MPI_Type_free(&type);
}

static void ev_reduce_export_result(struct CommBuffer * exportbuf, struct ImpExpCounts * counts, TreeWalk * tw)
{
    int64_t i;
    /* Notice that we build the dataindex table individually
     * on each thread, so we are ordered by particle and have memory locality.*/
    if(tw->reduce != NULL) {
        int * real_recv_count = ta_malloc("tmp_recv_count", int, tw->NTask);
        memset(real_recv_count, 0, sizeof(int)*tw->NTask);
        for(i = 0; i < tw->NThread; i++)
        {
            size_t k;
            for(k = 0; k < tw->Nexport_thread[i]; k++) {
                const int place = tw->ExportTable_thread[i][k].Index;
                const int task = tw->ExportTable_thread[i][k].Task;
                const int64_t bufpos = real_recv_count[task] + counts->Export_offset[task];
                real_recv_count[task]++;
                TreeWalkResultBase * output = (TreeWalkResultBase*) (exportbuf->databuf + tw->result_type_elsize * bufpos);
                treewalk_reduce_result(tw, output, place, TREEWALK_GHOSTS);
#ifdef DEBUG
                if(output->ID != P[place].ID)
                    endrun(8, "Error in communication: IDs mismatch %ld %ld\n", output->ID, P[place].ID);
#endif
            }
        }
        myfree(real_recv_count);
    }
}

/* run a treewalk on an active_set.
 *
 * active_set : a list of indices of particles. If active_set is NULL,
 *              all (NumPart) particles are used.
 *
 * */
void
treewalk_run(TreeWalk * tw, int * active_set, size_t size)
{
    if(!force_tree_allocated(tw->tree)) {
        endrun(0, "Tree has been freed before this treewalk.\n");
    }

    double tstart, tend;
#ifdef DEBUG
    GDB_current_ev = tw;
#endif

    tstart = second();
    ev_begin(tw, active_set, size);

    if(tw->preprocess) {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < tw->WorkSetSize; i ++) {
            const int p_i = tw->WorkSet ? tw->WorkSet[i] : i;
            tw->preprocess(p_i, tw);
        }
    }

    tend = second();
    tw->timecomp3 += timediff(tstart, tend);

    if(tw->visit) {
        tw->Nexportfull = 0;
        tw->Nexport_sum = 0;
        tw->Ninteractions = 0;
        int Ndone = 0;
        /* Needs to be outside loop because it allocates restart information*/
        alloc_export_memory(tw);
        do
        {
            tstart = second();
            /* First do the toptree and export particles for sending.*/
            ev_toptree(tw);
            /* All processes sync via alltoall.*/
            struct ImpExpCounts counts = ev_export_import_counts(tw, MPI_COMM_WORLD);
            Ndone = ev_ndone(tw, MPI_COMM_WORLD);
            /* Send the exported particle data */
            struct CommBuffer exports = {0}, imports = {0};
            /* exports is allocated first, then imports*/
            ev_send_recv_export_import(&counts, tw, &exports, &imports);
            tend = second();
            tw->timecomp0 += timediff(tstart, tend);
            /* Only do this on the first iteration, as we only need to do it once.*/
            tstart = second();
            if(tw->Nexportfull == 0)
                ev_primary(tw); /* do local particles and prepare export list */
            tend = second();
            tw->timecomp1 += timediff(tstart, tend);
            /* Do processing of received particles. We implement a queue that
             * checks each incoming task in turn and processes them as they arrive.*/
            tstart = second();
            /* Posts recvs to get the export results (which are sent in ev_secondary).*/
            struct CommBuffer res_exports = {0};
            ev_recv_export_result(&res_exports, &counts, tw);
            struct CommBuffer res_imports = ev_secondary(&imports, &counts, tw);
            // report_memory_usage(tw->ev_label);
            free_commbuffer(&imports);
            tend = second();
            tw->timecomp2 += timediff(tstart, tend);
            /* Now clear the sent data buffer, waiting for the send to complete.
             * This needs to be after the other end has called recv.*/
            tstart = second();
            wait_commbuffer(&res_exports);
            tend = second();
            tw->timewait1 += timediff(tstart, tend);
            tstart = second();
            ev_reduce_export_result(&res_exports, &counts, tw);
            wait_commbuffer(&exports);
            free_commbuffer(&exports);
            wait_commbuffer(&res_imports);
            tend = second();
            tw->timecommsumm += timediff(tstart, tend);
            free_commbuffer(&res_imports);
            free_commbuffer(&res_exports);
            free_impexpcount(&counts);
            /* Free export memory*/
            tw->Nexportfull++;
            /* Note there is no sync at the end!*/
        } while(Ndone < tw->NTask);
        free_export_memory(tw);
    }

    tstart = second();
    if(tw->postprocess) {
        int64_t i;
        #pragma omp parallel for
        for(i = 0; i < tw->WorkSetSize; i ++) {
            const int p_i = tw->WorkSet ? tw->WorkSet[i] : i;
            tw->postprocess(p_i, tw);
        }
    }
    tend = second();
    tw->timecomp3 += timediff(tstart, tend);
    ev_finish(tw);
    tw->Niteration++;
}

void
treewalk_add_counters(LocalTreeWalk * lv, const int64_t ninteractions)
{
    if(lv->maxNinteractions < ninteractions)
        lv->maxNinteractions = ninteractions;
    if(lv->minNinteractions > ninteractions)
        lv->minNinteractions = ninteractions;
    lv->Ninteractions += ninteractions;
}

/**********
 *
 * This particular TreeWalkVisitFunction that uses the nbgiter memeber of
 * The TreeWalk object to iterate over the neighbours of a Query.
 *
 * All Pairwise interactions are implemented this way.
 *
 * Note: Short range gravity is not based on pair enumeration.
 * We may want to port it over and see if gravtree.c receives any speed up.
 *
 * Required fields in TreeWalk: ngbiter, ngbiter_type_elsize.
 *
 * Before the iteration starts, ngbiter is called with iter->base.other == -1.
 * The callback function shall initialize the interator with Hsml, mask, and symmetric.
 *
 *****/
int treewalk_visit_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv)
{

    TreeWalkNgbIterBase * iter = (TreeWalkNgbIterBase *) alloca(lv->tw->ngbiter_type_elsize);

    /* Kick-start the iteration with other == -1 */
    iter->other = -1;
    lv->tw->ngbiter(I, O, iter, lv);
    /* Check whether the tree contains the particles we are looking for*/
    if((lv->tw->tree->mask & iter->mask) != iter->mask)
        endrun(5, "Treewalk for particles with mask %d but tree mask is only %d overlap %d.\n", iter->mask, lv->tw->tree->mask, lv->tw->tree->mask & iter->mask);
    /* If symmetric, make sure we did hmax first*/
    if(iter->symmetric == NGB_TREEFIND_SYMMETRIC && !lv->tw->tree->hmax_computed_flag)
        endrun(3, "%s tried to do a symmetric treewalk without computing hmax!\n", lv->tw->ev_label);
    const double BoxSize = lv->tw->tree->BoxSize;

    int64_t ninteractions = 0;
    int inode = 0;

    for(inode = 0; inode < NODELISTLENGTH && I->NodeList[inode] >= 0; inode++)
    {
        int numcand = ngb_treefind_threads(I, iter, I->NodeList[inode], lv);
        /* Export buffer is full end prematurally */
        if(numcand < 0)
            return numcand;

        /* If we are here, export is successful. Work on this particle -- first
         * filter out all of the candidates that are actually outside. */
        int numngb;

        for(numngb = 0; numngb < numcand; numngb ++) {
            int other = lv->ngblist[numngb];

            /* Skip garbage*/
            if(P[other].IsGarbage)
                continue;
            /* In case the type of the particle has changed since the tree was built.
             * Happens for wind treewalk for gas turned into stars on this timestep.*/
            if(!((1<<P[other].Type) & iter->mask)) {
                continue;
            }

            double dist;

            if(iter->symmetric == NGB_TREEFIND_SYMMETRIC) {
                dist = DMAX(P[other].Hsml, iter->Hsml);
            } else {
                dist = iter->Hsml;
            }

            double r2 = 0;
            int d;
            double h2 = dist * dist;
            for(d = 0; d < 3; d ++) {
                /* the distance vector points to 'other' */
                iter->dist[d] = NEAREST(I->Pos[d] - P[other].Pos[d], BoxSize);
                r2 += iter->dist[d] * iter->dist[d];
                if(r2 > h2) break;
            }
            if(r2 > h2) continue;

            /* update the iter and call the iteration function*/
            iter->r2 = r2;
            iter->r = sqrt(r2);
            iter->other = other;

            lv->tw->ngbiter(I, O, iter, lv);
        }

        ninteractions += numngb;
    }

    treewalk_add_counters(lv, ninteractions);

    return 0;
}

/**
 * Cull a node.
 *
 * Returns 1 if the node shall be opened;
 * Returns 0 if the node has no business with this query.
 */
static int
cull_node(const TreeWalkQueryBase * const I, const TreeWalkNgbIterBase * const iter, const struct NODE * const current, const double BoxSize)
{
    double dist;
    if(iter->symmetric == NGB_TREEFIND_SYMMETRIC) {
        dist = DMAX(current->mom.hmax, iter->Hsml) + 0.5 * current->len;
    } else {
        dist = iter->Hsml + 0.5 * current->len;
    }

    double r2 = 0;
    double dx = 0;
    /* do each direction */
    int d;
    for(d = 0; d < 3; d ++) {
        dx = NEAREST(current->center[d] - I->Pos[d], BoxSize);
        if(dx > dist) return 0;
        if(dx < -dist) return 0;
        r2 += dx * dx;
    }
    /* now test against the minimal sphere enclosing everything */
    dist += FACT1 * current->len;

    if(r2 > dist * dist) {
        return 0;
    }
    return 1;
}
/*****
 * This is the internal code that looks for particles in the ngb tree from
 * searchcenter upto hsml. if iter->symmetric is NGB_TREE_FIND_SYMMETRIC, then upto
 * max(P[other].Hsml, iter->Hsml).
 *
 * Particle that intersects with other domains are marked for export.
 * The hosting nodes (leaves of the global tree) are exported as well.
 *
 * For all 'other' particle within the neighbourhood and are local on this processor,
 * this function calls the ngbiter member of the TreeWalk object.
 * iter->base.other, iter->base.dist iter->base.r2, iter->base.r, are properly initialized.
 *
 * */
static int
ngb_treefind_threads(TreeWalkQueryBase * I,
        TreeWalkNgbIterBase * iter,
        int startnode,
        LocalTreeWalk * lv)
{
    int no;
    int numcand = 0;

    const ForceTree * tree = lv->tw->tree;
    const double BoxSize = tree->BoxSize;

    no = startnode;

    while(no >= 0)
    {
        if(node_is_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Particles should be added before getting here! no = %d, father = %d (ptype = %d) start=%d mode = %d\n", no, fat, tree->Nodes[fat].f.ChildType, startnode, lv->mode);
        }
        if(node_is_pseudo_particle(no, tree)) {
            int fat = force_get_father(no, tree);
            endrun(12312, "Pseudo-Particles should be added before getting here! no = %d, father = %d (ptype = %d)\n", no, fat, tree->Nodes[fat].f.ChildType);
        }

        struct NODE *current = &tree->Nodes[no];

        /* When walking exported particles we start from the encompassing top-level node,
         * so if we get back to a top-level node again we are done.*/
        if(lv->mode == TREEWALK_GHOSTS) {
            /* The first node is always top-level*/
            if(current->f.TopLevel && no != startnode) {
                /* we reached a top-level node again, which means that we are done with the branch */
                break;
            }
        }

        /* Cull the node */
        if(0 == cull_node(I, iter, current, BoxSize)) {
            /* in case the node can be discarded */
            no = current->sibling;
            continue;
        }

        if(lv->mode == TREEWALK_TOPTREE) {
            if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* Export the pseudo particle*/
                if(-1 == treewalk_export_particle(lv, current->s.suns[0]))
                    return -1;
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            /* Only walk toptree nodes here*/
            if(current->f.TopLevel && !current->f.InternalTopLevel) {
                no = current->sibling;
                continue;
            }
        }
        else {
            /* Node contains relevant particles, add them.*/
            if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                int i;
                int * suns = current->s.suns;
                for (i = 0; i < current->s.noccupied; i++) {
                    lv->ngblist[numcand++] = suns[i];
                }
                /* Move sideways*/
                no = current->sibling;
                continue;
            }
            else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                /* pseudo particle */
                if(lv->mode == TREEWALK_GHOSTS) {
                    endrun(12312, "Secondary for particle %d from node %d found pseudo at %d.\n", lv->target, startnode, no);
                } else {
                    /* This has already been evaluated with the toptree. Move sideways.*/
                    no = current->sibling;
                    continue;
                }
            }
        }
        /* ok, we need to open the node */
        no = current->s.suns[0];
    }

    return numcand;
}

/*****
 * Variant of ngbiter that doesn't use the Ngblist.
 * The ngblist is generally preferred for memory locality reasons.
 * Use this variant if the evaluation
 * wants to change the search radius, such as for knn algorithms
 * or some density code. Don't use it if the treewalk modifies other particles.
 * */
int treewalk_visit_nolist_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv)
{
    TreeWalkNgbIterBase * iter = (TreeWalkNgbIterBase *) alloca(lv->tw->ngbiter_type_elsize);

    /* Kick-start the iteration with other == -1 */
    iter->other = -1;
    lv->tw->ngbiter(I, O, iter, lv);

    int64_t ninteractions = 0;
    int inode;
    for(inode = 0; inode < NODELISTLENGTH && I->NodeList[inode] >= 0; inode++)
    {
        int no = I->NodeList[inode];
        const ForceTree * tree = lv->tw->tree;
        const double BoxSize = tree->BoxSize;

        while(no >= 0)
        {
            struct NODE *current = &tree->Nodes[no];

            /* When walking exported particles we start from the encompassing top-level node,
            * so if we get back to a top-level node again we are done.*/
            if(lv->mode == TREEWALK_GHOSTS) {
                /* The first node is always top-level*/
                if(no > tree->lastnode)
                    endrun(7, "Node is after lastnode. no %d lastnode %ld start %d first %ld\n", no, tree->lastnode, I->NodeList[inode], tree->firstnode);
                if(current->f.TopLevel && no != I->NodeList[inode]) {
                    /* we reached a top-level node again, which means that we are done with the branch */
                    break;
                }
            }

            /* Cull the node */
            if(0 == cull_node(I, iter, current, BoxSize)) {
                /* in case the node can be discarded */
                no = current->sibling;
                continue;
            }
            if(lv->mode == TREEWALK_TOPTREE) {
                if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* Export the pseudo particle*/
                    if(-1 == treewalk_export_particle(lv, current->s.suns[0]))
                        return -1;
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                /* Only walk toptree nodes here*/
                if(current->f.TopLevel && !current->f.InternalTopLevel) {
                    no = current->sibling;
                    continue;
                }
            }
            /* Node contains relevant particles, add them.*/
            else {
                if(current->f.ChildType == PARTICLE_NODE_TYPE) {
                    int i;
                    int * suns = current->s.suns;
                    for (i = 0; i < current->s.noccupied; i++) {
                        /* Now evaluate a particle for the list*/
                        int other = suns[i];
                        /* Skip garbage*/
                        if(P[other].IsGarbage)
                            continue;
                        /* In case the type of the particle has changed since the tree was built.
                        * Happens for wind treewalk for gas turned into stars on this timestep.*/
                        if(!((1<<P[other].Type) & iter->mask))
                            continue;

                        double dist = iter->Hsml;
                        double r2 = 0;
                        int d;
                        double h2 = dist * dist;
                        for(d = 0; d < 3; d ++) {
                            /* the distance vector points to 'other' */
                            iter->dist[d] = NEAREST(I->Pos[d] - P[other].Pos[d], BoxSize);
                            r2 += iter->dist[d] * iter->dist[d];
                            if(r2 > h2) break;
                        }
                        if(r2 > h2) continue;

                        /* update the iter and call the iteration function*/
                        iter->r2 = r2;
                        iter->other = other;
                        iter->r = sqrt(r2);
                        lv->tw->ngbiter(I, O, iter, lv);
                        ninteractions++;
                    }
                    /* Move sideways*/
                    no = current->sibling;
                    continue;
                }
                else if(current->f.ChildType == PSEUDO_NODE_TYPE) {
                    /* pseudo particle */
                    if(lv->mode == TREEWALK_GHOSTS) {
                        endrun(12312, "Secondary for particle %d from node %d found pseudo at %d.\n", lv->target, I->NodeList[inode], no);
                    } else {
                        /* This has already been evaluated with the toptree. Move sideways.*/
                        no = current->sibling;
                        continue;
                    }
                }
            }
            /* ok, we need to open the node */
            no = current->s.suns[0];
        }
    }

    treewalk_add_counters(lv, ninteractions);

    return 0;
}

/* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
 * This loop is used primarily in density estimation.*/
void
treewalk_do_hsml_loop(TreeWalk * tw, int * queue, int64_t queuesize, int update_hsml)
{
    int NumThreads = omp_get_max_threads();
    tw->maxnumngb = ta_malloc("numngb", double, NumThreads);
    tw->minnumngb = ta_malloc("numngb2", double, NumThreads);

    /* Build the first queue */
    double tstart = second();
    treewalk_build_queue(tw, queue, queuesize, 0);
    double tend = second();

    /* Next call to treewalk_run will over-write these pointers*/
    int64_t size = tw->WorkSetSize;
    int * ReDoQueue = tw->WorkSet;
    /* First queue is allocated low*/
    int alloc_high = 0;
    /* We don't need to redo the queue generation
     * but need to keep track of allocated memory.*/
    int orig_queue_alloc = (tw->haswork != NULL);
    tw->haswork = NULL;
    tw->timecomp3 += timediff(tstart, tend);
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do {
        /* The RedoQueue needs enough memory to store every workset particle on every thread, because
         * we cannot guarantee that the sph particles are evenly spread across threads!*/
        int * CurQueue = ReDoQueue;
        int i;
        for(i = 0; i < NumThreads; i++) {
            tw->maxnumngb[i] = 0;
            tw->minnumngb[i] = 1e50;
        }
        /* The ReDoQueue swaps between high and low allocations so we can have two allocated alternately*/
        if(!alloc_high)
            alloc_high = 1;
        else
            alloc_high = 0;
        gadget_thread_arrays loop = gadget_setup_thread_arrays("ReDoQueue", alloc_high, size);
        tw->NPRedo = loop.srcs;
        tw->NPLeft = loop.sizes;
        tw->Redo_thread_alloc = loop.total_size;
        treewalk_run(tw, CurQueue, size);

        /* Now done with the current queue*/
        if(orig_queue_alloc || tw->Niteration > 1)
            myfree(CurQueue);

        size = gadget_compact_thread_arrays(&ReDoQueue, &loop);
        /* We can stop if we are not updating hsml or if we are done.*/
        if(!update_hsml || !MPIU_Any(size > 0, MPI_COMM_WORLD)) {
            myfree(ReDoQueue);
            break;
        }
        for(i = 1; i < NumThreads; i++) {
            if(tw->maxnumngb[0] < tw->maxnumngb[i])
                tw->maxnumngb[0] = tw->maxnumngb[i];
            if(tw->minnumngb[0] > tw->minnumngb[i])
                tw->minnumngb[0] = tw->minnumngb[i];
        }
        double minngb, maxngb;
        MPI_Reduce(&tw->maxnumngb[0], &maxngb, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&tw->minnumngb[0], &minngb, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        message(0, "Max ngb=%g, min ngb=%g\n", maxngb, minngb);
#ifdef DEBUG
        treewalk_print_stats(tw);
#endif

        /*Shrink memory*/
        ReDoQueue = (int *) myrealloc(ReDoQueue, sizeof(int) * size);

        /*
        if(ntot < 1 ) {
            foreach(ActiveParticle)
            {
                if(density_haswork(i)) {
                    MyFloat Left = DENSITY_GET_PRIV(tw)->Left[i];
                    MyFloat Right = DENSITY_GET_PRIV(tw)->Right[i];
                    message (1, "i=%d task=%d ID=%llu type=%d, Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)\n",
                         i, ThisTask, P[i].ID, P[i].Type, P[i].Hsml, Left, Right,
                         (float) P[i].NumNgb, Right - Left, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                }
            }

        }
        */
#ifdef DEBUG
        if(size < 10 && tw->Niteration > 20 ) {
            int pp = ReDoQueue[0];
            message(1, "Remaining i=%d, t %d, pos %g %g %g, hsml: %g\n", pp, P[pp].Type, P[pp].Pos[0], P[pp].Pos[1], P[pp].Pos[2], P[pp].Hsml);
        }
#endif

        if(size > 0 && tw->Niteration > MAXITER) {
            endrun(1155, "failed to converge density for %ld particles\n", size);
        }
    } while(1);
    ta_free(tw->minnumngb);
    ta_free(tw->maxnumngb);
}


/* find the closest index from radius and numNgb list, update left and right bound, return new hsml */
double
ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize)
{
    int j;
    int close = 0;
    double ngbdist = fabs(numNgb[0] - desnumngb);
    for(j = 1; j < maxcmpt; j++){
        double newdist = fabs(numNgb[j] - desnumngb);
        if(newdist < ngbdist){
            ngbdist = newdist;
            close = j;
        }
    }
    if(closeidx)
        *closeidx = close;

    for(j = 0; j < maxcmpt; j++){
        if(numNgb[j] < desnumngb)
            *left = radius[j];
        if(numNgb[j] > desnumngb){
            *right = radius[j];
            break;
        }
    }

    double hsml = radius[close];

    if(*right > 0.99 * BoxSize){
        double dngbdv = 0;
        if(maxcmpt > 1 && (radius[maxcmpt-1]>radius[maxcmpt-2]))
            dngbdv = (numNgb[maxcmpt-1]-numNgb[maxcmpt-2])/(pow(radius[maxcmpt-1],3) - pow(radius[maxcmpt-2],3));
        /* Increase hsml by a maximum factor to avoid madness. We can be fairly aggressive about this factor.*/
        double newhsml = 4 * hsml;
        if(dngbdv > 0) {
            double dngb = (desnumngb - numNgb[maxcmpt-1]);
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            if(pow(newvolume, 1./3) < newhsml)
                newhsml = pow(newvolume, 1./3);
        }
        hsml = newhsml;
    }
    if(hsml > *right)
        hsml = *right;

    if(*left == 0) {
        /* Extrapolate using volume, ie locally constant density*/
        double dngbdv = 0;
        if(radius[1] > radius[0])
            dngbdv = (numNgb[1] - numNgb[0]) / (pow(radius[1],3) - pow(radius[0],3));
        /* Derivative is not defined for minimum, so use 0.*/
        if(maxcmpt == 1 && radius[0] > 0)
            dngbdv = numNgb[0] / pow(radius[0],3);

        if(dngbdv > 0) {
            double dngb = desnumngb - numNgb[0];
            double newvolume = pow(hsml,3) + dngb / dngbdv;
            hsml = pow(newvolume, 1./3);
        }
    }
    if(hsml < *left)
        hsml = *left;

    return hsml;
}

void
treewalk_print_stats(const TreeWalk * tw)
{
    int64_t NExportTargets;
    int64_t minNinteractions, maxNinteractions, Ninteractions, Nlistprimary, Nexport;
    MPI_Reduce(&tw->minNinteractions, &minNinteractions, 1, MPI_INT64, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->maxNinteractions, &maxNinteractions, 1, MPI_INT64, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->Ninteractions, &Ninteractions, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->WorkSetSize, &Nlistprimary, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->Nexport_sum, &Nexport, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tw->NExportTargets, &NExportTargets, 1, MPI_INT64, MPI_SUM, 0, MPI_COMM_WORLD);
    message(0, "%s Ngblist: min %ld max %ld avg %g average exports: %g avg target ranks: %g\n", tw->ev_label, minNinteractions, maxNinteractions,
            (double) Ninteractions / Nlistprimary, ((double) Nexport)/ tw->NTask, ((double) NExportTargets)/ tw->NTask);
}
