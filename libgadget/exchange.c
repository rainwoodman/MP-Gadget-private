#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "exchange.h"
#include "forcetree.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "walltime.h"
#include "timefac.h"

#include "utils.h"
#include "utils/mpsort.h"

/* New Exchange Algorithm:
 * We send a buffer of particle and slot data in a single point to point MPI request to from each process to each other process.
 * Alltoallv is avoided and MPI_requests are done as soon as the buffer is packed. Free slots are explicitly kept track of so
 * that incoming data can slot directly into a garbage slot.
 * The main loop is non-blocking: it packs up sends and receives and then busy-loops, checking for either sends or receives to be done.
 * If one of them is done then it packs up more sends and receives until none are available.
 *
 * Algorithm:
    - Work out desired send-receive number counts and corresponding bytes.
    - Do alltoall for send-receive counts (this is the only sync point!)
    - Busy-loop while there is data to send or receive or a pending ISend/IRecv.
        - Pack up sends in half the exchange space and ISend. This makes extra space in the particle table.
        - Irecv for receives into the other half.
        - If there is not enough space for all data to a single task (either packing space or free slots in the particle table),
          just do part of a single task. This decision is usually collective because packing space is collective. However, it is not
          important because IRecv will happily accept less data than expected. As long as we are expecting something, it doesn't matter if less is sent.
        - Check the completion of IRecvs and unpack. The send buffer goes directly into a garbage slot for the right particle type, ensuring
          that the particles are still ordered by type.
        - Check the completion of sends.
    This does not block because the sends cannot block the recvs (and vice versa). Also the sends and recvs are ordered:
    sends are done 'forwards' to the task 1 in front of the current task and receives are done 'backwards' to the task 1 before the current task.
    Because each send is paired with a receive no deadlock is possible.

    If we are blocked on a recv, we just do more sends (always allowed as we have separate space allocations).
    If we are blocked on a send, and we have space in the particle table (should always be the case as we made space for the blocked send), we do a recv.
 */


/*Number of structure types for particles*/
typedef struct {
    size_t totalbytes;
    int64_t base;
    int64_t slots[6];
} ExchangePlanEntry;

static MPI_Datatype MPI_TYPE_PLAN_ENTRY = 0;

typedef struct {
    ExchangePlanEntry * toGo;
    ExchangePlanEntry * toGet;
    ExchangePlanEntry toGoSum;
    ExchangePlanEntry toGetSum;
    int NTask;
    int ThisTask;
    /*Number of garbage particles of each type*/
    int64_t ngarbage[6];
    /* List of entries in particle table which are garbage particles of each type*/
    int * garbage_list[6];
    /* Per-task list of particles to send.*/
    int ** target_list;
} ExchangePlan;

/* Status of this exchange iteration.*/
enum ExchangeIterStatus
{
    TASK = 0,/* Normal status where we are working on tasks*/
    SUBTASK = 1, /* We didn't have enough space for a whole task, so we are just sending small amounts of particles.*/
    LASTSUBTASK = 2,/* This is the last set of particles in this task!*/
    DONE = 3, /*We have finished all transfers*/
};

/* Structure to store the tasks that need to be exchanged this iteration*/
struct ExchangeIter
{
    /* Start and end of the tasks for this iteration. Sends start from the current task and move forwards.
     * Recvs start from before the current task and move backwards.*/
    int StartTask;
    /* Note that this task should not be sent so the condition is < for send and > for recv.*/
    int EndTask;
    enum ExchangeIterStatus estat;
    /* If we only have one task this iteration, which particles do we send/recv?*/
    int64_t StartPart;
    int64_t EndPart;
    /* Total transfer this time*/
    size_t transferbytes;
};

struct CombinedBuffer
{
    struct CommBuffer Sends;
    struct CommBuffer Recvs;
    MPI_Request * rdata_all; /* Array of requests*/
};

/* Allocate/free a commbuffer with space for N tasks.*/
void alloc_comms(struct CommBuffer * buffer, int NTask)
{
    buffer->rqst_task = ta_malloc("rqst", int, NTask);
    buffer->displs = ta_malloc("displs", int, NTask);
    buffer->nrequest_all = 0;
    buffer->totcomplete = 0;
    buffer->databuf = NULL;
}

void free_comms(struct CommBuffer * buffer)
{
    if(buffer->databuf) {
        myfree(buffer->databuf);
        buffer->databuf = NULL;
    }
    if(buffer->displs) {
        ta_free(buffer->displs);
        ta_free(buffer->rqst_task);
    }
}

void alloc_combinedbuffer(struct CombinedBuffer *all, int NTask)
{
    alloc_comms(&all->Sends, NTask);
    alloc_comms(&all->Recvs, NTask);
    all->rdata_all = ta_malloc("requests", MPI_Request, 2*NTask);
    /* Initialise requests as NULL so we can safely wait on them without posting anything*/
    int i;
    for(i = 0; i < 2 * NTask; i++)
        all->rdata_all[i] = MPI_REQUEST_NULL;
    all->Recvs.rdata_all = all->rdata_all;
    all->Sends.rdata_all = all->rdata_all + NTask;
}

void free_combinedbuffer(struct CombinedBuffer *all)
{
    ta_free(all->rdata_all);
    free_comms(&all->Recvs);
    free_comms(&all->Sends);
}

/*
 * exchange particles according to layoutfunc.
 * layoutfunc gives the target task of particle p.
*/
static int domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, const size_t maxexch, MPI_Comm Comm);
static int domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);
static void domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm);

static ExchangePlan
domain_init_exchangeplan(MPI_Comm Comm)
{
    ExchangePlan plan = {0};
    MPI_Comm_size(Comm, &plan.NTask);
    MPI_Comm_rank(Comm, &plan.ThisTask);
    /*! toGo[0][task*NTask + partner] gives the number of particles in task 'task'
     *  that have to go to task 'partner'
     *  toGo[1] is SPH, toGo[2] is BH and toGo[3] is stars
     */
    plan.toGo = ta_malloc("toGo", ExchangePlanEntry, plan.NTask);
    plan.toGet = ta_malloc("toGet", ExchangePlanEntry, plan.NTask);
    return plan;
}

static void
domain_free_exchangeplan(ExchangePlan * plan)
{
    /* Free the lists.*/
    int n;
    if(plan->target_list) {
        for(n = plan->NTask -1; n >=0; n--) {
            if(plan->target_list[n])
                myfree(plan->target_list[n]);
        }
        myfree(plan->target_list);
        for(n = 5; n >=0; n--) {
            if(plan->garbage_list[n])
                myfree(plan->garbage_list[n]);
        }
    }
    myfree(plan->toGet);
    myfree(plan->toGo);
}

/* We want to avoid doing an alltoall with
    * more than 2GB of material as this hangs.*/
static size_t MaxExch = 2040L*1024L*1024L;
/* For tests*/
void
domain_set_max_exchange(const size_t maxexch)
{
    MaxExch = maxexch;
}

/* Init the exchange iteration. Note that for sends offset is +1 for recvs it is -1*/
void
domain_init_exchange_iter(struct ExchangeIter * iter, const int offset, const ExchangePlan * const plan)
{
    /* + plan->NTask ensures we get a positive number for offset = -1*/
    iter->StartTask = (plan->ThisTask + offset + plan->NTask) % plan->NTask;
    iter->EndTask = (plan->ThisTask + offset + plan->NTask) % plan->NTask;
    iter->StartPart = 0;
    iter->EndPart = 0;
}

/* Find the amount of space available for the domain exchange.
 * Collective so we can make the same decisions on all ranks.*/
size_t
domain_get_exchange_space(const int NTask, MPI_Comm Comm)
{
    size_t nlimit = mymalloc_freebytes();
    /* Save some memory for memory headers and wasted space at the end of each allocation.
     * Need max. 2*4096 for each heap-allocated array, max 1 send, 1 recv.*/
    nlimit -= 4096 * 4L * NTask;
    if(nlimit <= 0)
        endrun(1, "Not enough memory free to store requests!\n");

    if(nlimit > MaxExch)
        nlimit = MaxExch;
    MPI_Allreduce(MPI_IN_PLACE, &nlimit, 1, MPI_UINT64, MPI_MIN, Comm);
    return nlimit;
}

/*Plan and execute a domain exchange, also performing a garbage collection if requested*/
int domain_exchange(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preexch, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm) {
    /* register the MPI types used in communication if not yet. */
    if (MPI_TYPE_PLAN_ENTRY == 0) {
        MPI_Type_contiguous(sizeof(ExchangePlanEntry), MPI_BYTE, &MPI_TYPE_PLAN_ENTRY);
        MPI_Type_commit(&MPI_TYPE_PLAN_ENTRY);
    }
    PreExchangeList preplan = {0};
    /* Use the pre-exchange list if we can*/
    if(!preexch || !preexch->ExchangeList) {
        preexch = &preplan;
        domain_build_exchange_list(layoutfunc, layout_userdata, preexch, pman, sman, Comm);
        walltime_measure("/Domain/exchange/exchangelist");
    }

    /*Structure for building a list of particles that will be exchanged*/
    ExchangePlan plan = domain_init_exchangeplan(Comm);
    int failure = domain_build_plan(layoutfunc, layout_userdata, &plan, preexch, pman, sman, Comm);
    /* Done with the pre-exchange list*/
    myfree(preexch->ExchangeList);
    walltime_measure("/Domain/exchange/plan");
    /* Do this after domain_build_plan so the target lists are already allocated*/
    size_t maxexch = domain_get_exchange_space(plan.NTask, Comm);
    /* Now to do an exchange if it will succeed*/
    if(!failure)
        failure = domain_exchange_once(&plan, pman, sman, 123000, maxexch, Comm);
    domain_free_exchangeplan(&plan);

#ifdef DEBUG
    domain_test_id_uniqueness(pman, Comm);
    slots_check_id_consistency(pman, sman);

    if(!failure) {
        PreExchangeList plan9 = {0};
        domain_build_exchange_list(layoutfunc, layout_userdata, &plan9, pman, sman, Comm);
        if(plan9.nexchange - plan9.ngarbage > 0)
            endrun(5, "Still have %ld particles in exchange list\n", plan9.nexchange - plan9.ngarbage);
        myfree(plan9.ExchangeList);
    }
    walltime_measure("/Domain/exchange/finalize");
#endif
    return failure;
}

/* Move some particle and slot data into an exchange buffer for sending.
   The format is:
   (base particle data 1 - n)
   (slot data for particle i: may be variable size as not necessarily ordered by type)
    Returns: size of buffer packed. Writes first not-packed particle to endpart
   */
static size_t
exchange_pack_buffer(char * exch, const int task, const size_t StartPart, ExchangePlan * const plan, struct part_manager_type * pman, struct slots_manager_type * sman, const size_t maxsendexch, int64_t * endpart)
{
    char * exchptr = exch;
    int64_t n;
    for(n = StartPart; n < plan->toGo[task].base; n++)
    {
        if(!plan->target_list[task])
            endrun(4, "target list not allocated for task %d with %ld togo", task, plan->toGo[task].base);
        const int i = plan->target_list[task][n];
        /* preparing for export */
        const int type = pman->Base[i].Type;
        size_t elsize = 0;
        if(sman->info[type].enabled)
             elsize = sman->info[type].elsize;
        if(exchptr - exch + sizeof(struct particle_data) + elsize > maxsendexch) {
            break;
        }
        memcpy(exchptr, pman->Base+i, sizeof(struct particle_data));
        exchptr += sizeof(struct particle_data);
        if(sman->info[type].enabled) {
            memcpy(exchptr,(char*) sman->info[type].ptr + pman->Base[i].PI * elsize, elsize);
        }
        exchptr += elsize;
        /* Add this particle to the garbage list so we can unpack something else into it.*/
        int64_t gslot = plan->ngarbage[type];
        plan->garbage_list[type][gslot] = i;
        plan->ngarbage[type]++;
        /* mark the particle for removal. Both secondary and base slots will be marked. */
        slots_mark_garbage(i, pman, sman);
    }
    // message(4, "after pack for task %d ngarbage %ld %ld %ld\n", task, plan->ngarbage[0], plan->ngarbage[1], plan->ngarbage[2]);
    *endpart = n;
    return exchptr - exch;
}

/* Take a received buffer and move the particle data back into the particle table.
 * If possible a garbage particle of the same type will be used for the new memory,
 * This routine is not parallel.
*/
static size_t
exchange_unpack_buffer(char * exch, int task, ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, const int64_t recvd_bytes)
{
    char * exchptr = exch;
    int64_t copybase = 0;
    int64_t i;
    for(i = 0; i < plan->toGet[task].base; i++)
    {
        /* Extract type*/
        const unsigned int type = ((struct particle_data *) exchptr)->Type;
        int PI = sman->info[type].size;
        int64_t elsize = 0;
        if(sman->info[type].enabled)
            elsize = sman->info[type].elsize;
        /* Stop if we have processed all incoming bytes*/
        if(exchptr + sizeof(struct particle_data) + elsize - exch > recvd_bytes)
            break;
        /* Find a garbage place in the particle table*/
        int64_t dest = pman->MaxPart + 2;
        /* Garbage particle of the same type is best*/
        if(plan->ngarbage[type] >= 1) {
            dest = plan->garbage_list[type][plan->ngarbage[type]-1];
            /* Copy PI so it is not over-written*/
            PI = pman->Base[dest].PI;
            /* No longer garbage!*/
            plan->ngarbage[type]--;
        }
        else {
            /* Just copy to the end of the table, incrementing the counter*/
            if(pman->NumPart+1 < pman->MaxPart) {
                dest = pman->NumPart;
                pman->NumPart++;
            }
            /* Try to find a garbage particle of a different type.
             * This is the last resort because it means the slot will not be aligned with the particle.*/
            else {
                int tt;
                for(tt = 0; tt < 6; tt++)
                {
                    if(plan->ngarbage[tt] >= 1) {
                        dest = plan->garbage_list[tt][plan->ngarbage[tt]-1];
                        /* No longer garbage!*/
                        plan->ngarbage[tt]--;
                        break;
                    }
                }
            }
        }
        if(dest >= pman->MaxPart)
            endrun(6, "Not enough room for particle %ld of type %d from task %d (need %ld). Garbage is 0=%ld 1=%ld 4=%ld 5=%ld\n", i, type, task, plan->toGet[task].base, plan->ngarbage[0], plan->ngarbage[1], plan->ngarbage[4], plan->ngarbage[5]);
        memcpy(pman->Base+dest, exchptr, sizeof(struct particle_data));

        exchptr += sizeof(struct particle_data);
        copybase++;
        /* Copy the slot if needed*/
        if(sman->info[type].enabled) {
            /* Enforce that we have enough slots*/
            if(PI == sman->info[type].size) {
                sman->info[type].size++;
                if(sman->info[type].size >= sman->info[type].maxsize)
                    endrun(6, "Not enough room for slot %d after exchange\n",type);
            }
            memcpy((char*) sman->info[type].ptr + PI * elsize, exchptr, elsize);
            exchptr += elsize;
            /* Update the PI to be correct*/
            pman->Base[dest].PI = PI;
#ifdef DEBUG
            if(BASESLOT_PI(PI, type, sman)->ID != pman->Base[dest].ID) {
                endrun(1, "Exchange: P[%ld].ID = %ld (type %d) != SLOT ID = %ld. garbage: %d ReverseLink: %d\n",dest,pman->Base[dest].ID, pman->Base[dest].Type, BASESLOT_PI(PI, type, sman)->ID, pman->Base[dest].IsGarbage, BASESLOT_PI(PI, type, sman)->ReverseLink);
            }
#endif
        }
    }
    // message(4, "after unpack ngarbage %ld %ld %ld\n", plan->ngarbage[0], plan->ngarbage[1], plan->ngarbage[2]);
    return copybase;
}


/*Find how many tasks we can send in current exchange iteration.
 If the current task does not fit in the buffer, work out how many particles can be sent. */
static void
domain_find_send_iter(ExchangePlan * plan, struct ExchangeIter * senditer,  int64_t * expected_freeslots, const size_t maxsendexch)
{
    /* Last loop was a subtask*/
    if(senditer->estat == SUBTASK) {
        senditer->StartPart = senditer->EndPart;
        // message(5, "SendIter fastreturn: startpart is now %ld end %ld togo %ld\n", senditer->StartPart, senditer->EndPart, plan->toGo[senditer->StartTask].base);
        return;
    }
    senditer->StartTask = senditer->EndTask;
    if(senditer->StartTask == plan->ThisTask) {
        senditer->estat = DONE;
        return;
    }
    /* Total bytes needed to send*/
    senditer->transferbytes = 0;
    /* Note LASTSUBTASK will not hit the conditions above and will just hit here*/
    senditer->estat = TASK;
    int n;
    for(n = 0 ; n < 6; n++)
        expected_freeslots[n] = plan->ngarbage[n];
    /* First find some data to send. This gives us space in the particle table.*/
    int task;
    for(task = 0; task < plan->NTask; task++) {
        const int sendtask = (senditer->StartTask + task) % plan->NTask;
        senditer->EndTask = sendtask;
        if(senditer->transferbytes + plan->toGo[sendtask].totalbytes > maxsendexch || sendtask == plan->ThisTask)
            break;
        senditer->transferbytes += plan->toGo[sendtask].totalbytes;
        // message(1, "togo %ld tot %ld send %d\n", plan->toGo[sendtask].totalbytes, senditer->transferbytes, sendtask);
        for(n = 0; n < 6; n++)
            expected_freeslots[n] += plan->toGo[sendtask].slots[n];
    }
    // message(1, "Using %ld bytes to send from %d to %d\n", senditer->transferbytes, senditer->StartTask, senditer->EndTask);
    if(senditer->StartTask == senditer->EndTask) {
        senditer->estat = SUBTASK;
        senditer->transferbytes = maxsendexch;
        /* End task is the next one*/
        senditer->EndTask = (senditer->StartTask + 1) % plan->NTask;
        /* Start at 0 particle: endpart is set at send time.*/
        senditer->StartPart = 0;
    }
    return;
}

/*Find how many tasks we can transfer in current exchange iteration.
 If the current task does not fit in the buffer, work out how many */
static void
domain_find_recv_iter(ExchangePlan * plan, struct ExchangeIter * recviter, int64_t freepart, int64_t * expected_freeslots, const size_t maxrecvexch)
{
    /* Last loop was a subtask*/
    if(recviter->estat == SUBTASK) {
        recviter->StartPart = recviter->EndPart;
        return;
    }
    recviter->StartTask = recviter->EndTask;
    if(recviter->StartTask == plan->ThisTask) {
        recviter->estat = DONE;
        return;
    }

    recviter->transferbytes = 0;
    recviter->estat = TASK;
    int task;
    for(task = 0; task < plan->NTask; task++) {
        const int recvtask = (recviter->StartTask - task + plan->NTask) % plan->NTask;
        recviter->EndTask = recvtask;
        /*This checks we have enough space in the particle table for each slot*/
        int n;
        for(n = 0; n < 6; n++) {
            expected_freeslots[n] -= plan->toGet[recvtask].slots[n];
            /* If we overflow the slots available in garbage, we use the global particle table.*/
            if(expected_freeslots[n] < 0) {
                freepart += expected_freeslots[n];
                expected_freeslots[n] = 0;
            }
        }
        /* We do not have enough slots here to continue!
         * This logic means that we may not receive all the things sent to us this iteration,
         * as different ranks may have different particle loads.*/
        if(freepart < 0)
            break;
        if(recviter->transferbytes + plan->toGet[recvtask].totalbytes > maxrecvexch || recvtask == plan->ThisTask)
            break;
        recviter->transferbytes += plan->toGet[recvtask].totalbytes;
        // message(1, "toget %ld tot %ld recv %d\n", plan->toGet[recvtask].totalbytes, recviter->transferbytes, recvtask);
    }
    // message(1, "Using %ld bytes to recv from %d to %d\n", recviter->transferbytes, recviter->StartTask, recviter->EndTask);
    if(recviter->StartTask == recviter->EndTask) {
        recviter->estat = SUBTASK;
        recviter->transferbytes = maxrecvexch;
        recviter->EndTask = (recviter->StartTask - 1) % plan->NTask;
        /* Start at 0 particle: endpart is set at send time.*/
        recviter->StartPart = 0;
    }
    return;
}

/* Post a single receive for the current iteration task when there is not enough buffer space for one task. */
static void
domain_post_single_recv(struct CommBuffer * recvs, struct ExchangeIter *thisiter, int tag, MPI_Comm Comm)
{
    /* Note that IRecv will happily accept less bytes than are requested.*/
    MPI_Irecv(recvs->databuf, thisiter->transferbytes, MPI_BYTE, thisiter->StartTask, tag, Comm, recvs->rdata_all);
    // message(1, "Partial recv task %d bytes %ld, startpart %lu\n", thisiter->StartTask, thisiter->transferbytes, thisiter->StartPart);

    recvs->rqst_task[0] = thisiter->StartTask;
    recvs->displs[0] = 0;
    recvs->nrequest_all=1;
    recvs->totcomplete = 0;
    return;
}

/* Post the receives for the current iteration, and fill up a CommBuffer with Irecvs.
 * Note that receives are posted in reverse order! First the task before this one, then all tasks before that, backwards.
 * The sends are posted forwards, so that all send/recv pairs are ordered.*/
static void
domain_post_recvs(ExchangePlan * plan, struct CommBuffer * recvs, struct ExchangeIter *recviter, int tag, MPI_Comm Comm)
{
    /* First post receives*/
    size_t displs = 0;
    recvs->nrequest_all = 0;
    recvs->totcomplete = 0;
    int task;
    for(task=0; task < plan->NTask; task++) {
        const int recvtask = (recviter->StartTask - task + plan->NTask) % plan->NTask;
        if(recvtask == recviter->EndTask)
            break;
        /* Skip zero-size receives*/
        if(plan->toGet[recvtask].totalbytes == 0)
            continue;
        MPI_Irecv(recvs->databuf + displs, plan->toGet[recvtask].totalbytes, MPI_BYTE, recvtask, tag, Comm, &recvs->rdata_all[recvs->nrequest_all]);
        recvs->rqst_task[recvs->nrequest_all] = recvtask;
        recvs->displs[recvs->nrequest_all] = displs;
        displs += plan->toGet[recvtask].totalbytes;
        if(displs >= recvs->databufsize)
            endrun(4, "Recv buffer overflow with %ld bytes > %ld allocated for task %d end %d", displs, recvs->databufsize, task, recviter->EndTask);
        // message(1, "Receive task %d toget %ld tot %ld\n", recvtask, plan->toGet[recvtask].totalbytes, recviter->transferbytes);
        recvs->nrequest_all ++;
    }
    if(displs != recviter->transferbytes)
        endrun(3, "Posted receives for %lu bytes expected %lu bytes.\n", displs, recviter->transferbytes);
    return;
}

/* Pack a single send buffer when we don't have space for all the particles on one task.*/
static void
domain_pack_single_send(ExchangePlan * plan, struct CommBuffer * sends, struct ExchangeIter *senditer, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, MPI_Comm Comm)
{
    /* Move the data into the buffer*/
    size_t packed_bytes = exchange_pack_buffer(sends->databuf, senditer->StartTask, senditer->StartPart, plan, pman, sman, senditer->transferbytes, &senditer->EndPart);
    MPI_Isend(sends->databuf, packed_bytes, MPI_BYTE, senditer->StartTask, tag, Comm, sends->rdata_all);
    // message(1, "Partial send task %d bytes %ld, startpart %lu endpart %lu total %ld\n", senditer->StartTask, senditer->transferbytes, senditer->StartPart, senditer->EndPart, plan->toGo[senditer->StartTask].base);
    sends->rqst_task[0] = senditer->StartTask;
    sends->displs[0] = 0;
    sends->nrequest_all = 1;
    sends->totcomplete = 0;
    return;
}

/* Pack the send buffers for each task and issue ISend requests for them*/
static void
domain_pack_sends(ExchangePlan * plan, struct CommBuffer * sends, struct ExchangeIter *senditer, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, MPI_Comm Comm)
{
    size_t displs = 0;
    sends->nrequest_all = 0;
    sends->totcomplete = 0;
    int task;
    for(task=0; task < plan->NTask; task++) {
        /* Move the data into the buffer*/
        const int sendtask = (senditer->StartTask + task) % plan->NTask;
        if(sendtask == senditer->EndTask)
            break;
        /* Skip zero-size sends*/
        if(plan->toGo[sendtask].totalbytes == 0)
            continue;
        /* The openmp parallel is done inside exchange_pack_buffer so that we can issue MPI_Isend as soon as possible*/
        exchange_pack_buffer(sends->databuf + displs, sendtask, 0, plan, pman, sman, senditer->transferbytes, &senditer->EndPart);
        if(senditer->EndPart < plan->toGo[sendtask].base)
            endrun(4, "Expected %ld particles but only packed %lu\n", plan->toGo[sendtask].base, senditer->EndPart);
        MPI_Isend(sends->databuf + displs, plan->toGo[sendtask].totalbytes, MPI_BYTE, sendtask, tag, Comm, &sends->rdata_all[sends->nrequest_all]);
        sends->rqst_task[sends->nrequest_all] = sendtask;
        sends->displs[sends->nrequest_all] = displs;
        displs += plan->toGo[sendtask].totalbytes;
        if(displs >= sends->databufsize)
            endrun(4, "Send buffer overflow with %ld bytes > %ld allocated for task %d end %d", displs, sends->databufsize, task, senditer->EndTask);
        sends->nrequest_all ++;
    }
    // message(3, "Packed sends for task %d to %d nrequest %d\n", senditer->StartTask, senditer->EndTask, sends->nrequest_all);
    if(displs != senditer->transferbytes)
        endrun(3, "Packed %lu bytes for sending but expected %lu bytes.\n", displs, senditer->transferbytes);
    return;
}

/* Wait for some sends and receives. Unpack received data into a buffer.
 * We busy-loop until either sends or receives are done and then we try to get more.*/
static size_t
domain_check_unpack(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, struct CombinedBuffer * all)
{
    /* Now wait for and unpack the receives as they arrive.*/
    int flag = 0;
    size_t recvd = 0;
    int * complete_array = ta_malloc("completes", int, 2 * plan->NTask);
    MPI_Status * stats = ta_malloc("stats", MPI_Status, 2 * plan->NTask);;
    // message(3, "reqs: %d\n", all->Recvs.nrequest_all + all->Sends.nrequest_all);
    /* We wait in a loop until we have either all the sends or all the recvs. Once we have those we can post more.*/
    while(all->Recvs.totcomplete < all->Recvs.nrequest_all || all->Sends.totcomplete < all->Sends.nrequest_all) {
        int recvd_bytes = 0;
        int complete_cnt = MPI_UNDEFINED;
        /* Check for some completed requests: note that cleanup is performed if the requests are complete.
         * There may be only 1 completed request, and we need to wait again until we have more.*/
        MPI_Waitsome(plan->NTask + all->Sends.nrequest_all, all->rdata_all, &complete_cnt, complete_array, stats);
        /* This happens if all requests are MPI_REQUEST_NULL. It should never be hit*/
        if (complete_cnt == MPI_UNDEFINED)
            break;
        int j;
        for(j = 0; j < complete_cnt; j++) {
            const int i = complete_array[j];
            /* This is a Send and we don't need to do anything. Cleanup is done by MPI in Waitsome.*/
            if(i >= plan->NTask) {
                all->Sends.totcomplete++;
                continue;
            }
            /* Note the index here is j because it is in the array of completed requests not the array of requests!*/
            MPI_Get_count(&stats[j], MPI_BYTE, &recvd_bytes);
            if(i >= all->Recvs.nrequest_all)
                endrun(5, "Bad task %d nreq %d rdata_all %p\n", i, all->Recvs.nrequest_all, all->Recvs.rdata_all);
            if(recvd_bytes == 0)
                endrun(4, "Received zero bytes, should not happen! flag %d rqst %d complete %d nrequest %d\n", flag, i, all->Recvs.totcomplete, all->Recvs.nrequest_all);
            /* task number index is not the index in the request array. The Recvs come first, then the Sends. This lets us avoid an offset to the index*/
            recvd += exchange_unpack_buffer(all->Recvs.databuf+all->Recvs.displs[i], all->Recvs.rqst_task[i], plan, pman, sman, recvd_bytes);
            all->Recvs.totcomplete++;
        }
        /* We have completed either all the Sends or all the Recvs, see if we can post more*/
        if(all->Recvs.totcomplete == all->Recvs.nrequest_all || all->Sends.totcomplete == all->Sends.nrequest_all)
            break;
    };
    ta_free(stats);
    ta_free(complete_array);
    return recvd;
}

static int
domain_exchange_once(ExchangePlan * plan, struct part_manager_type * pman, struct slots_manager_type * sman, int tag, const size_t maxexch, MPI_Comm Comm)
{
    double tpack = 0, tunpack = 0;
    /* Flags when any pending sends or recvs are finished, so we can do more*/
    int no_sends_pending = 1, no_recvs_pending = 1;
    /* How many slots will we have available for new particles due to sends?*/
    int64_t expected_freeslots[6];
    memcpy(expected_freeslots, plan->ngarbage, 6 * sizeof(plan->ngarbage[0]));
    /* determine for each rank how many particles have to be shifted to other ranks */
    struct ExchangeIter senditer = {0}, recviter = {0};
    /* CommBuffers. init zero ensures no requests stored*/
    struct CombinedBuffer all = {0};
    alloc_combinedbuffer(&all, plan->NTask);

    domain_init_exchange_iter(&senditer, 1, plan);
    domain_init_exchange_iter(&recviter, -1, plan);
    walltime_measure("/Domain/exchange/misc");

    /* Loop. There is no blocking inside this loop. We keep track of the completion status
     * of the send and receives so we can do more the moment a buffer is free*/
    do {
        if(no_sends_pending && senditer.estat != DONE)
            domain_find_send_iter(plan, &senditer,  expected_freeslots, maxexch/2);
        if(no_recvs_pending && recviter.estat != DONE) {
            /* No recvs are pending, try to get more*/
            domain_find_recv_iter(plan, &recviter, pman->MaxPart - pman->NumPart, expected_freeslots, maxexch/2);
            /* Need check in case receives finished but still sends to do*/
            if(recviter.estat != DONE) {
                all.Recvs.databufsize = recviter.transferbytes+sizeof(struct particle_data);
                /* Free any earlier buffer. If not called will be freed in free_combinedbuffer after loop end.*/
                if(all.Recvs.databuf)
                    myfree(all.Recvs.databuf);
                all.Recvs.databuf = mymalloc("recvbuffer",all.Recvs.databufsize * sizeof(char));
                /* Receiving less than one task!*/
                if(recviter.estat == SUBTASK) {
                    domain_post_single_recv(&all.Recvs, &recviter, tag, Comm);
                } else if(recviter.estat == TASK)
                    domain_post_recvs(plan, &all.Recvs, &recviter, tag, Comm);
                if(all.Recvs.nrequest_all > 0)
                    no_recvs_pending = 0;
            }
        }
        /* Now post sends: note that the sends are done in reverse order to the receives.
         * This ensures that partial sends and receives can complete early.*/
        if(no_sends_pending && senditer.estat != DONE) {
            all.Sends.databufsize = senditer.transferbytes+sizeof(struct particle_data);
            if(all.Sends.databuf)
                myfree(all.Sends.databuf);
            all.Sends.databuf = mymalloc2("sendbuffer",all.Sends.databufsize * sizeof(char));
            double tstart = second();
            if(senditer.estat == SUBTASK) {
                domain_pack_single_send(plan, &all.Sends, &senditer, pman, sman, tag, Comm);
                if(senditer.EndPart == plan->toGo[senditer.StartTask].base)
                    senditer.estat = LASTSUBTASK;
            } else if(senditer.estat == TASK)
                domain_pack_sends(plan, &all.Sends, &senditer, pman, sman, tag, Comm);
            if(all.Sends.nrequest_all > 0)
                no_sends_pending = 0;
            double tend = second();
            tpack += timediff(tstart, tend);
        }
        /* Now wait for and unpack the receives as they arrive.*/
        if(!no_recvs_pending || !no_sends_pending) {
            double tstart = second();
            size_t recvd = domain_check_unpack(plan, pman, sman, &all);
            if(!no_recvs_pending && all.Recvs.totcomplete == all.Recvs.nrequest_all ) {
                no_recvs_pending = 1;
                if(recviter.estat == SUBTASK) {
                    recviter.EndPart = recviter.StartPart + recvd;
                    // message(2, "Done Partial Received %ld task %d sp %ld ep %ld end task %d\n", recvd, recviter.StartTask, recviter.StartPart, recviter.EndPart, recviter.EndTask);
                    /* Check whether this was the final part of a task, in which case move to next task.*/
                    if(recviter.EndPart == plan->toGet[recviter.StartTask].base) {
                        recviter.estat = LASTSUBTASK;
                        // message(2, "Finished recv task %d sp %ld ep %ld end task %d\n", recviter.StartTask, recviter.StartPart, recviter.EndPart, recviter.EndTask);
                    }
                    if(recviter.EndPart > plan->toGet[recviter.StartTask].base)
                        endrun(5, "Received %ld particles from task %d more than %ld expected\n", recviter.EndPart, recviter.StartTask, plan->toGet[recviter.StartTask].base);
                }
            }
            /* Done with sends, let's get more! */
            if(!no_sends_pending && all.Sends.totcomplete == all.Sends.nrequest_all) {
                no_sends_pending = 1;
                // message(2, "Finished send task %d sp %ld ep %ld end task %d\n", senditer.StartTask, senditer.StartPart, senditer.EndPart, senditer.EndTask);
            }
            double tend = second();
            tunpack += timediff(tstart, tend);
        }
    } while(!no_sends_pending || !no_recvs_pending || recviter.estat != DONE || senditer.estat != DONE);

    free_combinedbuffer(&all);
    double timeall = walltime_measure(WALLTIME_IGNORE);
    walltime_add("/Domain/exchange/pack", tpack);
    walltime_add("/Domain/exchange/unpack", tunpack);
    walltime_add("/Domain/exchange/wait", timeall - tpack - tunpack);
    return 0;
}

/* This function builds the list of particles to be exchanged.
 * All particles are processed every time, space is not considered.
 * The exchange list needs to be rebuilt every time gc is run. */
static void
domain_build_exchange_list(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
{
    /*Garbage particles are counted so we have an accurate memory estimate*/
    int ngarbage = 0;
    gadget_thread_arrays gthread = gadget_setup_thread_arrays("exchangelist", 1, pman->NumPart);
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    /* flag the particles that need to be exported */
    #pragma omp parallel
    {
        int i;
        size_t nexthr_local = 0;
        const int tid = omp_get_thread_num();
        int * threx_local = gthread.srcs[tid];
        #pragma omp for schedule(static, gthread.schedsz) reduction(+: ngarbage)
        for(i=0; i < pman->NumPart; i++)
        {
            if(pman->Base[i].IsGarbage) {
                ngarbage++;
                threx_local[nexthr_local] = i;
                nexthr_local++;
                continue;
            }
            int target = layoutfunc(i, layout_userdata);
            if(target != ThisTask) {
                threx_local[nexthr_local] = i;
                nexthr_local++;
            }
        }
        gthread.sizes[tid] = nexthr_local;
    }
    preplan->ngarbage = ngarbage;
    /*Merge step for the queue.*/
    preplan->nexchange = gadget_compact_thread_arrays(&preplan->ExchangeList, &gthread);
    /*Shrink memory*/
    preplan->ExchangeList = (int *) myrealloc(preplan->ExchangeList, sizeof(int) * preplan->nexchange);
}

/*This function populates the various lists for the domain plan.
 *These are: toGo and toGet arrays (counts).
 * target_list arrays (positions of particles to send).
 * garbage_list array (positions of garbage particles by type).*/
static int
domain_build_plan(ExchangeLayoutFunc layoutfunc, const void * layout_userdata, ExchangePlan * plan, PreExchangeList * preplan, struct part_manager_type * pman, struct slots_manager_type * sman, MPI_Comm Comm)
{
    int ptype;
    size_t n;

    memset(plan->toGo, 0, sizeof(plan->toGo[0]) * plan->NTask);

    int * layouts = (int *) mymalloc2("layoutcache",sizeof(int) * preplan->nexchange);
    int * tmp_garbage_list = mymalloc2("tmp_garbage",sizeof(int)* 6 * preplan->ngarbage);
    ExchangePlanEntry * toGoThread = (ExchangePlanEntry *) mymalloc2("toGoThread",sizeof(ExchangePlanEntry) * plan->NTask* omp_get_max_threads());
    memset(toGoThread, 0, sizeof(toGoThread[0]) * plan->NTask * omp_get_max_threads());
    memset(plan->ngarbage, 0, 6*sizeof(plan->ngarbage[0]));

    /* Compute exchange particle counts for each process*/
    #pragma omp parallel for
    for(n = 0; n < preplan->nexchange; n++)
    {
        const int tid = omp_get_thread_num();
        const int i = preplan->ExchangeList[n];
        const int ptype = pman->Base[i].Type;
        /* Add this to the proto-garbage list*/
        if(pman->Base[i].IsGarbage) {
            int gslot = atomic_fetch_and_add_64(&plan->ngarbage[ptype], 1);
            tmp_garbage_list[ptype * preplan->ngarbage + gslot] = i;
            layouts[n] = plan->ThisTask;
            continue;
        }
        const int target = layoutfunc(i, layout_userdata);
        if(target >= plan->NTask || target < 0)
            endrun(4, "layoutfunc for %d returned unreasonable %d for %d tasks\n", i, target, plan->NTask);
        toGoThread[tid * plan->NTask + target ].base++;
        toGoThread[tid * plan->NTask + target].slots[ptype]++;
        /* Compute total size being sent on this process*/
        toGoThread[tid * plan->NTask + target].totalbytes += sizeof(struct particle_data);
        if(sman->info[ptype].enabled){
            toGoThread[tid * plan->NTask + target].totalbytes += sman->info[ptype].elsize;
        }
        layouts[n] = target;
    }

    // message(4, "ngarbage %ld %ld %ld\n", plan->ngarbage[0], plan->ngarbage[1], plan->ngarbage[2]);

    /*Do the sum*/
    int tid;
    for(tid = 0; tid < omp_get_max_threads(); tid++)
    {
        int target;
        #pragma omp parallel for
        for(target = 0; target < plan->NTask; target++) {
            plan->toGo[target].base += toGoThread[tid * plan->NTask + target].base;
            int i;
            for(i = 0; i < 6; i++)
                plan->toGo[target].slots[i] += toGoThread[tid * plan->NTask + target].slots[i];
            plan->toGo[target].totalbytes += toGoThread[tid * plan->NTask + target].totalbytes;
        }
    }
    myfree(toGoThread);

    /* Get the send counts from every processor*/
    MPI_Alltoall(plan->toGo, 1, MPI_TYPE_PLAN_ENTRY, plan->toGet, 1, MPI_TYPE_PLAN_ENTRY, Comm);
    memcpy(&plan->toGoSum, &plan->toGo[0], sizeof(plan->toGoSum));
    memcpy(&plan->toGetSum, &plan->toGet[0], sizeof(plan->toGetSum));

    int rank;
    int64_t maxbasetoget=-1, maxbasetogo=-1;
    for(rank = 1; rank < plan->NTask; rank ++) {
        /* Direct assignment breaks compilers like icc */
        plan->toGoSum.base += plan->toGo[rank].base;
        plan->toGetSum.base += plan->toGet[rank].base;
        if(plan->toGo[rank].base > maxbasetogo)
            maxbasetogo = plan->toGo[rank].base;
        if(plan->toGet[rank].base > maxbasetoget)
            maxbasetoget = plan->toGet[rank].base;
        for(ptype = 0; ptype < 6; ptype++) {
            plan->toGoSum.slots[ptype] += plan->toGo[rank].slots[ptype];
            plan->toGetSum.slots[ptype] += plan->toGet[rank].slots[ptype];
        }
    }

    // message(1, "Particles togo: %ld, toget %ld\n", plan->toGoSum.base, plan->toGetSum.base);
    int64_t maxbasetogomax, maxbasetogetmax, sumtogo;
    MPI_Reduce(&maxbasetogo, &maxbasetogomax, 1, MPI_INT64, MPI_MAX, 0, Comm);
    MPI_Reduce(&maxbasetoget, &maxbasetogetmax, 1, MPI_INT64, MPI_MAX, 0, Comm);
    MPI_Reduce(&plan->toGoSum.base, &sumtogo, 1, MPI_INT64, MPI_SUM, 0, Comm);
    message(0, "Total particles in flight: %ld Largest togo: %ld, toget %ld\n", sumtogo, maxbasetogomax, maxbasetogetmax);

    int64_t finalNumPart = pman->NumPart;
    /* finalNumPart calculation is not completely optimal but reflects the current algorithm.
     * The achievable balance could be a little better as we can make extra slots without extra particles,
     * but then the extra particles would not be in type order.*/
    int64_t slots_needed[6] = {0};
    int need_slot_reserve = 0;
    for(n = 0; n < 6; n++) {
        finalNumPart += plan->toGetSum.slots[n]- plan->toGoSum.slots[n]- plan->ngarbage[n];
        if(sman->info[n].enabled) {
            slots_needed[n] = sman->info[n].size + plan->toGetSum.slots[n]- plan->toGoSum.slots[n]- plan->ngarbage[n];
            if(slots_needed[n] >= sman->info[n].maxsize) {
                need_slot_reserve = 1;
            }
        }
    }
    /* Make sure we have reserved enough slots to fit the new particles into.*/
    if(need_slot_reserve) {
        slots_reserve(1, slots_needed, sman);
    }

    /* Garbage lists with enough space for those from this exchange and those already present.*/
    for(n = 0; n < 6; n++) {
        if(plan->toGoSum.slots[n] + plan->ngarbage[n] == 0) {
            plan->garbage_list[n] = NULL;
            continue;
        }
        plan->garbage_list[n] = (int *) mymalloc("garbage",sizeof(int) * (plan->toGoSum.slots[n] + plan->ngarbage[n]+1));
        /* Copy over the existing garbage*/
        memcpy(plan->garbage_list[n], &tmp_garbage_list[n * preplan->ngarbage], sizeof(int)* plan->ngarbage[n]);
    }
    myfree(tmp_garbage_list);

    /* Transpose to per-target lists*/
    plan->target_list = (int **) ta_malloc("target_list",int*, plan->NTask);
    int target;
    for(target = 0; target < plan->NTask; target++) {
        if(plan->toGo[target].base == 0) {
            plan->target_list[target] = NULL;
            continue;
        }
        plan->target_list[target] = mymalloc("exchangelist",(plan->toGo[target].base+1) * sizeof(int));
    }
    int64_t * counts = ta_malloc("counts", int64_t, plan->NTask);
    memset(counts, 0, sizeof(counts[0]) * plan->NTask);
    for(n = 0; n < preplan->nexchange; n++) {
        const int target = layouts[n];
        /* This is garbage*/
        if(target == plan->ThisTask)
            continue;
        plan->target_list[target][counts[target]] = preplan->ExchangeList[n];
        counts[target]++;
        if(counts[target] > plan->toGo[target].base)
            endrun(5, "Corruption in target list n %lu target %d count %ld togo %ld nexchange %ld\n", n, target, counts[target], plan->toGo[target].base, preplan->nexchange);
    }
    for(target = 0; target < plan->NTask; target++) {
        if(counts[target] != plan->toGo[target].base)
            endrun(1, "Expected %ld in target list for task %d from plan but got %ld layout %d\n", plan->toGo[target].base, target, counts[target], layouts[0]);
    }

    myfree(counts);
    myfree(layouts);

    int failure = 0;
    if(finalNumPart > pman->MaxPart) {
        message(1, "Exchange will not succeed, need %ld particle slots but only have %ld. Current NumPart %ld\n", finalNumPart, pman->MaxPart, pman->NumPart);
        failure = 1;
    }

    /* Make failure collective*/
    MPI_Allreduce(MPI_IN_PLACE, &failure, 1, MPI_INT, MPI_SUM, Comm);

    return failure;
}

/* used only by test uniqueness */
static void
mp_order_by_id(const void * data, void * radix, void * arg) {
    ((uint64_t *) radix)[0] = ((MyIDType*) data)[0];
}

void
domain_test_id_uniqueness(struct part_manager_type * pman, MPI_Comm Comm)
{
    int64_t i, totnumpart;
    MyIDType *ids;
    int NTask, ThisTask;
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    MPI_Reduce(&pman->NumPart, &totnumpart, 1,MPI_INT64, MPI_SUM, 0, Comm);
    message(0, "Testing ID uniqueness for %ld particles on %d tasks\n", totnumpart, NTask);

    ids = (MyIDType *) mymalloc("ids", pman->NumPart * sizeof(MyIDType));

    #pragma omp parallel for
    for(i = 0; i < pman->NumPart; i++) {
        ids[i] = pman->Base[i].ID;
        if(pman->Base[i].IsGarbage)
            ids[i] = (MyIDType) -1;
    }

    mpsort_mpi(ids, pman->NumPart, sizeof(MyIDType), mp_order_by_id, 8, NULL, Comm);

    /*Remove garbage from the end*/
    int64_t nids = pman->NumPart;
    while(nids > 0 && (ids[nids-1] == (MyIDType)-1)) {
        nids--;
    }

    #pragma omp parallel for
    for(i = 1; i < nids; i++) {
        if(ids[i] <= ids[i - 1])
        {
            endrun(12, "non-unique (or non-ordered) ID=%013ld found on task=%d (i=%ld NumPart=%ld)\n",
                    ids[i], ThisTask, i, nids);
        }
    }

    MyIDType * prev = ta_malloc("prev", MyIDType, 1);
    memset(prev, 0, sizeof(MyIDType));
    const int TAG = 0xdead;

    if(NTask > 1) {
        if(ThisTask == 0) {
            MyIDType * ptr = prev;
            if(nids > 0) {
                ptr = ids;
            }
            MPI_Send(ptr, sizeof(MyIDType), MPI_BYTE, ThisTask + 1, TAG, Comm);
        }
        else if(ThisTask == NTask - 1) {
            MPI_Recv(prev, sizeof(MyIDType), MPI_BYTE,
                    ThisTask - 1, TAG, Comm, MPI_STATUS_IGNORE);
        }
        else if(nids == 0) {
            /* simply pass through whatever we get */
            MPI_Recv(prev, sizeof(MyIDType), MPI_BYTE, ThisTask - 1, TAG, Comm, MPI_STATUS_IGNORE);
            MPI_Send(prev, sizeof(MyIDType), MPI_BYTE, ThisTask + 1, TAG, Comm);
        }
        else
        {
            MPI_Sendrecv(
                    ids+(nids - 1), sizeof(MyIDType), MPI_BYTE,
                    ThisTask + 1, TAG,
                    prev, sizeof(MyIDType), MPI_BYTE,
                    ThisTask - 1, TAG, Comm, MPI_STATUS_IGNORE);
        }
    }

    if(ThisTask > 1) {
        if(nids > 0) {
            if(ids[0] <= *prev && ids[0])
                endrun(13, "non-unique ID=%ld found on task=%d\n", ids[0], ThisTask);
        }
    }

    myfree(prev);
    myfree(ids);
}

