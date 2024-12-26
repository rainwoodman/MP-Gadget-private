#include <string.h>
#include "slotsmanager.h"
#include "partmanager.h"

#include "utils.h"

struct slots_manager_type SlotsManager[1];

#define SLOTS_ENABLED(ptype, sman) (sman->info[ptype].enabled)

static struct sph_particle_data * GDB_SphP;
static struct star_particle_data * GDB_StarP;
static struct bh_particle_data * GDB_BhP;

static int
slots_gc_base(struct part_manager_type * pman);

static int
slots_gc_slots(int * compact_slots, struct part_manager_type * pman, struct slots_manager_type * sman);

/* Initialise a new slot with type at index pi
 * for the particle at index i.
 * This will modify both P[i] and the slot at pi in type.*/
static void
slots_connect_new_slot(int i, int pi, int type, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    /* Fill slot with a meaningless
     * poison value ('e') so we will recognise
     * if it is uninitialised.*/
    memset(BASESLOT_PI(pi, type, sman), 101, sman->info[type].elsize);
    /* book keeping ID: debug only */
#ifdef DEBUG
    BASESLOT_PI(pi, type, sman)->ID = pman->Base[i].ID;
#endif
    /*Update the particle's pointer*/
    pman->Base[i].PI = pi;
}

/* This will change a particle type. The original particle_data structure is preserved,
 * but the old slot is made garbage and a new one (with the new type) is created.
 * No data is copied, but the slot is created.
 *
 * Assumes the particle is protected by locks in threaded env.
 *
 * Note that the 'new particle' event is not emitted, as there is no new particle!
 * If you do something on a new slot, this needs a new event.
 *
 * Arguments:
 * parent - particle whose type is changing.
 * ptype - type to change it to
 * placement - if this is not -1, we use a specific numbered slot.
 *             if this is -1, get a new slot atomically from the pre-allocated heap.
 * discardold - if this is true, the pre-conversion slot will be marked as garbage.
 *              If you are really converting then this should be true.
 *              If you are using this function on the output of slots_split_particle, it should be false.
 * */
int
slots_convert(int parent, int ptype, int placement, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    /*Explicitly mark old slot as garbage*/
    int oldtype = pman->Base[parent].Type;
    int oldPI = pman->Base[parent].PI;
    if(oldPI >= 0 && SLOTS_ENABLED(oldtype, sman))
        BASESLOT_PI(oldPI, oldtype, sman)->ReverseLink = pman->MaxPart + 100;

    /*Make a new slot*/
    if(SLOTS_ENABLED(ptype, sman)) {
        int newPI = placement;
        /* if enabled, alloc a new Slot for secondary data */
        if(placement < 0)
            newPI = atomic_fetch_and_add_64(&sman->info[ptype].size, 1);

        /* There is no way clearly to safely grow the slots during this, because the memory may be deep in the heap.*/
        if(newPI >= sman->info[ptype].maxsize) {
            endrun(1, "Tried to use non-allocated slot %d (> %ld)\n", newPI, sman->info[ptype].maxsize);
        }
        slots_connect_new_slot(parent, newPI, ptype, pman, sman);
    }
    /*Type changed after slot updated*/
    pman->Base[parent].Type = ptype;
    return parent;
}

/* This will split a new particle out from an existing one, conserving mass.
 * The type is the same and the slot PI on the new particle is set to -1.
 * You should call slots_convert on the child afterwards to create a new slot.
 *
 * Assumes the particle is protected by locks in threaded env.
 *
 * The Generation of parent is incremented.
 * The child carries the incremented generation number.
 * The ID of the child is modified, with the new generation number set
 * at the highest 8 bits.
 *
 * the new particle's index is returned.
 *
 * Its mass and ptype can be then adjusted using slots_convert.
 * The 'new particle' event is emitted.
 * */
int
slots_split_particle(int parent, double childmass, struct part_manager_type * pman)
{
    int64_t child = atomic_fetch_and_add_64(&pman->NumPart, 1);

    if(child >= pman->MaxPart)
        endrun(8888, "Tried to spawn: NumPart=%ld MaxPart = %ld. Sorry, no space left.\n", child, pman->MaxPart);

    pman->Base[parent].Generation ++;
    uint64_t g = pman->Base[parent].Generation;
    pman->Base[child] = pman->Base[parent];

    /* change the child ID according to the generation. */
    pman->Base[child].ID = (pman->Base[parent].ID & 0x00ffffffffffffffL) + (g << 56L);
    if(g >= (1 << (64-56L)))
        endrun(1, "Particle %d (ID: %ld) generated too many particles: generation %ld wrapped.\n", parent, pman->Base[parent].ID, g);

    pman->Base[child].Mass = childmass;
    pman->Base[parent].Mass -= childmass;

    /*Invalidate the slot of the child. Call slots_convert soon afterwards!*/
    pman->Base[child].PI = -1;

    return child;
}

/* remove garbage particles, holes in sph chunk and holes in bh buffer.
 * This algorithm is O(n), and shifts particles over the holes.
 * compact_slots is a 6-member array, 1 if that slot should be compacted, 0 otherwise.
 * As slots_gc_base preserves the order of the slots, one may usually skip compaction.*/
int
slots_gc(int * compact_slots, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    /* tree is invalidated if the sequence on P is reordered; */

    int tree_invalid = 0;

    /* TODO: in principle we can track this change and modify the tree nodes;
     * But doing so requires cleaning up the TimeBin link lists, and the tree
     * link lists first. likely worth it, since GC happens only in domain decompose
     * and snapshot IO, both take far more time than rebuilding the tree. */
    tree_invalid |= slots_gc_base(pman);
    tree_invalid |= slots_gc_slots(compact_slots, pman, sman);

    MPI_Allreduce(MPI_IN_PLACE, &tree_invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return tree_invalid;
}


#define GARBAGE(i, ptype, pman, sman) (sman ? BASESLOT_PI(i,ptype, sman)->ReverseLink > pman->MaxPart : pman->Base[i].IsGarbage)
#define PART(i, ptype, pman, sman) (sman ? (void *) BASESLOT_PI(i, ptype, sman) : (void *) &pman->Base[i])

/*Find the next garbage particle*/
static int
slots_find_next_garbage(int start, int used, int ptype, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int i, nextgc = used;
    /*Find another garbage particle*/
    for(i = start; i < used; i++)
        if(GARBAGE(i, ptype, pman, sman)) {
            nextgc = i;
            break;
        }
    return nextgc;
}

/*Find the next non-garbage particle*/
static int
slots_find_next_nongarbage(int start, int used, int ptype, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int i, nextgc = used;
    /*Find another garbage particle*/
    for(i = start; i < used; i++)
        if(!GARBAGE(i, ptype, pman, sman)) {
            nextgc = i;
            break;
        }
    return nextgc;
}

/*Compaction algorithm*/
static int
slots_gc_compact(const int used, int ptype, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    /*Find first garbage particle: can't use bisection here as not sorted.*/
    int nextgc = slots_find_next_garbage(0, used, ptype, pman, sman);
    size_t size = sizeof(struct particle_data);
    if(sman)
        size = sman->info[ptype].elsize;
    int ngc = 0;
    /*Note each particle is tested exactly once*/
    while(nextgc < used) {
        /*Now lastgc contains a garbage*/
        int lastgc = nextgc;
        /*Find a non-garbage after it*/
        int src = slots_find_next_nongarbage(lastgc+1, used, ptype, pman, sman);
        /*If no more non-garbage particles, don't bother copying, just add a skip*/
        if(src == used) {
            ngc += src - lastgc;
            break;
        }
        /*Destination is shifted already*/
        int dest = lastgc - ngc;

        /*Find another garbage particle*/
        nextgc = slots_find_next_garbage(src + 1, used, ptype, pman, sman);

        /*Add number of particles we skipped*/
        ngc += src - lastgc;
        int nmove = nextgc - src +1;
//         message(1,"i = %d, PI = %d-> %d, nm=%d\n",i, src, dest, nmove);
        memmove(PART(dest, ptype, pman, sman),PART(src, ptype, pman, sman),nmove*size);
    }
    if(ngc > used)
        endrun(1, "ngc = %d > used = %d!\n", ngc, used);
    return ngc;
}

static int
slots_gc_base(struct part_manager_type * pman)
{
    int64_t total0, total;

    MPI_Allreduce(&pman->NumPart, &total0, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /*Compactify the P array: this invalidates the ReverseLink, so
        * that ReverseLink is valid only within gc.*/
    int64_t ngc = slots_gc_compact(pman->NumPart, -1, pman, NULL);

    pman->NumPart -= ngc;

    MPI_Allreduce(&pman->NumPart, &total, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    if(total != total0) {
        message(0, "GC : Reducing Particle slots from %ld to %ld\n", total0, total);
        return 1;
    }
    return 0;
}

static int slot_cmp_reverse_link(const void * b1in, const void * b2in) {
    const struct particle_data_ext * b1 = (struct particle_data_ext *) b1in;
    const struct particle_data_ext * b2 = (struct particle_data_ext *) b2in;
    return (b1->ReverseLink > b2->ReverseLink) - (b1->ReverseLink < b2->ReverseLink);
}

static int
slots_gc_mark(const struct part_manager_type * pman, const struct slots_manager_type * sman)
{
    int64_t i;
    if(!(sman->info[0].enabled ||
       sman->info[1].enabled ||
       sman->info[2].enabled ||
       sman->info[3].enabled ||
       sman->info[4].enabled ||
       sman->info[5].enabled))
        return 0;

#ifdef DEBUG
    int ptype;
    /*Initially set all reverse links to an obviously invalid value*/
    for(ptype = 0; ptype < 6; ptype++)
    {
        struct slot_info info = sman->info[ptype];
        if(!info.enabled)
            continue;
        #pragma omp parallel for
        for(i = 0; i < info.size; i++) {
            struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * i);
            sdata->ReverseLink = pman->MaxPart + 100;
        }
    }
#endif

#pragma omp parallel for
    for(i = 0; i < pman->NumPart; i++) {
        struct slot_info info = sman->info[pman->Base[i].Type];
        if(!info.enabled)
            continue;
        int sind = pman->Base[i].PI;
        if(sind >= info.size || sind < 0)
            endrun(1, "Particle %ld, type %d has PI index %d beyond max slot size %ld.\n", i, pman->Base[i].Type, sind, info.size);
        struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * sind);
        sdata->ReverseLink = i;
        /* Make the PI of garbage particles invalid*/
        if(pman->Base[i].IsGarbage)
            sdata->ReverseLink = pman->MaxPart + 100;
    }
    return 0;
}

/* sweep removes unused entries in the slot list. */
static int
slots_gc_sweep(int ptype, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    if(!SLOTS_ENABLED(ptype, sman)) return 0;
    int64_t used = sman->info[ptype].size;

    int64_t ngc = slots_gc_compact(used, ptype, pman, sman);

    sman->info[ptype].size -= ngc;

    return ngc;
}

/* update new pointers. */
static void
slots_gc_collect(int ptype, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int64_t i;
    if(!SLOTS_ENABLED(ptype, sman)) return;

    /* Now update the link in BhP */
#pragma omp parallel for
    for(i = 0;
        i < sman->info[ptype].size;
        i ++) {

#ifdef DEBUG
        if(BASESLOT_PI(i, ptype, sman)->ReverseLink >= pman->MaxPart) {
            endrun(1, "Shall not happen: i=%ld ptype = %d\n", i,ptype);
        }
#endif

        pman->Base[BASESLOT_PI(i, ptype, sman)->ReverseLink].PI = i;
    }
}


static int
slots_gc_slots(int * compact_slots, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int ptype;

    int64_t total0[6];
    int64_t total1[6];

    int disabled = 1;
    for(ptype = 0; ptype < 6; ptype ++) {
        MPI_Allreduce(&sman->info[ptype].size, &total0[ptype], 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
        if(compact_slots[ptype])
            disabled = 0;
    }

#ifdef DEBUG
    slots_check_id_consistency(pman, sman);
#endif

    if(!disabled) {
        slots_gc_mark(pman, sman);

        for(ptype = 0; ptype < 6; ptype++) {
            if(!compact_slots[ptype])
                continue;
            slots_gc_sweep(ptype, pman, sman);
            slots_gc_collect(ptype, pman, sman);
        }
    }
#ifdef DEBUG
    slots_check_id_consistency(pman, sman);
#endif
    for(ptype = 0; ptype < 6; ptype ++) {
        MPI_Allreduce(&sman->info[ptype].size, &total1[ptype], 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

        if(total1[ptype] != total0[ptype])
            message(0, "GC: Reducing number of slots for %d from %ld to %ld\n", ptype, total0[ptype], total1[ptype]);
    }

    /* slot gc never invalidates the tree */
    return 0;
}

/*Returns the number of non-Garbage particles in an array with garbage sorted to the end.
 * The index returned always points to a garbage particle.
 * If ptype < 0, find the last garbage particle in the P array.
 * If ptype >= 0, find the last garbage particle in the slot associated with ptype. */
int
slots_get_last_garbage(int nfirst, int nlast, int ptype, const struct part_manager_type * pman, const struct slots_manager_type * sman)
{
    /* Enforce that we don't get a negative number for an empty array*/
    if(nlast < 0)
        return 0;
    /* nfirst is always not garbage, nlast is always garbage*/
    if(GARBAGE(nfirst, ptype, pman, sman))
        return nfirst;
    if(!GARBAGE(nlast, ptype, pman, sman))
        return nlast+1;
    /*Bisection*/
    do {
        int nmid = (nfirst + nlast)/2;
        if(GARBAGE(nmid, ptype, pman, sman))
            nlast = nmid;
        else
            nfirst = nmid;
    }
    while(nlast - nfirst > 1);

    return nlast;
}

struct PeanoOrder
{
    peano_t Key;
    int TypeKey;
    int Pindex;
};

static int
order_by_type_and_key(const void *a, const void *b)
{
    const struct PeanoOrder * pa  = (const struct PeanoOrder *) a;
    const struct PeanoOrder * pb  = (const struct PeanoOrder *) b;
    /* Note garbage types have their values set to something large here*/
    if(pa->TypeKey < pb->TypeKey)
        return -1;
    if(pa->TypeKey > pb->TypeKey)
        return +1;
    if(pa->Key < pb->Key)
        return -1;
    if(pa->Key > pb->Key)
        return +1;

    return 0;
}

/* Sort the particles and their slots by type and peano order.
 * This does a gc by sorting the Garbage to the end of the array and then trimming.
 * It is a different algorithm to slots_gc, somewhat slower,
 * but delivers a spatially compact sort. It always compacts the slots*/
void
slots_gc_sorted(struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int ptype, i;
    /* Resort the particles such that those of the same type and key are close by.
     * The locality is broken by the exchange. */
    int64_t garbage=0;
    struct PeanoOrder * peanokeys = (struct PeanoOrder *)mymalloc("Keydata", pman->NumPart * sizeof(struct PeanoOrder));
    #pragma omp parallel for reduction(+: garbage)
    for(i = 0; i < pman->NumPart; i++) {
        peanokeys[i].Key = PEANO(pman->Base[i].Pos, pman->BoxSize);
        peanokeys[i].TypeKey = pman->Base[i].Type;
        if(pman->Base[i].IsGarbage) {
            garbage++;
            /* Set it to a type far beyond the existing types, so it sorts to the end.*/
            peanokeys[i].TypeKey = 255;
        }
        peanokeys[i].Pindex = i;
    }
    /* Sort the keys*/
    qsort_openmp(peanokeys, pman->NumPart, sizeof(struct PeanoOrder), order_by_type_and_key);
    /* Now sort the base with a cycle leader permutation algorithm, like qsort.*/
    for(i = 0; i < pman->NumPart; i++) {
        int k = peanokeys[i].Pindex;
        /* This element already in the right place*/
        if(k == i)
            continue;
        /* Copy the wrongly placed element to a tmp*/
        struct particle_data tmp_p = pman->Base[i];
        int j = i;
        do {
            /* Update index so that we know this element is permuted.*/
            peanokeys[j].Pindex = j;
            /* Copy the right element in.*/
            pman->Base[j] = pman->Base[k];
            /* k is the new j*/
            j = k;
            k = peanokeys[j].Pindex;
        } while(i != k);

        /* We finished when we found the right spot for the first object, the one we copied into tmp*/
        peanokeys[j].Pindex = j;
        pman->Base[j] = tmp_p;
    }
    // message(1, "garbage %ld\n", garbage);
    /*Remove garbage particles*/
    pman->NumPart -= garbage;

    myfree(peanokeys);
    /*Set up ReverseLink*/
    slots_gc_mark(pman, sman);

    for(ptype = 0; ptype < 6; ptype++) {
        if(!SLOTS_ENABLED(ptype, sman))
            continue;
        /* sort the used ones
         * by their location in the P array */
        qsort_openmp(sman->info[ptype].ptr,
                 sman->info[ptype].size,
                 sman->info[ptype].elsize,
                 slot_cmp_reverse_link);

        /*Reduce slots used*/
        SlotsManager->info[ptype].size = slots_get_last_garbage(0, sman->info[ptype].size-1, ptype, pman, sman);
        slots_gc_collect(ptype, pman, sman);
    }
#ifdef DEBUG
    slots_check_id_consistency(pman, sman);
#endif
}

size_t
slots_reserve(int where, int64_t atleast[6], struct slots_manager_type * sman)
{
    int64_t newMaxSlots[6];
    int ptype;
    int good = 1;

    if(sman->Base == NULL) {
        sman->Base = (char*) mymalloc("SlotsBase", sizeof(struct sph_particle_data));
        /* This is so the ptr is never null! Avoid undefined behaviour. */
        for(ptype = 5; ptype >= 0; ptype--) {
            sman->info[ptype].ptr = sman->Base;
        }
    }

    int64_t add = sman->increase;
    if (add < 8192) add = 8192;

    /* FIXME: allow shrinking; need to tweak the memmove later. */
    for(ptype = 0; ptype < 6; ptype ++) {
        newMaxSlots[ptype] = sman->info[ptype].maxsize;
        if(!SLOTS_ENABLED(ptype, sman)) continue;
        /* if current empty slots is less than half of add, need to grow */
        if (newMaxSlots[ptype] <= atleast[ptype] + add / 2) {
            newMaxSlots[ptype] = atleast[ptype] + add;
            good = 0;
        }
    }

    size_t total_bytes = 0;
    size_t offsets[6];
    size_t bytes[6] = {0};

    for(ptype = 0; ptype < 6; ptype++) {
        offsets[ptype] = total_bytes;
        bytes[ptype] = sman->info[ptype].elsize * newMaxSlots[ptype];
        total_bytes += bytes[ptype];
    }
    /* no need to grow, already have enough */
    if (good) {
        return total_bytes;
    }
    char * newSlotsBase = (char *) myrealloc(sman->Base, total_bytes);

    /* If we are using VALGRIND the allocator is system malloc, and so realloc may move the base pointer.
     * Thus we need to also move the slots pointers before doing the memmove. If we are using our own
     * memory allocator the base address never moves, so this is unnecessary (but we do it anyway).*/
    for(ptype = 0; ptype < 6; ptype++) {
        sman->info[ptype].ptr = sman->info[ptype].ptr - sman->Base + newSlotsBase;
    }

    message(where, "SLOTS: Reserved %g MB for %ld sph, %ld stars and %ld BHs (disabled: %ld %ld %ld)\n", total_bytes / (1024.0 * 1024.0),
            newMaxSlots[0], newMaxSlots[4], newMaxSlots[5], newMaxSlots[1], newMaxSlots[2], newMaxSlots[3]);

    /* move the last block first since we are only increasing sizes, moving items forward.
     * No need to move the 0 block, since it is already moved to newSlotsBase in realloc.*/
    for(ptype = 5; ptype > 0; ptype--) {
        if(!SLOTS_ENABLED(ptype, sman)) continue;
        memmove(newSlotsBase + offsets[ptype],
            sman->info[ptype].ptr,
            sman->info[ptype].elsize * sman->info[ptype].size);
    }

    sman->Base = newSlotsBase;

    for(ptype = 0; ptype < 6; ptype++) {
        sman->info[ptype].ptr = newSlotsBase + offsets[ptype];
        sman->info[ptype].maxsize = newMaxSlots[ptype];
    }
    GDB_SphP = (struct sph_particle_data *) sman->info[0].ptr;
    GDB_StarP = (struct star_particle_data *) sman->info[4].ptr;
    GDB_BhP = (struct bh_particle_data *) sman->info[5].ptr;
    return total_bytes;
}

void
slots_init(double increase, struct slots_manager_type * sman)
{
    memset(sman, 0, sizeof(sman[0]));
    sman->increase = increase;
}

void
slots_set_enabled(int ptype, size_t elsize, struct slots_manager_type * sman)
{
    sman->info[ptype].enabled = 1;
    sman->info[ptype].elsize = elsize;
}


void
slots_free(struct slots_manager_type * sman)
{
    myfree(sman->Base);
}

/* mark the i-th base particle as a garbage. */
void
slots_mark_garbage(int i, struct part_manager_type * pman, struct slots_manager_type * sman)
{
    pman->Base[i].IsGarbage = 1;
    int type = pman->Base[i].Type;
    if(SLOTS_ENABLED(type, sman)) {
        BASESLOT_PI(pman->Base[i].PI, type, sman)->ReverseLink = pman->MaxPart + 100;
    }
}

#ifdef DEBUG
void
slots_check_id_consistency(struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int64_t used[6] = {0};
    int64_t i;

    for(i = 0; i < pman->NumPart; i++) {
        int type = pman->Base[i].Type;
        if(pman->Base[i].IsGarbage)
            continue;
        struct slot_info info = sman->info[type];
        if(!info.enabled)
            continue;

        int PI = pman->Base[i].PI;
        if(PI >= info.size) {
            endrun(1, "slot PI consistency failed: PI=%d type = %d size %ld\n", PI, type, info.size);
        }
        if(BASESLOT_PI(PI, type, sman)->ID != pman->Base[i].ID) {
            endrun(1, "slot id consistency failed2: i=%ld PI=%d type = %d P.ID = %ld SLOT.ID=%ld\n",i, PI, pman->Base[i].Type, pman->Base[i].ID, BASESLOT_PI(PI, type, sman)->ID);
        }
        used[type] ++;
    }
    int64_t NTotal[6];

    MPI_Allreduce(used, NTotal, 6, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    int ptype;
    for(ptype = 0; ptype < 6; ptype ++) {
        if(NTotal[ptype] > 0) {
            /* Watch out: we print per rank here, but the condition must be global*/
            message(0, "Task 0: GC: Used slots for type %d is %ld\n", ptype, used[ptype]);
        }
    }
}
#endif

/* this function needs the Type of P[i] to be setup */
void
slots_setup_topology(struct part_manager_type * pman, int64_t * NLocal, struct slots_manager_type * sman)
{
    /* initialize particle types */
    int ptype;
    int64_t offset = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        int64_t i;
        struct slot_info info = sman->info[ptype];
        #pragma omp parallel for
        for(i = 0; i < NLocal[ptype]; i++)
        {
            size_t j = offset + i;
            pman->Base[j].Type = ptype;
            pman->Base[j].IsGarbage = 0;
            if(info.enabled)
                pman->Base[j].PI = i;
        }
        offset += NLocal[ptype];
    }

    for(ptype = 0; ptype < 6; ptype ++) {
        struct slot_info info = sman->info[ptype];
        if(!info.enabled)
            continue;
        sman->info[ptype].size = NLocal[ptype];
    }
}

void
slots_setup_id(const struct part_manager_type * pman, struct slots_manager_type * sman)
{
    int64_t i;
    /* set up the cross check for child IDs */
    #pragma omp parallel for
    for(i = 0; i < pman->NumPart; i++)
    {
        struct slot_info info = sman->info[pman->Base[i].Type];
        if(!info.enabled)
            continue;

        int sind = pman->Base[i].PI;
        if(sind >= info.size || sind < 0)
            endrun(1, "Particle %ld, type %d has PI index %d beyond max slot size %ld.\n", i, pman->Base[i].Type, sind, info.size);
        struct particle_data_ext * sdata = (struct particle_data_ext * )(info.ptr + info.elsize * (size_t) sind);
        sdata->ReverseLink = i;
#ifdef DEBUG
        sdata->ID = pman->Base[i].ID;
#endif
    }
}
