#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>

#include <bigfile-mpi.h>

#include "utils.h"
#include "utils/mpsort.h"

#include "partmanager.h"
#include "slotsmanager.h"
#include "petaio.h"
#include "exchange.h"
#include "fof.h"
#include "walltime.h"

static void fof_register_io_blocks(int MetalReturnOn, struct IOTable * IOTable);
static void fof_write_header(BigFile * bf, int64_t TotNgroups, const double atime, const double * MassTable, Cosmology * CP, MPI_Comm Comm);
static void build_buffer_fof(FOFGroups * fof, BigArray * array, IOTableEntry * ent, struct conversions * conv);
/* Allocate a new halo structure and move particles there*/
static int fof_distribute_particles(struct part_manager_type * halo_pman, struct slots_manager_type * halo_sman, int64_t NpigLocal, int64_t * atleast, MPI_Comm Comm);

static void fof_radix_Group_GrNr(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup*) a;
    u[0] = f->GrNr;
}

int fof_select_func(int i, const struct particle_data * Parts)
{
    return Parts[i].GrNr >= 0 && Parts[i].Swallowed == 0;
}

int fof_save_particles(FOFGroups * fof, char * fname, int SaveParticles, Cosmology * CP, double atime, const double * MassTable, int MetalReturnOn, MPI_Comm Comm) {
    int i;
    struct IOTable FOFIOTable = {0};

    fof_register_io_blocks(MetalReturnOn, &FOFIOTable);
    /* sort the groups according to group-number */
    mpsort_mpi(fof->Group, fof->Ngroups, sizeof(struct Group),
            fof_radix_Group_GrNr, 8, NULL, Comm);

    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, Comm)) {
        endrun(0, "Failed to open file at %s\n", fname);
    }
    myfree(fname);
    struct conversions conv = {0};
    conv.atime = atime;
    conv.hubble = hubble_function(CP, atime);

    fof_write_header(&bf, fof->TotNgroups, atime, MassTable, CP, Comm);

    for(i = 0; i < FOFIOTable.used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = FOFIOTable.ent[i].ptype;
        BigArray array = {0};
        if(ptype == PTYPE_FOF_GROUP) {
            sprintf(blockname, "FOFGroups/%s", FOFIOTable.ent[i].name);
            build_buffer_fof(fof, &array, &FOFIOTable.ent[i], &conv);
            message(0, "Writing Block %s\n", blockname);

            petaio_save_block(&bf, blockname, &array, 1);
            petaio_destroy_buffer(&array);
        }
    }
    destroy_io_blocks(&FOFIOTable);
    walltime_measure("/FOF/IO/WriteFOF");

    /* Store whether we need a new domain_maintain after we return*/
    int domain_needed = 0;
    if(SaveParticles) {
        struct IOTable IOTable = {0};
        register_io_blocks(&IOTable, 1, MetalReturnOn);
        struct part_manager_type * halo_pman = NULL;
        struct slots_manager_type * halo_sman = NULL;
        int64_t NpigLocal = 0;
        int64_t atleast[6]={0};
        /* Count how many particles we have: array reductions are an openmp 4.5 feature.*/
        #pragma omp parallel for reduction(+: NpigLocal, atleast[:6])
        for(i = 0; i < PartManager->NumPart; i ++) {
            if(P[i].GrNr >= 0) {
                NpigLocal++;
                int type = P[i].Type;
                /* How many of slot type?*/
                if(type < 6 && type >= 0 && SlotsManager->info[type].enabled)
                    atleast[type]++;
            }
        }

        /* Find the fraction of particles in halos. If it is largeish,
         * it is easiest to reuse the current particle table*/
        int64_t NpigGlobal, NumPartGlobal;
        MPI_Allreduce(&NpigLocal, &NpigGlobal, 1, MPI_INT64, MPI_SUM, Comm);
        MPI_Allreduce(&PartManager->NumPart, &NumPartGlobal, 1, MPI_INT64, MPI_SUM, Comm);

        struct part_manager_type npartman = {0};
        struct slots_manager_type nslotman = {0};
        if(NpigGlobal > 0.25 * NumPartGlobal) {
            halo_pman = PartManager;
            halo_sman = SlotsManager;
            message(0, "Re-using partmanager for FOF: total pig %ld, global %ld\n", NpigGlobal, NumPartGlobal);
            domain_needed = 1;
        }
        else {
            message(0, "Using new partmanager for FOF: total pig %ld, global %ld\n", NpigGlobal, NumPartGlobal);
            halo_pman = &npartman;
            halo_sman = &nslotman;
        }
        if(fof_distribute_particles(halo_pman, halo_sman, NpigLocal, atleast, Comm)) {
            destroy_io_blocks(&IOTable);
            return domain_needed;
        }

        int * selection = (int *) mymalloc("Selection", sizeof(int) * halo_pman->NumPart);

        int64_t ptype_offset[6]={0};
        int64_t ptype_count[6]={0};
        petaio_build_selection(selection, ptype_offset, ptype_count, halo_pman->Base, halo_pman->NumPart, fof_select_func);

        walltime_measure("/FOF/IO/argind");

        for(i = 0; i < IOTable.used; i ++) {
            /* only process the particle blocks */
            char blockname[128];
            int ptype = IOTable.ent[i].ptype;
            BigArray array = {0};
            if(ptype < 6 && ptype >= 0) {
                sprintf(blockname, "%d/%s", ptype, IOTable.ent[i].name);
                petaio_build_buffer(&array, &IOTable.ent[i], selection + ptype_offset[ptype], ptype_count[ptype], halo_pman->Base, halo_sman, &conv);

                message(0, "Writing Block %s\n", blockname);

                petaio_save_block(&bf, blockname, &array, 1);
                petaio_destroy_buffer(&array);
            }
        }
        myfree(selection);
        /* If we allocated new particle arrays, just free them*/
        if(halo_pman != PartManager) {
            myfree(halo_sman->Base);
            myfree(halo_pman->Base);
        }
        walltime_measure("/FOF/IO/WriteParticles");
        destroy_io_blocks(&IOTable);
    }

    big_file_mpi_close(&bf, Comm);

    /* Done saving particles*/
    MPIU_Barrier(Comm);
    message(0, "Group catalogues saved.\n");
    return domain_needed;
}

struct PartIndex {
    uint64_t origin;
    union {
        int64_t sortKey;
        int targetTask;
    };
};

static int fof_sorted_layout(int i, const void * userdata) {
    struct part_manager_type * halo_pman = (struct part_manager_type *) userdata;
    /* Reuse the topleaf info to be a task. We will regenerate it with a domain_maintain afterwards.*/
    return halo_pman->Base[i].TopLeaf;
}

static void fof_radix_sortkey(const void * c1, void * out, void * arg) {
    uint64_t * u = (uint64_t *) out;
    const struct PartIndex * pi = (const struct PartIndex *) c1;
    *u = pi->sortKey;
}
static void fof_radix_origin(const void * c1, void * out, void * arg) {
    uint64_t * u = (uint64_t *) out;
    const struct PartIndex * pi = (const struct PartIndex *) c1;
    *u = pi->origin;
}
#if 0
/*Unused functions*/
static int fof_select_particle(int i) {
    return P[i].GrNr > 0;
}
static int fof_cmp_sortkey(const void * c1, const void * c2) {
    const struct PartIndex * p1 = c1;
    const struct PartIndex * p2 = c2;
    return (p1->sortKey > p2->sortKey) - (p1->sortKey < p2->sortKey);
}
static int fof_cmp_origin(const void * c1, const void * c2) {
    const struct PartIndex * p1 = c1;
    const struct PartIndex * p2 = c2;
    return (p1->origin > p2->origin) - (p1->origin < p2->origin);
}
#endif

static int
order_by_type_and_grnr(const void *a, const void *b)
{
    const struct particle_data * pa  = (const struct particle_data *) a;
    const struct particle_data * pb  = (const struct particle_data *) b;

    if(pa->Type < pb->Type)
        return -1;
    if(pa->Type > pb->Type)
        return +1;
    if(pa->GrNr < pb->GrNr)
        return -1;
    if(pa->GrNr > pb->GrNr)
        return +1;

    return 0;
}

/* Build the target task structure by doing a double parallel sort */
static void
fof_find_target_task(struct PartIndex * pi, int64_t pi_size, const uint64_t task_origin_offset, MPI_Comm Comm)
{
    int64_t i;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    /* sort pi to decide targetTask */
    mpsort_mpi(pi, pi_size, sizeof(struct PartIndex),
            fof_radix_sortkey, 8, NULL, Comm);

    #pragma omp parallel for
    for(i = 0; i < pi_size; i ++) {
        /* YU: let's see if we keep the FOF particle load on the processes, IO would be faster
           (as at high z many ranks has no FOF), communication becomes sparse. */
        pi[i].targetTask = ThisTask;
    }
    /* return pi to the original processors */
    mpsort_mpi(pi, pi_size, sizeof(struct PartIndex), fof_radix_origin, 8, NULL, Comm);
    /* Target task is copied into the particle table, unioned with Dthsml.
     * This is a bit of a hack: probably the elegant thing to do is to unify slot
     * and main structure, then mpsort the combination directly. */
#ifdef DEBUG
    int NTask;
    MPI_Comm_size(Comm, &NTask);
    #pragma omp parallel for
    for(i = 0; i < pi_size; i ++) {
        if(pi[i].targetTask >= NTask || pi[i].targetTask < 0)
            endrun(23, "pi %ld is impossible %d of %d tasks\n",i,pi[i].targetTask, NTask);
    }
#endif
}

static void
fof_copy_target_task(struct part_manager_type * halo_pman, struct PartIndex * pi, const int64_t pi_size, const uint64_t task_origin_offset, MPI_Comm Comm)
{
    int64_t i = 0;
#ifdef DEBUG
    #pragma omp parallel for
    for(i = 0; i < halo_pman->NumPart; i ++) {
        halo_pman->Base[i].TopLeaf = -1;
    }
#endif
    /* Ensure that particles outside groups do not move*/
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    #pragma omp parallel for
    for(i = 0; i < halo_pman->NumPart; i ++) {
        if(halo_pman->Base[i].GrNr < 0)
            halo_pman->Base[i].TopLeaf = ThisTask;
    }
    #pragma omp parallel for
    for(i = 0; i < pi_size; i ++) {
        size_t index = pi[i].origin % task_origin_offset;
        if(index >= (size_t) halo_pman->NumPart)
            endrun(23, "entry %ld has index %lu (npiglocal %ld)\n", i, index, halo_pman->NumPart);
        halo_pman->Base[index].TopLeaf = pi[i].targetTask;
    }

#ifdef DEBUG
    #pragma omp parallel for
    for(i = 0; i < halo_pman->NumPart; i ++) {
        if(halo_pman->Base[i].TopLeaf < 0)
            endrun(4, "TargetTask %ld not changed %d! neighbours: %d %d\n", i, halo_pman->Base[i].TopLeaf, halo_pman->Base[i-1].TopLeaf, halo_pman->Base[i+1].TopLeaf);
    }
#endif
    walltime_measure("/FOF/IO/Distribute");
}

static int
fof_distribute_particles(struct part_manager_type * halo_pman, struct slots_manager_type * halo_sman, int64_t NpigLocal, int64_t * atleast, MPI_Comm Comm)
{
    int64_t i;
    int64_t GrNrMaxGlobal = 0, GrNrMax = -1;   /* will mark particles that are not in any group */

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);
    const uint64_t task_origin_offset = PartManager->MaxPart + 1Lu;

    struct PartIndex * pi = (struct PartIndex *) mymalloc2("PartIndex", sizeof(struct PartIndex) * NpigLocal);
    /* Build the index: unfortunately not parallel*/
    NpigLocal = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].GrNr >= 0) {
            pi[NpigLocal].sortKey = PartManager->Base[i].GrNr;
            /* This should be the index of the current particle in the particle_data struct.
             * If we are re-using the PartManager, we need it to be the index in P.
             * If we have a new struct, we will do a copy. */
            if(halo_pman == PartManager)
                pi[NpigLocal].origin = task_origin_offset * ((uint64_t) ThisTask) + i;
            else
                pi[NpigLocal].origin = task_origin_offset * ((uint64_t) ThisTask) + NpigLocal;
            NpigLocal++;
            if(P[i].GrNr > GrNrMax)
                GrNrMax = P[i].GrNr;
        }
    }
    MPI_Allreduce(&GrNrMax, &GrNrMaxGlobal, 1, MPI_INT64, MPI_MAX, Comm);
    message(0, "GrNrMax is %ld\n", GrNrMaxGlobal);

    fof_find_target_task(pi, NpigLocal, task_origin_offset, Comm);
    const double FOFPartAllocFactor = (double) PartManager->MaxPart / PartManager->NumPart;

    /* Initialise the new halo structure*/
    if(halo_pman != PartManager) {
        halo_pman->MaxPart = NpigLocal * FOFPartAllocFactor;
        struct particle_data * halopart = (struct particle_data *) mymalloc("HaloParticle", sizeof(struct particle_data) * halo_pman->MaxPart);
        halo_pman->Base = halopart;
        halo_pman->NumPart = NpigLocal;
        halo_pman->BoxSize = PartManager->BoxSize;
        memcpy(halo_pman->CurrentParticleOffset, PartManager->CurrentParticleOffset, 3 * sizeof(PartManager->CurrentParticleOffset[0]));
    }

    /* Now copy the TargetTask we constructed in pi into the new halo_pman*/
    fof_copy_target_task(halo_pman, pi, NpigLocal, task_origin_offset, Comm);
    /* Free pi so that we have space for slots*/
    myfree(pi);

    /* SlotsManager Needs initializing!*/
    if(halo_sman != SlotsManager) {
        /* We leave extra space in the hope that we can avoid compacting slots in the fof exchange*/
        atleast[0] *= 1.05;
        atleast[4]*= FOFPartAllocFactor;
        atleast[5]*= FOFPartAllocFactor;

        memcpy(halo_sman, SlotsManager, sizeof(struct slots_manager_type));

        halo_sman->Base = NULL;
        for(i = 0; i < 6; i ++) {
            halo_sman->info[i].size = 0;
            halo_sman->info[i].maxsize = 0;
        }
        slots_reserve(0, atleast, halo_sman);

        report_memory_usage("FOF_PetaIO");
        NpigLocal = 0;

        for(i = 0; i < PartManager->NumPart; i ++) {
            if(P[i].GrNr < 0)
                continue;
            /* Store TargetTask as it will be over-written.
            * We freed pi above so that we could overlap PI with the slots memory.*/
            int TargetTask = halo_pman->Base[NpigLocal].TopLeaf;
            memcpy(&halo_pman->Base[NpigLocal], &P[i], sizeof(P[i]));
            halo_pman->Base[NpigLocal].TopLeaf = TargetTask;
            struct slot_info * info = &(halo_sman->info[P[i].Type]);
            char * oldslotptr = SlotsManager->info[P[i].Type].ptr;
            if(info->enabled) {
                memcpy(info->ptr + info->size * info->elsize, oldslotptr+P[i].PI * info->elsize, info->elsize);
                halo_pman->Base[NpigLocal].PI = info->size;
                info->size++;
            }
            NpigLocal ++;
        }
        if(NpigLocal != halo_pman->NumPart)
            endrun(3, "Error in NpigLocal %ld != %ld!\n", NpigLocal, halo_pman->NumPart);
    }
    /* Do a domain exchange. No pre-computed list here. Maybe a different particle table.*/
    if(domain_exchange(fof_sorted_layout, halo_pman, NULL, halo_pman, halo_sman, Comm)) {
        message(1930, "Failed to exchange and write particles for the FOF. This is non-fatal, continuing.\n");
        if(halo_pman != PartManager) {
            myfree(halo_sman->Base);
            myfree(halo_pman->Base);
        }
        return 1;
    }

    /* Sort locally by group number*/
    qsort_openmp(halo_pman->Base, halo_pman->NumPart, sizeof(struct particle_data), order_by_type_and_grnr);
#ifdef DEBUG
    GrNrMax = -1;
    int64_t GrNrMaxGlobalAfter = -1;
    #pragma omp parallel for reduction(max: GrNrMax)
    for(i = 0; i < halo_pman->NumPart; i ++) {
        if(halo_pman->Base[i].GrNr > GrNrMax)
            GrNrMax = halo_pman->Base[i].GrNr;
    }
    MPI_Allreduce(&GrNrMax, &GrNrMaxGlobalAfter, 1, MPI_INT64, MPI_MAX, Comm);
    if(GrNrMaxGlobalAfter != GrNrMaxGlobal)
        endrun(2, "GrNrMax after exchange is %ld, before was %ld\n", GrNrMaxGlobalAfter, GrNrMaxGlobal);
#endif
    return 0;
}

static void build_buffer_fof(FOFGroups * fof, BigArray * array, IOTableEntry * ent, struct conversions * conv) {

    int64_t npartLocal = fof->Ngroups;

    petaio_alloc_buffer(array, ent, npartLocal);
    /* fill the buffer */
    char * p = (char *) array->data;
    int i;
    for(i = 0; i < fof->Ngroups; i ++) {
        ent->getter(i, p, fof->Group, NULL, conv);
        p += array->strides[0];
    }
}

static void fof_write_header(BigFile * bf, int64_t TotNgroups, const double atime, const double * MassTable, Cosmology * CP, MPI_Comm Comm) {
    BigBlock bh;
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, Comm)) {
        endrun(0, "Failed to create header\n");
    }
    int i;
    int k;
    int64_t npartLocal[6];
    int64_t npartTotal[6];

    for (k = 0; k < 6; k ++) {
        npartLocal[k] = 0;
    }
    #pragma omp parallel for reduction(+: npartLocal[:6])
    for (i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].GrNr < 0) continue; /* skip those not in groups */
        npartLocal[P[i].Type] ++;
    }

    MPI_Allreduce(npartLocal, npartTotal, 6, MPI_INT64, MPI_SUM, Comm);

    /* conversion from peculiar velocity to RSD */
    const double hubble = hubble_function(CP, atime);
    double RSD = 1.0 / (atime * hubble);

    int pecvel = GetUsePeculiarVelocity();
    if(!pecvel) {
        RSD /= atime; /* Conversion from internal velocity to RSD */
    }
    big_block_set_attr(&bh, "NumPartInGroupTotal", npartTotal, "u8", 6);
    big_block_set_attr(&bh, "NumFOFGroupsTotal", &TotNgroups, "u8", 1);
    big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1);
    big_block_set_attr(&bh, "MassTable", MassTable, "f8", 6);
    big_block_set_attr(&bh, "Time", &atime, "f8", 1);
    big_block_set_attr(&bh, "BoxSize", &PartManager->BoxSize, "f8", 1);
    big_block_set_attr(&bh, "OmegaLambda", &CP->OmegaLambda, "f8", 1);
    big_block_set_attr(&bh, "Omega0", &CP->Omega0, "f8", 1);
    big_block_set_attr(&bh, "HubbleParam", &CP->HubbleParam, "f8", 1);
    big_block_set_attr(&bh, "CMBTemperature", &CP->CMBTemperature, "f8", 1);
    big_block_set_attr(&bh, "OmegaBaryon", &CP->OmegaBaryon, "f8", 1);
    big_block_set_attr(&bh, "UsePeculiarVelocity", &pecvel, "i4", 1);
    big_block_mpi_close(&bh, Comm);
}


#define SIMPLE_PROPERTY_FOF(name, field, type, items) \
    SIMPLE_GETTER(GT ## name , field, type, items, struct Group ) \
    SIMPLE_SETTER(ST ## name , field, type, items, struct Group) \

SIMPLE_PROPERTY_FOF(GroupID, base.GrNr, uint32_t, 1)
SIMPLE_PROPERTY_FOF(MinID, base.MinID, uint64_t, 1)
SIMPLE_PROPERTY_FOF(Imom, Imom[0][0], float, 9)
/* FIXME: set Jmom to use peculiar velocity */
SIMPLE_PROPERTY_FOF(Jmom, Jmom[0], float, 3)

static void GTFirstPos(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    /* Remove the particle offset before saving*/
    struct Group * grp = (struct Group *) baseptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = grp[i].base.FirstPos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STFirstPos(int i, float * out, void * baseptr, void * smanptr, const struct conversions * params) {
    int d;
    struct Group * grp = (struct Group *) baseptr;
    for(d = 0; d < 3; d ++) {
        grp->base.FirstPos[i] = out[d];
    }
}

static void GTMassCenterPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    /* Remove the particle offset before saving*/
    struct Group * grp = (struct Group *) baseptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = grp[i].CM[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > PartManager->BoxSize) out[d] -= PartManager->BoxSize;
        while(out[d] <= 0) out[d] += PartManager->BoxSize;
    }
}

static void STMassCenterPosition(int i, double * out, void * baseptr, void * smanptr, const struct conversions * params) {
    int d;
    struct Group * grp = (struct Group *) baseptr;
    for(d = 0; d < 3; d ++) {
        grp->CM[d] = out[d];
    }
}

static void GTMassCenterVelocity(int i, float * out, void * baseptr, void * slotptr, const struct conversions * params) {
    double fac;
    struct Group * Group = (struct Group *) baseptr;
    if (GetUsePeculiarVelocity()) {
        fac = 1.0 / params->atime;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = fac * Group[i].Vel[d];
    }
}
SIMPLE_PROPERTY_FOF(Mass, Mass, float, 1)
SIMPLE_PROPERTY_FOF(MassByType, MassType[0], float, 6)
SIMPLE_PROPERTY_FOF(LengthByType, LenType[0], uint32_t , 6)
SIMPLE_PROPERTY_FOF(StarFormationRate, Sfr, float, 1)
SIMPLE_PROPERTY_FOF(GasMetalMass, GasMetalMass, float, 1)
SIMPLE_PROPERTY_FOF(StellarMetalMass, StellarMetalMass, float, 1)
SIMPLE_PROPERTY_FOF(GasMetalElemMass, GasMetalElemMass[0], float, NMETALS)
SIMPLE_PROPERTY_FOF(StellarMetalElemMass, StellarMetalElemMass[0], float, NMETALS)
SIMPLE_PROPERTY_FOF(BlackholeMass, BH_Mass, float, 1)
SIMPLE_PROPERTY_FOF(BlackholeAccretionRate, BH_Mdot, float, 1)
SIMPLE_PROPERTY_FOF(MassHeIonized, MassHeIonized, float, 1)

static void fof_register_io_blocks(int MetalReturnOn, struct IOTable * IOTable) {
    IOTable->used = 0;
    IOTable->allocated = 100;
    /* Allocate high so we can do a domain exchange,
     * potentially increasing the slots, around this*/
    IOTable->ent = (struct IOTableEntry *) mymalloc2("IOTable", IOTable->allocated* sizeof(IOTableEntry));

    IO_REG(GroupID, "u4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Mass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MassCenterPosition, "f8", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(FirstPos, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MinID, "u8", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Imom, "f4", 9, PTYPE_FOF_GROUP, IOTable);
    IO_REG(Jmom, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG_WRONLY(MassCenterVelocity, "f4", 3, PTYPE_FOF_GROUP, IOTable);
    IO_REG(LengthByType, "u4", 6, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MassByType, "f4", 6, PTYPE_FOF_GROUP, IOTable);
    IO_REG(MassHeIonized, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    /* Zero if star formation is not on*/
    IO_REG(StarFormationRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    if(MetalReturnOn) {
        IO_REG(GasMetalMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        IO_REG(StellarMetalMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
        /* Zero if metal return is not on*/
        IO_REG(GasMetalElemMass, "f4", NMETALS, PTYPE_FOF_GROUP, IOTable);
        IO_REG(StellarMetalElemMass, "f4", NMETALS, PTYPE_FOF_GROUP, IOTable);
    }
    /* Zero if black hole is not on*/
    IO_REG(BlackholeMass, "f4", 1, PTYPE_FOF_GROUP, IOTable);
    IO_REG(BlackholeAccretionRate, "f4", 1, PTYPE_FOF_GROUP, IOTable);
}
