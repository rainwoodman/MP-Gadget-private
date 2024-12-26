/*Simple test for the exchange function*/

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

#define qsort_openmp qsort

#include <libgadget/exchange.h>
#include <libgadget/domain.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/partmanager.h>
#include "stub.h"
#include <libgadget/walltime.h>

int NTask, ThisTask;
int TotNumPart;

static struct ClockTable Clocks;

#define NUMPART1 8
static int
setup_particles(int64_t NType[6])
{
    walltime_init(&Clocks);
    MPI_Barrier(MPI_COMM_WORLD);
    PartManager->MaxPart = 1024;
    int ptype;
    PartManager->NumPart = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        PartManager->NumPart += NType[ptype];
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    P = (struct particle_data *) mymalloc("P", PartManager->MaxPart * sizeof(struct particle_data));
    memset(P, 0, sizeof(struct particle_data) * PartManager->MaxPart);

    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    slots_set_enabled(4, sizeof(struct star_particle_data), SlotsManager);
    slots_set_enabled(5, sizeof(struct bh_particle_data), SlotsManager);


    slots_reserve(1, NType, SlotsManager);

    slots_setup_topology(PartManager, NType, SlotsManager);

    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
    }

    slots_setup_id(PartManager, SlotsManager);

    MPI_Allreduce(&PartManager->NumPart, &TotNumPart, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return 0;
}

static int
teardown_particles(void **state)
{
    int TotNumPart2;

    int i;
    int nongarbage = 0, garbage = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(!P[i].IsGarbage) {
            nongarbage++;
            assert_true (P[i].ID % NTask == 1Lu * ThisTask);
            continue;
        }
        else
            garbage++;
    }
    message(2, "curpart %d (np %ld) tot %d garbage %d\n", nongarbage, PartManager->NumPart, TotNumPart, garbage);
    MPI_Allreduce(&nongarbage, &TotNumPart2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(TotNumPart2, TotNumPart);

    slots_free(SlotsManager);
    myfree(P);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}


static int
test_exchange_layout_func(int i, const void * userdata)
{
    return P[i].ID % NTask;
}

static void
test_exchange(void **state)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager,MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);
    teardown_particles(state);
    return;
}

static void
test_exchange_multiple(void **state)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    domain_set_max_exchange(8*1024L);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager,MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);
    domain_set_max_exchange(2048L*1024L*1024L);

    teardown_particles(state);
    return;
}

static void
test_exchange_smallbuffer(void **state)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    domain_set_max_exchange(1024L);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager,MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);
    domain_set_max_exchange(2048L*1024L*1024L);

    teardown_particles(state);
    return;
}

static void
test_exchange_zero_slots(void **state)
{
    int64_t newSlots[6] = {NUMPART1, 0, NUMPART1, 0, NUMPART1, 0};

    setup_particles(newSlots);

    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager, MPI_COMM_WORLD);

    assert_all_true(!fail);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);

    teardown_particles(state);
    return;
}

static void
test_exchange_with_garbage(void **state)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);

    slots_mark_garbage(0, PartManager, SlotsManager); /* watch out! this propagates the garbage flag to children */
    TotNumPart -= NTask;
    int fail = domain_exchange(&test_exchange_layout_func, NULL, NULL, PartManager, SlotsManager, MPI_COMM_WORLD);

    assert_all_true(!fail);

    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);
#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    teardown_particles(state);
    return;
}

static int
test_exchange_layout_func_uneven(int i, const void * userdata)
{
    if(P[i].Type == 0) return 0;

    return P[i].ID % NTask;
}

static void
test_exchange_uneven(void **state)
{
    int64_t newSlots[6] = {NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1, NUMPART1};

    setup_particles(newSlots);
    int i;

    /* this will trigger a slot growth on slot type 0 due to the inbalance */
    int fail = domain_exchange(&test_exchange_layout_func_uneven, NULL, NULL, PartManager, SlotsManager, MPI_COMM_WORLD);

    assert_all_true(!fail);

    if(ThisTask == 0) {
        /* the slot type must have grown automatically to handle the new particles. */
        assert_int_equal(SlotsManager->info[0].size, NUMPART1 * NTask);
    }

#ifdef DEBUG
    slots_check_id_consistency(PartManager, SlotsManager);
#endif
    domain_test_id_uniqueness(PartManager, MPI_COMM_WORLD);

    int TotNumPart2;

    int nongarbage = 0;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(!P[i].IsGarbage) {
            nongarbage++;
            if(P[i].Type == 0) {
                assert_true (ThisTask == 0);
            } else {
                assert_true(P[i].ID % NTask == 1Lu * ThisTask);
            }
            continue;
        }
    }
    MPI_Allreduce(&nongarbage, &TotNumPart2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    assert_int_equal(TotNumPart2, TotNumPart);

    slots_free(SlotsManager);
    myfree(P);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_exchange),
        cmocka_unit_test(test_exchange_multiple),
        cmocka_unit_test(test_exchange_smallbuffer),
        cmocka_unit_test(test_exchange_with_garbage),
        cmocka_unit_test(test_exchange_zero_slots),
        cmocka_unit_test(test_exchange_uneven),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
