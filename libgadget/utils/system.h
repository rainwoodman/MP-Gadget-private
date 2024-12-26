#ifndef __UTILS_SYSTEM_H__
#define __UTILS_SYSTEM_H__

#include <stdint.h>
#include <mpi.h>
#include <stdarg.h>

/* Note on a 32-bit architecture MPI_LONG may be 32-bit,
 * so these should be MPI_LONG_LONG. But in
 * the future MPI_LONG_LONG may become 128-bit.*/
#define MPI_UINT64 MPI_UNSIGNED_LONG
#define MPI_INT64 MPI_LONG

/* Check the version of OPENMP. We now require OpenMP 4.5 for array reductions. */
#if _OPENMP < 201511
#error MP-Gadget requires OpenMP >= 4.5. Use a newer compiler (gcc >= 6.0, intel >= 17 clang >= 7).
#endif

typedef struct _Rnd_Table
{
  double * Table;
  size_t tablesize;
} RandTable;

int cluster_get_num_hosts(void);
double get_physmem_bytes(void);

/* Gets a random number in the range [0, 1) from the table. The id is used modulo the size of the table,
 * so only the lowest bits are used.
 * Random deviates are taken from a pre-seeded table generated in set_random_numbers, so that they are
 * independent of processor.*/
double get_random_number(const uint64_t id, const RandTable * const rnd);
/* Generate the random number table. The seed should be the same on each processor so the output is invariant to
 * To quote the GSL documentation: 'Note that the most generators only accept 32-bit seeds, with higher values being reduced modulo 2^32.'
 * It is important that each timestep uses a new seed value, so the seed should change by less than 2^32 each timestep.
 * The random number table is heap-allocated high, and random numbers are uniform doubles between 0 and 1.*/
RandTable set_random_numbers(uint64_t seed, const size_t rndtablesize);
/* Free the random number table and set Table to NULL*/
void free_random_numbers(RandTable * rnd);
int64_t count_sum(int64_t countLocal);

/* Returns true if condition is true on ANY processor*/
int MPIU_Any(int condition, MPI_Comm comm);

void MPIU_write_pids(char * filename);

typedef struct _gadget_thread_arrays {
  int * dest;
  int ** srcs;
  size_t *sizes;
  int narrays;
  size_t total_size;
  size_t schedsz;
} gadget_thread_arrays;
/* Compact an array which has segments (usually corresponding to different threads).
 * After this is run, it will be a single contiguous array. The memory can then be realloced.
 * Function returns size of the final array, pointer to final array is stored in dest.
 * The temporary arrays in gadget_thread_arrays are freed. */
size_t gadget_compact_thread_arrays(int ** dest, gadget_thread_arrays * arrays);

/* Set up pointers to different parts of a single segmented array, evenly spaced and corresponding queue space for different threads.*/
gadget_thread_arrays gadget_setup_thread_arrays(const char * destname, const int alloc_high, const size_t total_size);

int MPI_Alltoallv_smart(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_sparse(void *sendbuf, int *sendcnts, int *sdispls,
        MPI_Datatype sendtype, void *recvbuf, int *recvcnts,
        int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

double timediff(double t0, double t1);
double second(void);
size_t sizemax(size_t a, size_t b);

static inline int64_t atomic_fetch_and_add_64(int64_t * ptr, int64_t value) {
    int64_t k;
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
    return k;
}

static inline int atomic_fetch_and_add(int * ptr, int value) {
    int k;
#pragma omp atomic capture
    {
      k = (*ptr);
      (*ptr)+=value;
    }
    return k;
}
static inline int atomic_add_and_fetch(int * ptr, int value) {
    int k;
#pragma omp atomic capture
    {
      (*ptr)+=value;
      k = (*ptr);
    }
    return k;
}

void MPIU_Trace(MPI_Comm comm, int where, const char * fmt, ...);
void MPIU_Tracev(MPI_Comm comm, int where, int error, const char * fmt, va_list va);

int _MPIU_Barrier(const char * fn, const int ln, MPI_Comm comm);


/* Fancy barrier which warns if there is a lot of imbalance. */
#define MPIU_Barrier(comm) _MPIU_Barrier(__FILE__, __LINE__, comm)

/* Structure for asynchronous communication routines in treewalk and exchange.
 *Stores an array of MPI requests, the task each one is from, and a char buffer
 * for the data associated with them.*/
struct CommBuffer
{
    char * databuf; /* Bytes to store the received/sent data.*/
    size_t databufsize; /* Number of bytes allocated in databuf*/
    int * displs; /* Displacement of each request's data structure in the above buffer*/
    int * rqst_task; /* Array storing task number for each request in rdata_all*/
    MPI_Request * rdata_all; /* Array of requests*/
    int nrequest_all; /* Number of requests that had MPI_Irecv/MPI_ISend called on them*/
    int totcomplete; /* Number of completed requests*/
};
#endif //_UTILS_SYSTEM_H
