#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "allvars.h"
#include "mymalloc.h"
#include "memory.h"
#include "system.h"
#include "endrun.h"

/* The main allocator is used to store large objects, e.g. tree, toptree */
Allocator A_MAIN[1];

/* The temp allocator is used to store objects that lives on the stack;
 * replacing alloca and similar cases to avoid stack induced memory fragmentation
 * */
Allocator A_TEMP[1];

void
mymalloc_init(double MaxMemSizePerNode)
{
    int Nhost = cluster_get_num_hosts();

    size_t n = 1.0 * MaxMemSizePerNode * (1.0 * Nhost / NTask) * 1024 * 1024;

    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_init(A_MAIN, "MAIN", n, 1), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the MAIN allocator on at least one nodes."
                  "Requestion %td bytes. Try reducing MaxMemSizePerNode. Also check the node health status.\n", n);
    }

    n = 1024 * NTask; /* reserve 1K per task for the TEMP storage */

    if (MPIU_Any(ALLOC_ENOMEMORY == allocator_init(A_TEMP, "TEMP", n, 1), MPI_COMM_WORLD)) {
        endrun(0, "Insufficient memory for the TEMP allocator on at least one nodes."
                  "Requestion %td bytes. Try reducing MaxMemSizePerNode. Also check the node health status.\n", n);

    }
}

void report_detailed_memory_usage(const char *label, const char * fmt, ...)
{
    va_list va;
    char buf[4096];
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    if (ThisTask == 0) {
        message(1, "%s\n", buf);
        allocator_print(A_MAIN);
    }
}
