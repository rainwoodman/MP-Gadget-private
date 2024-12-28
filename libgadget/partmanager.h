#ifndef _PART_DATA_H
#define _PART_DATA_H

#include "types.h"
#include "utils/peano.h"
#include "utils/system.h"

/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
struct particle_data
{
    double Pos[3];   /*!< particle position at its current time */
    /* TopLeaf this particle belongs to. Set to find destinations in the exchange. Used to accelerate the tree build.
     During fof particle exchange this actually points to the target task, not the topleaf.*/
    int TopLeaf;
    float Mass;     /*!< particle mass */
    int PI; /* particle property index; used by BH, SPH and STAR.
                        points to the corresponding structure in (SPH|BH|STAR)P array.*/
    struct {
        unsigned int IsGarbage            :1; /* True for a garbage particle. readonly: Use slots_mark_garbage to mark this.*/
        unsigned int Swallowed            :1; /* True if the particle is a black hole which has been swallowed; these particles stay around so we have a merger tree.*/
        unsigned int HeIIIionized        :1; /* True if the particle has undergone helium reionization.*/
        unsigned int BHHeated              :1; /* Flags that particle was heated by a BH this timestep*/
        unsigned char Generation : 4; /* How many particles it has spawned; used to generate unique particle ID.
                                     We limit to sfr_params.Generations + 1 and enforce at max fitting into 4 bits in sfr_params. */
        unsigned char TimeBinHydro; /* Time step bin for hydro; 0 for unassigned. Must be smaller than the gravity timebin.
                                     * Star formation, cooling, and BH accretion takes place on the hydro timestep.*/
        unsigned char TimeBinGravity; /* Time step bin for gravity; 0 for unassigned.*/
        /* particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
        unsigned char Type;
        /* (jdavies): I moved this out of the bitfield because i need to access it by pointer in petapm.c
         * This could also be done by passing a struct pointer instead of void* as the petapm pstruct */
    };
    /* Cacheline is here: above data is needed for the treebuild*/
    MyFloat Vel[3];   /* particle velocity at its current time */
#ifdef SIDM
    MyFloat SIDMAccel[3]; /* DM Acceleration from self-interactions*/
#endif
    MyFloat FullTreeGravAccel[3]; /* Short-range tree acceleration at the most recent timestep
                                 which included all particles (ie, PM steps). Does not include PM acceleration.
                                 At time of writing this
                                 is used to test whether the particles are bound during
                                 black hole mergers for black holes, for predicted velocities for SPH particles
                                 and for predicting velocities for wind particle velocity dispersions.
                                 For non-hierarchical gravity this stores the gravitational acceleration from the current timestep.
                                 Note that changes during a short timestep this may not be noticed immediately, as
                                 the acceleration is not updated. On short timesteps gravitational
                                 accelerations are only from other active particles.
                                 * For SPH predicted velocities, we will be using a slightly out of date gravitational acceleration,
                                 * but the Gadget-4 paper says this is a negligible effect (I suspect that where the artificial viscosity
                                 * is important the gravitational acceleration is small compared to hydro force anyway).
                                 */
    MyFloat GravPM[3];      /* particle acceleration due to long-range PM gravity force */
    inttime_t Ti_drift;       /*!< current time of the particle position. The same for all particles. */
    MyFloat Hsml;
    /* Cacheline is here: data above needed for kick*/
    /* DtHsml is 1/3 DivVel * Hsml evaluated at the last active timestep for this particle.
     * This predicts Hsml during the current timestep in the way used in Gadget-4, more accurate
     * than the Gadget-2 prediction which could run away in deep timesteps. Used also
     * to limit timesteps by density change. */
    MyFloat DtHsml;
    MyIDType ID;
    /* FOF Group number: only has meaning during FOF.*/
    /* Transient but hard to move to private arrays because it needs to 
     * travel with the particle during exchange*/
    int64_t GrNr;
    MyFloat Potential;		/* Gravitational potential. This is the total potential only on a PM timestep,
                             * after gravtree+gravpm is called. We do not save the potential on short timesteps
                             * for hierarchical gravity as it would only be from active particles.*/
#ifdef DEBUG
    /* Kick times for both hydro and grav*/
    inttime_t Ti_kick_hydro;
    inttime_t Ti_kick_grav;
#endif
};

extern struct part_manager_type {
    struct particle_data *Base; /* Pointer to particle data on local processor. */
    /*!< number of particles on the LOCAL processor: number of valid entries in P array. */
    int64_t NumPart;
    /*!< Amount of memory we have available for particles locally: maximum size of P array. */
    int64_t MaxPart;
    /* Random shift applied to the box. This is changed
     * every domain decomposition to prevent correlated
     * errors building up in the tree force. */
    double CurrentParticleOffset[3];
    /* Current box size so we can work out periodic boundaries*/
    double BoxSize;
} PartManager[1];

/*Compatibility define*/
#define P PartManager->Base

/*Allocate memory for the particles*/
void particle_alloc_memory(struct part_manager_type * PartManager, double BoxSize, int64_t MaxPart);

/* Updates the global storing the current random offset of the particles,
 * and stores the relative offset from the last random offset in rel_random_shift.
 * RandomParticleOffset is the max adjustment as a fraction of the box. */
void update_random_offset(struct part_manager_type * PartManager, double * rel_random_shift, double RandomParticleOffset, const uint64_t seed);

/* Finds the correct relative position accounting for periodicity*/
#define NEAREST(x, BoxSize) (((x)>0.5*BoxSize)?((x)-BoxSize):(((x)<-0.5*BoxSize)?((x)+BoxSize):(x)))

static inline double DMAX(double a, double b) {
    if(a > b) return a;
    return b;
}
static inline double DMIN(double a, double b) {
    if(a < b) return a;
    return b;
}
#endif
