#ifndef HYDRA_H
#define HYDRA_H

#include "forcetree.h"
#include "types.h"
#include "timestep.h"
#include "density.h"
#include "utils/paramset.h"

/*Function to compute SIDM acceleration.*/
void sidm_force(const ActiveParticles * act, const double atime, Cosmology * CP, const ForceTree * const tree);

void set_sidm_params(ParameterSet * ps);

#endif
