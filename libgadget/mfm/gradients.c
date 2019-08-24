#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"
#ifdef PTHREADS_NUM_THREADS
#include <pthread.h>
#endif



/*! \file gradients.c
 *  \brief calculate gradients of hydro quantities
 *
 *  This file contains the "second hydro loop", where the gas hydro quantity
 *   gradients are calculated. All gradients now use the second-order accurate
 *   moving-least-squares formulation, and are calculated here consistently.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


#define ASSIGN_ADD_PRESET(x,y,mode) (x+=y)
#define MINMAX_CHECK(x,xmin,xmax) ((x<xmin)?(xmin=x):((x>xmax)?(xmax=x):(1)))
#define SHOULD_I_USE_SPH_GRADIENTS(condition_number) ((condition_number > CONDITION_NUMBER_DANGER) ? (1):(0))



#if defined(MHD_CONSTRAINED_GRADIENT)
#if (MHD_CONSTRAINED_GRADIENT > 1)
#define NUMBER_OF_GRADIENT_ITERATIONS 3
#else
#define NUMBER_OF_GRADIENT_ITERATIONS 2
#endif
#else
#define NUMBER_OF_GRADIENT_ITERATIONS 1
#endif

#if defined(RT_EVOLVE_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_NGAMMA)
#define E_gamma_Pred E_gamma
#endif


#ifdef PTHREADS_NUM_THREADS
extern pthread_mutex_t mutex_nexport;
extern pthread_mutex_t mutex_partnodedrift;
#define LOCK_NEXPORT     pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT   pthread_mutex_unlock(&mutex_nexport);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#endif

#define NV_MYSIGN(x) (( x > 0 ) - ( x < 0 ))

/* define a common 'gradients' structure to hold
 everything we're going to take derivatives of */
struct Quantities_for_Gradients
{
    MyDouble Density;
    MyDouble Pressure;
    MyDouble Velocity[3];
};

struct kernel_GasGrad
{
    double dp[3],r,wk_i, wk_j, dwk_i, dwk_j,h_i;
};

struct GasGraddata_in
{
    MyDouble Pos[3];
    MyFloat Mass;
    MyFloat Hsml;
    integertime Timestep;
    int NodeList[NODELISTLENGTH];
    struct Quantities_for_Gradients GQuant;

}
*GasGradDataIn, *GasGradDataGet;



struct GasGraddata_out
{
    struct Quantities_for_Gradients Gradients[3];
    struct Quantities_for_Gradients Maxima;
    struct Quantities_for_Gradients Minima;
    MyFloat MaxDistance;
}
*GasGradDataResult, *GasGradDataOut;



struct GasGraddata_out_iter
{
    MyFloat dummy;
}
*GasGradDataResult_iter, *GasGradDataOut_iter;



/* this is a temporary structure for quantities used ONLY in the loop below,
 for example for computing the slope-limiters (for the Reimann problem) */
static struct temporary_data_topass
{
    struct Quantities_for_Gradients Maxima;
    struct Quantities_for_Gradients Minima;
    MyFloat MaxDistance;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    MyFloat GlassAcc[3];
#endif
}
*GasGradDataPasser;



static inline void particle2in_GasGrad(struct GasGraddata_in *in, int i, int gradient_iteration);
static inline void out2particle_GasGrad(struct GasGraddata_out *out, int i, int mode, int gradient_iteration);
static inline void out2particle_GasGrad_iter(struct GasGraddata_out_iter *out, int i, int mode, int gradient_iteration);



static inline void particle2in_GasGrad(struct GasGraddata_in *in, int i, int gradient_iteration)
{
    int k;
    for(k = 0; k < 3; k++)
        in->Pos[k] = P[i].Pos[k];
    in->Hsml = PPP[i].Hsml;
    in->Mass = P[i].Mass;
    if(in->Mass < 0) {in->Mass = 0;}

    if(SHOULD_I_USE_SPH_GRADIENTS(SphP[i].ConditionNumber)) {in->Mass *= -1;}
    in->Timestep = (P[i].TimeBin ? (((integertime) 1) << P[i].TimeBin) : 0);
    if(gradient_iteration == 0)
    {
        in->GQuant.Density = SphP[i].Density;
        in->GQuant.Pressure = SphP[i].Pressure;
        for(k = 0; k < 3; k++)
            in->GQuant.Velocity[k] = SphP[i].VelPred[k];
    } // gradient_iteration == 0
}



//#define MAX_ADD(x,y,mode) (mode == 0 ? (x=y) : (((x)<(y)) ? (x=y) : (x))) // these definitions applied before the symmetric re-formulation of this routine
//#define MIN_ADD(x,y,mode) (mode == 0 ? (x=y) : (((x)>(y)) ? (x=y) : (x)))
#define MAX_ADD(x,y,mode) ((y > x) ? (x = y) : (1)) // simpler definition now used
#define MIN_ADD(x,y,mode) ((y < x) ? (x = y) : (1))


static inline void out2particle_GasGrad_iter(struct GasGraddata_out_iter *out, int i, int mode, int gradient_iteration)
{
}



static inline void out2particle_GasGrad(struct GasGraddata_out *out, int i, int mode, int gradient_iteration)
{
    
    if(gradient_iteration == 0)
    {
        int j,k;
        MAX_ADD(GasGradDataPasser[i].MaxDistance,out->MaxDistance,mode);
        
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
        for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(GasGradDataPasser[i].GlassAcc[k],out->GlassAcc[k],mode);}
#endif
        MAX_ADD(GasGradDataPasser[i].Maxima.Density,out->Maxima.Density,mode);
        MIN_ADD(GasGradDataPasser[i].Minima.Density,out->Minima.Density,mode);
        MAX_ADD(GasGradDataPasser[i].Maxima.Pressure,out->Maxima.Pressure,mode);
        MIN_ADD(GasGradDataPasser[i].Minima.Pressure,out->Minima.Pressure,mode);
        for(k=0;k<3;k++)
        {
            ASSIGN_ADD_PRESET(SphP[i].Gradients.Density[k],out->Gradients[k].Density,mode);
            ASSIGN_ADD_PRESET(SphP[i].Gradients.Pressure[k],out->Gradients[k].Pressure,mode);
        }
        for(j=0;j<3;j++)
        {
            MAX_ADD(GasGradDataPasser[i].Maxima.Velocity[j],out->Maxima.Velocity[j],mode);
            MIN_ADD(GasGradDataPasser[i].Minima.Velocity[j],out->Minima.Velocity[j],mode);
            for(k=0;k<3;k++)
            {
                ASSIGN_ADD_PRESET(SphP[i].Gradients.Velocity[j][k],out->Gradients[k].Velocity[j],mode);
            }
        }
    } // gradient_iteration == 0
}




void local_slopelimiter(double *grad, double valmax, double valmin, double alim, double h, double shoot_tol)
{
    int k;
    double d_abs = 0.0;
    for(k=0;k<3;k++) {d_abs += grad[k]*grad[k];}
    if(d_abs > 0)
    {
        double cfac = 1 / (alim * h * sqrt(d_abs));
        double fabs_max = fabs(valmax);
        double fabs_min = fabs(valmin);
        double abs_min = DMIN(fabs_max,fabs_min);
        if(shoot_tol > 0)
        {
            double abs_max = DMAX(fabs_max,fabs_min);
            cfac *= DMIN(abs_min + shoot_tol*abs_max, abs_max);
        } else {
            cfac *= abs_min;
        }
        if(cfac < 1) {for(k=0;k<3;k++) {grad[k] *= cfac;}}
    }
}

void construct_gradient(double *grad, int i);

void construct_gradient(double *grad, int i)
{
    /* check if the matrix is well-conditioned: otherwise we will use the 'standard SPH-like' derivative estimation */
    if(SHOULD_I_USE_SPH_GRADIENTS(SphP[i].ConditionNumber))
    {
        /* the condition number was bad, so we used SPH-like gradients */
        int k; for(k=0;k<3;k++) {grad[k] *= PPP[i].DhsmlNgbFactor / SphP[i].Density;}
    } else {
        /* ok, the condition number was good so we used the matrix-like gradient estimator */
        int k; double v_tmp[3];
        for(k=0;k<3;k++) {v_tmp[k] = grad[k];}
        for(k=0;k<3;k++) {grad[k] = SphP[i].NV_T[k][0]*v_tmp[0] + SphP[i].NV_T[k][1]*v_tmp[1] + SphP[i].NV_T[k][2]*v_tmp[2];}
    }
}




void hydro_gradient_calc(void)
{
    int i, j, k, k1, ngrp, ndone, ndone_flag;
    int recvTask, place;
    double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 = 0, timewait2 = 0;
    double timecomp, timecomm, timewait, tstart, tend, t0, t1;
    int save_NextParticle;
    long long n_exported = 0;
 
    /* allocate buffers to arrange communication */
    long long NTaskTimesNumPart;
    GasGradDataPasser = (struct temporary_data_topass *) mymalloc("GasGradDataPasser",N_gas * sizeof(struct temporary_data_topass));
    NTaskTimesNumPart = maxThreads * NumPart;
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                                             sizeof(struct GasGraddata_in) +
                                                             sizeof(struct GasGraddata_out) +
                                                             sizemax(sizeof(struct GasGraddata_in),sizeof(struct GasGraddata_out))));
    CPU_Step[CPU_DENSMISC] += measure_time();
    t0 = my_second();
    Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    
    /* before doing any operations, need to zero the appropriate memory so we can correctly do pair-wise operations */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type==0)
        {
            int k2;
            memset(&GasGradDataPasser[i], 0, sizeof(struct temporary_data_topass));
            /* and zero out the gradients structure itself */
            for(k=0;k<3;k++)
            {
                SphP[i].Gradients.Density[k] = 0;
                SphP[i].Gradients.Pressure[k] = 0;
                for(k2=0;k2<3;k2++) {SphP[i].Gradients.Velocity[k2][k] = 0;}
            }
        }
    
    
    
    /* prepare to do the requisite number of sweeps over the particle distribution */
    int gradient_iteration;
    for(gradient_iteration = 0; gradient_iteration < NUMBER_OF_GRADIENT_ITERATIONS; gradient_iteration++)
    {
        // now we actually begin the main gradient loop //
        NextParticle = FirstActiveParticle;	/* begin with this index */
        do
        {
            
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;
            
            for(j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }
            
            /* do local particles and prepare export list */
            tstart = my_second();
            
#ifdef PTHREADS_NUM_THREADS
            pthread_t mythreads[PTHREADS_NUM_THREADS - 1];
            int threadid[PTHREADS_NUM_THREADS - 1];
            pthread_attr_t attr;
            
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            pthread_mutex_init(&mutex_nexport, NULL);
            pthread_mutex_init(&mutex_partnodedrift, NULL);
            
            TimerFlag = 0;
            
            for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            {
                threadid[j] = j + 1;
                pthread_create(&mythreads[j], &attr, GasGrad_evaluate_primary, &threadid[j]);
            }
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else
                int mainthreadid = 0;
#endif
                GasGrad_evaluate_primary(&mainthreadid, gradient_iteration);	/* do local particles and prepare export list */
            }
            
#ifdef PTHREADS_NUM_THREADS
            for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
                pthread_join(mythreads[j], NULL);
#endif
            
            
            tend = my_second();
            timecomp1 += timediff(tstart, tend);
            
            if(BufferFullFlag)
            {
                int last_nextparticle = NextParticle;
                
                NextParticle = save_NextParticle;
                
                while(NextParticle >= 0)
                {
                    if(NextParticle == last_nextparticle)
                        break;
                    
                    if(ProcessedFlag[NextParticle] != 1)
                        break;
                    
                    ProcessedFlag[NextParticle] = 2;
                    
                    NextParticle = NextActiveParticle[NextParticle];
                }
                
                if(NextParticle == save_NextParticle)
                {
                    /* in this case, the buffer is too small to process even a single particle */
                    endrun(113308);
                }
                
                int new_export = 0;
                
                for(j = 0, k = 0; j < Nexport; j++)
                    if(ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if(k < j + 1)
                            k = j + 1;
                        
                        for(; k < Nexport; k++)
                            if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                            {
                                int old_index = DataIndexTable[j].Index;
                                
                                DataIndexTable[j] = DataIndexTable[k];
                                DataNodeList[j] = DataNodeList[k];
                                DataIndexTable[j].IndexGet = j;
                                new_export++;
                                
                                DataIndexTable[k].Index = old_index;
                                k++;
                                break;
                            }
                    }
                    else
                        new_export++;
                
                Nexport = new_export;
                
            }
            
            n_exported += Nexport;
            
            for(j = 0; j < NTask; j++)
                Send_count[j] = 0;
            for(j = 0; j < Nexport; j++)
                Send_count[DataIndexTable[j].Task]++;
            
            MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
            
            tstart = my_second();
            
            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
            
            tend = my_second();
            timewait1 += timediff(tstart, tend);
            
            for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
            {
                Nimport += Recv_count[j];
                
                if(j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }
            
            GasGradDataGet = (struct GasGraddata_in *) mymalloc("GasGradDataGet", Nimport * sizeof(struct GasGraddata_in));
            GasGradDataIn = (struct GasGraddata_in *) mymalloc("GasGradDataIn", Nexport * sizeof(struct GasGraddata_in));
            
            /* prepare particle data for export */
            
            for(j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                particle2in_GasGrad(&GasGradDataIn[j], place, gradient_iteration);
                memcpy(GasGradDataIn[j].NodeList,
                       DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
            }
            
            /* exchange particle data */
            tstart = my_second();
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;
                
                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* get the particles */
                        MPI_Sendrecv(&GasGradDataIn[Send_offset[recvTask]],
                                     Send_count[recvTask] * sizeof(struct GasGraddata_in), MPI_BYTE,
                                     recvTask, TAG_GRADLOOP_A,
                                     &GasGradDataGet[Recv_offset[recvTask]],
                                     Recv_count[recvTask] * sizeof(struct GasGraddata_in), MPI_BYTE,
                                     recvTask, TAG_GRADLOOP_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            tend = my_second();
            timecommsumm1 += timediff(tstart, tend);
            
            myfree(GasGradDataIn);
            if(gradient_iteration==0)
            {
                GasGradDataResult = (struct GasGraddata_out *) mymalloc("GasGradDataResult", Nimport * sizeof(struct GasGraddata_out));
                GasGradDataOut = (struct GasGraddata_out *) mymalloc("GasGradDataOut", Nexport * sizeof(struct GasGraddata_out));
                report_memory_usage(&HighMark_GasGrad, "GRADIENTS_LOOP");
            } else {
                GasGradDataResult_iter = (struct GasGraddata_out_iter *) mymalloc("GasGradDataResult_iter", Nimport * sizeof(struct GasGraddata_out_iter));
                GasGradDataOut_iter = (struct GasGraddata_out_iter *) mymalloc("GasGradDataOut_iter", Nexport * sizeof(struct GasGraddata_out_iter));
            }
            
            /* now do the particles that were sent to us */
            tstart = my_second();
            NextJ = 0;
            
#ifdef PTHREADS_NUM_THREADS
            for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
                pthread_create(&mythreads[j], &attr, GasGrad_evaluate_secondary, &threadid[j]);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else
                int mainthreadid = 0;
#endif
                GasGrad_evaluate_secondary(&mainthreadid, gradient_iteration);
            }
            
#ifdef PTHREADS_NUM_THREADS
            for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
                pthread_join(mythreads[j], NULL);
            
            pthread_mutex_destroy(&mutex_partnodedrift);
            pthread_mutex_destroy(&mutex_nexport);
            pthread_attr_destroy(&attr);
#endif
            
            tend = my_second();
            timecomp2 += timediff(tstart, tend);
            
            if(NextParticle < 0)
                ndone_flag = 1;
            else
                ndone_flag = 0;
            
            tstart = my_second();
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            tend = my_second();
            timewait2 += timediff(tstart, tend);
            
            /* get the result */
            tstart = my_second();
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;
                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* send the results */
                        if(gradient_iteration==0)
                        {
                            MPI_Sendrecv(&GasGradDataResult[Recv_offset[recvTask]],
                                         Recv_count[recvTask] * sizeof(struct GasGraddata_out),
                                         MPI_BYTE, recvTask, TAG_GRADLOOP_B,
                                         &GasGradDataOut[Send_offset[recvTask]],
                                         Send_count[recvTask] * sizeof(struct GasGraddata_out),
                                         MPI_BYTE, recvTask, TAG_GRADLOOP_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } else {
                            MPI_Sendrecv(&GasGradDataResult_iter[Recv_offset[recvTask]],
                                         Recv_count[recvTask] * sizeof(struct GasGraddata_out_iter),
                                         MPI_BYTE, recvTask, TAG_GRADLOOP_C,
                                         &GasGradDataOut_iter[Send_offset[recvTask]],
                                         Send_count[recvTask] * sizeof(struct GasGraddata_out_iter),
                                         MPI_BYTE, recvTask, TAG_GRADLOOP_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }
                }
            }
            tend = my_second();
            timecommsumm2 += timediff(tstart, tend);
            
            /* add the result to the local particles */
            tstart = my_second();
            for(j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                if(gradient_iteration==0)
                {
                    out2particle_GasGrad(&GasGradDataOut[j], place, 1, gradient_iteration);
                } else {
                    out2particle_GasGrad_iter(&GasGradDataOut_iter[j], place, 1, gradient_iteration);
                }
            }
            tend = my_second();
            timecomp1 += timediff(tstart, tend);
            if(gradient_iteration==0)
            {
                myfree(GasGradDataOut);
                myfree(GasGradDataResult);
            } else {
                myfree(GasGradDataOut_iter);
                myfree(GasGradDataResult_iter);
            }
            myfree(GasGradDataGet);
        }
        while(ndone < NTask);
        
        
        /* here, we insert intermediate operations on the results, from the iterations we have completed */
    } // closes gradient_iteration
    
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
    
    
    /* do final operations on results: these are operations that can be done after the complete set of iterations */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type == 0)
        {
            /* now we can properly calculate (second-order accurate) gradients of hydrodynamic quantities from this loop */
            construct_gradient(SphP[i].Gradients.Density,i);
            construct_gradient(SphP[i].Gradients.Pressure,i);
            for(k=0;k<3;k++) {construct_gradient(SphP[i].Gradients.Velocity[k],i);}
            /* now the gradients are calculated: below are simply useful operations on the results */
            
#ifdef HYDRO_SPH
            /* compute the traditional Balsara limiter (now that we have velocity gradients) */
            double divVel = All.cf_a2inv * fabs(SphP[i].Gradients.Velocity[0][0] + SphP[i].Gradients.Velocity[1][1] + SphP[i].Gradients.Velocity[2][2]);
            if(All.ComovingIntegrationOn) divVel += 3*All.cf_hubble_a; // hubble-flow correction added (physical units)
            double CurlVel[3];
            double MagCurl;
            CurlVel[0] = SphP[i].Gradients.Velocity[1][2] - SphP[i].Gradients.Velocity[2][1];
            CurlVel[1] = SphP[i].Gradients.Velocity[2][0] - SphP[i].Gradients.Velocity[0][2];
            CurlVel[2] = SphP[i].Gradients.Velocity[0][1] - SphP[i].Gradients.Velocity[1][0];
            MagCurl = All.cf_a2inv * sqrt(CurlVel[0]*CurlVel[0] + CurlVel[1]*CurlVel[1] + CurlVel[2]*CurlVel[2]);
            double fac_mu = 1 / (All.cf_afac3 * All.cf_atime);
            SphP[i].alpha_limiter = divVel / (divVel + MagCurl + 0.0001 * Particle_effective_soundspeed_i(i) /
                                              (Get_Particle_Size(i)) / fac_mu);
#endif

            /* finally, we need to apply a sensible slope limiter to the gradients, to prevent overshooting */
            double stol = 0.0;
            double stol_tmp, stol_diffusion;
            stol_diffusion = 0.1; stol_tmp = stol;
            double h_lim = PPP[i].Hsml;
//#if (defined(MAGNETIC) && defined(COOLING)) ||
            h_lim = DMAX(PPP[i].Hsml,GasGradDataPasser[i].MaxDistance);
//#else
//            h_lim = DMIN(GasGradDataPasser[i].MaxDistance , 4.0*PPP[i].Hsml);
//#endif
            /* fraction of H at which maximum reconstruction is allowed (=0.5 for 'standard'); for pure hydro we can
             be a little more aggresive and the equations are still stable (but this is as far as you want to push it) */
            double a_limiter = 0.25; if(SphP[i].ConditionNumber>100) a_limiter=DMIN(0.5, 0.25 + 0.25 * (SphP[i].ConditionNumber-100)/100);
#if (SLOPE_LIMITER_TOLERANCE == 2)
            h_lim = PPP[i].Hsml; a_limiter *= 0.5; stol = 0.125;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 0)
            a_limiter *= 2.0; stol = 0.0;
#endif

            local_slopelimiter(SphP[i].Gradients.Density,GasGradDataPasser[i].Maxima.Density,GasGradDataPasser[i].Minima.Density,a_limiter,h_lim,0);
            local_slopelimiter(SphP[i].Gradients.Pressure,GasGradDataPasser[i].Maxima.Pressure,GasGradDataPasser[i].Minima.Pressure,a_limiter,h_lim,stol);
            stol_tmp = stol;
            for(k1=0;k1<3;k1++) {local_slopelimiter(SphP[i].Gradients.Velocity[k1],GasGradDataPasser[i].Maxima.Velocity[k1],GasGradDataPasser[i].Minima.Velocity[k1],a_limiter,h_lim,stol_tmp);}

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
            /* if the mesh motion is specified to be glass-generating, this is where we apply the appropriate mesh velocity */
            if(All.Time > 0)
            {
                double cs_invelunits = Particle_effective_soundspeed_i(i) * All.cf_afac3 * All.cf_atime; // soundspeed, converted to units of code velocity
                double L_i_code = Get_Particle_Size(i); // particle effective size (in code units)
                double dvel[3]={0}, velnorm=0; for(k=0;k<3;k++) {dvel[k] = L_i_code*L_i_code*GasGradDataPasser[i].GlassAcc[k]; velnorm += dvel[k]*dvel[k];} // calculate quantities to use for glass
                double dtx = P[i].dt_step * All.Timebase_interval / All.cf_hubble_a; // need timestep for limiter below
                if(velnorm > 0 && dtx > 0)
                {
                    velnorm = sqrt(velnorm); // normalization for glass 'force'
                    double v00 = 0.5 * DMIN(cs_invelunits*(0.5*velnorm) , All.CourantFac*(L_i_code/dtx)/All.cf_a2inv); // limit added velocity of mesh-generating point to Courant factor
                    for(k=0;k<3;k++) {SphP[i].ParticleVel[k] += v00 * (dvel[k]/velnorm);} // actually add the correction velocity to the mesh velocity
                }
            }
#endif
        }
    

    /* free the temporary structure we created for the MinMax and additional data passing */
    myfree(GasGradDataPasser);
    
    /* collect some timing information */
    t1 = WallclockTime = my_second();
    timeall += timediff(t0, t1);
    timecomp = timecomp1 + timecomp2;
    timewait = timewait1 + timewait2;
    timecomm = timecommsumm1 + timecommsumm2;
    
    CPU_Step[CPU_DENSCOMPUTE] += timecomp;
    CPU_Step[CPU_DENSWAIT] += timewait;
    CPU_Step[CPU_DENSCOMM] += timecomm;
    CPU_Step[CPU_DENSMISC] += timeall - (timecomp + timewait + timecomm);
}


int GasGrad_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                     int *ngblist, int gradient_iteration)
{
    int startnode, numngb, listindex = 0;
    int j, k, k2, n, swap_to_j;
    double hinv, hinv3, hinv4, r2, u, hinv_j, hinv3_j, hinv4_j;
    struct kernel_GasGrad kernel;
    struct GasGraddata_in local;
    struct GasGraddata_out out;
    struct GasGraddata_out_iter out_iter;
    if(gradient_iteration==0)
    {
        memset(&out, 0, sizeof(struct GasGraddata_out));
    } else {
        memset(&out_iter, 0, sizeof(struct GasGraddata_out_iter));
    }
    memset(&kernel, 0, sizeof(struct kernel_GasGrad));
    
    if(mode == 0)
        particle2in_GasGrad(&local, target, gradient_iteration);
    else
        local = GasGradDataGet[target];
    
    /* check if we should bother doing a neighbor loop */
    if(local.Hsml <= 0) return 0;
    if(local.Mass == 0) return 0;
    if(gradient_iteration == 0)
        if(local.GQuant.Density <= 0) return 0;
    
    /* now set particle-i centric quantities so we don't do it inside the loop */
    kernel.h_i = local.Hsml;
    double h2_i = kernel.h_i*kernel.h_i;
    kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
    int sph_gradients_flag_i = 0;
    int sph_gradients_flag_j = 0;
    if(local.Mass < 0) {sph_gradients_flag_i=1; local.Mass*=-1;}
    double V_i;
    V_i = local.Mass / local.GQuant.Density;
    
    int kernel_mode_i = -1; // only need to calculate wk, by default
    if(sph_gradients_flag_i) kernel_mode_i = 0; // for sph, only need dwk
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
    kernel_mode_i = 0; // for some circumstances, we require both wk and dwk //
#endif
    
    /* Now start the actual neighbor computation for this particle */
    
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = GasGradDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            if(numngb < 0)
                return -1;
            
            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;
                if(j >= N_gas) continue;

                integertime TimeStep_J = (P[j].TimeBin ? (((integertime) 1) << P[j].TimeBin) : 0);
#ifndef BOX_SHEARING // (shearing box means the fluxes at the boundaries are not actually symmetric, so can't do this) //
                if(local.Timestep > TimeStep_J) continue; /* compute from particle with smaller timestep */
                /* use relative positions to break degeneracy */
                if(local.Timestep == TimeStep_J)
                {
                    int n0=0; if(local.Pos[n0] == P[j].Pos[n0]) {n0++; if(local.Pos[n0] == P[j].Pos[n0]) n0++;}
                    if(local.Pos[n0] < P[j].Pos[n0]) continue;
                }
                swap_to_j = TimeBinActive[P[j].TimeBin];
#else
                swap_to_j = 0;
#endif
                if(P[j].Mass <= 0) continue;
                if(SphP[j].Density <= 0) continue;
                
                kernel.dp[0] = local.Pos[0] - P[j].Pos[0];
                kernel.dp[1] = local.Pos[1] - P[j].Pos[1];
                kernel.dp[2] = local.Pos[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC			/*  now find the closest image in the given box size  */
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
#endif
                r2 = kernel.dp[0] * kernel.dp[0] + kernel.dp[1] * kernel.dp[1] + kernel.dp[2] * kernel.dp[2];
                double h_j = PPP[j].Hsml;
#if !defined(HYDRO_SPH) && !defined(KERNEL_CRK_FACES)
                if(r2 <= 0) continue;
#else
                if(r2 <= 0) {swap_to_j = 0;}
#endif
#ifdef TURB_DIFF_DYNAMIC
#ifdef GALSF_SUBGRID_WINDS
                if (gradient_iteration == 0 && ((SphP[j].DelayTime == 0 && local.DelayTime == 0) || (SphP[j].DelayTime > 0 && local.DelayTime > 0))) {
#else
                if (gradient_iteration == 0) {
#endif
                    hhat_i = All.TurbDynamicDiffFac * kernel.h_i; hhat_j = All.TurbDynamicDiffFac * h_j;
                    if((r2 >= (hhat_i * hhat_i)) && (r2 >= (hhat_j * hhat_j))) continue;
                    double h_avg = 0.5 * (hhat_i + hhat_j), particle_distance = sqrt(r2);
                    kernel_hinv(h_avg, &hhatinv_i, &hhatinv3_i, &hhatinv4_i); u = DMIN(particle_distance * hhatinv_i, 1.0);
                    kernel_main(u, hhatinv3_i, hhatinv4_i, &wkhat_i, &dwkhat_i, 0); /* wkhat is symmetric in this case W_{ij} = W_{ji} */
                    double mean_weight = wkhat_i * 0.5 * (SphP[j].Norm_hat + local.Norm_hat) / (local.Norm_hat * SphP[j].Norm_hat);
                    double weight_i = P[j].Mass * mean_weight, weight_j = local.Mass * mean_weight, Velocity_bar_diff[3];
                    if(particle_distance < h_avg) {
                        for(k=0;k<3;k++) {Velocity_bar_diff[k] = SphP[j].Velocity_bar[k] - local.GQuant.Velocity_bar[k]; out.Velocity_hat[k] += Velocity_bar_diff[k] * weight_i;}
                        if(swap_to_j) {for(k=0;k<3;k++) {SphP[j].Velocity_hat[k] -= Velocity_bar_diff[k] * weight_j;}}
                    }
                } /* closes gradient_iteration == 0 */
#endif
                if((r2 >= h2_i) && (r2 >= h_j * h_j)) continue;
                
                kernel.r = sqrt(r2);
                if(kernel.r < kernel.h_i)
                {
                    u = kernel.r * hinv;
                    kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, kernel_mode_i);
                }
                else
                {
                    kernel.dwk_i = kernel.wk_i = 0;
                }
#if defined(MHD_CONSTRAINED_GRADIENT) || defined(KERNEL_CRK_FACES)
                if(kernel.r < h_j)
#else
                if((kernel.r < h_j) && (swap_to_j))
#endif
                {
                    /* ok, we need the j-particle weights, but first check what kind of gradient we are calculating */
                    sph_gradients_flag_j = SHOULD_I_USE_SPH_GRADIENTS(SphP[j].ConditionNumber);
                    int kernel_mode_j;
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
                    kernel_mode_j = 0; // for some circumstances, we require both wk and dwk //
#else
                    if(sph_gradients_flag_j) {kernel_mode_j=0;} else {kernel_mode_j=-1;}
#endif
                    kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
                    u = kernel.r * hinv_j;
                    kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, kernel_mode_j);
                }
                else
                {
                    kernel.dwk_j = kernel.wk_j = 0;
                }
                
                
#if defined(MHD_CONSTRAINED_GRADIENT)
                double V_j = P[j].Mass / SphP[j].Density;
                double Face_Area_Vec[3];
                double wt_i,wt_j;
#ifdef COOLING
                //wt_i=wt_j = 2.*V_i*V_j / (V_i + V_j); // more conservatively, could use DMIN(V_i,V_j), but that is less accurate
                if((fabs(V_i-V_j)/DMIN(V_i,V_j))/NUMDIMS > 1.25) {wt_i=wt_j=2.*V_i*V_j/(V_i+V_j);} else {wt_i=V_i; wt_j=V_j;}
#else
                //wt_i=wt_j = (V_i*PPP[j].Hsml + V_j*local.Hsml) / (local.Hsml+PPP[j].Hsml); // should these be H, or be -effective sizes- //
                if((fabs(V_i-V_j)/DMIN(V_i,V_j))/NUMDIMS > 1.50) {wt_i=wt_j=(V_i*PPP[j].Hsml+V_j*local.Hsml)/(local.Hsml+PPP[j].Hsml);} else {wt_i=V_i; wt_j=V_j;}
#endif
                for(k=0;k<3;k++)
                {
                    /* calculate the face area between the particles (must match what is done in the actual hydro routine! */
                    Face_Area_Vec[k] = kernel.wk_i * wt_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2])
                                     + kernel.wk_j * wt_j * (SphP[j].NV_T[k][0]*kernel.dp[0] + SphP[j].NV_T[k][1]*kernel.dp[1] + SphP[j].NV_T[k][2]*kernel.dp[2]);
                    if(All.ComovingIntegrationOn) {Face_Area_Vec[k] *= All.cf_atime*All.cf_atime;} /* Face_Area_Norm has units of area, need to convert to physical */
                    /* on the first pass, we need to save the face information to be used to correct the gradients; this only needs to be done once */
                    if(gradient_iteration == 0)
                    {
                        out.Face_Area[k] += Face_Area_Vec[k];
                        if(swap_to_j) SphP[j].Face_Area[k] -= Face_Area_Vec[k];
                        
                        for(k2=0;k2<3;k2++)
                        {
                            double q = -0.5 * Face_Area_Vec[k] * kernel.dp[k2];
                            out.FaceCrossX[k][k2] += q;
                            if(swap_to_j) GasGradDataPasser[j].FaceCrossX[k][k2] += q;
                        }
                    }
                    
                    /* now use the gradients to construct the B_L,R states */
                    double Bjk = Get_Particle_BField(j,k);
                    double db_c=0, db_cR=0;
                    for(k2=0;k2<3;k2++)
                    {
                        db_c += 0.5 * SphP[j].Gradients.B[k][k2] * kernel.dp[k2];
                        db_cR -= 0.5 * local.BGrad[k][k2]  * kernel.dp[k2];
                    }
                    
                    /* now we apply our slope-limiter to the B_L,R reconstruction */
                    double Q_L, Q_R;
                    if(Bjk == local.GQuant.B[k])
                    {
                        Q_L = Q_R = Bjk;
                    } else {
                        Q_L = Bjk + db_c;
                        Q_R = local.GQuant.B[k] + db_cR;
                        double Qmax, Qmin, Qmed = 0.5*(local.GQuant.B[k] + Bjk);
                        if(local.GQuant.B[k] < Bjk) {Qmax=Bjk; Qmin=local.GQuant.B[k];} else {Qmax=local.GQuant.B[k]; Qmin=Bjk;}
                        double fac = MHD_CONSTRAINED_GRADIENT_FAC_MINMAX * (Qmax-Qmin);
                        fac += MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM * fabs(Qmed);
                        double Qmax_eff = Qmax + fac;
                        double Qmin_eff = Qmin - fac;
                        fac = MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV * (Qmax-Qmin);
                        fac += MHD_CONSTRAINED_GRADIENT_FAC_MED_PM * fabs(Qmed);
                        double Qmed_max = Qmed + fac;
                        double Qmed_min = Qmed - fac;
                        if(Qmed_max>Qmax_eff) Qmed_max=Qmax_eff;
                        if(Qmed_min<Qmin_eff) Qmed_min=Qmin_eff;
                        if(local.GQuant.B[k] < Bjk)
                        {
                            if(Q_L>Qmax_eff) Q_L=Qmax_eff;
                            if(Q_L<Qmed_min) Q_L=Qmed_min;
                            if(Q_R<Qmin_eff) Q_R=Qmin_eff;
                            if(Q_R>Qmed_max) Q_R=Qmed_max;
                        } else {
                            if(Q_L<Qmin_eff) Q_L=Qmin_eff;
                            if(Q_L>Qmed_max) Q_L=Qmed_max;
                            if(Q_R>Qmax_eff) Q_R=Qmax_eff;
                            if(Q_R<Qmed_min) Q_R=Qmed_min;
                        }
                    }
                    
                    if(gradient_iteration==0)
                    {
                        out.FaceDotB += Face_Area_Vec[k] * (local.GQuant.B[k] + Q_L);
                    } else {
                        out_iter.FaceDotB += Face_Area_Vec[k] * (local.GQuant.B[k] + Q_L);
                    }
                    if(swap_to_j) GasGradDataPasser[j].FaceDotB -= Face_Area_Vec[k] * (Bjk + Q_R);
                }
                
#if defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
                /* this will fit the gradient at the -midpoint- as opposed to at the j locations, i.e.
                 attempting to minimize the quantity phi_L - phi_R, at face locations */
                double dphi = Get_Particle_PhiField(j) - local.GQuant.Phi;
                if(gradient_iteration == 0)
                {
                    MINMAX_CHECK(dphi,out.Minima.Phi,out.Maxima.Phi);
                    if(swap_to_j) {MINMAX_CHECK(-dphi,GasGradDataPasser[j].Minima.Phi,GasGradDataPasser[j].Maxima.Phi);}
                }

                // dphi = phi_j - phi_i :: if phi_i = 0, dphi = phi_j //
                double dphi_grad_j = 0, dphi_grad_i = 0;
                for(k=0;k<3;k++)
                {
                    dphi_grad_j += 0.5 * kernel.dp[k] * SphP[j].Gradients.Phi[k];
                    dphi_grad_i -= 0.5 * kernel.dp[k] * local.PhiGrad[k];
                }
                if(dphi > 0)
                {
                    if(dphi_grad_j>0) {dphi_grad_j=0;} else {if(dphi_grad_j<0.5*dphi) dphi_grad_j=0.5*dphi;}
                    if(dphi_grad_i<0) {dphi_grad_i=0;} else {if(dphi_grad_i>0.5*dphi) dphi_grad_i=0.5*dphi;}
                } else {
                    if(dphi_grad_j<0) {dphi_grad_j=0;} else {if(dphi_grad_j>0.5*dphi) dphi_grad_j=0.5*dphi;}
                    if(dphi_grad_i>0) {dphi_grad_i=0;} else {if(dphi_grad_i<0.5*dphi) dphi_grad_i=0.5*dphi;}
                }
                double dphi_j = dphi + dphi_grad_j;
                double dphi_i = dphi - dphi_grad_i;
                if(sph_gradients_flag_i) {dphi_j *= -2*kernel.wk_i;} else {dphi_j *= kernel.dwk_i/kernel.r * P[j].Mass;}
                if(sph_gradients_flag_j) {dphi_i *= -2*kernel.wk_j;} else {dphi_i *= kernel.dwk_j/kernel.r * local.Mass;}
                if(gradient_iteration == 0) {for(k=0;k<3;k++) {out.Gradients[k].Phi += dphi_j * kernel.dp[k];}} else {for(k=0;k<3;k++) {out_iter.PhiGrad[k] += dphi_j * kernel.dp[k];}}
                if(swap_to_j) {for(k=0;k<3;k++) {GasGradDataPasser[j].PhiGrad[k] += dphi_i * kernel.dp[k];}}
#endif
#endif
                
                if(gradient_iteration == 0)
                {
                    /* ------------------------------------------------------------------------------------------------ */
                    /* DIFFERENCE & SLOPE LIMITING: need to check maxima and minima of particle values in the kernel, to avoid
                     'overshoot' with our gradient estimators. this check should be among all interacting pairs */

                    if(kernel.r > out.MaxDistance) {out.MaxDistance = kernel.r;}
                    if(swap_to_j) {if(kernel.r > GasGradDataPasser[j].MaxDistance) {GasGradDataPasser[j].MaxDistance = kernel.r;}}

                    double d_rho = SphP[j].Density - local.GQuant.Density;
                    MINMAX_CHECK(d_rho,out.Minima.Density,out.Maxima.Density);
                    if(swap_to_j) {MINMAX_CHECK(-d_rho,GasGradDataPasser[j].Minima.Density,GasGradDataPasser[j].Maxima.Density);}

                    double dp = SphP[j].Pressure - local.GQuant.Pressure;
                    MINMAX_CHECK(dp,out.Minima.Pressure,out.Maxima.Pressure);
                    if(swap_to_j) {MINMAX_CHECK(-dp,GasGradDataPasser[j].Minima.Pressure,GasGradDataPasser[j].Maxima.Pressure);}

#ifdef TURB_DIFF_DYNAMIC
                    double dv_bar[3]; /* Need to calculate the filtered velocity gradient for the filtered shear */
                    for (k = 0; k < 3; k++) {
                        dv_bar[k] = SphP[j].Velocity_bar[k] - local.GQuant.Velocity_bar[k];
#ifdef SHEARING_BOX
                        if (k == SHEARING_BOX_PHI_COORDINATE) {
                            if (local.Pos[0] - P[j].Pos[0] > +boxHalf_X) {dv_bar[k] -= Shearing_Box_Vel_Offset;}
                            if (local.Pos[0] - P[j].Pos[0] < -boxHalf_X) {dv_bar[k] += Shearing_Box_Vel_Offset;}}
#endif
                        MINMAX_CHECK(dv_bar[k], out.Minima.Velocity_bar[k], out.Maxima.Velocity_bar[k]);
                        if (swap_to_j) {MINMAX_CHECK(-dv_bar[k], GasGradDataPasser[j].Minima.Velocity_bar[k], GasGradDataPasser[j].Maxima.Velocity_bar[k]);}
                    }
#endif

#if defined(KERNEL_CRK_FACES)
                    {
                        double V_i = local.Mass/local.GQuant.Density, V_j = P[j].Mass/SphP[j].Density;
                        double wk_ij = 0.5*(kernel.wk_i + kernel.wk_j), dwk_ij = 0.5*(kernel.dwk_i + kernel.dwk_j), rinv = 1./(MIN_REAL_NUMBER + kernel.r);
                        double Vj_wki = V_j*wk_ij, Vj_dwki = V_j*dwk_ij*rinv, Vi_wkj = V_i*wk_ij, Vi_dwkj = V_i*dwk_ij*rinv;
                        out.m0 += Vj_wki;
                        for(k=0;k<3;k++) {out.dm0[k] += Vj_dwki*kernel.dp[k];}
                        for(k2=0;k2<3;k2++)
                        {
                            out.m1[k2] += Vj_wki*kernel.dp[k2];
                            for(k=0;k<3;k++) {out.dm1[k2][k] += Vj_dwki*kernel.dp[k2]*kernel.dp[k];}
                        }
                        for(k2=0;k2<6;k2++)
                        {
                            int kk0[6]={0,1,2,0,0,1};
                            int kk1[6]={0,1,2,1,2,2};
                            out.m2[k2] += Vj_wki*kernel.dp[kk0[k2]]*kernel.dp[kk1[k2]];
                            for(k=0;k<3;k++) {out.dm2[k2][k] += Vj_dwki*kernel.dp[kk0[k2]]*kernel.dp[kk1[k2]]*kernel.dp[k];}
                        }
                        if(swap_to_j)
                        {
                            GasGradDataPasser[j].m0 += Vi_wkj;
                            for(k=0;k<3;k++) {GasGradDataPasser[j].dm0[k] -= Vi_dwkj*kernel.dp[k];}
                            for(k2=0;k2<3;k2++)
                            {
                                GasGradDataPasser[j].m1[k2] -= Vi_wkj*kernel.dp[k2];
                                for(k=0;k<3;k++) {GasGradDataPasser[j].dm1[k2][k] += Vi_dwkj*kernel.dp[k2]*kernel.dp[k];}
                            }
                            for(k2=0;k2<6;k2++)
                            {
                                int kk0[6]={0,1,2,0,0,1};
                                int kk1[6]={0,1,2,1,2,2};
                                GasGradDataPasser[j].m2[k2] += Vi_wkj*kernel.dp[kk0[k2]]*kernel.dp[kk1[k2]];
                                for(k=0;k<3;k++) {GasGradDataPasser[j].dm2[k2][k] -= Vi_dwkj*kernel.dp[kk0[k2]]*kernel.dp[kk1[k2]]*kernel.dp[k];}
                            }
                        }
                    }
#endif
                    
                    double dv[3];
                    for(k=0;k<3;k++)
                    {
                        dv[k] = SphP[j].VelPred[k] - local.GQuant.Velocity[k];
#ifdef BOX_SHEARING
                        if(k==BOX_SHEARING_PHI_COORDINATE)
                        {
                            if(local.Pos[0] - P[j].Pos[0] > +boxHalf_X) {dv[k] -= Shearing_Box_Vel_Offset;}
                            if(local.Pos[0] - P[j].Pos[0] < -boxHalf_X) {dv[k] += Shearing_Box_Vel_Offset;}
                        }
#endif
                        MINMAX_CHECK(dv[k],out.Minima.Velocity[k],out.Maxima.Velocity[k]);
                        if(swap_to_j) {MINMAX_CHECK(-dv[k],GasGradDataPasser[j].Minima.Velocity[k],GasGradDataPasser[j].Maxima.Velocity[k]);}
                    }

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
                    for(k=0;k<3;k++)
                    {
                        double GlassAcc = kernel.dp[k] / (kernel.r*kernel.r*kernel.r); // acceleration to apply to force cells into a glass
                        out.GlassAcc[k] += GlassAcc;
                        if(swap_to_j) {GasGradDataPasser[j].GlassAcc[k] -= GlassAcc;}
                    }
#endif
                    
#ifdef DOGRAD_INTERNAL_ENERGY
                    double du = SphP[j].InternalEnergyPred - local.GQuant.InternalEnergy;
                    MINMAX_CHECK(du,out.Minima.InternalEnergy,out.Maxima.InternalEnergy);
                    if(swap_to_j) {MINMAX_CHECK(-du,GasGradDataPasser[j].Minima.InternalEnergy,GasGradDataPasser[j].Maxima.InternalEnergy);}
#endif
#ifdef DOGRAD_SOUNDSPEED
                    double dc = Particle_effective_soundspeed_i(j) - local.GQuant.SoundSpeed;
                    MINMAX_CHECK(dc,out.Minima.SoundSpeed,out.Maxima.SoundSpeed);
                    if(swap_to_j) {MINMAX_CHECK(-dc,GasGradDataPasser[j].Minima.SoundSpeed,GasGradDataPasser[j].Maxima.SoundSpeed);}
#endif
#ifdef MAGNETIC
                    double Bj[3],dB[3];
                    for(k=0;k<3;k++)
                    {
                        Bj[k] = Get_Particle_BField(j,k);
                        dB[k] = Bj[k] - local.GQuant.B[k];
                        MINMAX_CHECK(dB[k],out.Minima.B[k],out.Maxima.B[k]);
                        if(swap_to_j) {MINMAX_CHECK(-dB[k],GasGradDataPasser[j].Minima.B[k],GasGradDataPasser[j].Maxima.B[k]);}
                    }
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
                    double dphi = Get_Particle_PhiField(j) - local.GQuant.Phi;
                    MINMAX_CHECK(dphi,out.Minima.Phi,out.Maxima.Phi);
                    if(swap_to_j) {MINMAX_CHECK(-dphi,GasGradDataPasser[j].Minima.Phi,GasGradDataPasser[j].Maxima.Phi);}
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
                    double dmetal[NUM_METAL_SPECIES];
                    for(k = 0; k < NUM_METAL_SPECIES; k++)
                    {
                        dmetal[k] = P[j].Metallicity[k] - local.GQuant.Metallicity[k];
                        MINMAX_CHECK(dmetal[k],out.Minima.Metallicity[k],out.Maxima.Metallicity[k]);
                        if(swap_to_j) {MINMAX_CHECK(-dmetal[k],GasGradDataPasser[j].Minima.Metallicity[k],GasGradDataPasser[j].Maxima.Metallicity[k]);}
                    }
#endif
#ifdef RT_EVOLVE_EDDINGTON_TENSOR
                    double dnET[N_RT_FREQ_BINS][6];
                    double dn[N_RT_FREQ_BINS];
                    double V_i_inv = 1/V_i, V_j_inv = SphP[j].Density/P[j].Mass;
                    for(k = 0; k < N_RT_FREQ_BINS; k++)
                    {
                        int k_dE; for(k_dE=0;k_dE<6;k_dE++) {dnET[k][k_dE] = SphP[j].E_gamma_Pred[k]*SphP[j].ET[k][k_dE]*V_j_inv - local.GQuant.E_gamma[k]*local.GQuant.E_gamma_ET[k][k_dE]*V_i_inv;}
                        dn[k] = SphP[j].E_gamma_Pred[k]*V_j_inv - local.GQuant.E_gamma[k]*V_i_inv;
                        MINMAX_CHECK(dn[k],out.Minima.E_gamma[k],out.Maxima.E_gamma[k]);
                        if(swap_to_j) {MINMAX_CHECK(-dn[k],GasGradDataPasser[j].Minima.E_gamma[k],GasGradDataPasser[j].Maxima.E_gamma[k]);}
                    }
#endif
                    /* end of difference and slope-limiter (min/max) block */
                    /* ------------------------------------------------------------------------------------------------ */


                    /* ------------------------------------------------------------------------------------------------ */
                    /*  Here we insert additional operations we want to fit into the gradients loop. at the moment, all of these are SPH-specific */
#ifdef HYDRO_SPH
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
                    out.alpha_limiter += NV_MYSIGN(SphP[j].NV_DivVel) * P[j].Mass * kernel.wk_i;
                    if(swap_to_j) SphP[j].alpha_limiter += NV_MYSIGN(local.NV_DivVel) * local.Mass * kernel.wk_j;
#endif
#ifdef MAGNETIC
                    double mji_dwk_r = P[j].Mass * kernel.dwk_i / kernel.r;
                    double mij_dwk_r = local.Mass * kernel.dwk_j / kernel.r;
                    for(k=0;k<3;k++)
                    {
                        for(k2=0;k2<3;k2++)
                        {
                            out.DtB[k] += local.GQuant.B[k2] * mji_dwk_r * kernel.dp[k2] * dv[k];
                            if(swap_to_j) SphP[j].DtB[k] += Bj[k2] * mij_dwk_r * kernel.dp[k2] * dv[k];
                        }
#ifdef DIVBCLEANING_DEDNER
                        out.divB += dB[k] * kernel.dp[k] * mji_dwk_r;
                        if(swap_to_j) SphP[j].divB += dB[k] * kernel.dp[k] * mij_dwk_r;
#endif
                    }
#endif
#endif
                    /* end of additional/miscellaneous operators block */
                    /* ------------------------------------------------------------------------------------------------ */

                    
                    /* ------------------------------------------------------------------------------------------------ */
                    /* Finally, save actual output for GRADIENTS */
                    
                    /* first do particle i */
                    if(kernel.r < kernel.h_i)
                    {
                        if(sph_gradients_flag_i) {kernel.wk_i = -kernel.dwk_i/kernel.r * P[j].Mass;} // sph-like weights for gradients //
                        for(k=0;k<3;k++)
                        {
                            double wk_xyz_i = -kernel.wk_i * kernel.dp[k]; /* sign is important here! */
                            out.Gradients[k].Density += wk_xyz_i * d_rho;
                            out.Gradients[k].Pressure += wk_xyz_i * dp;
                            for(k2=0;k2<3;k2++) {out.Gradients[k].Velocity[k2] += wk_xyz_i * dv[k2];}
#ifdef TURB_DIFF_DYNAMIC
                            for (k2 = 0; k2 < 3; k2++) {out.Gradients[k].Velocity_bar[k2] += wk_xyz_i * dv_bar[k2];}
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
                            out.Gradients[k].InternalEnergy += wk_xyz_i * du;
#endif
#ifdef DOGRAD_SOUNDSPEED
                            out.Gradients[k].SoundSpeed += wk_xyz_i * dc;
#endif
#ifdef MAGNETIC
                            for(k2=0;k2<3;k2++) {out.Gradients[k].B[k2] += wk_xyz_i * dB[k2];}
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
                            out.Gradients[k].Phi += wk_xyz_i * dphi;
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
                            for(k2=0;k2<NUM_METAL_SPECIES;k2++) {out.Gradients[k].Metallicity[k2] += wk_xyz_i * dmetal[k2];}
#endif
#ifdef RT_EVOLVE_EDDINGTON_TENSOR
                            for(k2=0;k2<N_RT_FREQ_BINS;k2++) 
                            {
                            	out.Gradients[k].E_gamma[k2] += wk_xyz_i * dn[k2];
                            	int k_et; for(k_et=0;k_et<6;k_et++) out.Gradients[k].E_gamma_ET[k2][k_et] += wk_xyz_i * dnET[k2][k_et];
                            } 
#endif
                        }
                    }
                    
                    /* next do particle j */
                    if((kernel.r < h_j) && (swap_to_j))
                    {
                        if(sph_gradients_flag_j) {kernel.wk_j = -kernel.dwk_j/kernel.r * local.Mass;} // sph-like weights for gradients //
                        for(k=0;k<3;k++)
                        {
                            double wk_xyz_j = -kernel.wk_j * kernel.dp[k]; /* sign is important here! (note dp-dd signs cancel) */
                            SphP[j].Gradients.Density[k] += wk_xyz_j * d_rho;
                            SphP[j].Gradients.Pressure[k] += wk_xyz_j * dp;
                            for(k2=0;k2<3;k2++) {SphP[j].Gradients.Velocity[k2][k] += wk_xyz_j * dv[k2];}
#ifdef TURB_DIFF_DYNAMIC
                            for (k2 = 0; k2 < 3; k2++) {GasGradDataPasser[j].GradVelocity_bar[k2][k] += wk_xyz_j * dv_bar[k2];}
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
                            SphP[j].Gradients.InternalEnergy[k] += wk_xyz_j * du;
#endif
#ifdef DOGRAD_SOUNDSPEED
                            SphP[j].Gradients.SoundSpeed[k] += wk_xyz_j * dc;
#endif
#ifdef MAGNETIC
#ifdef MHD_CONSTRAINED_GRADIENT
                            for(k2=0;k2<3;k2++) {GasGradDataPasser[j].BGrad[k2][k] += wk_xyz_j * dB[k2];}
#else
                            for(k2=0;k2<3;k2++) {SphP[j].Gradients.B[k2][k] += wk_xyz_j * dB[k2];}
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
                            SphP[j].Gradients.Phi[k] += wk_xyz_j * dphi;
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
                            for(k2=0;k2<NUM_METAL_SPECIES;k2++) {SphP[j].Gradients.Metallicity[k2][k] += wk_xyz_j * dmetal[k2];}
#endif
#ifdef RT_EVOLVE_EDDINGTON_TENSOR
                            for(k2=0;k2<N_RT_FREQ_BINS;k2++) 
                            {
                            	GasGradDataPasser[j].Gradients_E_gamma[k2][k] += wk_xyz_j * dn[k2];
								/* below we have the gradient dotted into the Eddington tensor (more complicated than a scalar gradient, but should recover full anisotropy */
								int k_freq=k2,k_xyz,j_xyz,i_xyz=k,k_et_loop[3]; // recall, for ET: 0=xx,1=yy,2=zz,3=xy,4=yz,5=xz
								for(k_xyz=0;k_xyz<3;k_xyz++)
								{
									if(k_xyz==0) {k_et_loop[0]=0; k_et_loop[1]=3; k_et_loop[2]=5;}
									if(k_xyz==1) {k_et_loop[0]=3; k_et_loop[1]=1; k_et_loop[2]=4;}
									if(k_xyz==2) {k_et_loop[0]=5; k_et_loop[1]=4; k_et_loop[2]=2;}
									for(j_xyz=0;j_xyz<3;j_xyz++)
									{
										SphP[j].Gradients.E_gamma_ET[k_freq][k_xyz] += SphP[j].NV_T[j_xyz][i_xyz] * wk_xyz_j * dnET[k_freq][k_et_loop[j_xyz]];
									}
								}
                            }
#endif
                        }
                    }

                    /* end of GRADIENTS calculation block */
                    /* ------------------------------------------------------------------------------------------------ */
                    
                        
                } // (r2 < h2i || r2 < h2j) && gradient_iteration==0
            } // numngb loop
        } // while(startnode)
        
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = GasGradDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
    }
    
    
    /* ------------------------------------------------------------------------------------------------ */
    /* Now collect the result at the right place */
    if(gradient_iteration==0)
    {
        if(mode == 0)
            out2particle_GasGrad(&out, target, 0, gradient_iteration);
        else
            GasGradDataResult[target] = out;
    } else {
        if(mode == 0)
            out2particle_GasGrad_iter(&out_iter, target, 0, gradient_iteration);
        else
            GasGradDataResult_iter[target] = out_iter;
    }
    /* ------------------------------------------------------------------------------------------------ */
    
    return 0;
}

void *GasGrad_evaluate_primary(void *p, int gradient_iteration)
{
#define CONDITION_FOR_EVALUATION if(P[i].Type==0)
#define EVALUATION_CALL GasGrad_evaluate(i,0,exportflag,exportnodecount,exportindex,ngblist,gradient_iteration)
#include "../system/code_block_primary_loop_evaluation.h"
#undef CONDITION_FOR_EVALUATION
#undef EVALUATION_CALL
}
void *GasGrad_evaluate_secondary(void *p, int gradient_iteration)
{
#define EVALUATION_CALL GasGrad_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, gradient_iteration);
#include "../system/code_block_secondary_loop_evaluation.h"
#undef EVALUATION_CALL
}

