#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <mpi.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <bigfile-mpi.h>

#include <libgadget/cosmology.h>
#include <libgadget/utils.h>

#include <libgadget/physconst.h>
#include "power.h"
#include "proto.h"
static double Delta_EH(double k);
static double Delta_Tabulated(double k, enum TransferType Type);
static double sigma2_int(double k, void * params);
static double TopHatSigma2(double R);
static double tk_eh(double k);

static double Norm;
static int WhichSpectrum;
/*Only used for tk_eh, WhichSpectrum == 0*/
static double PrimordialIndex;
static double UnitLength_in_cm;
static Cosmology * CP;

/* Small factor so a zero in the power spectrum is not
 * a log(0) = -INF*/
#define NUGGET 1e-30
#define MAXCOLS 9

struct table
{
    int Nentry;
    double * logk;
    double * logD[MAXCOLS];
    gsl_interp * mat_intp[MAXCOLS];
};

/*Typedef for a function that parses the table from text*/
typedef void (*_parse_fn)(int i, double k, char * line, struct table *, int *InputInLog10, const double InitTime, int NumCol);


static struct table power_table;
/*Columns: 0 == baryon, 1 == CDM, 2 == neutrino, 3 == baryon velocity, 4 == CDM velocity, 5 = neutrino velocity*/
static struct table transfer_table;

static const char * tnames[MAXCOLS] = {"DELTA_BAR", "DELTA_CDM", "DELTA_NU", "DELTA_CB", "VEL_BAR", "VEL_CDM", "VEL_NU", "VEL_CB", "VEL_TOT"};

double DeltaSpec(double k, enum TransferType Type)
{
  double power;

  if(WhichSpectrum == 2)
      power = Delta_Tabulated(k, Type);
  else
      power = Delta_EH(k);

  /*Normalise the power spectrum*/
  power *= Norm;

  return power;
}

/* Internal helper function that performs interpolation for a row of the
 * tabulated transfer/mater power table*/
static double get_Tabulated(double k, enum TransferType Type)
{
    /*Convert k to Mpc/h*/
    const double scale = (CM_PER_MPC / UnitLength_in_cm);
    double logk = log10(k*scale);
    double maxlogk = power_table.logk[power_table.Nentry - 1];
    /* For interpolation*/
    double intlogk = logk;

    if(logk < power_table.logk[0]-0.1 || logk > maxlogk + 0.25)
        endrun(6, "Requested k = %g h/Mpc for type %d but power table is from %g to %g h/Mpc\n",
               pow(10, logk), Type, pow(10, power_table.logk[0]), pow(10, maxlogk));
    if(logk < power_table.logk[0])
        intlogk = power_table.logk[0];
    if(logk > maxlogk)
        intlogk = maxlogk;

    double logD = gsl_interp_eval(power_table.mat_intp[0], power_table.logk, power_table.logD[0], intlogk, NULL);
    /* If we are past the end of the table, assume the power scales like k^-3 log (k) and the transfer function is constant*/
    if(logk > maxlogk)
        logD += -3 * (logk - intlogk) + log(logk / intlogk);
    double trans = 1;
    /*Transfer table stores (T_type(k) / T_tot(k))*/
    if(transfer_table.Nentry > 0)
       if(Type >= DELTA_BAR && Type < DELTA_TOT)
          trans = gsl_interp_eval(transfer_table.mat_intp[Type], transfer_table.logk, transfer_table.logD[Type], intlogk, NULL);

    /*Convert delta from (Mpc/h)^3/2 to kpc/h^3/2*/
    logD += 1.5 * log10(scale);
    double delta = (pow(10.0, logD)-NUGGET) * trans;
    if(!isfinite(delta))
        endrun(1,"infinite delta or growth: %g for k = %g, Type = %d (tk = %g, logD = %g)\n",delta, k, Type, trans, logD);
    return delta;
}

double Delta_Tabulated(double k, enum TransferType Type)
{
    if(Type >= VEL_BAR && Type <= VEL_TOT)
        endrun(1, "Velocity Type %d passed to Delta_Tabulated\n", Type);

    return get_Tabulated(k, Type);
}

double dlogGrowth(double kmag, enum TransferType Type)
{
    /*Default to total growth: type 3 is cdm + baryons.*/
    if(Type < DELTA_BAR || Type > DELTA_CB)
        Type = VEL_TOT;
    else
        /*Type should be an offset from the first velocity*/
        Type = (enum TransferType) ((int) VEL_BAR + ((int) Type - (int) DELTA_BAR));
    return get_Tabulated(kmag, Type);
}

/*Save a transfer function table to the IC file*/
static void save_transfer(BigFile * bf, int ncol, struct table * ttable, const char * bname, int ThisTask, const char * colnames[])
{
    BigBlock btransfer;
    int i;
    if(0 != big_file_mpi_create_block(bf, &btransfer, bname, NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "failed to create block %s:%s", bname,
                big_file_get_error_message());
    }

    if ( (0 != big_block_set_attr(&btransfer, "Nentry", &(ttable->Nentry), "u8", 1)) ) {
        endrun(0, "Failed to write table size %s\n",
                    big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&btransfer, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
    size_t dims[2] = {0 , 1};
    /*The transfer state is shared between all processors,
     *so only write on master task*/
    if(ThisTask == 0) {
        dims[0] = ttable->Nentry;
    }

    char buf[100];
    snprintf(buf, 100, "%s/logk", bname);

    _bigfile_utils_create_block_from_c_array(bf, ttable->logk, buf, "f8", dims, sizeof(double), 1, 1, MPI_COMM_WORLD);

    for(i = 0; i < ncol; i++)
    {
        snprintf(buf, 100, "%s/%s", bname, colnames[i]);
        _bigfile_utils_create_block_from_c_array(bf, ttable->logD[i], buf, "f8", dims, sizeof(double), 1, 1, MPI_COMM_WORLD);
    }
}

/*Save both transfer function tables to the IC file*/
void save_all_transfer_tables(BigFile * bf, int ThisTask)
{
    const char * pname = "DELTA_MAT";
    save_transfer(bf, 1, &power_table, "ICPower", ThisTask, &pname);
    save_transfer(bf, MAXCOLS, &transfer_table, "ICTransfers", ThisTask, tnames);
}


void parse_power(int i, double k, char * line, struct table *out_tab, int * InputInLog10, const double InitTime, int NumCol)
{
    char * retval;
    if((*InputInLog10) == 0) {
        if(k < 0) {
            message(1, "some input k is negative, guessing the file is in log10 units\n");
            *InputInLog10 = 1;
        }
        else
            k = log10(k);
    }
    out_tab->logk[i] = k;
    retval = strtok(NULL, " \t");
    if(!retval)
        endrun(1,"Incomplete line in power spectrum: %s\n",line);
    double p = atof(retval);
    if ((*InputInLog10) == 0)
        p = log10(p+NUGGET);
    /*Store delta, square root of power*/
    out_tab->logD[0][i] = p/2;
}

void parse_transfer(int i, double k, char * line, struct table *out_tab, int * InputInLog10, const double InitTime, int NumCol)
{
    int j;
    int ncols = NumCol - 1; /* The first column k is already read in read_power_table. */
    /* Detect dark energy fluid perturbations.*/
    int defld = 0;
    if(NumCol > 22)
        defld = 1;
    else if(NumCol > 24)
        endrun(2,"Transfer function has %d columns, expected maximum 22!\n", NumCol);
    int nnu = round((ncols - 15 - defld *2)/2);

    double * transfers = (double *) mymalloc("transfers", sizeof(double) * ncols);
    k = log10(k);
    out_tab->logk[i] = k;
    /* Note: the ncdm entries change depending on the number of neutrino species. The first row, k,
     * is read in read_power_table and passed as a parameter.
     * but only the first is used for particles.
     * 1:k (h/Mpc)              2:d_g                    3:d_b                    4:d_cdm      (5: d_fld)            5:d_ur
     * 6:d_ncdm[0]              7:d_ncdm[1]              8:d_ncdm[2]              9:d_tot                 10:phi
     * 11:psi                   12:h                     13:h_prime               14:eta                   15:eta_prime
     * 16:t_g                   17:t_b                   18:t_ur                  (19:t_fld)
     * 19:t_ncdm[0]             20:t_ncdm[1]             21:t_ncdm[2]             22:t_tot*/
    for(j = 0; j< ncols; j++) {
        char * retval = strtok(NULL, " \t");
        /*This happens if we do not have as many neutrino species as we expect, or we don't find h_prime.*/
        if(!retval)
            endrun(1,"Incomplete line in power spectrum: only %d columns, expecting %d. Did you remember to set extra metric transfer functions=y?\n",j, ncols);
        transfers[j] = atof(retval);
    }
    /*Order of the transfer table matches the particle types:
     * 0 is baryons, 1 is CDM, 2 is massive neutrinos (we use the most massive neutrino species).
     * 3 is growth function for baryons, 4 is growth function for CDM, 5 is growth function for massive neutrinos.
     * We use the formulae for velocities from fastpm in synchronous gauge:
     * https://github.com/rainwoodman/fastpm-python/blob/02ce2ff87897f713c7b9204630f4e0257d703784/fastpm/multi.py#L185
     * CDM = - h_prime / 2 / d_cdm
     * bar = - (h_prime / 2  + t_b) / d_b
     * nu = - (h_prime / 2 + t_ncdm) / d_ncdm
     * and there is a normalisation factor of (1+z)/ hubble applied later on.
     *
     * See http://adsabs.harvard.edu/abs/1995ApJ...455....7M
     * Eq. 42 (cdm), not using 49(ur), eq. 66 (baryons, and ncdm, truncation at the same order as baryon).
     * These are in CLASS units: they are converted to Gadget units in init_transfer_table().
     * */
    out_tab->logD[DELTA_BAR][i] = -1*transfers[1];
    out_tab->logD[DELTA_CDM][i] = -1*transfers[2];
    /*This should be the weighted average sum of the three neutrino species*/
    const _omega_nu * Onu = &CP->ONu;
    out_tab->logD[DELTA_NU][i] = 0;
    /* The DE fluid moves the neutrinos up one*/
    for(j=0; j < nnu; j++)
        out_tab->logD[DELTA_NU][i] = -1*transfers[4+j+defld] * omega_nu_single(Onu, InitTime, j);
    const double onu = get_omega_nu(&CP->ONu, InitTime);
    /*Should be weighted by omega_nu*/
    out_tab->logD[DELTA_NU][i] /= onu;
    /*h_prime is entry 8 + nnu. t_b is 12 + nnu, t_ncdm[2] is 13 + nnu * 2.*/
    out_tab->logD[VEL_BAR][i] = transfers[12+nnu + defld];
    out_tab->logD[VEL_CDM][i] = transfers[8+nnu + defld] * 0.5;
    out_tab->logD[VEL_NU][i] = 0;
    for(j=0; j < nnu; j++)
        out_tab->logD[VEL_NU][i] = transfers[13 + nnu + defld * 2 + j] * omega_nu_single(Onu, InitTime, j);
    /*Should be weighted by omega_nu*/
    out_tab->logD[VEL_NU][i] /= onu;
    myfree(transfers);
}

void read_power_table(int ThisTask, const char * inputfile, const int ncols, struct table * out_tab, const double InitTime, _parse_fn parse_line)
{
    FILE *fd = NULL;
    int j;
    int InputInLog10 = 0;

    if(ThisTask == 0) {
        if(!(fd = fopen(inputfile, "r")))
            endrun(1, "can't read input spectrum in file '%s' on task %d\n", inputfile, ThisTask);

        out_tab->Nentry = 0;
        do
        {
            char buffer[1024];
            char * retval = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!retval)
                break;
            retval = strtok(buffer, " \t");
            if(!retval || retval[0] == '#')
                continue;
            out_tab->Nentry++;
        }
        while(1);
        rewind(fd);
    }
    MPI_Bcast(&(out_tab->Nentry), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(out_tab->Nentry < 2)
        endrun(1, "Input spectrum too short\n");
    out_tab->logk = (double *) mymalloc("Powertable", (ncols+1)*out_tab->Nentry * sizeof(double));
    for(j=0; j<ncols; j++)
        out_tab->logD[j] = out_tab->logk + (j+1)*out_tab->Nentry;

    if(ThisTask == 0)
    {
        /* detect the columns of the input file */
        char line1[1024];

        while(fgets(line1,1024,fd))
        {
            char * content = strtok(line1, " \t");
            if(content[0] != '#') /*Find the first line*/
                break;
        }
        int Ncolumns = 0;
        char *c;
        do
        {
            Ncolumns++;
            c = strtok(NULL," \t");
        }
        while(c != NULL);

        rewind(fd);
        message(0, "Detected %d columns in file '%s'. \n", Ncolumns, inputfile);

        int i = 0;
        do
        {
            char buffer[1024];
            char * line = fgets(buffer, 1024, fd);
            /*Happens on end of file*/
            if(!line)
                break;
            char * retval = strtok(line, " \t");
            if(!retval || retval[0] == '#')
                continue;
            double k = atof(retval);
            parse_line(i, k, line, out_tab, &InputInLog10, InitTime, Ncolumns);
            i++;
        }
        while(1);

        fclose(fd);
    }

    MPI_Bcast(out_tab->logk, (ncols+1)*out_tab->Nentry, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(j=0; j<ncols; j++) {
        out_tab->mat_intp[j] = gsl_interp_alloc(gsl_interp_cspline,out_tab->Nentry);
    }
}

int
init_transfer_table(int ThisTask, double InitTime, const struct power_params * const ppar)
{
    int i, t;
    const int nnu = (CP->MNu[0] > 0) + (CP->MNu[1] > 0) + (CP->MNu[2] > 0);
    if(strlen(ppar->FileWithTransferFunction) > 0) {
        read_power_table(ThisTask, ppar->FileWithTransferFunction, MAXCOLS, &transfer_table, InitTime, parse_transfer);
    }
    if(transfer_table.Nentry == 0) {
        endrun(1, "Could not read transfer table at: '%s'\n",ppar->FileWithTransferFunction);
    }
    /*Normalise the transfer functions.*/

    /*Now normalise the velocity transfer functions: divide by a * hubble, where hubble is the hubble function in Mpc^-1, so H0/c*/
    const double fac = InitTime * hubble_function(CP, InitTime)/CP->Hubble * 100 * CP->HubbleParam/(LIGHTCGS / 1e5);
    const double onu = get_omega_nu(&CP->ONu, InitTime)*pow(InitTime,3);
    double meangrowth[VEL_TOT-VEL_BAR+1] = {0};
    /* At this point the transfer table contains: (3,4,5) t_b, 0.5 * h_prime, t_ncdm.
     * After, if t_cdm = 0.5 h_prime / (a H(a) / H0 /c) we need:
     * (t_b + t_cdm) / d_b, t_cdm/d_cdm, (t_ncdm + t_cdm) / d_ncdm*/
    for(i=0; i< transfer_table.Nentry; i++) {
        /* Now row 4 is t_cdm*/
        transfer_table.logD[VEL_CDM][i] /= fac;
        transfer_table.logD[VEL_BAR][i] /= fac;
        transfer_table.logD[VEL_NU][i] /= fac;
        /* Now change to t bar and tnu*/
        transfer_table.logD[VEL_BAR][i] += transfer_table.logD[VEL_CDM][i];
        transfer_table.logD[VEL_NU][i] += transfer_table.logD[VEL_CDM][i];

        /*Set up the CDM + baryon rows*/
        transfer_table.logD[DELTA_CB][i] = CP->OmegaBaryon * transfer_table.logD[DELTA_BAR][i] + CP->OmegaCDM * transfer_table.logD[DELTA_CDM][i];
        transfer_table.logD[VEL_CB][i] = CP->OmegaBaryon * transfer_table.logD[VEL_BAR][i] + CP->OmegaCDM * transfer_table.logD[VEL_CDM][i];
        /*total growth and delta: start as CDM + baryon and then add nu if needed.*/
        transfer_table.logD[VEL_TOT][i] = transfer_table.logD[VEL_CB][i];
        double T_tot = transfer_table.logD[DELTA_CB][i];
        /*Normalise the cb rows*/
        double Omega0a3 = CP->OmegaBaryon + CP->OmegaCDM;
        transfer_table.logD[DELTA_CB][i] /= Omega0a3;
        transfer_table.logD[VEL_CB][i] /= Omega0a3;
        /*Total matter density in T_tot. Neutrinos may be slightly relativistic, so
         * Omega0a3 >= CP->Omega0 if neutrinos are massive.*/
        if(nnu > 0) {
            /*Add neutrino growth to total growth*/
            transfer_table.logD[VEL_TOT][i] += onu *  transfer_table.logD[VEL_NU][i];
            T_tot += onu * transfer_table.logD[DELTA_NU][i];
            Omega0a3 += onu;
        }
        /*Normalise the totals now we have neutrinos*/
        transfer_table.logD[VEL_TOT][i] /= Omega0a3;
        T_tot /= Omega0a3;
        /*Normalize growth_i and delta_i by delta_tot */
        for(t = DELTA_BAR; t <= VEL_TOT; t++) {
            transfer_table.logD[t][i] /= T_tot;
        }
    }

    /*Now compute mean growths*/
    for(t = VEL_BAR; t <= VEL_TOT; t++) {
        int nmean=0;
        for(i=0; i< transfer_table.Nentry; i++)
            if(transfer_table.logk[i] > power_table.logk[0]) {
                meangrowth[t-VEL_BAR] += transfer_table.logD[t][i];
                nmean++;
            }
        if(nmean > 0)
            meangrowth[t-VEL_BAR]/= nmean;
    }
    /*Initialise the interpolation*/
    for(t = 0; t < MAXCOLS; t++)
        gsl_interp_init(transfer_table.mat_intp[t],transfer_table.logk, transfer_table.logD[t],transfer_table.Nentry);

    message(0,"Scale-dependent growth calculated. Mean = %g %g %g %g %g\n",meangrowth[0], meangrowth[1], meangrowth[2], meangrowth[3], meangrowth[4]);
    message(0, "Power spectrum rows: %d, Transfer: %d (%g -> %g)\n", power_table.Nentry, transfer_table.Nentry, transfer_table.logD[DELTA_BAR][0],transfer_table.logD[DELTA_BAR][transfer_table.Nentry-1]);
    return transfer_table.Nentry;
}

int init_powerspectrum(int ThisTask, double InitTime, double UnitLength_in_cm_in, Cosmology * CPin, struct power_params * ppar)
{
    WhichSpectrum = ppar->WhichSpectrum;
    /*Used only for tk_eh*/
    PrimordialIndex = ppar->PrimordialIndex;
    UnitLength_in_cm = UnitLength_in_cm_in;
    CP = CPin;

    if(ppar->WhichSpectrum == 2) {
        read_power_table(ThisTask, ppar->FileWithInputSpectrum, 1, &power_table, InitTime, parse_power);
        /*Initialise the interpolation*/
        gsl_interp_init(power_table.mat_intp[0],power_table.logk, power_table.logD[0],power_table.Nentry);
        transfer_table.Nentry = 0;
        if(ppar->DifferentTransferFunctions || ppar->ScaleDepVelocity) {
            init_transfer_table(ThisTask, InitTime, ppar);
        }
    }

    Norm = 1.0;
    if(ppar->InputPowerRedshift >= 0 || ppar->Sigma8 > 0) {
        const double R8 = 8 * (CM_PER_MPC / UnitLength_in_cm);	/* 8 Mpc/h */
        if(ppar->Sigma8 > 0) {
            double res = TopHatSigma2(R8);
            if(isfinite(res) && res > 0)
                Norm = ppar->Sigma8 / sqrt(res);
            else
                endrun(1, "Could not normalize P(k) to Sigma8=%g! Measured Sigma8^2 is %g\n", ppar->Sigma8, res);
        }
        double Dplus = GrowthFactor(CP, InitTime, 1/(1+ppar->InputPowerRedshift));
        if(ppar->InputPowerRedshift >= 0) {
            Norm *= Dplus;
            message(0,"Growth factor from z=%g (InputPowerRedshift) to z=%g (Init): %g \n", ppar->InputPowerRedshift, 1. / InitTime - 1, Dplus);
        }
        message(0, "Normalization adjusted to  Sigma8=%g (at z=0)  (Normfac=%g). \n", sqrt(TopHatSigma2(R8))/Dplus, Norm);
    }
    return power_table.Nentry;
}

double Delta_EH(double k)	/* Eisenstein & Hu */
{
  return sqrt(k * pow(tk_eh(k), 2)* pow(k, PrimordialIndex - 1.0));
}


double tk_eh(double k)		/* from Martin White */
{
  double q, theta, ommh2, a, s, gamma, L0, C0;
  double tmp;
  double omegam, ombh2, hubble;

  /* other input parameters */
  hubble = CP->HubbleParam;

  omegam = CP->Omega0;
  ombh2 = CP->OmegaBaryon * CP->HubbleParam * CP->HubbleParam;

  if(CP->OmegaBaryon == 0)
    ombh2 = 0.044 * CP->HubbleParam * CP->HubbleParam;

  k *= (CM_PER_MPC / UnitLength_in_cm);	/* convert to h/Mpc */

  theta = 2.728 / 2.7;
  ommh2 = omegam * hubble * hubble;
  s = 44.5 * log(9.83 / ommh2) / sqrt(1. + 10. * exp(0.75 * log(ombh2))) * hubble;
  a = 1. - 0.328 * log(431. * ommh2) * ombh2 / ommh2
    + 0.380 * log(22.3 * ommh2) * (ombh2 / ommh2) * (ombh2 / ommh2);
  gamma = a + (1. - a) / (1. + exp(4 * log(0.43 * k * s)));
  gamma *= omegam * hubble;
  q = k * theta * theta / gamma;
  L0 = log(2. * exp(1.) + 1.8 * q);
  C0 = 14.2 + 731. / (1. + 62.5 * q);
  tmp = L0 / (L0 + C0 * q * q);
  return (tmp);
}


double TopHatSigma2(double R)
{
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  double result,abserr;
  gsl_function F;
  F.function = &sigma2_int;
  F.params = &R;

  /* Integral is oscillatory but almost zero kr = (n +1/2) pi for odd n. With P(k) ~ n^-3 the integrand is close to zero kr = 20.5 pi */
  double maxk = M_PI * (20 + 0.5) / R;
  double maxtabk = pow(10, 0.24 + power_table.logk[power_table.Nentry - 1]);
  if(maxk > maxtabk)
      endrun(3, "Trying to do sigma8 integral for rescaling, but need k = %g and largest k in power table is %g\n", maxk, pow(10, power_table.logk[power_table.Nentry - 1]));
  gsl_integration_qags (&F, 0, M_PI * (20 + 0.5) / R, 0, 1e-4,1000,w,&result, &abserr);
  // printf("gsl_integration_qng in TopHatSigma2. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
  gsl_integration_workspace_free (w);
  return result;
}


double sigma2_int(double k, void * params)
{
  double w, x;

  double r_tophat = *(double *) params;
  const double kr = r_tophat * k;
  const double kr2 = kr * kr;

  /*Series expansion; actually good until kr~1*/
  if(kr < 1e-3)
      w = 1./3. - kr2/30. +kr2*kr2/840.;
  else
      w = 3 * (sin(kr) / kr - cos(kr)) / kr2;
  x = 4 * M_PI / (2 * M_PI * 2 * M_PI * 2 * M_PI) * k * k * w * w * pow(DeltaSpec(k, DELTA_TOT),2);

  return x;

}
