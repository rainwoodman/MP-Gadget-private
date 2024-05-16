#include "lenstools.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h> 

#ifdef USE_CFITSIO
#include "fitsio.h"
#endif

#include <string.h>
#include "partmanager.h"
#include "cosmology.h"
#include "physconst.h"

void linspace(double start, double stop, int num, double *result) {
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
}

double ***allocate_3d_array(int nx, int ny, int nz) {
    // Allocate memory for the 3D array
    double ***array = malloc(nx * sizeof(double **));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nx; i++) {
        array[i] = malloc(ny * sizeof(double *));
        if (array[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for layer %d\n", i);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < ny; j++) {
            array[i][j] = malloc(nz * sizeof(double));
            if (array[i][j] == NULL) {
                fprintf(stderr, "Memory allocation failed for layer %d, row %d\n", i, j);
                exit(EXIT_FAILURE);
            }
            // Initialize elements (optional)
            for (int k = 0; k < nz; k++) {
                array[i][j][k] = 0.0; // Example initialization
            }
        }
    }

    return array;
}

void free_3d_array(double ***array, int nx, int ny) {
    // Free each row
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}


// Function to allocate a 2D array
double **allocate_2d_array(int Nx, int Ny) {
    double **array = malloc(Nx * sizeof(double *));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < Nx; i++) {
        array[i] = malloc(Ny * sizeof(double));
    }
    return array;
}

// Function to free a 2D array
void free_2d_array(double **array, int Nx) {
    for (int i = 0; i < Nx; i++) {
        free(array[i]);
    }
    free(array);
}

typedef struct {
    int nx, ny, nz;
} GridDimensions;

// Function to determine the bin index for a given value
int find_bin(double value, double *bins, int num_bins) {
    for (int i = 0; i < num_bins - 1; i++) {
        if (value >= bins[i] && value < bins[i + 1]) {
            return i;
        }
    }
    return -1;  // Return the last bin if the value is on the boundary or beyond
}

void grid3d_nfw(const struct particle_data * Parts, int num_particles, double **binning, GridDimensions dims, double ***density) {
    
    double position[3];
    // Process each particle
    for (int p = 0; p < num_particles; p++) {

        // remove offset
        for(int d = 0; d < 3; d ++) {
            position[d] = Parts[p].Pos[d] - PartManager->CurrentParticleOffset[d];
            while(position[d] > PartManager->BoxSize) position[d] -= PartManager->BoxSize;
            while(position[d] <= 0) position[d] += PartManager->BoxSize;
        }
        int ix = find_bin(position[0], binning[0], dims.nx + 1);
        int iy = find_bin(position[1], binning[1], dims.ny + 1);
        int iz = find_bin(position[2], binning[2], dims.nz + 1);

        // continue if the particle is outside the grid
        if (ix == -1 || iy == -1 || iz == -1) {
            continue;
        }
        // Increment the density in the appropriate bin
        density[ix][iy][iz]++;
    }

}

void projectDensity(double ***density, double **density_projected, GridDimensions dims, int normal) {
    // z; x, y
    // y; z, x
    // x; y, z
    int DimNorm = (normal == 0) ? dims.nx : (normal == 1) ? dims.ny : dims.nz;
    int Dim0 = (normal == 2) ? dims.nx : (normal == 0) ? dims.ny : dims.nx;
    int Dim1 = (normal == 2) ? dims.ny : (normal == 1) ? dims.nz : dims.nz;

    for (int i = 0; i < Dim0; i++) {
        for (int j = 0; j < Dim1; j++) {
            density_projected[i][j] = 0;
            for (int k = 0; k < DimNorm; k++) {
                if (normal == 0) {
                    density_projected[i][j] += density[k][i][j];
                } else if (normal == 1) {
                    density_projected[i][j] += density[i][k][j];
                } else {
                    density_projected[i][j] += density[i][j][k];
                }
            }
        }
    }
}

void calculate_lensing_potential(double **density_projected, int plane_resolution, double bin_resolution_0, double bin_resolution_1, double chi,double smooth, double **lensing_potential) {
    // Allocate the complex FFT output array
    fftw_complex *density_ft = fftw_malloc(sizeof(fftw_complex) * plane_resolution * (plane_resolution / 2 + 1));
    // Initialize density_ft to zero
    for (int i = 0; i < plane_resolution * (plane_resolution / 2 + 1); i++) {
        density_ft[i][0] = 0.0;  // Real part
        density_ft[i][1] = 0.0;  // Imaginary part
    }

    double **l_squared = allocate_2d_array(plane_resolution, plane_resolution / 2 + 1);

    // Create a temporary array to hold the real-valued lensing potential, for the backward FFT
    double *temp_lensing_potential = malloc(plane_resolution * plane_resolution * sizeof(double));
    if (temp_lensing_potential == NULL) {
        fprintf(stderr, "Memory allocation failed for temp_lensing_potential\n");
        // Free other allocated resources
        fftw_free(density_ft);
        free_2d_array(l_squared, plane_resolution);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Initialize temp_lensing_potential to zero
    for (int i = 0; i < plane_resolution * plane_resolution; i++) {
        temp_lensing_potential[i] = 0.0;
    }

    // Allocate memory for a temporary 1D array to store the 2D data (for contiguous memory layout)
    double *temp_density_projected = malloc(plane_resolution * plane_resolution * sizeof(double));
    if (!temp_density_projected) {
        perror("Failed to allocate memory for temporary data storage");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < plane_resolution; i++) {
        memcpy(temp_density_projected + i * plane_resolution, density_projected[i], plane_resolution * sizeof(double));
    }

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(plane_resolution, plane_resolution, temp_density_projected, density_ft, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(plane_resolution, plane_resolution, density_ft, temp_lensing_potential, FFTW_ESTIMATE);

    // Compute l_squared (multipoles)
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution / 2 + 1; j++) {
            double lx = i < plane_resolution / 2 ? i : -(plane_resolution - i);
            lx /= plane_resolution;
            double ly = j;  // Since rfftn outputs only the non-negative frequencies
            ly /= plane_resolution;
            l_squared[i][j] = lx * lx + ly * ly;
        }
    }
    l_squared[0][0] = 1.0;  // Avoid division by zero at the DC component

    // Perform the forward FFT
    fftw_execute(forward_plan);

    // Solve the Poisson equation and apply Gaussian smoothing in the frequency domain
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution / 2 + 1; j++) {
            int idx = i * (plane_resolution / 2 + 1) + j;
            double factor = -2.0 * (bin_resolution_0 * bin_resolution_1 / (chi * chi)) / (l_squared[i][j] * 4 * M_PI * M_PI);
            density_ft[idx][0] *= factor * exp(-0.5 * ((2.0 * M_PI * smooth) * (2.0 * M_PI * smooth)) * l_squared[i][j]);
            density_ft[idx][1] *= factor * exp(-0.5 * ((2.0 * M_PI * smooth) * (2.0 * M_PI * smooth)) * l_squared[i][j]);
        }
    }

    // Perform the inverse FFT
    fftw_execute(backward_plan);

    // Normalize the output of the inverse FFT and copy to the lensing_potential 2D array
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            lensing_potential[i][j] = temp_lensing_potential[i * plane_resolution + j] / (plane_resolution * plane_resolution);
        }
    }
    
    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(density_ft);
    free(temp_lensing_potential);
    // for (int i = 0; i < plane_resolution; i++) {
    //     free(l_squared[i]);
    // }
    // free(l_squared);
    free_2d_array(l_squared, plane_resolution);
}

int64_t cutPlaneGaussianGrid(int num_particles_tot, double comoving_distance, double Lbox, const Cosmology * CP, const double atime, const int normal, const double center, const double thickness, const double *left_corner, const int plane_resolution, double **lensing_potential) {
    // Get the rank of the current process
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    // smooth
    float smooth = 1.0; // fixed in our case

    int num_particles_rank = PartManager->NumPart;  // dark matter-only simulation: NumPart = number of dark matter particles

    double **density_projected = allocate_2d_array(plane_resolution, plane_resolution);

    // printf("cutPlaneGaussianGrid called with rank %d\n", rank);
    // int i, j;
    int thickness_resolution = 1;  // Number of bins along the thickness direction, fixed to 1 for now

    // cosmological normalization factor
    double H0 = 100 * CP->HubbleParam * 3.2407793e-20;  // Hubble constant in cgs units
    double cosmo_normalization = 1.5 * pow(H0, 2) * CP->Omega0 / pow(LIGHTCGS, 2);  

    // Binning for directions perpendicular to 'normal'
    double *binning[3];
    int plane_directions[2] = { (normal + 1) % 3, (normal + 2) % 3 };

    for (int i = 0; i < 3; i++) {
        int resolution = (i == normal) ? thickness_resolution : plane_resolution;
        binning[i] = malloc((resolution + 1) * sizeof(double));
        if (binning[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for binning %d on rank %d\n", i, ThisTask);
            // Free other allocated resources
            for (int j = 0; j < i; j++) {
                free(binning[j]);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        double start = (i == normal) ? (center - thickness / 2) : left_corner[i];
        double stop = (i == normal) ? (center + thickness / 2) : (left_corner[i] + Lbox);
        linspace(start, stop, resolution + 1, binning[i]);
    }


    // bin resolution (cell size in kpc/h)
    double bin_resolution[3];
    bin_resolution[plane_directions[0]] = Lbox / plane_resolution;
    bin_resolution[plane_directions[1]] = Lbox / plane_resolution;
    bin_resolution[normal] = thickness / thickness_resolution;

    // density normalization
    double density_normalization = bin_resolution[normal] * comoving_distance * pow(CM_PER_KPC/CP->HubbleParam, 2) / atime;


    // density 3D array
    GridDimensions dims;
    // Set dimensions based on the orientation specified by 'normal'
    dims.nx = (normal == 0) ? thickness_resolution : plane_resolution;
    dims.ny = (normal == 1) ? thickness_resolution : plane_resolution;
    dims.nz = (normal == 2) ? thickness_resolution : plane_resolution;
    // printf("nx: %d, ny: %d, nz: %d\n", dims.nx, dims.ny, dims.nz);

    double ***density = allocate_3d_array(dims.nx, dims.ny, dims.nz);

    grid3d_nfw(P, num_particles_rank, binning, dims, density);

    projectDensity(density, density_projected, dims, normal);

    free_3d_array(density, dims.nx, dims.ny);

    //number of particles on the plane
    int64_t num_particles_plane = 0;
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            num_particles_plane += density_projected[i][j];
        }
    }

    if (num_particles_plane == 0) {
        return 0;
    }
    // normalize the density to the density fluctuation
    double density_norm_factor = 1. / num_particles_tot * (pow(Lbox,3) / (bin_resolution[0] * bin_resolution[1] * bin_resolution[2]));

    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            density_projected[i][j] *= density_norm_factor;
        }
    }

    // Calculate the lensing potential by solving the Poisson equation
    calculate_lensing_potential(density_projected, plane_resolution, bin_resolution[plane_directions[0]], bin_resolution[plane_directions[1]], comoving_distance, smooth, lensing_potential);

    // normalize the lensing potential
    for (int i = 0; i < plane_resolution; i++) {
        for (int j = 0; j < plane_resolution; j++) {
            lensing_potential[i][j] *= cosmo_normalization * density_normalization;
        }
    }
    free_2d_array(density_projected, plane_resolution);
    // Free the binning arrays
    for (int i = 0; i < 3; i++) {
        free(binning[i]);
    }
    return num_particles_plane;
}

#ifdef USE_CFITSIO
void savePotentialPlane(double **data, int rows, int cols, const char filename[128], double Lbox, const Cosmology * CP, double redshift, double comoving_distance, int64_t num_particles) {
    fitsfile *fptr;       // Pointer to the FITS file; defined in fitsio.h
    int status = 0;       // Status must be initialized to zero.
    long naxes[2] = {cols, rows};  // image dimensions
    char newFilename[128];
    // Format the filename to include '!' to overwrite existing files
    sprintf(newFilename, "!%s", filename);

    // Create the file
    if (fits_create_file(&fptr, newFilename, &status)) {
        printf("Error creating FITS file: %s\n", filename);
        fits_report_error(stderr, status);
        return;
    }
    
    // Create the primary image (double precision)
    if (fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        return;
    }
    double H0 = CP->HubbleParam * 100;
    double Lbox_Mpc = Lbox / 1e3;
    double comoving_distance_Mpc = comoving_distance / 1e3;
    double Ode0 = CP->OmegaLambda > 0 ? CP->OmegaLambda : CP->Omega_fld;
    // Insert a blank line as a separator
    fits_write_record(fptr, "        ", &status);
    // Add headers to the FITS file
    fits_update_key(fptr, TDOUBLE, "H0", &H0, "Hubble constant in km/s*Mpc", &status);
    // fits_update_key(fptr, TSTRING, " ", &cosmo.h, "Dimensionless Hubble constant", &status);
    fits_update_key(fptr, TDOUBLE, "h", &CP->HubbleParam, "Dimensionless Hubble constant", &status);
    fits_update_key(fptr, TDOUBLE, "OMEGA_M", &CP->Omega0, "Dark Matter density", &status);
    fits_update_key(fptr, TDOUBLE, "OMEGA_L", &Ode0, "Dark Energy density", &status);
    fits_update_key(fptr, TDOUBLE, "W0", &CP->w0_fld, "Dark Energy equation of state", &status);
    fits_update_key(fptr, TDOUBLE, "WA", &CP->wa_fld, "Dark Energy running equation of state", &status);

    fits_update_key(fptr, TDOUBLE, "Z", &redshift, "Redshift of the lens plane", &status);
    fits_update_key(fptr, TDOUBLE, "CHI", (&comoving_distance_Mpc), "Comoving distance in Mpc/h", &status);
    fits_update_key(fptr, TDOUBLE, "SIDE", &(Lbox_Mpc), "Side length in Mpc/h", &status);
    fits_update_key(fptr, TLONGLONG, "NPART", &num_particles, "Number of particles on the plane", &status);
    fits_update_key(fptr, TSTRING, "UNIT", "rad2    ", "Pixel value unit", &status);

    // Allocate memory for a temporary 1D array to store the 2D data (for contiguous memory layout)
    double *tempData = malloc(rows * cols * sizeof(double));
    if (!tempData) {
        perror("Failed to allocate memory for temporary data storage");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        memcpy(tempData + i * cols, data[i], cols * sizeof(double));
    }

    // Write the 2D array of doubles to the image
    long fpixel[2] = {1, 1};  // first pixel to write (1-based indexing)
    if (fits_write_pix(fptr, TDOUBLE, fpixel, rows * cols, tempData, &status)) {
        fits_report_error(stderr, status);
        free(tempData);
        return;
    }

    free(tempData);


    // Close the FITS file
    if (fits_close_file(fptr, &status)) {
        printf("Error closing FITS file.\n");
        fits_report_error(stderr, status);
    }
}
#endif

