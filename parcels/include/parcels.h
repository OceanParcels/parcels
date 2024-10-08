#ifndef _PARCELS_H
#define _PARCELS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "random.h"
#include "index_search.h"
#include "interpolation_utils.h"

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct
{
  int xdim, ydim, zdim, tdim, igrid, allow_time_extrapolation, time_periodic, gridindexingtype, interp_method;
  float ****data_chunks;
  CGrid *grid;
} CField;

/* Bilinear interpolation routine for 2D grid */
static inline StatusCode spatial_interpolation_bilinear(double xsi, double eta, float data[2][2], float *value)
{
  *value = (1-xsi)*(1-eta) * data[0][0]
         +    xsi *(1-eta) * data[0][1]
         +    xsi *   eta  * data[1][1]
         + (1-xsi)*   eta  * data[1][0];
  return SUCCESS;
}

/* Bilinear interpolation routine for 2D grid for tracers with squared inverse distance weighting near land*/
static inline StatusCode spatial_interpolation_bilinear_invdist_land(double xsi, double eta, float data[2][2], float *value)
{
  int i, j, k, l, nb_land = 0, land[2][2] = {{0}};
  float w_sum = 0.;
  // count the number of surrounding land points (assume land is where the value is close to zero)
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      if (is_zero_flt(data[i][j])) {
	    land[i][j] = 1;
	    nb_land++;
      }
      else {
	    // record the coordinates of the last non-land point
	    // (for the case where this is the only location with valid data)
	    k = i;
	    l = j;
      }
    }
  }
  switch (nb_land) {
  case 0:  // no land, use usual routine
    return spatial_interpolation_bilinear(xsi, eta, data, value);
  case 3:  // single non-land point
    *value = data[k][l];
    return SUCCESS;
  case 4:  // only land
    *value = 0.;
    return SUCCESS;
  default:
    break;
  }
  // interpolate with 1 or 2 land points
  *value = 0.;
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      float distance = pow((xsi - j), 2) + pow((eta - i), 2);
      if (is_zero_flt(distance)) {
	    if (land[i][j] == 1) { // index search led us directly onto land
          *value = 0.;
          return SUCCESS;
	    }
	    else {
	      *value = data[i][j];
	      return SUCCESS;
	    }
      }
      else if (land[i][j] == 0) {
	    *value += data[i][j] / distance;
	    w_sum += 1 / distance;
      }
    }
  }
  *value /= w_sum;
  return SUCCESS;
}

/* Trilinear interpolation routine for 3D grid */
static inline StatusCode spatial_interpolation_trilinear(double xsi, double eta, double zeta,
                                                        float data[2][2][2], float *value)
{
  float f0, f1;
  f0 = (1-xsi)*(1-eta) * data[0][0][0]
     +    xsi *(1-eta) * data[0][0][1]
     +    xsi *   eta  * data[0][1][1]
     + (1-xsi)*   eta  * data[0][1][0];
  f1 = (1-xsi)*(1-eta) * data[1][0][0]
     +    xsi *(1-eta) * data[1][0][1]
     +    xsi *   eta  * data[1][1][1]
     + (1-xsi)*   eta  * data[1][1][0];
  *value = (1-zeta) * f0 + zeta * f1;
  return SUCCESS;
}

/* Trilinear interpolation routine for MOM surface 3D grid */
static inline StatusCode spatial_interpolation_trilinear_surface(double xsi, double eta, double zeta,
                                                                 float data[2][2][2], float *value)
{
  float f1;
  f1 = (1-xsi)*(1-eta) * data[0][0][0]
     +    xsi *(1-eta) * data[0][0][1]
     +    xsi *   eta  * data[0][1][1]
     + (1-xsi)*   eta  * data[0][1][0];
  *value = zeta * f1;
  return SUCCESS;
}

static inline StatusCode spatial_interpolation_trilinear_bottom(double xsi, double eta, double zeta,
                                                                 float data[2][2][2], float *value)
{
  float f1;
  f1 = (1-xsi)*(1-eta) * data[1][0][0]
     +    xsi *(1-eta) * data[1][0][1]
     +    xsi *   eta  * data[1][1][1]
     + (1-xsi)*   eta  * data[1][1][0];
  *value = (1 - zeta) * f1;
  return SUCCESS;
}

/* Trilinear interpolation routine for 3D grid for tracers with squared inverse distance weighting near land*/
static inline StatusCode spatial_interpolation_trilinear_invdist_land(double xsi, double eta, double zeta, float data[2][2][2], float *value)
{
  int i, j, k, l, m, n, nb_land = 0, land[2][2][2] = {{{0}}};
  float w_sum = 0.;
  // count the number of surrounding land points (assume land is where the value is close to zero)
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 2; k++) {
        if(is_zero_flt(data[i][j][k])) {
	      land[i][j][k] = 1;
	      nb_land++;
        }
        else {
	    // record the coordinates of the last non-land point
	    // (for the case where this is the only location with valid data)
          l = i;
          m = j;
          n = k;
        }
      }
    }
  }
  switch (nb_land) {
  case 0:  // no land, use usual routine
    return spatial_interpolation_trilinear(xsi, eta, zeta, data, value);
  case 7:  // single non-land point
    *value = data[l][m][n];
    return SUCCESS;
  case 8:  // only land
    *value = 0.;
    return SUCCESS;
  default:
    break;
  }
  // interpolate with 1 to 6 land points
  *value = 0.;
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) {
        for (k = 0; k < 2; k++) {
          float distance = pow((zeta - i), 2) + pow((eta - j), 2) + pow((xsi - k), 2);
          if (is_zero_flt(distance)) {
	        if (land[i][j][k] == 1) {
	          // index search led us directly onto land
              *value = 0.;
              return SUCCESS;
	        } else {
	          *value = data[i][j][k];
	          return SUCCESS;
	        }
        }
        else if (land[i][j][k] == 0) {
	      *value += data[i][j][k] / distance;
	      w_sum += 1 / distance;
        }
      }
    }
  }
  *value /= w_sum;
  return SUCCESS;
}

/* Nearest neighbor interpolation routine for 2D grid */
static inline StatusCode spatial_interpolation_nearest2D(double xsi, double eta,
                                                        float data[2][2], float *value)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int i, j;
  if (xsi < .5) {i = 0;} else {i = 1;}
  if (eta < .5) {j = 0;} else {j = 1;}
  *value = data[j][i];
  return SUCCESS;
}

/* C grid interpolation routine for tracers on 2D grid */
static inline StatusCode spatial_interpolation_tracer_bc_grid_2D(double _xsi, double _eta,
								float data[2][2], float *value)
{
  *value = data[1][1];
  return SUCCESS;
}

/* C grid interpolation routine for tracers on 3D grid */
static inline StatusCode spatial_interpolation_tracer_bc_grid_3D(double _xsi, double _eta, double _zeta,
								float data[2][2][2], float *value)
{
  *value = data[0][1][1];
  return SUCCESS;
}

static inline StatusCode spatial_interpolation_tracer_bc_grid_bottom(double _xsi, double _eta, double _zeta,
								float data[2][2][2], float *value)
{
  *value = data[1][1][1];
  return SUCCESS;
}

/* Nearest neighbor interpolation routine for 3D grid */
static inline StatusCode spatial_interpolation_nearest3D(double xsi, double eta, double zeta,
                                                        float data[2][2][2], float *value)
{
  int i, j, k;
  if (xsi < .5) {i = 0;} else {i = 1;}
  if (eta < .5) {j = 0;} else {j = 1;}
  if (zeta < .5) {k = 0;} else {k = 1;}
  *value = data[k][j][i];
  return SUCCESS;
}

static inline int getBlock2D(int *chunk_info, int yi, int xi, int *block, int *index_local)
{
  int ndim = chunk_info[0];
  if (ndim != 2)
    exit(-1);
  int i, j;

  int shape[ndim];
  int index[2] = {yi, xi};
  for(i=0; i<ndim; ++i){
    int chunk_sum = 0, shift = 0;
    for (j = 0; j < i; j++) shift += chunk_info[1+j];
    shape[i] = chunk_info[1+i];
    for (j=0; j<shape[i]; j++) {
      chunk_sum += chunk_info[1+ndim+shift+j];
      if (index[i] < chunk_sum) {
        chunk_sum -= chunk_info[1+ndim+shift+j];
        break;
      }
    }
    block[i] = j;
    index_local[i] = index[i] - chunk_sum;
  }

  int bid =  block[0]*shape[1] +
             block[1];
  return bid;
}

static inline StatusCode getCell2D(CField *f, int xi, int yi, int ti, float cell_data[2][2][2], int first_tstep_only)
{
  CStructuredGrid *grid = f->grid->grid;
  int *chunk_info = grid->chunk_info;
  int ndim = chunk_info[0];
  int block[ndim];
  int ilocal[ndim];

  int tii, yii, xii;

  int blockid = getBlock2D(chunk_info, yi, xi, block, ilocal);
  if (grid->load_chunk[blockid] < 2){
    grid->load_chunk[blockid] = 1;
    return REPEAT;
  }
  grid->load_chunk[blockid] = 2;
  int zdim = 1;
  int ydim = chunk_info[1+ndim+block[0]];
  int yshift = chunk_info[1];
  int xdim = chunk_info[1+ndim+yshift+block[1]];

  if (((ilocal[0] == ydim-1) && (ydim > 1)) || ((ilocal[1] == xdim-1) && (xdim > 1)))
  {
    // Cell is on multiple chunks
    for (tii=0; tii<2; ++tii){
      for (yii=0; yii<2; ++yii){
        for (xii=0; xii<2; ++xii){
          blockid = getBlock2D(chunk_info, yi+yii, xi+xii, block, ilocal);
          if (grid->load_chunk[blockid] < 2){
            grid->load_chunk[blockid] = 1;
            return REPEAT;
          }
          grid->load_chunk[blockid] = 2;
          zdim = 1;
          ydim = chunk_info[1+ndim+block[0]];
          yshift = chunk_info[1];
          xdim = chunk_info[1+ndim+yshift+block[1]];
          float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
          float (*data)[xdim] = (float (*)[xdim]) (data_block[ti+tii]);
          cell_data[tii][yii][xii] = data[ilocal[0]][ilocal[1]];
        }
      }
      if (first_tstep_only == 1)
         break;
    }
  }
  else
  {
    float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
    for (tii=0; tii<2; ++tii){
      float (*data)[xdim] = (float (*)[xdim]) (data_block[ti+tii]);
      int xiid = ((xdim==1) ? 0 : 1);
      int yiid = ((ydim==1) ? 0 : 1);
      for (yii=0; yii<2; yii++)
        for (xii=0; xii<2; xii++)
          cell_data[tii][yii][xii] = data[ilocal[0]+(yii*yiid)][ilocal[1]+(xii*xiid)];
      if (first_tstep_only == 1)
         break;
    }
  }
  return SUCCESS;
}

static inline int getBlock3D(int *chunk_info, int zi, int yi, int xi, int *block, int *index_local)
{
  int ndim = chunk_info[0];
  if (ndim != 3)
    exit(-1);
  int i, j;

  int shape[ndim];
  int index[3] = {zi, yi, xi};
  for(i=0; i<ndim; ++i){
    int chunk_sum = 0, shift = 0;
    for (j = 0; j < i; j++) shift += chunk_info[1+j];
    shape[i] = chunk_info[1+i];
    for (j=0; j<shape[i]; j++) {
      chunk_sum += chunk_info[1+ndim+shift+j];
      if (index[i] < chunk_sum) {
        chunk_sum -= chunk_info[1+ndim+shift+j];
        break;
      }
    }
    block[i] = j;
    index_local[i] = index[i] - chunk_sum;
  }

  int bid =  block[0]*shape[1]*shape[2] +
             block[1]*shape[2] +
             block[2];
  return bid;
}

static inline StatusCode getCell3D(CField *f, int xi, int yi, int zi, int ti, float cell_data[2][2][2][2], int first_tstep_only)
{
  CStructuredGrid *grid = f->grid->grid;
  int *chunk_info = grid->chunk_info;
  int ndim = chunk_info[0];
  int block[ndim];
  int ilocal[ndim];

  int tii, zii, yii, xii;

  int blockid = getBlock3D(chunk_info, zi, yi, xi, block, ilocal);
  if (grid->load_chunk[blockid] < 2){
    grid->load_chunk[blockid] = 1;
    return REPEAT;
  }
  grid->load_chunk[blockid] = 2;
  int zdim = chunk_info[1+ndim+block[0]];
  int zshift = chunk_info[1];
  int ydim = chunk_info[1+ndim+zshift+block[1]];
  int yshift = chunk_info[1+1];
  int xdim = chunk_info[1+ndim+zshift+yshift+block[2]];

  if (((ilocal[0] == zdim-1) && zdim > 1) || ((ilocal[1] == ydim-1) && ydim > 1) || ((ilocal[2] == xdim-1) && xdim >1))
  {
    // Cell is on multiple chunks\n
    for (tii=0; tii<2; ++tii){
      for (zii=0; zii<2; ++zii){
        for (yii=0; yii<2; ++yii){
          for (xii=0; xii<2; ++xii){
            blockid = getBlock3D(chunk_info, zi+zii, yi+yii, xi+xii, block, ilocal);
            if (grid->load_chunk[blockid] < 2){
              grid->load_chunk[blockid] = 1;
              return REPEAT;
            }
            grid->load_chunk[blockid] = 2;
            zdim = chunk_info[1+ndim+block[0]];
            zshift = chunk_info[1];
            ydim = chunk_info[1+ndim+zshift+block[1]];
            yshift = chunk_info[1+1];
            xdim = chunk_info[1+ndim+zshift+yshift+block[2]];
            float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
            float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) (data_block[ti+tii]);
            cell_data[tii][zii][yii][xii] = data[ilocal[0]][ilocal[1]][ilocal[2]];
          }
        }
      }
      if (first_tstep_only == 1)
         break;
    }
  }
  else
  {
    float (*data_block)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) f->data_chunks[blockid];
    for (tii=0; tii<2; ++tii){
      float (*data)[ydim][xdim] = (float (*)[ydim][xdim]) (data_block[ti+tii]);
      int xiid = ((xdim==1) ? 0 : 1);
      int yiid = ((ydim==1) ? 0 : 1);
      int ziid = ((zdim==1) ? 0 : 1);
      for (zii=0; zii<2; zii++)
        for (yii=0; yii<2; yii++)
          for (xii=0; xii<2; xii++)
            cell_data[tii][zii][yii][xii] = data[ilocal[0]+(zii*ziid)][ilocal[1]+(yii*yiid)][ilocal[2]+(xii*xiid)];
      if (first_tstep_only == 1)
         break;
    }
  }
  return SUCCESS;
}


/* Linear interpolation along the time axis */
static inline StatusCode temporal_interpolation_structured_grid(type_coord x, type_coord y, type_coord z, double time, CField *f,
                                                               GridType gtype, int *xi, int *yi, int *zi, int *ti,
                                                               float *value)
{
  StatusCode status;
  CStructuredGrid *grid = f->grid->grid;
  GridIndexingType gridindexingtype = f->gridindexingtype;
  InterpMethod interp_method = f->interp_method;
  int igrid = f->igrid;

  /* Find time index for temporal interpolation */
  if (f->time_periodic == 0 && f->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERRORTIMEEXTRAPOLATION;
  }
  status = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], f->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

  double xsi, eta, zeta;

  float data2D[2][2][2];
  float data3D[2][2][2][2];

  // if we're in between time indices, and not at the end of the timeseries,
  // we'll make sure to interpolate data between the two time values
  // otherwise, we'll only use the data at the current time index
  int tii = (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) ? 2 : 1;

  float val[2] = {0.0f, 0.0f};
  double t0 = grid->time[ti[igrid]];
  // we set our second time bound and search time depending on the
  // index critereon above
  double t1 = (tii == 2) ? grid->time[ti[igrid]+1] : t0+1;
  double tsrch = (tii == 2) ? time : t0;

  status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid],
			  &xsi, &eta, &zeta, gtype, ti[igrid],
			  tsrch, t0, t1, interp_method, gridindexingtype);
  CHECKSTATUS(status);

  if (grid->zdim == 1) {
    // last param is a flag, which denotes that we only want the first timestep
    // (rather than both)
    status = getCell2D(f, xi[igrid], yi[igrid], ti[igrid], data2D, tii == 1); CHECKSTATUS(status);
  } else {
    if ((gridindexingtype == MOM5) && (zi[igrid] == -1)) {
      status = getCell3D(f, xi[igrid], yi[igrid], 0, ti[igrid], data3D, tii == 1); CHECKSTATUS(status);
    } else if ((gridindexingtype == POP) && (zi[igrid] == grid->zdim-2)) {
      status = getCell3D(f, xi[igrid], yi[igrid], zi[igrid]-1, ti[igrid], data3D, tii == 1); CHECKSTATUS(status);
    } else {
      status = getCell3D(f, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D, tii == 1); CHECKSTATUS(status);
    }
  }

  // define a helper macro that will select the appropriate interpolation method
  // depending on whether we need 2D or 3D
#define INTERP(fn_2d, fn_3d)                                            \
  do {                                                                  \
    if (grid->zdim == 1) {                                              \
      for (int i = 0; i < tii; i++) {                                   \
        status = fn_2d(xsi, eta, data2D[i], &val[i]);                   \
        CHECKSTATUS(status);                                            \
      }                                                                 \
    } else {                                                            \
      for (int i = 0; i < tii; i++) {                                   \
        status = fn_3d(xsi, eta, zeta, data3D[i], &val[i]);             \
        CHECKSTATUS(status);                                            \
      }                                                                 \
    }                                                                   \
  } while (0)

  if ((interp_method == LINEAR) || (interp_method == CGRID_VELOCITY) ||
      (interp_method == BGRID_VELOCITY) || (interp_method == BGRID_W_VELOCITY)) {
    // adjust the normalised coordinate for flux-based interpolation methods
    if ((interp_method == CGRID_VELOCITY) || (interp_method == BGRID_W_VELOCITY)) {
      if ((gridindexingtype == NEMO)   || (gridindexingtype == MOM5) || (gridindexingtype == POP)) {
        // velocity is on the northeast of a tracer cell
        xsi = 1;
        eta = 1;
      } else if (gridindexingtype == MITGCM) {
        // velocity is on the southwest of a tracer cell
        xsi = 0;
        eta = 0;
      }
    } else if (interp_method == BGRID_VELOCITY) {
      if (gridindexingtype == MOM5) {
        zeta = 1;
      } else {
        zeta = 0;
      }
    }
    if ((gridindexingtype == MOM5) && (zi[igrid] == -1)) {
      INTERP(spatial_interpolation_bilinear, spatial_interpolation_trilinear_surface);
    } else if ((gridindexingtype == POP) && (zi[igrid] == grid->zdim-2)) {
      INTERP(spatial_interpolation_bilinear, spatial_interpolation_trilinear_bottom);
    } else {
      INTERP(spatial_interpolation_bilinear, spatial_interpolation_trilinear);
    }
  } else if (interp_method == NEAREST) {
    INTERP(spatial_interpolation_nearest2D, spatial_interpolation_nearest3D);
  } else if ((interp_method == CGRID_TRACER) || (interp_method == BGRID_TRACER)) {
    if ((gridindexingtype == POP) && (zi[igrid] == grid->zdim-2)) {
      INTERP(spatial_interpolation_tracer_bc_grid_2D, spatial_interpolation_tracer_bc_grid_bottom);
    } else {
      INTERP(spatial_interpolation_tracer_bc_grid_2D, spatial_interpolation_tracer_bc_grid_3D);
    }
  } else if (interp_method == LINEAR_INVDIST_LAND_TRACER) {
    INTERP(spatial_interpolation_bilinear_invdist_land, spatial_interpolation_trilinear_invdist_land);
  } else {
    return ERROR;
  }

  // tsrch = t0 in the case where val[1] isn't populated, so this
  // gives the right interpolation in either case
  *value = val[0] + (val[1] - val[0]) * (float)((tsrch - t0) / (t1 - t0));

  return SUCCESS;
#undef INTERP
}

static double dist(double lon1, double lon2, double lat1, double lat2, int sphere_mesh, double lat)
{
  if (sphere_mesh == 1){
    double rad = M_PI / 180.;
    double deg2m = 1852 * 60.;
    return sqrt((lon2-lon1)*(lon2-lon1) * deg2m * deg2m * cos(rad * lat) * cos(rad * lat) + (lat2-lat1)*(lat2-lat1) * deg2m * deg2m);
  }
  else{
    return sqrt((lon2-lon1)*(lon2-lon1) + (lat2-lat1)*(lat2-lat1));
  }
}

/* Linear interpolation routine for 2D C grid */
static inline StatusCode spatial_interpolation_UV_c_grid(double xsi, double eta, int xi, int yi, CStructuredGrid *grid,
                                                        GridType gtype, float dataU[2][2], float dataV[2][2], float *u, float *v)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int xdim = grid->xdim;

  double xgrid_loc[4];
  double ygrid_loc[4];
  int iN;
  if( (gtype == RECTILINEAR_Z_GRID) || (gtype == RECTILINEAR_S_GRID) ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[yi+iN/2][xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }


  double phi[4];
  phi2D_lin(0., eta, phi);
  double U0 = dataU[1][0] * dist(xgrid_loc[3], xgrid_loc[0], ygrid_loc[3], ygrid_loc[0], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(1., eta, phi);
  double U1 = dataU[1][1] * dist(xgrid_loc[1], xgrid_loc[2], ygrid_loc[1], ygrid_loc[2], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 0., phi);
  double V0 = dataV[0][1] * dist(xgrid_loc[0], xgrid_loc[1], ygrid_loc[0], ygrid_loc[1], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  phi2D_lin(xsi, 1., phi);
  double V1 = dataV[1][1] * dist(xgrid_loc[2], xgrid_loc[3], ygrid_loc[2], ygrid_loc[3], grid->sphere_mesh, dot_prod(phi, ygrid_loc, 4));
  double U = (1-xsi) * U0 + xsi * U1;
  double V = (1-eta) * V0 + eta * V1;

  double dphidxsi[4] = {eta-1, 1-eta, eta, -eta};
  double dphideta[4] = {xsi-1, -xsi, xsi, 1-xsi};
  double dxdxsi = 0; double dxdeta = 0;
  double dydxsi = 0; double dydeta = 0;
  int i;
  for(i=0; i<4; ++i){
    dxdxsi += xgrid_loc[i] *dphidxsi[i];
    dxdeta += xgrid_loc[i] *dphideta[i];
    dydxsi += ygrid_loc[i] *dphidxsi[i];
    dydeta += ygrid_loc[i] *dphideta[i];
  }
  double meshJac = 1;
  if (grid->sphere_mesh == 1){
    double deg2m = 1852 * 60.;
    double rad = M_PI / 180.;
    phi2D_lin(xsi, eta, phi);
    double lat = dot_prod(phi, ygrid_loc, 4);
    meshJac = deg2m * deg2m * cos(rad * lat);
  }
  double jac = (dxdxsi*dydeta - dxdeta * dydxsi) * meshJac;

  *u = ( (-(1-eta) * U - (1-xsi) * V ) * xgrid_loc[0] +
         ( (1-eta) * U -  xsi    * V ) * xgrid_loc[1] +
         (    eta  * U +  xsi    * V ) * xgrid_loc[2] +
         (   -eta  * U + (1-xsi) * V ) * xgrid_loc[3] ) / jac;
  *v = ( (-(1-eta) * U - (1-xsi) * V ) * ygrid_loc[0] +
         ( (1-eta) * U -  xsi    * V ) * ygrid_loc[1] +
         (    eta  * U +  xsi    * V ) * ygrid_loc[2] +
         (   -eta  * U + (1-xsi) * V ) * ygrid_loc[3] ) / jac;

  return SUCCESS;
}



static inline StatusCode temporal_interpolationUV_c_grid(type_coord x, type_coord y, type_coord z, double time, CField *U, CField *V,
                                                         GridType gtype, int *xi, int *yi, int *zi, int *ti,
                                                         float *u, float *v)
{
  StatusCode status;
  CStructuredGrid *grid = U->grid->grid;
  GridIndexingType gridindexingtype = U->gridindexingtype;

  int igrid = U->igrid;

  /* Find time index for temporal interpolation */
  if (U->time_periodic == 0 && U->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERRORTIMEEXTRAPOLATION;
  }
  status = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

  double xsi, eta, zeta;


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gtype, ti[igrid], time, t0, t1, CGRID_VELOCITY, gridindexingtype); CHECKSTATUS(status);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2];
      if (gridindexingtype == NEMO) {
        status = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 0); CHECKSTATUS(status);
        status = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 0); CHECKSTATUS(status);
      }
      else if (gridindexingtype == MITGCM) {
        status = getCell2D(U, xi[igrid], yi[igrid]-1, ti[igrid], data2D_U, 0); CHECKSTATUS(status);
        status = getCell2D(V, xi[igrid]-1, yi[igrid], ti[igrid], data2D_V, 0); CHECKSTATUS(status);
      }
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data2D_U[0], data2D_V[0], &u0, &v0); CHECKSTATUS(status);
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data2D_U[1], data2D_V[1], &u1, &v1); CHECKSTATUS(status);

    } else {
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2];
      if (gridindexingtype == NEMO) {
        status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 0); CHECKSTATUS(status);
        status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKSTATUS(status);
      }
      else if (gridindexingtype == MITGCM) {
        status = getCell3D(U, xi[igrid], yi[igrid]-1, zi[igrid], ti[igrid], data3D_U, 0); CHECKSTATUS(status);
        status = getCell3D(V, xi[igrid]-1, yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKSTATUS(status);
      }
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data3D_U[0][0], data3D_V[0][0], &u0, &v0); CHECKSTATUS(status);
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data3D_U[1][0], data3D_V[1][0], &u1, &v1); CHECKSTATUS(status);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gtype, ti[igrid], t0, t0, t0+1, CGRID_VELOCITY, gridindexingtype); CHECKSTATUS(status);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2];
      if (gridindexingtype == NEMO) {
        status = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 1); CHECKSTATUS(status);
        status = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 1); CHECKSTATUS(status);
      }
      else if (gridindexingtype == MITGCM) {
        status = getCell2D(U, xi[igrid], yi[igrid]-1, ti[igrid], data2D_U, 1); CHECKSTATUS(status);
        status = getCell2D(V, xi[igrid]-1, yi[igrid], ti[igrid], data2D_V, 1); CHECKSTATUS(status);
      }
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data2D_U[0], data2D_V[0], u, v); CHECKSTATUS(status);
    }
    else{
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2];
      if (gridindexingtype == NEMO) {
        status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 1); CHECKSTATUS(status);
        status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKSTATUS(status);
      }
      else if (gridindexingtype == MITGCM){
        status = getCell3D(U, xi[igrid], yi[igrid]-1, zi[igrid], ti[igrid], data3D_U, 1); CHECKSTATUS(status);
        status = getCell3D(V, xi[igrid]-1, yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKSTATUS(status);
      }
      status = spatial_interpolation_UV_c_grid(xsi, eta, xi[igrid], yi[igrid], grid, gtype, data3D_U[0][0], data3D_V[0][0], u, v); CHECKSTATUS(status);
    }
    return SUCCESS;
  }
}

/* Quadratic interpolation routine for 3D C grid */
static inline StatusCode spatial_interpolation_UVW_c_grid(double xsi, double eta, double zet, int xi, int yi, int zi, int ti, CStructuredGrid *grid,
                                                        GridType gtype, float dataU[2][2][2], float dataV[2][2][2], float dataW[2][2][2], float *u, float *v, float *w)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  int xdim = grid->xdim;
  int ydim = grid->ydim;
  int zdim = grid->zdim;

  float xgrid_loc[4];
  float ygrid_loc[4];
  int iN;
  if( gtype == RECTILINEAR_S_GRID ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      xgrid_loc[iN] = xgrid[yi+iN/2][xi+min(1, (iN%3))];
      ygrid_loc[iN] = ygrid[yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  int i4;
  for (i4 = 1; i4 < 4; ++i4){
    if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
    if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
  }

  float u0 = dataU[0][1][0];
  float u1 = dataU[0][1][1];
  float v0 = dataV[0][0][1];
  float v1 = dataV[0][1][1];
  float w0 = dataW[0][1][1];
  float w1 = dataW[1][1][1];

  double px[8] = {xgrid_loc[0], xgrid_loc[1], xgrid_loc[2], xgrid_loc[3],
                  xgrid_loc[0], xgrid_loc[1], xgrid_loc[2], xgrid_loc[3]};
  double py[8] = {ygrid_loc[0], ygrid_loc[1], ygrid_loc[2], ygrid_loc[3],
                  ygrid_loc[0], ygrid_loc[1], ygrid_loc[2], ygrid_loc[3]};
  double pz[8];
  if (grid->z4d == 1){
    float (*zvals)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) grid->depth;
    for (iN=0; iN < 4; ++iN){
      pz[iN] = zvals[ti][zi][yi+iN/2][xi+min(1, (iN%3))];
      pz[iN+4] = zvals[ti][zi+1][yi+iN/2][xi+min(1, (iN%3))];
    }
  }
  else{
    float (*zvals)[ydim][xdim] = (float (*)[ydim][xdim]) grid->depth;
    for (iN=0; iN < 4; ++iN){
      pz[iN] = zvals[zi][yi+iN/2][xi+min(1, (iN%3))];
      pz[iN+4] = zvals[zi+1][yi+iN/2][xi+min(1, (iN%3))];
    }
  }

  double U0 = u0 * jacobian3D_lin_face(px, py, pz, 0, eta, zet, ZONAL, grid->sphere_mesh);
  double U1 = u1 * jacobian3D_lin_face(px, py, pz, 1, eta, zet, ZONAL, grid->sphere_mesh);
  double V0 = v0 * jacobian3D_lin_face(px, py, pz, xsi, 0, zet, MERIDIONAL, grid->sphere_mesh);
  double V1 = v1 * jacobian3D_lin_face(px, py, pz, xsi, 1, zet, MERIDIONAL, grid->sphere_mesh);
  double W0 = w0 * jacobian3D_lin_face(px, py, pz, xsi, eta, 0, VERTICAL, grid->sphere_mesh);
  double W1 = w1 * jacobian3D_lin_face(px, py, pz, xsi, eta, 1, VERTICAL, grid->sphere_mesh);

  // Computing fluxes in half left hexahedron -> flux_u05
  double xxu[8] = {px[0], (px[0]+px[1])/2, (px[2]+px[3])/2, px[3], px[4], (px[4]+px[5])/2, (px[6]+px[7])/2, px[7]};
  double yyu[8] = {py[0], (py[0]+py[1])/2, (py[2]+py[3])/2, py[3], py[4], (py[4]+py[5])/2, (py[6]+py[7])/2, py[7]};
  double zzu[8] = {pz[0], (pz[0]+pz[1])/2, (pz[2]+pz[3])/2, pz[3], pz[4], (pz[4]+pz[5])/2, (pz[6]+pz[7])/2, pz[7]};
  double flux_u0 = u0 * jacobian3D_lin_face(xxu, yyu, zzu, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0_halfx = v0 * jacobian3D_lin_face(xxu, yyu, zzu, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_v1_halfx = v1 * jacobian3D_lin_face(xxu, yyu, zzu, .5, 1, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0_halfx = w0 * jacobian3D_lin_face(xxu, yyu, zzu, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w1_halfx = w1 * jacobian3D_lin_face(xxu, yyu, zzu, .5, .5, 1, VERTICAL, grid->sphere_mesh);
  double flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx;

  // Computing fluxes in half front hexahedron -> flux_v05
  double xxv[8] = {px[0], px[1], (px[1]+px[2])/2, (px[0]+px[3])/2, px[4], px[5], (px[5]+px[6])/2, (px[4]+px[7])/2};
  double yyv[8] = {py[0], py[1], (py[1]+py[2])/2, (py[0]+py[3])/2, py[4], py[5], (py[5]+py[6])/2, (py[4]+py[7])/2};
  double zzv[8] = {pz[0], pz[1], (pz[1]+pz[2])/2, (pz[0]+pz[3])/2, pz[4], pz[5], (pz[5]+pz[6])/2, (pz[4]+pz[7])/2};
  double flux_u0_halfy = u0 * jacobian3D_lin_face(xxv, yyv, zzv, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_u1_halfy = u1 * jacobian3D_lin_face(xxv, yyv, zzv, 1, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0 = v0 * jacobian3D_lin_face(xxv, yyv, zzv, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0_halfy = w0 * jacobian3D_lin_face(xxv, yyv, zzv, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w1_halfy = w1 * jacobian3D_lin_face(xxv, yyv, zzv, .5, .5, 1, VERTICAL, grid->sphere_mesh);
  double flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy;

  // Computing fluxes in half lower hexahedron -> flux_w05
  double xx[8] = {px[0], px[1], px[2], px[3], (px[0]+px[4])/2, (px[1]+px[5])/2, (px[2]+px[6])/2, (px[3]+px[7])/2};
  double yy[8] = {py[0], py[1], py[2], py[3], (py[0]+py[4])/2, (py[1]+py[5])/2, (py[2]+py[6])/2, (py[3]+py[7])/2};
  double zz[8] = {pz[0], pz[1], pz[2], pz[3], (pz[0]+pz[4])/2, (pz[1]+pz[5])/2, (pz[2]+pz[6])/2, (pz[3]+pz[7])/2};
  double flux_u0_halfz = u0 * jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_u1_halfz = u1 * jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, ZONAL, grid->sphere_mesh);
  double flux_v0_halfz = v0 * jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_v1_halfz = v1 * jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, MERIDIONAL, grid->sphere_mesh);
  double flux_w0 = w0 * jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, VERTICAL, grid->sphere_mesh);
  double flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0;

  double surf_u05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, ZONAL, grid->sphere_mesh);
  double jac_u05 = jacobian3D_lin_face(px, py, pz, .5, eta, zet, ZONAL, grid->sphere_mesh);
  double U05 = flux_u05 / surf_u05 * jac_u05;

  double surf_v05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, MERIDIONAL, grid->sphere_mesh);
  double jac_v05 = jacobian3D_lin_face(px, py, pz, xsi, .5, zet, MERIDIONAL, grid->sphere_mesh);
  double V05 = flux_v05 / surf_v05 * jac_v05;

  double surf_w05 = jacobian3D_lin_face(px, py, pz, .5, .5, .5, VERTICAL, grid->sphere_mesh);
  double jac_w05 = jacobian3D_lin_face(px, py, pz, xsi, eta, .5, VERTICAL, grid->sphere_mesh);
  double W05 = flux_w05 / surf_w05 * jac_w05;

  double jac = jacobian3D_lin(px, py, pz, xsi, eta, zet, grid->sphere_mesh);

  double phi[3];
  phi1D_quad(xsi, phi);
  double uvec[3] = {U0, U05, U1};
  double dxsidt = dot_prod(phi, uvec, 3) / jac;
  phi1D_quad(eta, phi);
  double vvec[3] = {V0, V05, V1};
  double detadt = dot_prod(phi, vvec, 3) / jac;
  phi1D_quad(zet, phi);
  double wvec[3] = {W0, W05, W1};
  double dzetdt = dot_prod(phi, wvec, 3) / jac;

  double dphidxsi[8], dphideta[8], dphidzet[8];
  dphidxsi3D_lin(xsi, eta, zet, dphidxsi, dphideta, dphidzet);

  *u = dot_prod(dphidxsi, px, 8) * dxsidt + dot_prod(dphideta, px, 8) * detadt + dot_prod(dphidzet, px, 8) * dzetdt;
  *v = dot_prod(dphidxsi, py, 8) * dxsidt + dot_prod(dphideta, py, 8) * detadt + dot_prod(dphidzet, py, 8) * dzetdt;
  *w = dot_prod(dphidxsi, pz, 8) * dxsidt + dot_prod(dphideta, pz, 8) * detadt + dot_prod(dphidzet, pz, 8) * dzetdt;

  return SUCCESS;
}

static inline StatusCode temporal_interpolationUVW_c_grid(type_coord x, type_coord y, type_coord z, double time, CField *U, CField *V, CField *W,
                                                         GridType gtype, int *xi, int *yi, int *zi, int *ti,
                                                         float *u, float *v, float *w)
{
  StatusCode status;
  CStructuredGrid *grid = U->grid->grid;
  GridIndexingType gridindexingtype = U->gridindexingtype;
  int igrid = U->igrid;

  /* Find time index for temporal interpolation */
  if (U->time_periodic == 0 && U->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERRORTIMEEXTRAPOLATION;
  }
  status = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

  double xsi, eta, zet;
  float data3D_U[2][2][2][2];
  float data3D_V[2][2][2][2];
  float data3D_W[2][2][2][2];


  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1, w0, w1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zet, gtype, ti[igrid], time, t0, t1, CGRID_VELOCITY, gridindexingtype); CHECKSTATUS(status);
    status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 0); CHECKSTATUS(status);
    status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKSTATUS(status);
    status = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 0); CHECKSTATUS(status);
    if (grid->zdim==1){
      return ERROR;
    } else {
      status = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid],   grid, gtype, data3D_U[0], data3D_V[0], data3D_W[0], &u0, &v0, &w0); CHECKSTATUS(status);
      status = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid]+1, grid, gtype, data3D_U[1], data3D_V[1], data3D_W[1], &u1, &v1, &w1); CHECKSTATUS(status);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    *w = w0 + (w1 - w0) * (float)((time - t0) / (t1 - t0));
    return SUCCESS;
  } else {
    double t0 = grid->time[ti[igrid]];
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zet, gtype, ti[igrid], t0, t0, t0+1, CGRID_VELOCITY, gridindexingtype); CHECKSTATUS(status);
    status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 1); CHECKSTATUS(status);
    status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKSTATUS(status);
    status = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 1); CHECKSTATUS(status);
    if (grid->zdim==1){
      return ERROR;
    }
    else{
      status = spatial_interpolation_UVW_c_grid(xsi, eta, zet, xi[igrid], yi[igrid], zi[igrid], ti[igrid], grid, gtype, data3D_U[0], data3D_V[0], data3D_W[0], u, v, w); CHECKSTATUS(status);
    }
    return SUCCESS;
  }
}

static inline StatusCode calculate_slip_conditions_2D(double xsi, double eta, float dataU[2][2],
                                                      float dataV[2][2], float dataW[2][2],
                                                      float *u, float *v, float *w, int interp_method, int withW)
{
      float f_u = 1, f_v = 1, f_w = 1;
      if ((is_zero_flt(dataU[0][0])) && (is_zero_flt(dataU[0][1])) &&
          (is_zero_flt(dataV[0][0])) && (is_zero_flt(dataV[0][1])) && eta > 0.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (.5 + .5 * eta) / eta;
          if (withW) {
            f_w = f_w * (.5 + .5 * eta) / eta;
          }
        } else if (interp_method == FREESLIP) {
          f_u = f_u / eta;
          if (withW) {
            f_w = f_w / eta;
          }
        }
      }
      if ((is_zero_flt(dataU[1][0])) && (is_zero_flt(dataU[1][1])) &&
          (is_zero_flt(dataV[1][0])) && (is_zero_flt(dataV[1][1])) && eta < 1.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (1 - .5 * eta) / (1 - eta);
          if (withW) {
            f_w = f_w * (1 - .5 * eta) / (1 - eta);
          }
        } else if (interp_method == FREESLIP) {
          f_u = f_u / (1 - eta);
          if (withW) {
            f_w = f_w / (1 - eta);
          }
        }
      }
      if ((is_zero_flt(dataU[0][0])) && (is_zero_flt(dataU[1][0])) &&
          (is_zero_flt(dataV[0][0])) && (is_zero_flt(dataV[1][0])) && xsi > 0.){
        if (interp_method == PARTIALSLIP) {
          f_v = f_v * (.5 + .5 * xsi) / xsi;
          if (withW) {
            f_w = f_w * (.5 + .5 * xsi) / xsi;
          }
        } else if (interp_method == FREESLIP) {
          f_v = f_v / xsi;
          if (withW) {
            f_w = f_w / xsi;
          }
        }
      }
      if ((is_zero_flt(dataU[0][1])) && (is_zero_flt(dataU[1][1])) &&
          (is_zero_flt(dataV[0][1])) && (is_zero_flt(dataV[1][1])) && xsi < 1.){
        if (interp_method == PARTIALSLIP) {
          f_v = f_v * (1 - .5 * xsi) / (1 - xsi);
          if (withW) {
            f_w = f_w * (1 - .5 * xsi) / (1 - xsi);
          }
        } else if (interp_method == FREESLIP) {
          f_v = f_v / (1 - xsi);
          if (withW) {
            f_w = f_w / (1 - xsi);
          }
        }
      }
      *u *= f_u;
      *v *= f_v;
      if (withW) {
        *w *= f_w;
      }

  return SUCCESS;
}

static inline StatusCode calculate_slip_conditions_3D(double xsi, double eta, double zeta, float dataU[2][2][2],
                                                      float dataV[2][2][2], float dataW[2][2][2],
                                                      float *u, float *v, float *w, int interp_method, int withW)
{
      float f_u = 1, f_v = 1, f_w = 1;
      if ((is_zero_flt(dataU[0][0][0])) && (is_zero_flt(dataU[0][0][1])) && (is_zero_flt(dataU[1][0][0])) && (is_zero_flt(dataU[1][0][1])) &&
          (is_zero_flt(dataV[0][0][0])) && (is_zero_flt(dataV[0][0][1])) && (is_zero_flt(dataV[1][0][0])) && (is_zero_flt(dataV[1][0][1])) &&
          eta > 0.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (.5 + .5 * eta) / eta;
          if (withW) {
            f_w = f_w * (.5 + .5 * eta) / eta;
          }
        } else if (interp_method == FREESLIP) {
          f_u = f_u / eta;
          if (withW) {
            f_w = f_w / eta;
          }
        }
      }
      if ((is_zero_flt(dataU[0][1][0])) && (is_zero_flt(dataU[0][1][1])) && (is_zero_flt(dataU[1][1][0])) && (is_zero_flt(dataU[1][1][1])) &&
          (is_zero_flt(dataV[0][1][0])) && (is_zero_flt(dataV[0][1][1])) && (is_zero_flt(dataV[1][1][0])) && (is_zero_flt(dataV[1][1][1])) &&
           eta < 1.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (1 - .5 * eta) / (1 - eta);
          if (withW) {
            f_w = f_w * (1 - .5 * eta) / (1 - eta);
          }
        } else if (interp_method == FREESLIP) {
          f_u = f_u / (1 - eta);
          if (withW) {
            f_w = f_w / (1 - eta);
          }
        }
      }
      if ((is_zero_flt(dataU[0][0][0])) && (is_zero_flt(dataU[0][1][0])) && (is_zero_flt(dataU[1][0][0])) && (is_zero_flt(dataU[1][1][0])) &&
          (is_zero_flt(dataV[0][0][0])) && (is_zero_flt(dataV[0][1][0])) && (is_zero_flt(dataV[1][0][0])) && (is_zero_flt(dataV[1][1][0])) &&
          xsi > 0.){
        if (interp_method == PARTIALSLIP) {
          f_v = f_v * (.5 + .5 * xsi) / xsi;
          if (withW) {
            f_w = f_w * (.5 + .5 * xsi) / xsi;
          }
        } else if (interp_method == FREESLIP) {
          f_v = f_v / xsi;
          if (withW) {
            f_w = f_w / xsi;
          }
        }
      }
      if ((is_zero_flt(dataU[0][0][1])) && (is_zero_flt(dataU[0][1][1])) && (is_zero_flt(dataU[1][0][1])) && (is_zero_flt(dataU[1][1][1])) &&
          (is_zero_flt(dataV[0][0][1])) && (is_zero_flt(dataV[0][1][1])) && (is_zero_flt(dataV[1][0][1])) && (is_zero_flt(dataV[1][1][1])) &&
          xsi < 1.){
        if (interp_method == PARTIALSLIP) {
          f_v = f_v * (1 - .5 * xsi) / (1 - xsi);
          if (withW) {
            f_w = f_w * (1 - .5 * xsi) / (1 - xsi);
          }
        } else if (interp_method == FREESLIP) {
          f_v = f_v / (1 - xsi);
          if (withW) {
            f_w = f_w / (1 - xsi);
          }
        }
      }
      if ((is_zero_flt(dataU[0][0][0])) && (is_zero_flt(dataU[0][0][1])) && (is_zero_flt(dataU[0][1][0])) && (is_zero_flt(dataU[0][1][1])) &&
          (is_zero_flt(dataV[0][0][0])) && (is_zero_flt(dataV[0][0][1])) && (is_zero_flt(dataV[0][1][0])) && (is_zero_flt(dataV[0][1][1])) &&
          zeta > 0.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (.5 + .5 * zeta) / zeta;
          f_v = f_v * (.5 + .5 * zeta) / zeta;
        } else if (interp_method == FREESLIP) {
          f_u = f_u / zeta;
          f_v = f_v / zeta;
        }
      }
      if ((is_zero_flt(dataU[1][0][0])) && (is_zero_flt(dataU[1][0][1])) && (is_zero_flt(dataU[1][1][0])) && (is_zero_flt(dataU[1][1][1])) &&
          (is_zero_flt(dataV[1][0][0])) && (is_zero_flt(dataV[1][0][1])) && (is_zero_flt(dataV[1][1][0])) && (is_zero_flt(dataV[1][1][1])) &&
          zeta < 1.){
        if (interp_method == PARTIALSLIP) {
          f_u = f_u * (1 - .5 * zeta) / (1 - zeta);
          f_v = f_v * (1 - .5 * zeta) / (1 - zeta);
        } else if (interp_method == FREESLIP) {
          f_u = f_u / (1 - zeta);
          f_v = f_v / (1 - zeta);
        }
      }
      *u *= f_u;
      *v *= f_v;
      if (withW) {
        *w *= f_w;
      }

  return SUCCESS;
}

static inline StatusCode temporal_interpolation_slip(type_coord x, type_coord y, type_coord z, double time, CField *U, CField *V, CField *W,
                                                         GridType gtype, int *xi, int *yi, int *zi, int *ti,
                                                         float *u, float *v, float *w, int withW)
{
  StatusCode status;
  CStructuredGrid *grid = U->grid->grid;
  GridIndexingType gridindexingtype = U->gridindexingtype;
  InterpMethod interp_method = U->interp_method;
  int igrid = U->igrid;

  /* Find time index for temporal interpolation */
  if (U->time_periodic == 0 && U->allow_time_extrapolation == 0 && (time < grid->time[0] || time > grid->time[grid->tdim-1])){
    return ERRORTIMEEXTRAPOLATION;
  }
  status = search_time_index(&time, grid->tdim, grid->time, &ti[igrid], U->time_periodic, grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

  double xsi, eta, zeta;

  if (ti[igrid] < grid->tdim-1 && time > grid->time[ti[igrid]]) {
    float u0, u1, v0, v1, w0, w1;
    double t0 = grid->time[ti[igrid]]; double t1 = grid->time[ti[igrid]+1];
    /* Identify grid cell to sample through local linear search */
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gtype, ti[igrid], time, t0, t1, interp_method, gridindexingtype); CHECKSTATUS(status);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2], data2D_W[2][2][2];
      status = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 0); CHECKSTATUS(status);
      status = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 0); CHECKSTATUS(status);
      if (withW){
        status = getCell2D(W, xi[igrid], yi[igrid], ti[igrid], data2D_W, 0); CHECKSTATUS(status);
        status = spatial_interpolation_bilinear(xsi, eta, data2D_W[0], &w0); CHECKSTATUS(status);
        status = spatial_interpolation_bilinear(xsi, eta, data2D_W[1], &w1); CHECKSTATUS(status);
      }

      status = spatial_interpolation_bilinear(xsi, eta, data2D_U[0], &u0); CHECKSTATUS(status);
      status = spatial_interpolation_bilinear(xsi, eta, data2D_V[0], &v0); CHECKSTATUS(status);
      status = calculate_slip_conditions_2D(xsi, eta, data2D_U[0], data2D_V[0], data2D_W[0], &u0, &v0, &w0, interp_method, withW); CHECKSTATUS(status);

      status = spatial_interpolation_bilinear(xsi, eta, data2D_U[1], &u1); CHECKSTATUS(status);
      status = spatial_interpolation_bilinear(xsi, eta, data2D_V[1], &v1); CHECKSTATUS(status);
      status = calculate_slip_conditions_2D(xsi, eta, data2D_U[1], data2D_V[1], data2D_W[1], &u1, &v1, &w1, interp_method, withW); CHECKSTATUS(status);
    } else {
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2], data3D_W[2][2][2][2];
      status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 0); CHECKSTATUS(status);
      status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 0); CHECKSTATUS(status);
      if (withW){
        status = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 0); CHECKSTATUS(status);
        status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_W[0], &w0); CHECKSTATUS(status);
        status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_W[1], &w1); CHECKSTATUS(status);
      }
      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_U[0], &u0); CHECKSTATUS(status);
      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_V[0], &v0); CHECKSTATUS(status);
      status = calculate_slip_conditions_3D(xsi, eta, zeta, data3D_U[0], data3D_V[0], data3D_W[0], &u0, &v0, &w0, interp_method, withW); CHECKSTATUS(status);

      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_U[1], &u1); CHECKSTATUS(status);
      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_V[1], &v1); CHECKSTATUS(status);
      status = calculate_slip_conditions_3D(xsi, eta, zeta, data3D_U[1], data3D_V[1], data3D_W[1], &u1, &v1, &w1, interp_method, withW); CHECKSTATUS(status);
    }
    *u = u0 + (u1 - u0) * (float)((time - t0) / (t1 - t0));
    *v = v0 + (v1 - v0) * (float)((time - t0) / (t1 - t0));
    if (withW){
      *w = w0 + (w1 - w0) * (float)((time - t0) / (t1 - t0));
    }

  } else {
    double t0 = grid->time[ti[igrid]];
    status = search_indices(x, y, z, grid, &xi[igrid], &yi[igrid], &zi[igrid], &xsi, &eta, &zeta, gtype, ti[igrid], t0, t0, t0+1, interp_method, gridindexingtype); CHECKSTATUS(status);
    if (grid->zdim==1){
      float data2D_U[2][2][2], data2D_V[2][2][2], data2D_W[2][2][2];
      status = getCell2D(U, xi[igrid], yi[igrid], ti[igrid], data2D_U, 1); CHECKSTATUS(status);
      status = getCell2D(V, xi[igrid], yi[igrid], ti[igrid], data2D_V, 1); CHECKSTATUS(status);
      if (withW){
        status = getCell2D(W, xi[igrid], yi[igrid], ti[igrid], data2D_W, 1); CHECKSTATUS(status);
        status = spatial_interpolation_bilinear(xsi, eta, data2D_W[0], w); CHECKSTATUS(status);
      }

      status = spatial_interpolation_bilinear(xsi, eta, data2D_U[0], u); CHECKSTATUS(status);
      status = spatial_interpolation_bilinear(xsi, eta, data2D_V[0], v); CHECKSTATUS(status);

      status = calculate_slip_conditions_2D(xsi, eta, data2D_U[0], data2D_V[0], data2D_W[0], u, v, w, interp_method, withW); CHECKSTATUS(status);
    } else {
      float data3D_U[2][2][2][2], data3D_V[2][2][2][2], data3D_W[2][2][2][2];
      status = getCell3D(U, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_U, 1); CHECKSTATUS(status);
      status = getCell3D(V, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_V, 1); CHECKSTATUS(status);
      if (withW){
        status = getCell3D(W, xi[igrid], yi[igrid], zi[igrid], ti[igrid], data3D_W, 1); CHECKSTATUS(status);
        status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_W[0], w); CHECKSTATUS(status);
      }
      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_U[0], u); CHECKSTATUS(status);
      status = spatial_interpolation_trilinear(xsi, eta, zeta, data3D_V[0], v); CHECKSTATUS(status);
      status = calculate_slip_conditions_3D(xsi, eta, zeta, data3D_U[0], data3D_V[0], data3D_W[0], u, v, w, interp_method, withW); CHECKSTATUS(status);
    }
  }
  return SUCCESS;
}

static inline StatusCode temporal_interpolation(type_coord x, type_coord y, type_coord z, double time, CField *f,
                                               int *xi, int *yi, int *zi, int *ti,
                                               float *value)
{
  CGrid *_grid = f->grid;
  GridType gtype = _grid->gtype;

  if (gtype == RECTILINEAR_Z_GRID || gtype == RECTILINEAR_S_GRID || gtype == CURVILINEAR_Z_GRID || gtype == CURVILINEAR_S_GRID)
    return temporal_interpolation_structured_grid(x, y, z, time, f, gtype, xi, yi, zi, ti, value);
  else{
    printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
    return ERROR;
  }
}

static inline StatusCode temporal_interpolationUV(type_coord x, type_coord y, type_coord z, double time,
                                                 CField *U, CField *V,
                                                 int *xi, int *yi, int *zi, int *ti,
                                                 float *valueU, float *valueV)
{
  StatusCode status;
  InterpMethod interp_method = U->interp_method;
  if (interp_method == CGRID_VELOCITY){
    CGrid *_grid = U->grid;
    GridType gtype = _grid->gtype;
    status = temporal_interpolationUV_c_grid(x, y, z, time, U, V, gtype, xi, yi, zi, ti, valueU, valueV); CHECKSTATUS(status);
    return SUCCESS;
  } else if ((interp_method == PARTIALSLIP) || (interp_method == FREESLIP)){
    CGrid *_grid = U->grid;
    CField *W = U;
    GridType gtype = _grid->gtype;
    int withW = 0;
    status = temporal_interpolation_slip(x, y, z, time, U, V, W, gtype, xi, yi, zi, ti, valueU, valueV, 0, withW); CHECKSTATUS(status);
    return SUCCESS;
  } else {
    status = temporal_interpolation(x, y, z, time, U, xi, yi, zi, ti, valueU); CHECKSTATUS(status);
    status = temporal_interpolation(x, y, z, time, V, xi, yi, zi, ti, valueV); CHECKSTATUS(status);
    return SUCCESS;
  }
}

static inline StatusCode temporal_interpolationUVW(type_coord x, type_coord y, type_coord z, double time,
                                                  CField *U, CField *V, CField *W,
                                                  int *xi, int *yi, int *zi, int *ti,
                                                  float *valueU, float *valueV, float *valueW)
{
  StatusCode status;
  InterpMethod interp_method = U->interp_method;
  if (interp_method == CGRID_VELOCITY){
    CGrid *_grid = U->grid;
    GridType gtype = _grid->gtype;
    if (gtype == RECTILINEAR_S_GRID || gtype == CURVILINEAR_S_GRID){
      status = temporal_interpolationUVW_c_grid(x, y, z, time, U, V, W, gtype, xi, yi, zi, ti, valueU, valueV, valueW); CHECKSTATUS(status);
      return SUCCESS;
    }
  } else if ((interp_method == PARTIALSLIP) || (interp_method == FREESLIP)){
    CGrid *_grid = U->grid;
    GridType gtype = _grid->gtype;
    int withW = 1;
    status = temporal_interpolation_slip(x, y, z, time, U, V, W, gtype, xi, yi, zi, ti, valueU, valueV, valueW, withW); CHECKSTATUS(status);
    return SUCCESS;
  }
  status = temporal_interpolationUV(x, y, z, time, U, V, xi, yi, zi, ti, valueU, valueV); CHECKSTATUS(status);
  if (interp_method == BGRID_VELOCITY)
    interp_method = BGRID_W_VELOCITY;
  status = temporal_interpolation(x, y, z, time, W, xi, yi, zi, ti, valueW); CHECKSTATUS(status);
  return SUCCESS;
}

#ifdef __cplusplus
}
#endif
#endif
