#ifndef _INDEX_SEARCH_H
#define _INDEX_SEARCH_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECKSTATUS(res) do {if (res != SUCCESS) return res;} while (0)
#define CHECKSTATUS_KERNELLOOP(res) {if (res == REPEAT) return res;}
#define rtol 1.e-5
#define atol 1.e-8

#ifdef DOUBLE_COORD_VARIABLES
typedef double type_coord;
#else
typedef float type_coord;
#endif

typedef enum
  {
    LINEAR=0, NEAREST=1, CGRID_VELOCITY=2, CGRID_TRACER=3, BGRID_VELOCITY=4, BGRID_W_VELOCITY=5, BGRID_TRACER=6, LINEAR_INVDIST_LAND_TRACER=7, PARTIALSLIP=8, FREESLIP=9
  } InterpCode;

typedef enum
  {
    NEMO = 0, MITGCM = 1, MOM5 = 2, POP = 3, CROCO = 4
  } GridIndexingType;

typedef struct
{
  int gtype;
  void *grid;
} CGrid;

typedef struct
{
  int xdim, ydim, zdim, tdim, z4d;
  int sphere_mesh, zonal_periodic;
  int *chunk_info;
  int *load_chunk;
  double tfull_min, tfull_max;
  int* periods;
  float *lonlat_minmax;
  float *lon, *lat, *depth;
  double *time;
} CStructuredGrid;


typedef enum
  {
    SUCCESS=0, EVALUATE=10, REPEAT=20, DELETE=30, STOPEXECUTION=40, STOPALLEXECUTION=41, ERROR=50, ERRORINTERPOLATION=51, ERROROUTOFBOUNDS=60, ERRORTHROUGHSURFACE=61, ERRORTIMEEXTRAPOLATION=70
  } StatusCode;

typedef enum
  {
    RECTILINEAR_Z_GRID=0, RECTILINEAR_S_GRID=1, CURVILINEAR_Z_GRID=2, CURVILINEAR_S_GRID=3
  } GridType;

// equal/closeness comparison that is equal to numpy (double)
static inline bool is_close_dbl(double a, double b) {
    return (fabs(a-b) <= (atol + rtol * fabs(b)));
}

// customisable equal/closeness comparison (double)
static inline bool is_close_dbl_tol(double a, double b, double tolerance) {
    return (fabs(a-b) <= (tolerance + fabs(b)));
}

// numerically accurate equal/closeness comparison (double)
static inline bool is_equal_dbl(double a, double b) {
    return (fabs(a-b) <= (DBL_EPSILON * fabs(b)));
}

// customisable equal/closeness comparison (float)
static inline bool is_close_flt_tol(float a, float b, float tolerance) {
    return (fabs(a-b) <= (tolerance + fabs(b)));
}

// equal/closeness comparison that is equal to numpy (float)
static inline bool is_close_flt(float a, float b) {
    return (fabs(a-b) <= ((float)(atol) + (float)(rtol) * fabs(b)));
}

// numerically accurate equal/closeness comparison (float)
static inline bool is_equal_flt(float a, float b) {
    return (fabs(a-b) <= (FLT_EPSILON * fabs(b)));
}

static inline bool is_zero_dbl(double a) {
    return (fabs(a) <= DBL_EPSILON * fabs(a));
}

static inline bool is_zero_flt(float a) {
    return (fabs(a) <= FLT_EPSILON * fabs(a));
}

static inline StatusCode search_indices_vertical_z(type_coord z, int zdim, float *zvals, int *zi, double *zeta, int gridindexingtype)
{
  if (zvals[zdim-1] > zvals[0]){
    if ((z < zvals[0]) && (gridindexingtype == MOM5) && (z > 2 * zvals[0] - zvals[1])){
      *zi = -1;
      *zeta = z / zvals[0];
      return SUCCESS;
    }
    if ((z > zvals[zdim-1]) && (z < 0) && (gridindexingtype == CROCO)){
      *zi = zdim-2;
      *zeta = 1;
      return SUCCESS;
    }
    if (z < zvals[0]) {return ERRORTHROUGHSURFACE;}
    if (z > zvals[zdim-1]) {return ERROROUTOFBOUNDS;}
    while (*zi < zdim-1 && z > zvals[*zi+1]) ++(*zi);
    while (*zi > 0 && z < zvals[*zi]) --(*zi);
  }
  else{
    if (z > zvals[0]) {return ERRORTHROUGHSURFACE;}
    if (z < zvals[zdim-1]) {return ERROROUTOFBOUNDS;}
    while (*zi < zdim-1 && z < zvals[*zi+1]) ++(*zi);
    while (*zi > 0 && z > zvals[*zi]) --(*zi);
  }
  if (*zi == zdim-1) {--*zi;}

  *zeta = (z - zvals[*zi]) / (zvals[*zi+1] - zvals[*zi]);
  return SUCCESS;
}

static inline StatusCode search_indices_vertical_s(type_coord z, int xdim, int ydim, int zdim, float *zvals,
                                    int xi, int yi, int *zi, double xsi, double eta, double *zeta,
                                    int z4d, int ti, int tdim, double time, double t0, double t1, int interp_method)
{
  if (interp_method == BGRID_VELOCITY || interp_method == BGRID_W_VELOCITY || interp_method == BGRID_TRACER){
    xsi = 1;
    eta = 1;
  }
  float zcol[zdim];
  int zii;
  if (z4d == 1){
    float (*zvalstab)[zdim][ydim][xdim] = (float (*)[zdim][ydim][xdim]) zvals;
    int ti1 = ti;
    if (ti < tdim-1)
       ti1= ti+1;
    double zt0, zt1;
    for (zii=0; zii < zdim; zii++){
      zt0 = (1-xsi)*(1-eta) * zvalstab[ti ][zii][yi  ][xi  ]
          + (  xsi)*(1-eta) * zvalstab[ti ][zii][yi  ][xi+1]
          + (  xsi)*(  eta) * zvalstab[ti ][zii][yi+1][xi+1]
          + (1-xsi)*(  eta) * zvalstab[ti ][zii][yi+1][xi  ];
      zt1 = (1-xsi)*(1-eta) * zvalstab[ti1][zii][yi  ][xi  ]
          + (  xsi)*(1-eta) * zvalstab[ti1][zii][yi  ][xi+1]
          + (  xsi)*(  eta) * zvalstab[ti1][zii][yi+1][xi+1]
          + (1-xsi)*(  eta) * zvalstab[ti1][zii][yi+1][xi  ];
      zcol[zii] = zt0 + (zt1 - zt0) * (float)((time - t0) / (t1 - t0));
    }

  }
  else{
    float (*zvalstab)[ydim][xdim] = (float (*)[ydim][xdim]) zvals;
    for (zii=0; zii < zdim; zii++){
      zcol[zii] = (1-xsi)*(1-eta) * zvalstab[zii][yi  ][xi  ]
                + (  xsi)*(1-eta) * zvalstab[zii][yi  ][xi+1]
                + (  xsi)*(  eta) * zvalstab[zii][yi+1][xi+1]
                + (1-xsi)*(  eta) * zvalstab[zii][yi+1][xi  ];
    }
  }

  if (zcol[zdim-1] > zcol[0]){
    if (z < zcol[0]) {return ERRORTHROUGHSURFACE;}
    if (z > zcol[zdim-1]) {return ERROROUTOFBOUNDS;}
    while (*zi < zdim-1 && z > zcol[*zi+1]) ++(*zi);
    while (*zi > 0 && z < zcol[*zi]) --(*zi);
  }
  else{
    if (z > zcol[0]) {return ERRORTHROUGHSURFACE;}
    if (z < zcol[zdim-1]) {return ERROROUTOFBOUNDS;}
    while (*zi < zdim-1 && z < zcol[*zi+1]) ++(*zi);
    while (*zi > 0 && z > zcol[*zi]) --(*zi);
  }
  if (*zi == zdim-1) {--*zi;}

  *zeta = (z - zcol[*zi]) / (zcol[*zi+1] - zcol[*zi]);
  return SUCCESS;
}

static inline void reconnect_bnd_indices(int *xi, int *yi, int xdim, int ydim, int onlyX, int sphere_mesh)
{
  if (*xi < 0){
    if (sphere_mesh)
      (*xi) = xdim-2;
    else
      (*xi) = 0;
  }
  if (*xi > xdim-2){
    if (sphere_mesh)
      (*xi) = 0;
    else
      (*xi) = xdim-2;
  }
  if (onlyX == 0){
    if (*yi < 0){
      (*yi) = 0;
    }
    if (*yi > ydim-2){
      (*yi) = ydim-2;
      if (sphere_mesh)
        (*xi) = xdim - (*xi);
    }
  }
}


static inline StatusCode search_indices_rectilinear(type_coord x, type_coord y, type_coord z, CStructuredGrid *grid, GridType gtype,
                                                   int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                                   int ti, double time, double t0, double t1, int interp_method,
                                                   int gridindexingtype)
{
  int xdim = grid->xdim;
  int ydim = grid->ydim;
  int zdim = grid->zdim;
  int tdim = grid->tdim;
  float *xvals = grid->lon;
  float *yvals = grid->lat;
  float *zvals = grid->depth;
  float *xy_minmax = grid->lonlat_minmax;
  int sphere_mesh = grid->sphere_mesh;
  int zonal_periodic = grid->zonal_periodic;
  int z4d = grid->z4d;

  if (zonal_periodic == 0){
    if ((xdim > 1) && ((x < xy_minmax[0]) || (x > xy_minmax[1])))
      return ERROROUTOFBOUNDS;
  }
  if ((ydim > 1) && ((y < xy_minmax[2]) || (y > xy_minmax[3])))
    return ERROROUTOFBOUNDS;

  if (xdim == 1){
    *xi = 0;
    *xsi = 0;
  }
  else if (sphere_mesh == 0){
    while (*xi < xdim-1 && x > xvals[*xi+1]) ++(*xi);
    while (*xi > 0 && x < xvals[*xi]) --(*xi);
    *xsi = (x - xvals[*xi]) / (xvals[*xi+1] - xvals[*xi]);
  }
  else{

    float xvalsi = xvals[*xi];
    // TODO: this will fail if longitude is e.g. only [-180, 180] (so length 2)
    if (xvalsi < x - 225) xvalsi += 360;
    if (xvalsi > x + 225) xvalsi -= 360;
    float xvalsi1 = xvals[*xi+1];
    if (xvalsi1 < xvalsi - 180) xvalsi1 += 360;
    if (xvalsi1 > xvalsi + 180) xvalsi1 -= 360;

    int itMax = 10000;
    int it = 0;
    while ( (xvalsi > x) || (xvalsi1 < x) ){
      if (xvalsi1 < x)
        ++(*xi);
      else if (xvalsi > x)
        --(*xi);
      reconnect_bnd_indices(xi, yi, xdim, ydim, 1, 1);
      xvalsi = xvals[*xi];
      if (xvalsi < x - 225) xvalsi += 360;
      if (xvalsi > x + 225) xvalsi -= 360;
      xvalsi1 = xvals[*xi+1];
      if (xvalsi1 < xvalsi - 180) xvalsi1 += 360;
      if (xvalsi1 > xvalsi + 180) xvalsi1 -= 360;
      it++;
      if (it > itMax){
        return ERROROUTOFBOUNDS;
      }
    }

    *xsi = (x - xvalsi) / (xvalsi1 - xvalsi);
  }

  if (ydim == 1){
    *yi = 0;
    *eta = 0;
  }
  else {
    while (*yi < ydim-1 && y > yvals[*yi+1]) ++(*yi);
    while (*yi > 0 && y < yvals[*yi]) --(*yi);
    *eta = (y - yvals[*yi]) / (yvals[*yi+1] - yvals[*yi]);
  }

  StatusCode status;
  if (zdim > 1){
    switch(gtype){
      case RECTILINEAR_Z_GRID:
        status = search_indices_vertical_z(z, zdim, zvals, zi, zeta, gridindexingtype);
        break;
      case RECTILINEAR_S_GRID:
        status = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *xi, *yi, zi, *xsi, *eta, zeta,
                                        z4d, ti, tdim, time, t0, t1, interp_method);
        break;
      default:
        status = ERRORINTERPOLATION;
    }
    CHECKSTATUS(status);
  }
  else
    *zeta = 0;

  if ( (*xsi < 0)  && (is_zero_dbl(*xsi)) )       {*xsi = 0.;}
  if ( (*xsi > 1)  && (is_close_dbl(*xsi, 1.)) )  {*xsi = 1.;}
  if ( (*eta < 0)  && (is_zero_dbl(*eta)) )       {*eta = 0.;}
  if ( (*eta > 1)  && (is_close_dbl(*eta, 1.)) )  {*eta = 1.;}
  if ( (*zeta < 0) && (is_zero_dbl(*zeta)) )      {*zeta = 0.;}
  if ( (*zeta > 1) && (is_close_dbl(*zeta, 1.)) ) {*zeta = 1.;}

  if ( (*xsi < 0) || (*xsi > 1) ) return ERRORINTERPOLATION;
  if ( (*eta < 0) || (*eta > 1) ) return ERRORINTERPOLATION;
  if ( (*zeta < 0) || (*zeta > 1) ) return ERRORINTERPOLATION;

  return SUCCESS;
}


static inline StatusCode search_indices_curvilinear(type_coord x, type_coord y, type_coord z, CStructuredGrid *grid, GridType gtype,
                                                   int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                                   int ti, double time, double t0, double t1, int interp_method,
                                                   int gridindexingtype)
{
  int xi_old = *xi;
  int yi_old = *yi;
  int xdim = grid->xdim;
  int ydim = grid->ydim;
  int zdim = grid->zdim;
  int tdim = grid->tdim;
  float *xvals = grid->lon;
  float *yvals = grid->lat;
  float *zvals = grid->depth;
  float *xy_minmax = grid->lonlat_minmax;
  int sphere_mesh = grid->sphere_mesh;
  int zonal_periodic = grid->zonal_periodic;
  int z4d = grid->z4d;

  // NEMO convention
  float (* xgrid)[xdim] = (float (*)[xdim]) xvals;
  float (* ygrid)[xdim] = (float (*)[xdim]) yvals;

  if (zonal_periodic == 0){
    if ((x < xy_minmax[0]) || (x > xy_minmax[1])){
      if (xgrid[0][0] < xgrid[0][xdim-1]) {return ERROROUTOFBOUNDS;}
      else if (x < xgrid[0][0] && x > xgrid[0][xdim-1]) {return ERROROUTOFBOUNDS;}
    }
  }
  if ((y < xy_minmax[2]) || (y > xy_minmax[3]))
    return ERROROUTOFBOUNDS;

  double a[4], b[4];

  *xsi = *eta = -1;
  int maxIterSearch = 1e6, it = 0;
  double tol = 1e-10;
  while ( (*xsi < -tol) || (*xsi > 1+tol) || (*eta < -tol) || (*eta > 1+tol) ){
    double xgrid_loc[4] = {xgrid[*yi][*xi], xgrid[*yi][*xi+1], xgrid[*yi+1][*xi+1], xgrid[*yi+1][*xi]};
    if (sphere_mesh){ //we are on the sphere
      int i4;
      if (xgrid_loc[0] < x - 225) xgrid_loc[0] += 360;
      if (xgrid_loc[0] > x + 225) xgrid_loc[0] -= 360;
      for (i4 = 1; i4 < 4; ++i4){
        if (xgrid_loc[i4] < xgrid_loc[0] - 180) xgrid_loc[i4] += 360;
        if (xgrid_loc[i4] > xgrid_loc[0] + 180) xgrid_loc[i4] -= 360;
      }
    }
    double ygrid_loc[4] = {ygrid[*yi][*xi], ygrid[*yi][*xi+1], ygrid[*yi+1][*xi+1], ygrid[*yi+1][*xi]};

    a[0] =  xgrid_loc[0];
    a[1] = -xgrid_loc[0]    + xgrid_loc[1];
    a[2] = -xgrid_loc[0]                                              + xgrid_loc[3];
    a[3] =  xgrid_loc[0]    - xgrid_loc[1]      + xgrid_loc[2]        - xgrid_loc[3];
    b[0] =  ygrid_loc[0];
    b[1] = -ygrid_loc[0]    + ygrid_loc[1];
    b[2] = -ygrid_loc[0]                                              + ygrid_loc[3];
    b[3] =  ygrid_loc[0]    - ygrid_loc[1]      + ygrid_loc[2]        - ygrid_loc[3];

    double aa = a[3]*b[2] - a[2]*b[3];
    double bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3];
    double cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1];
    if (fabs(aa) < 1e-12)  // Rectilinear  cell, or quasi
      *eta = -cc / bb;
    else{
      double det = sqrt(bb*bb-4*aa*cc);
      if (det == det)  // so, if det is nan we keep the xsi, eta from previous iter
        *eta = (-bb+det)/(2*aa);
    }
    if ( fabs(a[1]+a[3]*(*eta)) < 1e-12 ) // this happens when recti cell rotated of 90deg
      *xsi = ( (y-ygrid_loc[0]) / (ygrid_loc[1]-ygrid_loc[0]) +
               (y-ygrid_loc[3]) / (ygrid_loc[2]-ygrid_loc[3]) ) * .5;
    else
      *xsi = (x-a[0]-a[2]* (*eta)) / (a[1]+a[3]* (*eta));
    if ( (*xsi < 0) && (*eta < 0) && (*xi == 0) && (*yi == 0) )
      return ERROROUTOFBOUNDS;
    if ( (*xsi > 1) && (*eta > 1) && (*xi == xdim-1) && (*yi == ydim-1) )
      return ERROROUTOFBOUNDS;
    if (*xsi < -tol)
      (*xi)--;
    if (*xsi > 1+tol)
      (*xi)++;
    if (*eta < -tol)
      (*yi)--;
    if (*eta > 1+tol)
      (*yi)++;
    reconnect_bnd_indices(xi, yi, xdim, ydim, 0, sphere_mesh);
    it++;
    if ( it > maxIterSearch){
      printf("Correct cell not found for (%f, %f) after %d iterations\n", x, y, maxIterSearch);
      printf("Debug info: old particle indices: (yi, xi) %d %d\n", yi_old, xi_old);
      printf("            new particle indices: (yi, xi) %d %d\n", *yi, *xi);
      printf("            Mesh 2d shape:  %d %d\n", ydim, xdim);
      printf("            Relative particle position:  (xsi, eta) %1.16e %1.16e\n", *xsi, *eta);
      return ERROROUTOFBOUNDS;
    }
  }
  if ( (*xsi != *xsi) || (*eta != *eta) ){  // check if nan
      printf("Correct cell not found for (%f, %f))\n", x, y);
      printf("Debug info: old particle indices: (yi, xi) %d %d\n", yi_old, xi_old);
      printf("            new particle indices: (yi, xi) %d %d\n", *yi, *xi);
      printf("            Mesh 2d shape:  %d %d\n", ydim, xdim);
      printf("            Relative particle position:  (xsi, eta) %1.16e %1.16e\n", *xsi, *eta);
      return ERROROUTOFBOUNDS;
  }
  if (*xsi < 0) *xsi = 0;
  if (*xsi > 1) *xsi = 1;
  if (*eta < 0) *eta = 0;
  if (*eta > 1) *eta = 1;

  StatusCode status;
  if (zdim > 1){
    switch(gtype){
      case CURVILINEAR_Z_GRID:
        status = search_indices_vertical_z(z, zdim, zvals, zi, zeta, gridindexingtype);
        break;
      case CURVILINEAR_S_GRID:
        status = search_indices_vertical_s(z, xdim, ydim, zdim, zvals,
                                        *xi, *yi, zi, *xsi, *eta, zeta,
                                        z4d, ti, tdim, time, t0, t1, interp_method);
        break;
      default:
        status = ERRORINTERPOLATION;
    }
    CHECKSTATUS(status);
  }
  else
    *zeta = 0;

  if ( (*xsi < 0) || (*xsi > 1) ) return ERRORINTERPOLATION;
  if ( (*eta < 0) || (*eta > 1) ) return ERRORINTERPOLATION;
  if ( (*zeta < 0) || (*zeta > 1) ) return ERRORINTERPOLATION;

  return SUCCESS;
}

/* Local linear search to update grid index
 * params ti, sizeT, time. t0, t1 are only used for 4D S grids
 * */
static inline StatusCode search_indices(type_coord x, type_coord y, type_coord z, CStructuredGrid *grid,
                                       int *xi, int *yi, int *zi, double *xsi, double *eta, double *zeta,
                                       GridType gtype, int ti, double time, double t0, double t1, int interp_method,
                                       int gridindexingtype)
{
  switch(gtype){
    case RECTILINEAR_Z_GRID:
    case RECTILINEAR_S_GRID:
      return search_indices_rectilinear(x, y, z, grid, gtype, xi, yi, zi, xsi, eta, zeta,
                                   ti, time, t0, t1, interp_method, gridindexingtype);
      break;
    case CURVILINEAR_Z_GRID:
    case CURVILINEAR_S_GRID:
      return search_indices_curvilinear(x, y, z, grid, gtype, xi, yi, zi, xsi, eta, zeta,
                                   ti, time, t0, t1, interp_method, gridindexingtype);
      break;
    default:
      printf("Only RECTILINEAR_Z_GRID, RECTILINEAR_S_GRID, CURVILINEAR_Z_GRID and CURVILINEAR_S_GRID grids are currently implemented\n");
      return ERROR;
  }
}

/* Local linear search to update time index */
static inline StatusCode search_time_index(double *t, int size, double *tvals, int *ti, int time_periodic, double tfull_min, double tfull_max, int *periods)
{
  if (*ti < 0)
    *ti = 0;
  if (time_periodic == 1){
    if (*t < tvals[0]){
      *ti = size-1;
      *periods = (int) floor( (*t-tfull_min)/(tfull_max-tfull_min));
      *t -= *periods * (tfull_max-tfull_min);
      if (*t < tvals[0]){ // e.g. t=5, tfull_min=0, t_full_max=5 -> periods=1 but we want periods = 0
        *periods -= 1;
        *t -= *periods * (tfull_max-tfull_min);
      }
      search_time_index(t, size, tvals, ti, time_periodic, tfull_min, tfull_max, periods);
    }
    else if (*t > tvals[size-1]){
      *ti = 0;
      *periods = (int) floor( (*t-tfull_min)/(tfull_max-tfull_min));
      *t -= *periods * (tfull_max-tfull_min);
      search_time_index(t, size, tvals, ti, time_periodic, tfull_min, tfull_max, periods);
    }
  }
  while (*ti < size-1 && *t > tvals[*ti+1]) ++(*ti);
  while (*ti > 0 && *t < tvals[*ti]) --(*ti);
  return SUCCESS;
}


#ifdef __cplusplus
}
#endif
#endif
