#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double compute_rs(double r, double B, double delta, double s_min, double tol){
  if (fabs(B) < tol){
    return -delta *s_min + r;
  } else {
    return (r + delta / B) * exp(-B * s_min) - delta / B;
  }
}

void compute_ds(double F0, double F1, double r, double direction, double tol,
                double *ds, double *B, double *delta){
  double up = F0 * (1-r) + F1 * r;
  double r_target = 0;
  if (direction * up >=0.){
    r_target = 1.;
  }
  *delta = -F0;
  *B = F0 - F1;
  double F_r1 = r_target + *delta / *B;
  double F_r0 = r + *delta / *B;
  if (fabs(*B) < tol){
    *B = 0;
    F_r1 = 0.0/0.0; // SET TO NAN
    F_r0 = 0.0/0.0; // SET TO NAN
  }

  if (fabs(*B) < tol && fabs(*delta) < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  } else if (*B == 0){
      *ds = -(r_target - r) / *delta;
  } else if (F_r1 * F_r0 < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  } else {
      *ds = - 1. / *B * log(F_r1 / F_r0);
  }

  if (fabs(*ds) < tol){
      *ds = 1.0 /0.0;  // SET TO INFINITY
  }
}


static inline StatusCode calcAdvectionAnalytical_JIT(CField *fu, CField *fv, CField *fw, double *flow3D_dbl,
                              int *xi, int *yi, int *zi, int *ti,
                              double *lon, double *lat, double *depth,
                              double *time, double *dt,
                              double *particle_dlon, double *particle_dlat, double *particle_ddepth)
{
  StatusCode status;
  CStructuredGrid *grid = fu->grid->grid;
  GridType gtype = fu->grid->gtype;
  GridIndexingType gridindexingtype = fu->gridindexingtype;
  int xdim = grid->xdim;
  int igrid = fu->igrid;

  double tol = 1e-10;
  double I_s = 10;  // number of intermediate time steps
  double direction = 1;
  if (*dt < 0)
    direction = -1;

  double xsi, rs_x, ds_x, B_x, delta_x;
  double eta, rs_y, ds_y, B_y, delta_y;
  double zeta, rs_z, ds_z, B_z, delta_z;

  double tau;
  double ds_t = *dt;
  int tii, first_tstep_only;

  if (grid->tdim == 1){
    tii = 0;
    tau = 0;
    status = search_indices(*lon, *lat, *depth, grid, &xi[igrid], &yi[igrid], &zi[igrid],
                            &xsi, &eta, &zeta, gtype, tii, 0, 0, 1, CGRID_VELOCITY, gridindexingtype);
    first_tstep_only = 1;
  } else {
    status = search_time_index(time, grid->tdim, grid->time, &ti[igrid], fu->time_periodic,
                               grid->tfull_min, grid->tfull_max, grid->periods); CHECKSTATUS(status);

    double t0 = grid->time[ti[igrid]];
    double t1 = grid->time[ti[igrid]+1];
    tau = (*time - t0) / (t1 - t0);
    for (double i=I_s; i>0; i--){
      if (*time - t0 < i / I_s * (t1 - t0)){
        ds_t = min(ds_t, i / I_s);
      }
    }
    tii = ti[igrid];
    status = search_indices(*lon, *lat, *depth, grid, &xi[igrid], &yi[igrid], &zi[igrid],
                            &xsi, &eta, &zeta, gtype, tii, *time, t0, t1, CGRID_VELOCITY, gridindexingtype);
    first_tstep_only = 0;
  }
  CHECKSTATUS(status);

  float dataU_2D[2][2][2];
  float dataV_2D[2][2][2];
  float dataU_3D[2][2][2][2];
  float dataV_3D[2][2][2][2];
  float dataW_3D[2][2][2][2];

  int flow3D = (int) *flow3D_dbl;

  if (flow3D == 0){
    status = getCell2D(fu, *xi, *yi, tii, dataU_2D, first_tstep_only); CHECKSTATUS(status);
    status = getCell2D(fv, *xi, *yi, tii, dataV_2D, first_tstep_only); CHECKSTATUS(status);

    bool updateCells = 0;
    if (fabs(xsi - 1) < tol){
      if (dataU_2D[0][1][1] > 0){
        *xi += 1;
        xsi = 0;
        updateCells = 1;
      }
    } else if (fabs(xsi) < tol){
      if (dataU_2D[0][0][0] < 0){
        *xi -= 1;
        xsi = 1;
        updateCells = 1;
      }
    }
    if (fabs(eta - 1) < tol){
      if (dataV_2D[0][1][1] > 0){
        *yi += 1;
        eta = 0;
        updateCells = 1;
      }
    } else if (fabs(eta) < tol){
      if (dataV_2D[0][0][0] < 0){
        *yi -= 1;
        eta = 1;
        updateCells = 1;
      }
    }
    if (updateCells == 1){
      status = getCell2D(fu, *xi, *yi, tii, dataU_2D, first_tstep_only); CHECKSTATUS(status);
      status = getCell2D(fv, *xi, *yi, tii, dataV_2D, first_tstep_only); CHECKSTATUS(status);
    }
  } else if (flow3D == 1){
    status = getCell3D(fu, *xi, *yi, *zi, tii, dataU_3D, first_tstep_only); CHECKSTATUS(status);
    status = getCell3D(fv, *xi, *yi, *zi, tii, dataV_3D, first_tstep_only); CHECKSTATUS(status);
    status = getCell3D(fw, *xi, *yi, *zi, tii, dataW_3D, first_tstep_only); CHECKSTATUS(status);
    bool updateCells = 0;
    if (fabs(xsi - 1) < tol){
      if (dataU_3D[0][1][1][1] > 0){
        *xi += 1;
        xsi = 0;
        updateCells = 1;
      }
    } else if (fabs(xsi) < tol){
      if (dataU_3D[0][0][0][0] < 0){
        *xi -= 1;
        xsi = 1;
        updateCells = 1;
      }
    }
    if (fabs(eta - 1) < tol){
      if (dataV_3D[0][1][1][1] > 0){
        *yi += 1;
        eta = 0;
        updateCells = 1;
      }
    } else if (fabs(eta) < tol){
      if (dataV_3D[0][0][0][0] < 0){
        *yi -= 1;
        eta = 1;
        updateCells = 1;
      }
    }
    if (fabs(zeta - 1) < tol){
      if (dataW_3D[0][1][1][1] > 0){
        *zi += 1;
        zeta = 0;
        updateCells = 1;
      }
    } else if (fabs(zeta) < tol){
      if (dataW_3D[0][0][0][0] < 0){
        *zi -= 1;
        zeta = 1;
        updateCells = 1;
      }
    }
    if (updateCells == 1){
      status = getCell3D(fu, *xi, *yi, *zi, tii, dataU_3D, first_tstep_only); CHECKSTATUS(status);
      status = getCell3D(fv, *xi, *yi, *zi, tii, dataV_3D, first_tstep_only); CHECKSTATUS(status);
      status = getCell3D(fw, *xi, *yi, *zi, tii, dataW_3D, first_tstep_only); CHECKSTATUS(status);
    }
  }

  double px[4];
  double py[4];
  int iN;
  if( (gtype == RECTILINEAR_Z_GRID) || (gtype == RECTILINEAR_S_GRID) ){
    float *xgrid = grid->lon;
    float *ygrid = grid->lat;
    for (iN=0; iN < 4; ++iN){
      px[iN] = xgrid[*xi+min(1, (iN%3))];
      py[iN] = ygrid[*yi+iN/2];
    }
  }
  else{
    float (* xgrid)[xdim] = (float (*)[xdim]) grid->lon;
    float (* ygrid)[xdim] = (float (*)[xdim]) grid->lat;
    for (iN=0; iN < 4; ++iN){
      px[iN] = xgrid[*yi+iN/2][*xi+min(1, (iN%3))];
      py[iN] = ygrid[*yi+iN/2][*xi+min(1, (iN%3))];
    }
    for (int i4 = 1; i4 < 4; ++i4){
      if (px[i4] < px[0] - 180) px[i4] += 360;
      if (px[i4] > px[0] + 180) px[i4] -= 360;
    }
  }

  double dphidxsi[4] = {eta-1, 1-eta, eta, -eta};
  double dphideta[4] = {xsi-1, -xsi, xsi, 1-xsi};
  double dxdxsi = 0; double dxdeta = 0;
  double dydxsi = 0; double dydeta = 0;
  for(int i=0; i<4; ++i){
    dxdxsi += px[i] *dphidxsi[i];
    dxdeta += px[i] *dphideta[i];
    dydxsi += py[i] *dphidxsi[i];
    dydeta += py[i] *dphideta[i];
  }
  double meshJac = 1;
  double phi[4];
  if (grid->sphere_mesh == 1){
    double deg2m = 1852 * 60.;
    double rad = M_PI / 180.;
    phi2D_lin(xsi, eta, phi);
    double lat = dot_prod(phi, py, 4);
    meshJac = deg2m * deg2m * cos(rad * lat);
  }
  double dxdy = (dxdxsi*dydeta - dxdeta * dydxsi) * meshJac;

  double pz[2];
  double dz = 1;
  if (flow3D == 1){
    pz[0] = grid->depth[*zi];
    pz[1] = grid->depth[*zi+1];
    dz = pz[1] - pz[0];
  }

  phi2D_lin(xsi, 0., phi);
  double c1 = dist(px[0], px[1], py[0], py[1], grid->sphere_mesh, dot_prod(phi, py, 4));
  phi2D_lin(1., eta, phi);
  double c2 = dist(px[1], px[2], py[1], py[2], grid->sphere_mesh, dot_prod(phi, py, 4));
  phi2D_lin(xsi, 1., phi);
  double c3 = dist(px[2], px[3], py[2], py[3], grid->sphere_mesh, dot_prod(phi, py, 4));
  phi2D_lin(0., eta, phi);
  double c4 = dist(px[3], px[0], py[3], py[0], grid->sphere_mesh, dot_prod(phi, py, 4));

  double U0, U1, V0, V1, W0, W1;
  if (flow3D == 0){
    U0 = ((1-tau)*dataU_2D[0][1][0] + tau*dataU_2D[1][1][0]) * c4 * direction;
    U1 = ((1-tau)*dataU_2D[0][1][1] + tau*dataU_2D[1][1][1]) * c2 * direction;
    V0 = ((1-tau)*dataV_2D[0][0][1] + tau*dataV_2D[1][0][1]) * c1 * direction;
    V1 = ((1-tau)*dataV_2D[0][1][1] + tau*dataV_2D[1][1][1]) * c3 * direction;
  } else if (flow3D == 1){
    U0 = ((1-tau)*dataU_3D[0][1][1][0] + tau*dataU_3D[1][1][1][0]) * c4 * dz * direction;
    U1 = ((1-tau)*dataU_3D[0][1][1][1] + tau*dataU_3D[1][1][1][1]) * c2 * dz * direction;
    V0 = ((1-tau)*dataV_3D[0][1][0][1] + tau*dataV_3D[1][1][0][1]) * c1 * dz * direction;
    V1 = ((1-tau)*dataV_3D[0][1][1][1] + tau*dataV_3D[1][1][1][1]) * c3 * dz * direction;
    W0 = ((1-tau)*dataW_3D[0][0][1][1] + tau*dataW_3D[1][0][1][1]) * dxdy * direction;
    W1 = ((1-tau)*dataW_3D[0][1][1][1] + tau*dataW_3D[1][1][1][1]) * dxdy * direction;
  }

  compute_ds(U0, U1, xsi, direction, tol, &ds_x, &B_x, &delta_x);
  compute_ds(V0, V1, eta, direction, tol, &ds_y, &B_y, &delta_y);
  if (flow3D == 1){
    compute_ds(W0, W1, zeta, direction, tol, &ds_z, &B_z, &delta_z);
  } else {
    ds_z = 1.0/0.0;
  }

  double s_min = min(min(min(fabs(ds_x), fabs(ds_y)), fabs(ds_z)), fabs(ds_t / (dxdy * dz)));

  rs_x = compute_rs(xsi, B_x, delta_x, s_min, tol);
  rs_y = compute_rs(eta, B_y, delta_y, s_min, tol);

  *particle_dlon += (1.-rs_x)*(1.-rs_y) * px[0] + rs_x * (1.-rs_y) * px[1] + rs_x * rs_y * px[2] + (1.-rs_x)*rs_y * px[3] - *lon;
  *particle_dlat += (1.-rs_x)*(1.-rs_y) * py[0] + rs_x * (1.-rs_y) * py[1] + rs_x * rs_y * py[2] + (1.-rs_x)*rs_y * py[3] - *lat;

  if (flow3D == 1){
    rs_z = compute_rs(zeta, B_z, delta_z, s_min, tol);
    *particle_ddepth += (1.-rs_z) * pz[0] + rs_z * pz[1] - *depth;
  }

  if (*dt > 0){
    *dt = max(direction * s_min * dxdy * dz, 1e-7);
  } else {
    *dt = min(direction * s_min * dxdy * dz, 1e-7);
  }

  return SUCCESS;
}
