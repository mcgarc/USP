#include <stdio.h>
#include <math.h>

double norm( double *vec) {
    double vec_sqd;
    vec_sqd = pow(vec[0], 2) + pow(vec[1], 2), pow(vec[2], 2);
    return sqrt(vec_sqd);
}

double dot( double *a, double *b) {
  double tot = 0;
  for (int i=0; i<3; i++) {
    tot = tot + a[i] * b[i];
  }
  return tot;
}

void sub(double *output, double *a, double *b) {
  for (int i=0; i<3; i++) {
    output[i] = a[i] - b[i];
  }
}

void add(double *output, double *a, double *b) {
  for (int i=0; i<3; i++) {
    output[i] = a[i] + b[i];
  }
}

void scalar_mult(double *output, double s, double *a) {
  for (int i=0; i<3; i++) {
    output[i] = s*a[i];
  }
}

void printf_vec( double* a ) {
  printf("[ %f, %f, %f ]\n", a[0], a[1], a[2]);
}


void cross(double *output, double *a, double *b) {
  output[0] = a[1]*b[2] - a[2]*b[1];
  output[1] = a[2]*b[0] - a[0]*b[2];
  output[2] = a[0]*b[1] - a[1]*b[0];
}

void wire_segment_field (
    double *output,
    double *r,
    double *w,
    double *start,
    double *end
    ) {
  double w_mag = norm(w);
  // Vector from wire start to r
  double r_start[3] = {0, 0, 0};
  sub(r_start, r, start);
  double r_start_mag = norm(r_start);
  // Vector from wire end to r
  double r_end[3] = {0, 0, 0};
  sub(r_end, r, end);
  double r_end_mag = norm(r_end);
  // Find the cosines
  double cos_start = dot(r_start, w);
  // TODO Check ramifications of abs
  cos_start = fabs(cos_start) / (r_start_mag * w_mag);
  double cos_end = dot(r_end, w);
  cos_end = fabs(cos_end) / (r_end_mag * w_mag);
  // Find the closest point on the wire to r
  double m[3] = {0, 0, 0};
  scalar_mult(m, (r_start_mag * cos_start / w_mag), w);
  add(m, start, m);
  // Find R, smallest vector from wire to R
  double R[3] = {0, 0, 0};
  sub(R, r, m);
  double R_mag = norm(R);
  // Find direction of B field
  double B[3] = {0, 0, 0};
  cross(B, w, R);
  // Find magnitude of B field (excluding current * u_0 / 4 pi)
  // This also includes normalisation for direction
  double B_mag = (cos_start + cos_end) / (R_mag * R_mag * w_mag);
  scalar_mult(B, B_mag, B);
  for (int i=0; i<3; i++) {
    output[i] = B[i];
  }
};

int main() {
  double output[3] = {0, 0, 0};
  double r[3] = {0, 1, 0};
  double w[3] = {2, 0, 0};
  double start[3] = {-1, 0, 0};
  double end[3] = {1, 0, 0};
  wire_segment_field(output, r, w, start, end);
  return 0;
}
