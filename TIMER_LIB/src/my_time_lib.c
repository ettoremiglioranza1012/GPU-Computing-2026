#include <math.h>
#include "my_time_lib.h"

double arithmetic_mean(double *v, int len) {
    double mu = 0.0;
    for (int i = 0; i < len; i++) mu += v[i];
    return mu / (double)len;
}

double geometric_mean(double *v, int len) {
    double mu = 1.0;
    for (int i = 0; i < len; i++)
        mu *= (v[i] > 0.0) ? v[i] : 1.0;
    return pow(mu, 1.0 / (double)len);
}

double sigma_fn(double *v, double mu, int len) {
    double sigma = 0.0;
    for (int i = 0; i < len; i++)
        sigma += (v[i] - mu) * (v[i] - mu);
    return sigma / (double)len;
}
