#ifndef MY_TIME_LIB_H
#define MY_TIME_LIB_H

#include <sys/time.h>
#include <math.h>

/* Macro-based timer using gettimeofday — microsecond resolution.
 * TIMER_ELAPSED returns elapsed time in microseconds (double).
 * Divide by 1.e6 to get seconds.
 */
#define TIMER_DEF(n)     struct timeval temp_1_##n = {0,0}, temp_2_##n = {0,0}
#define TIMER_START(n)   gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)    gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec  - temp_1_##n.tv_sec ) * 1.e6 + \
                          (temp_2_##n.tv_usec - temp_1_##n.tv_usec))

double arithmetic_mean(double *v, int len);
double geometric_mean (double *v, int len);
double sigma_fn       (double *v, double mu, int len);

#endif /* MY_TIME_LIB_H */
