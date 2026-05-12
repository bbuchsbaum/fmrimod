#include <math.h>
#include <stddef.h>

/*
 * Segment-aware ARMA whitening kernel.
 *
 * y:          input matrix (row-major), shape (n, v)
 * phi:        AR coefficients length p
 * theta:      MA coefficients length q
 * seg_starts: segment starts length n_seg (0-based, must include 0)
 * out:        output matrix (row-major), shape (n, v)
 *
 * Returns 0 on success, non-zero on invalid arguments.
 */
int arma_whiten_segments_c(
    const double *y,
    int n,
    int v,
    const double *phi,
    int p,
    const double *theta,
    int q,
    const int *seg_starts,
    int n_seg,
    int do_exact,
    double *out
) {
    if (y == NULL || out == NULL || seg_starts == NULL) {
        return 1;
    }
    if (n <= 0 || v <= 0 || n_seg <= 0) {
        return 2;
    }
    if (p < 0 || q < 0) {
        return 3;
    }

    for (int si = 0; si < n_seg; ++si) {
        int s0 = seg_starts[si];
        int s1 = (si + 1 < n_seg) ? seg_starts[si + 1] : n;
        if (s0 < 0) {
            s0 = 0;
        }
        if (s1 > n) {
            s1 = n;
        }
        if (s1 <= s0) {
            continue;
        }

        for (int col = 0; col < v; ++col) {
            for (int t = s0; t < s1; ++t) {
                size_t idx = (size_t)t * (size_t)v + (size_t)col;
                double val = y[idx];

                /* AR contribution uses original y. */
                for (int k = 0; k < p; ++k) {
                    int tt = t - (k + 1);
                    if (tt >= s0) {
                        size_t src = (size_t)tt * (size_t)v + (size_t)col;
                        val -= phi[k] * y[src];
                    }
                }

                /* MA contribution uses prior innovations (out). */
                for (int j = 0; j < q; ++j) {
                    int tt = t - (j + 1);
                    if (tt >= s0) {
                        size_t src = (size_t)tt * (size_t)v + (size_t)col;
                        val -= theta[j] * out[src];
                    }
                }

                out[idx] = val;
            }

            if (do_exact && p == 1 && q == 0) {
                size_t first = (size_t)s0 * (size_t)v + (size_t)col;
                double s = 1.0 - phi[0] * phi[0];
                if (s < 0.0) {
                    s = 0.0;
                }
                out[first] *= sqrt(s);
            }
        }
    }

    return 0;
}
