/**
 * BUFF FPGA Inference — Euler ODE solver.
 *
 * N-step forward Euler integration with feedback loop:
 *     x_{i+1} = x_i + h * v(t_i, x_i)
 *
 * The outer loop (over timesteps) is sequential because step i+1
 * depends on step i.  Within each step, all d BDT evaluations
 * run fully in parallel via velocity_eval().
 *
 * The solver is wrapped in HLS DATAFLOW at the top level (buff_top.cpp)
 * so that the scaler / inverse-scaler stages of consecutive events
 * overlap with the ODE loop of the current event.
 */
#ifndef EULER_SOLVER_H
#define EULER_SOLVER_H

#include "buff_types.h"
#include "bdt_ensemble.h"

/**
 * Euler ODE solver: integrates from t=0 to t=1 in N_STEPS-1 steps.
 *
 * @param x0     Scaled input state (d features)
 * @param x_out  Solved output state (d features)
 */
void euler_solve(
    const data_t x0[N_FEATURES],
    data_t x_out[N_FEATURES]
) {
    #pragma HLS INLINE off

    // Working state — fully partitioned for parallel access
    data_t x[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x complete dim=1

    // Copy input to working state
    init_copy:
    for (int k = 0; k < N_FEATURES; k++) {
        #pragma HLS UNROLL
        x[k] = x0[k];
    }

    // ── Main Euler loop ──────────────────────────────────────────────
    // Cannot pipeline: step i+1 depends on step i (true data dependency).
    // But within each iteration, velocity_eval is fully pipelined/unrolled.
    euler_loop:
    for (int step = 0; step < N_STEPS - 1; step++) {
        // Velocity at current state
        data_t v[N_FEATURES];
        #pragma HLS ARRAY_PARTITION variable=v complete dim=1

        velocity_eval(x, v, step);

        // Euler update: x += h * v
        euler_accum:
        for (int k = 0; k < N_FEATURES; k++) {
            #pragma HLS UNROLL
            accum_t x_wide = accum_t(x[k]);
            accum_t step_val = accum_t(STEP_H) * accum_t(v[k]);
            x[k] = data_t(x_wide + step_val);
        }
    }

    // Copy output
    out_copy:
    for (int k = 0; k < N_FEATURES; k++) {
        #pragma HLS UNROLL
        x_out[k] = x[k];
    }
}

#endif  // EULER_SOLVER_H
