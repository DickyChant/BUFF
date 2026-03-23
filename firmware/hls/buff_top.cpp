/**
 * BUFF FPGA Inference — Top-level implementation with DATAFLOW.
 *
 * Three-stage pipeline:
 *   Stage 1: Read from AXI-Stream + forward MinMax scale
 *   Stage 2: Euler ODE solve (N_STEPS-1 sequential iterations)
 *   Stage 3: Inverse MinMax scale + write to AXI-Stream
 *
 * With HLS DATAFLOW, Stage 1 of event N+1 overlaps with Stage 2 of
 * event N and Stage 3 of event N-1.  Since Stage 2 dominates latency,
 * throughput ~= 1 event per ODE-solve time.
 */

#include "buff_top.h"
#include "euler_solver.h"

// ── Scaler parameters (defaults: identity) ───────────────────────────
// Overwritten by export_conifer.py with actual trained scaler values.
// Using weak symbols so generated params can override at link time.
#ifndef BUFF_PARAMS_GENERATED
const data_t SCALE[N_FEATURES]      = {1, 1, 1, 1};
const data_t OFFSET[N_FEATURES]     = {0, 0, 0, 0};
const data_t INV_SCALE[N_FEATURES]  = {1, 1, 1, 1};
const data_t INV_OFFSET[N_FEATURES] = {0, 0, 0, 0};
#endif


// ── Stage 1: Read + Forward Scale ────────────────────────────────────
void read_and_scale(
    hls::stream<data_t> &x_stream_in,
    data_t x_scaled[N_FEATURES]
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1

    read_scale_loop:
    for (int k = 0; k < N_FEATURES; k++) {
        #pragma HLS PIPELINE II=1
        data_t raw = x_stream_in.read();
        // MinMax forward: x_scaled = raw * scale + offset
        accum_t scaled = accum_t(raw) * accum_t(SCALE[k]) + accum_t(OFFSET[k]);
        x_scaled[k] = data_t(scaled);
    }
}


// ── Stage 3: Inverse Scale + Write ───────────────────────────────────
void inv_scale_and_write(
    const data_t x_solved[N_FEATURES],
    hls::stream<data_t> &x_stream_out
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1

    write_inv_loop:
    for (int k = 0; k < N_FEATURES; k++) {
        #pragma HLS PIPELINE II=1
        // Inverse MinMax: x_orig = x_solved * inv_scale + inv_offset
        accum_t orig = accum_t(x_solved[k]) * accum_t(INV_SCALE[k])
                     + accum_t(INV_OFFSET[k]);
        x_stream_out.write(data_t(orig));
    }
}


// ── Top-level with DATAFLOW ──────────────────────────────────────────
void buff_top(
    hls::stream<data_t> &x_stream_in,
    hls::stream<data_t> &x_stream_out
) {
    // AXI-Stream interfaces
    #pragma HLS INTERFACE axis port=x_stream_in
    #pragma HLS INTERFACE axis port=x_stream_out
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Enable inter-stage overlap for event-level pipelining
    #pragma HLS DATAFLOW

    // Stage 1: Read from stream + forward MinMax scale
    data_t x_scaled[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x_scaled complete dim=1
    read_and_scale(x_stream_in, x_scaled);

    // Stage 2: Euler ODE solver (dominates latency)
    data_t x_solved[N_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=x_solved complete dim=1
    euler_solve(x_scaled, x_solved);

    // Stage 3: Inverse scale + write to output stream
    inv_scale_and_write(x_solved, x_stream_out);
}
