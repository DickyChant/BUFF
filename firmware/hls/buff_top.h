/**
 * BUFF FPGA Inference — Top-level function declaration.
 *
 * AXI-Stream interface: d values per event in/out.
 * HLS DATAFLOW enables inter-event pipelining:
 *   Event N scaler | Event N-1 ODE | Event N-2 inv-scaler
 * overlap in time.
 */
#ifndef BUFF_TOP_H
#define BUFF_TOP_H

#include <hls_stream.h>
#include "buff_types.h"

/**
 * Top-level BUFF inference function.
 *
 * Reads N_FEATURES values from input stream, runs:
 *   1. Forward MinMax scaler
 *   2. Euler ODE solver (N_STEPS-1 iterations)
 *   3. Inverse MinMax scaler
 * Writes N_FEATURES result values to output stream.
 *
 * @param x_stream_in   AXI-Stream input  (d values per event)
 * @param x_stream_out  AXI-Stream output (d values per event)
 */
void buff_top(
    hls::stream<data_t> &x_stream_in,
    hls::stream<data_t> &x_stream_out
);

/**
 * Read d values from stream and apply forward MinMax scaling.
 * x_scaled[k] = x_raw[k] * SCALE[k] + OFFSET[k]
 */
void read_and_scale(
    hls::stream<data_t> &x_stream_in,
    data_t x_scaled[N_FEATURES]
);

/**
 * Apply inverse MinMax scaling and write d values to stream.
 * x_orig[k] = x_solved[k] * INV_SCALE[k] + INV_OFFSET[k]
 */
void inv_scale_and_write(
    const data_t x_solved[N_FEATURES],
    hls::stream<data_t> &x_stream_out
);

#endif  // BUFF_TOP_H
