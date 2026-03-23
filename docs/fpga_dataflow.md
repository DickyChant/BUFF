# BUFF FPGA Dataflow Pipeline — Design Document

## Overview

This document describes the FPGA implementation of BUFF inference for L1 trigger deployment. The design converts trained XGBoost BDT regressors to HLS using conifer and implements an async dataflow architecture for inter-event pipelining.

## Architecture

BUFF inference runs an ODE solver (Euler) where each step evaluates `d` independent BDT regressors as the velocity field, then accumulates. Steps are sequential (`x_{n+1}` depends on `x_n`), but within each step the `d` BDT evaluations are fully parallel.

### Time-Multiplexed ODE Pipeline

Full spatial unrolling of all `n_t x d` BDTs is infeasible for large configs (e.g. 360 BDTs at ~2k LUTs each = 720k LUTs). Instead, we instantiate `d` BDT evaluation units and time-multiplex across timesteps by loading tree parameters from BRAM each step.

```
                        +-- BRAM bank 0 (trees for feature 0, all timesteps)
                        |   BRAM bank 1 (trees for feature 1, all timesteps)
                        |   ...
                        |   BRAM bank d-1
                        v
x_in --> [Scaler] --> [BDT_0] --\
  (d)    (fixed-pt     [BDT_1] --> [Accumulate: x += h*v] --> x_out
          linear)      ...    --/      (d multiplies + adds)     |
                       [BDT_{d-1}]                               |
                            ^                                    |
                            |     feedback (N-1 times)           |
                            +------------------------------------+
                                      step counter

         After N-1 iterations: x_out --> [InvScaler] --> result
```

### Dataflow (Inter-Event Pipelining)

The `#pragma HLS DATAFLOW` directive enables stage overlap between consecutive events:

```
Event 0:  |--scale--|--step0--|--step1--| ... |--step28--|--invscale--|
Event 1:            |--scale--|--step0--| ... |--step27--|--step28--|--invscale--|
Event 2:                      |--scale--|--step0--| ...
```

Three pipeline stages:
1. **Read + Forward Scale**: Read d values from AXI-Stream, apply MinMax normalization
2. **Euler ODE Solve**: N-1 sequential iterations (dominates latency)
3. **Inverse Scale + Write**: Undo normalization, write to AXI-Stream

Since Stage 2 dominates, throughput = ~1 event per ODE-solve time.

## BDT Evaluation Unit

Each BDT evaluates 100 trees of depth 4. Tree traversal is a 4-comparison chain, fully unrolled by HLS. The 100 tree scores are summed (pipelined accumulation).

### Tree Storage (Breadth-First Array Layout)

For a depth-4 tree:
- 15 internal nodes: `threshold[0..14]`, `feature_idx[0..14]`
- 16 leaves: `leaf_value[0..15]`
- Navigation: left child of node i = `2*i + 1`, right child = `2*i + 2`

Parameters indexed as `[feature][timestep][tree][node]`, stored in BRAM.

### Per-Step Evaluation

All `d` BDTs evaluate in parallel on the same input vector:

```cpp
velocity_features:
for (int k = 0; k < N_FEATURES; k++) {
    #pragma HLS UNROLL
    v_out[k] = bdt_predict(x_in, step_idx, k);
}
```

## Latency Budget

Target: < 1 us for L1 trigger at 200 MHz (5 ns clock period).

### Per-Step Breakdown

| Component | Cycles | Time (200 MHz) |
|-----------|--------|----------------|
| BDT tree traversal (depth 4) | ~1-2 | ~5-10 ns |
| Sum 100 trees (pipelined) | ~3-5 | ~15-25 ns |
| Accumulate (d multiply-adds) | ~2-3 | ~10-15 ns |
| **Total per step** | **~6-10** | **~30-50 ns** |

### Total Latency by Configuration

| Config (d, N) | Steps | Est. Latency | L1 Budget |
|---------------|-------|--------------|-----------|
| d=4, N=15 | 14 | ~0.56 us | OK |
| d=4, N=30 | 29 | ~1.16 us | Marginal |
| d=6, N=15 | 14 | ~0.56 us | OK |
| d=12, N=15 | 14 | ~0.56 us | OK |
| d=12, N=30 | 29 | ~1.16 us | Marginal |

Note: Latency is independent of `d` since all features evaluate in parallel.

## Resource Estimates

### Per BDT Unit (depth 4, 100 trees)

| Resource | Usage |
|----------|-------|
| Comparators | ~100 (one per tree root, pipelined through 4 levels) |
| BRAM (thresholds) | 100 trees x 15 nodes x 16 bits = 24 Kb |
| BRAM (leaf values) | 100 trees x 16 leaves x 16 bits = 25.6 Kb |
| BRAM (feature indices) | 100 trees x 15 nodes x 4 bits = 6 Kb |
| LUTs | ~2,000 (combinational tree traversal + adder tree) |

### Total by Configuration

| Config | d units | BRAM (18Kb blocks) | LUTs | Latency |
|--------|---------|-------------------|------|---------|
| d=4, N=15 | 4 | ~60 | ~8k | ~0.6 us |
| d=4, N=30 | 4 | ~120 | ~8k | ~1.2 us |
| d=12, N=15 | 12 | ~180 | ~24k | ~0.6 us |
| d=12, N=30 | 12 | ~360 | ~24k | ~1.2 us |

LUT count stays constant (d units reused). BRAM scales with `n_t x d`. All configs fit on Virtex UltraScale+ (VU9P: 6840 BRAMs, 1,182k LUTs).

## Fixed-Point Precision

Default: `ap_fixed<16,6>` for data/thresholds, `ap_fixed<24,12>` for accumulators.

| Type | Width | Integer | Fractional | Range | Resolution |
|------|-------|---------|------------|-------|------------|
| `data_t` | 16 | 6 | 10 | [-32, 31.999] | ~0.001 |
| `score_t` | 16 | 6 | 10 | [-32, 31.999] | ~0.001 |
| `accum_t` | 24 | 12 | 12 | [-2048, 2047.999] | ~0.0002 |

The wider accumulator prevents overflow during the `h * v` multiply and multi-step ODE integration. For `N=30` steps with velocities in [-1, 1], the state stays well within `accum_t` range.

### Precision Tradeoffs

- **Narrower (12,4)**: Saves LUTs/BRAM, but quantization error may degrade physics metrics (W1 distance)
- **Default (16,6)**: Good balance; ~0.001 resolution sufficient for MinMax-scaled [-1,1] features
- **Wider (20,8)**: Minimal quantization error, but 25% more BRAM, diminishing returns

Recommended validation: sweep `W,I` and compare W1 distance of FPGA-generated samples against float reference.

## File Structure

```
BUFF/firmware/
    export_conifer.py           # Python: model conversion + code generation
    hls/
        buff_types.h            # Fixed-point types, configuration constants
        bdt_ensemble.h          # BDT evaluation (tree traversal + velocity)
        euler_solver.h          # N-step Euler ODE loop
        buff_top.h              # Top-level header
        buff_top.cpp            # Top-level with DATAFLOW + scaler stages
        testbench.cpp           # C-sim testbench with golden vectors
        build_hls.tcl           # Vitis HLS build script
    generated/                  # Output from export_conifer.py
        hls/
            bdt_params.h        # Generated BDT tree parameters
            scaler_params.h     # Generated scaler constants
            golden_vectors.h    # Generated test vectors
        export_config.json      # Export configuration record
```

## How to Run

### 1. Train Models (existing workflow)

```bash
uv run python -m BUFF.runner.train_and_sample \
    --data path/to/data.h5 \
    --n-timesteps 30 --solver euler --solver-steps 30 \
    --output-dir results/my_model/
```

### 2. Export to HLS

```bash
uv run python -m BUFF.firmware.export_conifer \
    --model-dir results/my_model/ \
    --output-dir BUFF/firmware/generated/ \
    --n-features 4 --n-steps 30 \
    --precision "ap_fixed<16,6>" \
    --n-test-events 100
```

### 3. C Simulation

```bash
cd BUFF/firmware/generated/hls
vitis_hls -f build_hls.tcl  # runs csim + csynth
```

Or standalone (without Vitis HLS):
```bash
g++ -std=c++14 -I$XILINX_HLS/include testbench.cpp buff_top.cpp -o tb && ./tb
```

### 4. Synthesis + Co-simulation

Uncomment `cosim_design` and `export_design` in `build_hls.tcl`, then re-run.

### 5. Precision Sweep

```bash
for prec in "ap_fixed<12,4>" "ap_fixed<16,6>" "ap_fixed<20,8>"; do
    python -m BUFF.firmware.export_conifer \
        --model-dir results/my_model/ \
        --output-dir BUFF/firmware/sweep_${prec}/ \
        --precision "$prec"
done
```

Compare W1 distance of generated distributions against float reference at each precision.

## Verification Checklist

1. **C simulation**: `testbench.cpp` compares HLS output against Python golden vectors. Must pass with tolerance (quantization error only).
2. **Co-simulation**: Vitis HLS cosim verifies RTL matches C sim cycle-accurately.
3. **Precision sweep**: Run export with different `ap_fixed<W,I>` widths, compare W1 distance.
4. **Latency check**: `csynth_design` report must show total latency < 1 us target.
5. **Resource check**: Post-synthesis utilization must fit target FPGA.

## Design Tradeoffs

### Full Unroll vs Time-Multiplex

| Approach | LUTs | BRAM | Latency | Throughput |
|----------|------|------|---------|------------|
| Full unroll (N*d units) | N*d*2k | Minimal | ~40 ns (1 step, all parallel) | 1 event / 40 ns |
| Time-mux (d units) | d*2k | N*d*~1 block | N*40 ns | 1 event / N*40 ns |

Full unroll gives ~30x better throughput but is infeasible for large configs (d=12, N=30 = 720k LUTs). Time-mux trades throughput for area, keeping LUTs constant while scaling BRAM linearly.

### Euler vs Higher-Order Solvers

Euler is simplest (1 velocity eval per step). Midpoint requires 2 evals per step (2x latency). DOPRI5 requires 6 evals per step (6x latency, but needs fewer steps). For FPGA L1 trigger, Euler with N=15-30 is the practical choice.
