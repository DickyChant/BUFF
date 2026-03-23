# ======================================================================
# BUFF FPGA Inference — Vitis HLS Build Script
#
# Usage:
#   vitis_hls -f build_hls.tcl
#
# Targets (comment/uncomment as needed):
#   csim_design    — C simulation with testbench
#   csynth_design  — HLS synthesis (produces latency/resource reports)
#   cosim_design   — RTL co-simulation (verifies RTL matches C sim)
#   export_design  — Export IP for Vivado integration
# ======================================================================

# ── Project setup ─────────────────────────────────────────────────────
open_project buff_hls
set_top buff_top

# Source files
add_files buff_top.cpp
add_files buff_types.h
add_files bdt_ensemble.h
add_files euler_solver.h
add_files buff_top.h

# Generated parameter files (produced by export_conifer.py)
# Uncomment after running export:
# add_files bdt_params.h
# add_files scaler_params.h

# Testbench
add_files -tb testbench.cpp
add_files -tb golden_vectors.h

# ── Solution configuration ────────────────────────────────────────────
open_solution "solution1" -flow_target vivado

# Target FPGA — Xilinx Virtex UltraScale+ (common for L1 trigger)
# Alternatives: xcu250, xcvu13p, xcku115
set_part {xcvu9p-flga2104-2L-e}

# Clock: 200 MHz (5 ns period) — typical L1 trigger clock
create_clock -period 5 -name default

# ── Optimization directives ───────────────────────────────────────────
# BDT parameter arrays → BRAM (avoid LUT mapping for large arrays)
# These are applied in the source via #pragma, but can also be set here:
# set_directive_resource -core RAM_1P "bdt_predict" bdt_threshold
# set_directive_resource -core RAM_1P "bdt_predict" bdt_leaf_value

# ── Build stages ──────────────────────────────────────────────────────
# Stage 1: C simulation (verify functional correctness)
csim_design

# Stage 2: HLS synthesis (get latency and resource estimates)
csynth_design

# Stage 3: RTL co-simulation (verify RTL matches C simulation)
# Uses the same testbench; can be slow for large designs.
# cosim_design

# Stage 4: Export as Vivado IP catalog entry
# export_design -format ip_catalog -description "BUFF flowBDT FPGA inference" -vendor "buff" -display_name "buff_top"

exit
