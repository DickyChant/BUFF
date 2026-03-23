"""
BUFF FPGA Export — Convert trained XGBoost models to HLS via conifer.

Takes a trained BUFF model directory (containing models.pkl and scaler.pkl)
and generates a complete Vitis HLS project with:
  1. BDT parameters packed into C arrays (BRAM-friendly layout)
  2. Scaler parameters as C constants
  3. Golden test vectors for C-simulation verification

The generated HLS project can be built with:
    cd <output_dir>/hls && vitis_hls -f build_hls.tcl

Usage
-----
    python -m BUFF.firmware.export_conifer \
        --model-dir results/jetnet_highlevel/ \
        --output-dir BUFF/firmware/generated/ \
        --n-features 4 --n-steps 30 \
        --precision "ap_fixed<16,6>" \
        --n-test-events 100

Dependencies
------------
    pip install conifer xgboost numpy scikit-learn
"""

import argparse
import copy
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

# Optional: conifer is only needed for the convert_via_conifer path.
# The direct extraction path (extract_tree_params) works without conifer.
try:
    import conifer
    HAS_CONIFER = True
except ImportError:
    HAS_CONIFER = False

import xgboost as xgb


def parse_args():
    p = argparse.ArgumentParser(
        description="Export trained BUFF models to FPGA HLS project"
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing models.pkl and scaler.pkl",
    )
    p.add_argument(
        "--output-dir",
        default="BUFF/firmware/generated",
        help="Output directory for generated HLS files",
    )
    p.add_argument("--n-features", type=int, default=None, help="Override feature count")
    p.add_argument("--n-steps", type=int, default=None, help="Override step count")
    p.add_argument("--n-trees", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument(
        "--precision",
        default="ap_fixed<16,6>",
        help="Fixed-point precision string for HLS",
    )
    p.add_argument("--n-test-events", type=int, default=100)
    p.add_argument("--class-idx", type=int, default=0, help="Class index to export")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use-conifer",
        action="store_true",
        help="Use conifer library for conversion (requires conifer installed)",
    )
    return p.parse_args()


# ── Tree parameter extraction ─────────────────────────────────────────

def extract_tree_params(xgb_model, max_depth):
    """
    Extract tree structure from an XGBoost model into flat arrays
    suitable for FPGA BDT evaluation.

    Returns:
        thresholds:  (n_trees, n_internal_nodes) — comparison values
        feature_ids: (n_trees, n_internal_nodes) — feature index at each node
        leaf_values: (n_trees, n_leaves)          — leaf scores
    """
    booster = xgb_model.get_booster()
    dump = booster.get_dump(dump_format="json")

    n_internal = (1 << max_depth) - 1
    n_leaves = 1 << max_depth
    n_trees = len(dump)

    thresholds = np.zeros((n_trees, n_internal), dtype=np.float64)
    feature_ids = np.zeros((n_trees, n_internal), dtype=np.int32)
    leaf_values = np.zeros((n_trees, n_leaves), dtype=np.float64)

    for tree_idx, tree_json_str in enumerate(dump):
        tree_json = json.loads(tree_json_str)
        _fill_tree_arrays(
            tree_json, 0, 0,
            thresholds[tree_idx],
            feature_ids[tree_idx],
            leaf_values[tree_idx],
            n_internal,
        )

    return thresholds, feature_ids, leaf_values


def _fill_tree_arrays(node, node_idx, depth, thresholds, feature_ids, leaf_values, n_internal):
    """Recursively fill flat arrays from XGBoost JSON tree dump."""
    if "leaf" in node:
        # Leaf node
        leaf_idx = node_idx - n_internal
        if 0 <= leaf_idx < len(leaf_values):
            leaf_values[leaf_idx] = node["leaf"]
        return

    # Internal node
    if node_idx < n_internal:
        # Parse feature index from "f0", "f1", etc.
        split_feat = node.get("split", "f0")
        if isinstance(split_feat, str) and split_feat.startswith("f"):
            feat_id = int(split_feat[1:])
        else:
            feat_id = int(split_feat)

        thresholds[node_idx] = node.get("split_condition", 0.0)
        feature_ids[node_idx] = feat_id

    # Recurse into children
    children = node.get("children", [])
    if len(children) >= 2:
        # XGBoost: children[0] = yes (left), children[1] = no (right)
        left_idx = 2 * node_idx + 1
        right_idx = 2 * node_idx + 2
        _fill_tree_arrays(children[0], left_idx, depth + 1,
                          thresholds, feature_ids, leaf_values, n_internal)
        _fill_tree_arrays(children[1], right_idx, depth + 1,
                          thresholds, feature_ids, leaf_values, n_internal)


# ── Conifer-based extraction (optional) ───────────────────────────────

def extract_via_conifer(xgb_model, precision, output_dir):
    """
    Use conifer to convert an XGBoost model, then extract parameters
    from the generated HLS.  Returns the same format as extract_tree_params.
    """
    if not HAS_CONIFER:
        raise ImportError("conifer is required for --use-conifer mode")

    cfg = conifer.backends.xilinxhls.auto_config()
    cfg["Precision"] = precision
    cfg["OutputDir"] = output_dir

    cnf_model = conifer.model(
        xgb_model, conifer.backends.xilinxhls, conifer.converters.xgboost, cfg
    )
    cnf_model.compile()
    cnf_model.write()
    return cnf_model


# ── Code generation ───────────────────────────────────────────────────

def format_fixed_array(values, name, dims_str, indent=0):
    """Format a C array initializer from numpy values."""
    prefix = " " * indent
    lines = [f"{prefix}const data_t {name}{dims_str} = {{"]

    if values.ndim == 1:
        vals = ", ".join(f"{v:.8f}" for v in values)
        lines.append(f"{prefix}    {vals}")
    else:
        # Multi-dimensional: flatten with nested braces
        lines.append(_format_nd_array(values, indent + 4))

    lines.append(f"{prefix}}};")
    return "\n".join(lines)


def _format_nd_array(arr, indent):
    """Recursively format multi-dimensional array."""
    prefix = " " * indent
    if arr.ndim == 1:
        vals = ", ".join(f"{v:.8f}" for v in arr)
        return f"{prefix}{{{vals}}}"
    else:
        inner = [_format_nd_array(arr[i], indent + 4) for i in range(arr.shape[0])]
        joined = ",\n".join(inner)
        return f"{prefix}{{\n{joined}\n{prefix}}}"


def generate_bdt_params_header(all_thresholds, all_feature_ids, all_leaf_values, config):
    """
    Generate bdt_params.h with all tree parameters packed into C arrays.

    all_thresholds:  [n_features][n_steps][n_trees][n_internal_nodes]
    all_feature_ids: [n_features][n_steps][n_trees][n_internal_nodes]
    all_leaf_values: [n_features][n_steps][n_trees][n_leaves]
    """
    n_feat = config["n_features"]
    n_steps = config["n_steps"]
    n_trees = config["n_trees"]
    n_internal = (1 << config["max_depth"]) - 1
    n_leaves = 1 << config["max_depth"]

    lines = [
        "/**",
        " * Auto-generated BDT parameters.",
        f" * Config: d={n_feat}, N={n_steps}, trees={n_trees}, depth={config['max_depth']}",
        " * Generated by BUFF/firmware/export_conifer.py",
        " */",
        "#ifndef BDT_PARAMS_H",
        "#define BDT_PARAMS_H",
        "",
        "#include \"buff_types.h\"",
        "",
    ]

    # Threshold array
    lines.append(f"const data_t bdt_threshold[{n_feat}][{n_steps}][{n_trees}][{n_internal}] = {{")
    for f in range(n_feat):
        lines.append(f"    {{ // feature {f}")
        for s in range(n_steps):
            lines.append(f"        {{ // step {s}")
            for t in range(n_trees):
                vals = ", ".join(f"{v:.8f}" for v in all_thresholds[f][s][t])
                lines.append(f"            {{{vals}}},")
            lines.append("        },")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    # Feature index array
    lines.append(f"const int bdt_feature_idx[{n_feat}][{n_steps}][{n_trees}][{n_internal}] = {{")
    for f in range(n_feat):
        lines.append(f"    {{ // feature {f}")
        for s in range(n_steps):
            lines.append(f"        {{ // step {s}")
            for t in range(n_trees):
                vals = ", ".join(f"{int(v)}" for v in all_feature_ids[f][s][t])
                lines.append(f"            {{{vals}}},")
            lines.append("        },")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    # Leaf value array
    lines.append(f"const score_t bdt_leaf_value[{n_feat}][{n_steps}][{n_trees}][{n_leaves}] = {{")
    for f in range(n_feat):
        lines.append(f"    {{ // feature {f}")
        for s in range(n_steps):
            lines.append(f"        {{ // step {s}")
            for t in range(n_trees):
                vals = ", ".join(f"{v:.8f}" for v in all_leaf_values[f][s][t])
                lines.append(f"            {{{vals}}},")
            lines.append("        },")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    lines.append("#endif  // BDT_PARAMS_H")
    return "\n".join(lines)


def generate_scaler_params_header(scaler_dict, config):
    """
    Generate scaler_params.h from sklearn MinMaxScaler state.

    MinMaxScaler with feature_range=(-1, 1):
        x_scaled = (x - data_min) / (data_max - data_min) * 2 - 1
                 = x * scale + offset
    where:
        scale  = 2 / (data_max - data_min)
        offset = -1 - 2 * data_min / (data_max - data_min)

    Inverse:
        x = (x_scaled - offset) / scale = x_scaled * inv_scale + inv_offset
    """
    scaler = scaler_dict["scaler"]
    n_feat = config["n_features"]

    # Extract sklearn MinMaxScaler internals
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    data_range = scaler.data_range_

    # Forward: x_scaled = x * scale_ + min_ (sklearn convention)
    scale = scaler.scale_       # 2 / data_range (for feature_range=(-1,1))
    offset = scaler.min_        # -1 - scale * data_min

    # Inverse: x = (x_scaled - offset) / scale = x_scaled / scale - offset / scale
    inv_scale = 1.0 / scale
    inv_offset = -offset / scale

    def fmt_array(name, values):
        vals = ", ".join(f"{v:.10f}" for v in values[:n_feat])
        return f"const data_t {name}[N_FEATURES] = {{{vals}}};"

    lines = [
        "/**",
        " * Auto-generated scaler parameters.",
        " * Generated by BUFF/firmware/export_conifer.py",
        " */",
        "#ifndef SCALER_PARAMS_H",
        "#define SCALER_PARAMS_H",
        "",
        "#include \"buff_types.h\"",
        "",
        "#define BUFF_PARAMS_GENERATED",
        "",
        fmt_array("SCALE", scale),
        fmt_array("OFFSET", offset),
        fmt_array("INV_SCALE", inv_scale),
        fmt_array("INV_OFFSET", inv_offset),
        "",
        "#endif  // SCALER_PARAMS_H",
    ]
    return "\n".join(lines)


def generate_golden_vectors(scaler_dict, regr, config, n_events, seed):
    """
    Generate golden test vectors by running the Python reference Euler
    solver on random inputs, then quantizing to the target fixed-point
    precision for bit-accurate comparison.

    Returns (golden_input, golden_output) as float arrays.
    """
    np.random.seed(seed)

    scaler = scaler_dict["scaler"]
    n_feat = config["n_features"]
    n_steps = config["n_steps"]
    class_idx = config["class_idx"]

    # Generate random inputs in the original data range
    x_min = scaler_dict.get("X_min", np.full(n_feat, -3.0))
    x_max = scaler_dict.get("X_max", np.full(n_feat, 3.0))
    golden_input = np.random.uniform(
        x_min[:n_feat], x_max[:n_feat], size=(n_events, n_feat)
    )

    golden_output = np.zeros_like(golden_input)
    h = 1.0 / (n_steps - 1)

    for ev in range(n_events):
        # Forward scale
        x = scaler.transform(golden_input[ev : ev + 1]).flatten()[:n_feat]

        # Quantize to simulate fixed-point (round to nearest representable)
        frac_bits = 10  # 16-6 = 10 fractional bits for ap_fixed<16,6>
        x = np.round(x * (2**frac_bits)) / (2**frac_bits)
        x = np.clip(x, -32.0, 31.999)  # ap_fixed<16,6> range

        # Euler integration
        for step in range(n_steps - 1):
            v = np.zeros(n_feat)
            for k in range(n_feat):
                model = regr[class_idx][step][k]
                v[k] = model.predict(x.reshape(1, -1))[0]
                # Quantize velocity
                v[k] = np.round(v[k] * (2**frac_bits)) / (2**frac_bits)
                v[k] = np.clip(v[k], -32.0, 31.999)

            # Euler update with quantized arithmetic
            x = x + h * v
            x = np.round(x * (2**frac_bits)) / (2**frac_bits)
            x = np.clip(x, -32.0, 31.999)

        # Inverse scale
        x_out = scaler.inverse_transform(x.reshape(1, -1)).flatten()[:n_feat]
        golden_output[ev] = x_out

    return golden_input, golden_output


def write_golden_vectors_header(golden_input, golden_output, output_path):
    """Write golden_vectors.h with test data as C arrays."""
    n_events, n_feat = golden_input.shape

    lines = [
        "/**",
        " * Auto-generated golden test vectors.",
        " * Generated by BUFF/firmware/export_conifer.py",
        " */",
        "#ifndef GOLDEN_VECTORS_H",
        "#define GOLDEN_VECTORS_H",
        "",
        f"const int N_TEST_EVENTS = {n_events};",
        "",
        f"const float golden_input[{n_events}][{n_feat}] = {{",
    ]
    for ev in range(n_events):
        vals = ", ".join(f"{v:.8f}" for v in golden_input[ev])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    lines.append(f"const float golden_output[{n_events}][{n_feat}] = {{")
    for ev in range(n_events):
        vals = ", ".join(f"{v:.8f}" for v in golden_output[ev])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")
    lines.append("#endif  // GOLDEN_VECTORS_H")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ── HLS template copy ────────────────────────────────────────────────

def copy_hls_templates(output_dir):
    """Copy static HLS source files to the output directory."""
    hls_src = Path(__file__).parent / "hls"
    hls_dst = Path(output_dir) / "hls"
    hls_dst.mkdir(parents=True, exist_ok=True)

    static_files = [
        "buff_types.h",
        "bdt_ensemble.h",
        "euler_solver.h",
        "buff_top.h",
        "buff_top.cpp",
        "testbench.cpp",
        "build_hls.tcl",
    ]

    for fname in static_files:
        src = hls_src / fname
        if src.exists():
            shutil.copy2(src, hls_dst / fname)
            print(f"  Copied {fname}")


# ── Main export pipeline ─────────────────────────────────────────────

def export(args):
    """
    Main export pipeline:
    1. Load trained models + scaler
    2. Extract tree parameters from all XGBoost models
    3. Generate HLS parameter headers
    4. Generate golden test vectors
    5. Copy static HLS sources
    """
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    hls_dir = output_dir / "hls"
    hls_dir.mkdir(parents=True, exist_ok=True)

    # ── Load trained models ───────────────────────────────────────────
    print("Loading trained models...")
    with open(model_dir / "models.pkl", "rb") as f:
        regr = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler_dict = pickle.load(f)

    # Determine dimensions from model structure
    # regr[class_idx][timestep][feature] = XGBoost model
    n_classes = len(regr)
    n_steps_trained = len(regr[0])
    n_feat_trained = len(regr[0][0])

    n_feat = args.n_features if args.n_features else n_feat_trained
    n_steps = args.n_steps if args.n_steps else n_steps_trained
    class_idx = args.class_idx

    print(f"  Classes: {n_classes}, Timesteps: {n_steps_trained}, Features: {n_feat_trained}")
    print(f"  Exporting: class={class_idx}, d={n_feat}, N={n_steps}")

    config = {
        "n_features": n_feat,
        "n_steps": n_steps,
        "n_trees": args.n_trees,
        "max_depth": args.max_depth,
        "precision": args.precision,
        "class_idx": class_idx,
    }

    # ── Extract tree parameters ───────────────────────────────────────
    print("Extracting tree parameters...")
    n_internal = (1 << args.max_depth) - 1
    n_leaves = 1 << args.max_depth

    all_thresholds = np.zeros((n_feat, n_steps, args.n_trees, n_internal))
    all_feature_ids = np.zeros((n_feat, n_steps, args.n_trees, n_internal), dtype=np.int32)
    all_leaf_values = np.zeros((n_feat, n_steps, args.n_trees, n_leaves))

    for feat in range(n_feat):
        for step in range(n_steps):
            # Map step index if n_steps != n_steps_trained
            step_mapped = min(step, n_steps_trained - 1)
            feat_mapped = min(feat, n_feat_trained - 1)

            model = regr[class_idx][step_mapped][feat_mapped]
            thresh, fids, lvals = extract_tree_params(model, args.max_depth)

            n_actual_trees = min(thresh.shape[0], args.n_trees)
            all_thresholds[feat, step, :n_actual_trees] = thresh[:n_actual_trees]
            all_feature_ids[feat, step, :n_actual_trees] = fids[:n_actual_trees]
            all_leaf_values[feat, step, :n_actual_trees] = lvals[:n_actual_trees]

        print(f"  Feature {feat}: extracted {n_steps} timesteps x {n_actual_trees} trees")

    # ── Generate headers ──────────────────────────────────────────────
    print("Generating HLS headers...")

    # BDT parameters
    bdt_header = generate_bdt_params_header(
        all_thresholds, all_feature_ids, all_leaf_values, config
    )
    bdt_path = hls_dir / "bdt_params.h"
    with open(bdt_path, "w") as f:
        f.write(bdt_header)
    print(f"  Written: {bdt_path}")

    # Scaler parameters
    scaler_header = generate_scaler_params_header(scaler_dict, config)
    scaler_path = hls_dir / "scaler_params.h"
    with open(scaler_path, "w") as f:
        f.write(scaler_header)
    print(f"  Written: {scaler_path}")

    # ── Generate golden vectors ───────────────────────────────────────
    print(f"Generating {args.n_test_events} golden test vectors...")
    golden_in, golden_out = generate_golden_vectors(
        scaler_dict, regr, config, args.n_test_events, args.seed
    )
    golden_path = hls_dir / "golden_vectors.h"
    write_golden_vectors_header(golden_in, golden_out, golden_path)
    print(f"  Written: {golden_path}")

    # ── Copy static HLS templates ─────────────────────────────────────
    print("Copying HLS template files...")
    copy_hls_templates(output_dir)

    # ── Generate config summary ───────────────────────────────────────
    config_summary = {
        **config,
        "n_internal_nodes": n_internal,
        "n_leaves": n_leaves,
        "total_bdts": n_feat * n_steps,
        "n_test_events": args.n_test_events,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
    }
    config_path = output_dir / "export_config.json"
    with open(config_path, "w") as f:
        json.dump(config_summary, f, indent=2)
    print(f"  Config: {config_path}")

    # ── Estimate resources ────────────────────────────────────────────
    bdt_luts = n_feat * 2000  # ~2k LUTs per BDT unit
    bram_per_step = n_feat * (args.n_trees * (n_internal + n_leaves) * 16) / 8 / 1024  # KB
    total_bram_kb = bram_per_step * n_steps
    latency_per_step_ns = 40  # ~8 cycles at 200 MHz
    total_latency_us = (n_steps - 1) * latency_per_step_ns / 1000

    print("\n── Resource Estimates ──────────────────────────────────")
    print(f"  BDT units:        {n_feat} (one per feature)")
    print(f"  LUTs (BDT logic): ~{bdt_luts}")
    print(f"  BRAM (params):    ~{total_bram_kb:.1f} KB ({total_bram_kb / 2.25:.0f} x 18Kb blocks)")
    print(f"  Latency:          ~{total_latency_us:.2f} us ({n_steps - 1} steps x ~{latency_per_step_ns} ns)")
    print(f"  Total BDTs:       {n_feat * n_steps} (time-multiplexed onto {n_feat} units)")
    print("")
    print("Done! To build:")
    print(f"  cd {hls_dir} && vitis_hls -f build_hls.tcl")


def main():
    args = parse_args()
    export(args)


if __name__ == "__main__":
    main()
