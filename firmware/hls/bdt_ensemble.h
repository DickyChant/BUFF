/**
 * BUFF FPGA Inference — BDT ensemble evaluation.
 *
 * Evaluates all d BDTs for a single ODE timestep.  Tree parameters are
 * stored in BRAM-backed arrays indexed by [feature][timestep][tree][node].
 * This allows time-multiplexing: d BDT evaluation units are instantiated
 * once and reused across all N timesteps by changing the index.
 */
#ifndef BDT_ENSEMBLE_H
#define BDT_ENSEMBLE_H

#include "buff_types.h"

// ── Single tree traversal ────────────────────────────────────────────
/**
 * Traverse one decision tree of depth MAX_DEPTH.
 *
 * The tree is stored in a breadth-first array layout:
 *   - Internal nodes [0 .. N_INTERNAL_NODES-1]:
 *       threshold[node], feature_idx[node]
 *   - Leaves [0 .. N_LEAVES-1]:
 *       leaf_value[leaf]
 *
 * Navigation: left child of node i  = 2*i + 1
 *             right child of node i = 2*i + 2
 *             leaf index            = node_idx - N_INTERNAL_NODES
 *
 * With MAX_DEPTH=4 the traversal is a fixed 4-comparison chain,
 * fully unrolled by HLS.
 */
inline score_t tree_predict(
    const data_t x[N_FEATURES],
    const data_t thresh[N_INTERNAL_NODES],
    const int feat_idx[N_INTERNAL_NODES],
    const score_t leaves[N_LEAVES]
) {
    #pragma HLS INLINE
    int node = 0;

    // Depth-4 traversal, fully unrolled
    tree_traverse:
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        #pragma HLS UNROLL
        int f = feat_idx[node];
        if (x[f] <= thresh[node]) {
            node = 2 * node + 1;  // left child
        } else {
            node = 2 * node + 2;  // right child
        }
    }

    // node is now in the leaf range [N_INTERNAL_NODES, N_INTERNAL_NODES + N_LEAVES)
    int leaf_idx = node - N_INTERNAL_NODES;
    return leaves[leaf_idx];
}


// ── Single BDT prediction (sum of N_TREES trees) ────────────────────
/**
 * Evaluate one BDT: sum the predictions of all N_TREES trees.
 *
 * @param x         Input feature vector (all d features)
 * @param step_idx  ODE timestep index (selects which timestep's parameters)
 * @param feat_out  Which output feature this BDT predicts
 * @return          Sum of leaf values across all trees
 */
inline score_t bdt_predict(
    const data_t x[N_FEATURES],
    int step_idx,
    int feat_out
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1

    accum_t sum = 0;

    tree_sum:
    for (int t = 0; t < N_TREES; t++) {
        #pragma HLS PIPELINE II=1
        sum += tree_predict(
            x,
            bdt_threshold[feat_out][step_idx][t],
            bdt_feature_idx[feat_out][step_idx][t],
            bdt_leaf_value[feat_out][step_idx][t]
        );
    }

    return score_t(sum);
}


// ── Velocity evaluation (all d BDTs in parallel) ─────────────────────
/**
 * Evaluate the velocity field v(t, x) at a single ODE timestep.
 *
 * All d BDTs run in parallel on the same input vector x.
 * Each BDT[k] predicts the k-th component of the velocity.
 *
 * @param x_in      Current state vector (d features)
 * @param v_out     Output velocity vector (d components)
 * @param step_idx  ODE timestep index (selects tree parameters from BRAM)
 */
void velocity_eval(
    const data_t x_in[N_FEATURES],
    data_t v_out[N_FEATURES],
    int step_idx
) {
    #pragma HLS INLINE off
    // Partition arrays so all d BDTs can read/write simultaneously
    #pragma HLS ARRAY_PARTITION variable=x_in complete dim=1
    #pragma HLS ARRAY_PARTITION variable=v_out complete dim=1

    // Evaluate all d BDTs in parallel
    velocity_features:
    for (int k = 0; k < N_FEATURES; k++) {
        #pragma HLS UNROLL
        v_out[k] = bdt_predict(x_in, step_idx, k);
    }
}

#endif  // BDT_ENSEMBLE_H
