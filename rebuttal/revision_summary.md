# Revision Summary — BUFF (DR13756/Jiang)

Comprehensive list of all changes made to `prd.tex` in response to referee report.

## Major Additions

### 1. Bootstrap Uncertainties (Tables 1-2)
- All evaluation metrics now report value +/- error (n=100 bootstrap resamples)
- Finite-sample truth baselines computed by evaluating metrics on two independent halves of real data
- Table 1 (JetNet HLV) recomputed end-to-end on a single self-consistent run so flowBDT and truth columns share statistics; supersedes paper-era numbers. Inline-truth-in-parentheses format follows the JetNet paper (Ref [58]) standard
- Implementation: `BUFF/evaluation/bootstrap.py`, results in `rebuttal/results/jetnet_v3/`

### 2. MLP Classifier Discriminator Test (JetNet HLV + CaloChallenge)
- MLP trained to distinguish real samples from flowBDT-generated samples
- JetNet 12-feature HLV: AUC = 0.9994 +/- 0.0002
- CaloChallenge 8-feature HLV: AUC = 0.9995 +/- 0.0002
- ROC curve embedded in paper (Fig. fig:calo_roc); near-unity AUC discussed honestly as a consequence of per-feature regression's inability to enforce joint correlations and min-max boundary-clipping artefacts at training-range edges (see `scripts/diagnose_discriminator.py`)
- Implementation: `BUFF/evaluation/discriminator.py`

### 3. Consistency Checks (tau21, tau32)
- Compare directly generated tau21 with tau2/tau1 computed from independently generated tau1, tau2
- Same for tau32 vs tau3/tau2
- Quantified: W1 ~ 0.014 (tau21), ~ 0.029 (tau32) -- both within the percent level of the underlying ratio support
- Implementation: `BUFF/evaluation/consistency.py`, plots in `rebuttal/results/jetnet_v3/consistency_tau{21,32}.pdf`

### 5. Quantitative JetNet 30x3 Metrics (new Table 2)
- Particle-level evaluation following Ref [58]: W1-M (jet mass / pT_jet), W1-Pt (sanity), W1-P(eta_rel, phi_rel, pT_rel)
- Two columns in Table 2: "Raw" generator output and "+ pt renorm" (post-hoc per-jet rescaling so sum_i pT_rel matches the empirical real-data distribution)
- Raw (W1 x 10^3): jet mass 276.9, jet pT 230.0, per-particle eta/phi 190/130, per-particle pT 10.7
- With renorm: jet mass 202.5, jet pT 19.8, per-particle eta/phi unchanged, per-particle pT 6.0
- Truth floor: jet mass 0.18, jet pT 0.38, per-particle eta/phi 0.70, per-particle pT 0.24
- Renormalization collapses W1-Pt to within ~50x the floor (cheap fix); jet mass and angular distributions remain three orders of magnitude above the floor in both columns because per-feature MSE does not constrain the angular structure that determines m_jet
- The higher-capacity retrain (depth=6, n_est=200, eta=0.05, multi-output; ran 15h 41min in job 52862779) is now the headline configuration in Table 2. Hicap improves every metric ~10-15% over baseline (W1-M 242 vs 277; W1-Pt 199 vs 230, etc.) but does NOT close the structural gap -- the jet-level kinematics remain two orders of magnitude above the truth floor regardless of capacity, consistent with the conclusion that a backbone with an explicit jet-level loss is required
- Pipeline: `scripts/run_jetnet_30x3_eval.py`, `scripts/slurm/run_jetnet_30x3.sh` (baseline), `scripts/slurm/run_jetnet_30x3_hicap.sh` (hicap headline); results at `rebuttal/results/jetnet_30x3_truthonly/` (truth floor), `rebuttal/results/jetnet_30x3/` (baseline), and `rebuttal/results/jetnet_30x3_hicap/` (headline)

### 6. Figure 8 reproduction (Pythia26 unfolding)
- Re-generated from scratch via a self-contained pipeline (`scripts/build_omnifold_unfolding.py` for data, `runner/train_and_sample.py` with the new `--source-data` flag for Flow-OT, `scripts/plot_fig8.py` for the 6-panel plot, `scripts/slurm/run_fig8.sh` for orchestration; ran 57 min in job 52870261)
- The y-axis label overlap from the original figure is fixed (now uses ratio sub-panels with proper labels)
- Quantitatively confirms the original narrative: Flow-OT W1(mass)=0.10 vs Flow-Diffu 0.51 vs Sim-prior 3.33; Flow-OT outperforms Flow-Diffu by factors of 2-12 across all six observables
- New PDF replaces `paper_src/FLOWBDT/figures1/flowbdt_plot/fold_comp.pdf` (the original is kept as `fold_comp_orig_backup.pdf` for reference)

### 4. Public Code Repository
- Training: `BUFF/runner/train_and_sample.py`
- Evaluation: `BUFF/evaluation/`
- ODE solvers: `BUFF/runner/ode_example.py`
- Code availability statement added to Conclusion

## Clarifications

### BUFF vs flowBDT vs Ref [47]
- **BUFF** = methodology (Boosted Decision Tree based Ultra-Fast Flow matching)
- **flowBDT** = specific trained model
- Four contributions beyond Ref [47]:
  1. First application to HEP datasets
  2. Higher-order ODE solvers (Midpoint, Dormand-Prince)
  3. Multi-output tree strategy (5-8x inference speedup)
  4. Conditional generation from non-Gaussian priors

### Data Handling
- New paragraph in Section II: all data is continuous, real-valued
- Variable-length data uses zero-padding with binary mask
- Min-max scaling to [-1, 1]
- Data shapes stated explicitly: (N, 12), (N, 368), (N, 90)

### Implementation
- Flow matching data preparation: torchcfm package
- GBT training and ODE solvers: own code (now public)

## Section-by-Section Fixes

### Section I (Introduction)
- Clarified BUFF = methodology, flowBDT = model
- Listed 4 explicit contributions vs Ref [47]
- Added code availability statement

### Section II (Method)
- Fixed Eq. 6 reference -> Eq. 4b
- Acknowledged data duplication follows Ref [47] explicitly
- Added data handling paragraph (continuous data, zero-padding, scaling)

### Section III (Datasets)
- Title: "Dataset" -> "Datasets"
- III.A (JetNet): Reworded to not imply authors created the dataset
- III.B (CaloChallenge): Added ATLAS geometry reference; clarified photons from dataset 1
- III.C (Unfolding): Fixed cross-reference to correct subsection; added Zenodo DOI
- III.D (GFlash): Corrected GFlash as "fast simulation tool"; added Zenodo DOI

### Section IV (Results, formerly "Tasks")
- Renamed section: "Tasks" -> "Results"
- Split former IV.C into IV.C (Unfolding) and IV.D (Schrodinger Bridge) -> 4 result sections match 4 datasets
- IV.A: Added Ref [13] citation; added N-subjettiness and ECF references
- IV.A: Fixed "f-divergence" -> "triangular discriminator, a specific f-divergence"
- IV.A: Removed "impressive"; replaced with quantitative statements
- IV.A: Added tau21/tau32 consistency check discussion
- IV.B: Added explicit data shapes; added quantitative Table 2 of JetNet 30x3 W1 metrics (mass, pT, per-particle eta/phi/pT) following Ref [58]; softened "excellent alignment" language to honestly reflect the large jet-level W1 gap; added CaloChallenge AUC + ROC figure
- Section V: softened "continues to perform well" to honestly reflect that 30x3 jet-level kinematics are three orders of magnitude above the truth floor; flagged multi-output backbone / explicit jet-level loss as the natural follow-up
- IV.C: Reorganized to describe setup before results
- IV.C: Referenced Figure 8 in text; defined Flow-OT and Flow-Diffu labels
- Figure 8: Fixed overlapping y-axis labels

### Grammar and Language
- "reasonale" -> "rationale"
- "simultanously" -> "simultaneously"
- "Jacobican" -> "Jacobian"
- "problems insist" -> "problems persist"
- "continuous differentiable" -> "continuously differentiable"
- "dimensonalities" -> "dimensionalities"
- Fixed spacing: "Equation.4" -> "Equation 4", "Table." -> "Table ", etc.
- Extensive rewriting of awkward phrasing throughout all sections
