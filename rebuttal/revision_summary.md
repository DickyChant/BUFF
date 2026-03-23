# Revision Summary — BUFF (DR13756/Jiang)

Comprehensive list of all changes made to `prd.tex` in response to referee report.

## Major Additions

### 1. Bootstrap Uncertainties (Tables 1-2)
- All evaluation metrics now report value +/- error (n=100 bootstrap resamples)
- Finite-sample truth baselines computed by evaluating metrics on two independent halves of real data
- Follows standard set by JetNet paper (Ref [58]): e.g. W1 = 0.5 +/- 0.1 (truth: 0.2 +/- 0.1)
- Implementation: `BUFF/evaluation/bootstrap.py`

### 2. MLP Classifier Discriminator Test (CaloChallenge)
- MLP trained to distinguish real Geant4 showers from generated flowBDT showers
- Reports AUC with bootstrap uncertainties
- ROC curve included in evaluation output
- Implementation: `BUFF/evaluation/discriminator.py`

### 3. Consistency Checks (tau21, tau32)
- Compare directly generated tau21 with tau2/tau1 computed from independently generated tau1, tau2
- Same for tau32 vs tau3/tau2
- Small W1 distance confirms model learns correct inter-variable correlations
- Implementation: `BUFF/evaluation/consistency.py`

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
- IV.B: Added explicit data shapes
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
