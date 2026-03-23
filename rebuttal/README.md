# BUFF Rebuttal — DR13756/Jiang

Organization of the referee response, paper revision, and supporting experimental results.

## Structure

```
rebuttal/
    README.md                   # This file
    referee_report.txt          # Original referee comments (verbatim)
    referee_reply.tex           # Point-by-point response
    referee_reply.pdf           # Compiled response
    revision_summary.md         # Concise summary of all changes to prd.tex
    evaluation_scripts/         # Scripts used to produce rebuttal results
        run_bootstrap.sh        # Reproduce bootstrap uncertainty tables
    results/                    # Experimental results referenced in reply
        bootstrap_results.md    # Bootstrap uncertainty results (Tables 1-2)
        discriminator_results.md # MLP classifier AUC for CaloChallenge
        consistency_results.md  # tau21/tau32 consistency check results
```

## Referee's Main Concerns

1. **No error estimates** in tables/plots — no discussion of finite-sample limitations
2. **Missing discriminator test** for CaloChallenge (true vs generated)
3. **Missing consistency checks** (generated tau21 vs tau2/tau1)
4. **Unclear contributions** vs Ref [47]
5. **Missing references** for N-subjettiness, energy correlation functions
6. **Structural issues**: section naming, missing figure references, grammar

## How We Addressed Them

| Concern | Action | Evidence |
|---------|--------|----------|
| No uncertainties | Bootstrap (n=100) on all metrics + truth baselines | `results/bootstrap_results.md` |
| No discriminator | MLP classifier AUC with bootstrap errors | `results/discriminator_results.md` |
| No consistency | tau21/tau32 generated vs derived comparison | `results/consistency_results.md` |
| Unclear contributions | 4 explicit contributions listed in intro | `referee_reply.tex` §1 |
| Missing refs | Added N-subjettiness, ECF, Ref [13] citations | `revision_summary.md` |
| Structure | Renamed Sec IV, split IV.C/IV.D, fixed Fig 8 ref | `revision_summary.md` |

## Reproducing Results

```bash
# Bootstrap uncertainties for JetNet
uv run python -m BUFF.evaluation.run_jetnet_eval \
    --real-data path/to/real.npy \
    --gen-data path/to/generated.npy \
    --n-bootstrap 100

# Bootstrap uncertainties for CaloChallenge
uv run python -m BUFF.evaluation.run_calo_eval \
    --real-data path/to/real.npy \
    --gen-data path/to/generated.npy \
    --n-bootstrap 100
```

## File Locations

- Paper source: `paper_src/prd.tex`
- Bibliography: `paper_src/prd.bib`
- Evaluation code: `BUFF/evaluation/`
- Training code: `BUFF/runner/train_and_sample.py`
