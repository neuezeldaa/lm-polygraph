# Rank-correlation matrix (33 estimators, n=150, quality: Accuracy)

Spearman rho on the per-sample uncertainty vectors. PRR asks which estimator
ranks best; this asks how many *distinct* measurements exist. Two estimators
correlating at |rho| >= 0.9 are one signal under two names, and
their PRR ordering is not a result.

## Near-duplicates of the top 8 by normalized PRR@0.5

| Rank | Estimator | PRR@0.5 | correlates >= 0.90 with |
|---:|---|---:|---|
| 1 | `SpilledEnergy_marginal_mean` | 0.8678 | `SelfCertainty (0.982)`, `SpilledEnergy_logit_mean_noterm (0.936)`, `SpilledEnergy_marginal_mean_noterm (0.934)`, `MeanTokenEntropy (0.909)` |
| 2 | `MeanTokenEntropy` | 0.8608 | `Pooled_log_likelihood_mean_noterm (0.937)`, `SpilledEnergy_logit_max_noterm (0.919)`, `Pooled_log_likelihood_max_noterm (0.915)`, `SpilledEnergy_marginal_mean (0.909)`, `SelfCertainty (0.901)` |
| 3 | `SelfCertainty` | 0.8596 | `SpilledEnergy_marginal_mean (0.982)`, `SpilledEnergy_logit_mean_noterm (0.961)`, `SpilledEnergy_marginal_mean_noterm (0.958)`, `MeanTokenEntropy (0.901)` |
| 4 | `SpilledEnergy_logit_min` | 0.8208 | `SpilledEnergy_logit_min_noterm (0.969)`, `SpilledEnergy_marginal_min_noterm (0.967)` |
| 5 | `Pooled_log_likelihood_mean_noterm` | 0.7931 | `MeanTokenEntropy (0.937)`, `Pooled_log_likelihood_max_noterm (0.933)` |
| 6 | `SpilledEnergy_logit_max_noterm` | 0.7681 | `SpilledEnergy_marginal_max (0.991)`, `SpilledEnergy_marginal_max_noterm (0.991)`, `MeanTokenEntropy (0.919)`, `Pooled_log_likelihood_max_noterm (0.901)` |
| 7 | `SpilledEnergy_logit_mean_noterm` | 0.7630 | `SpilledEnergy_marginal_mean_noterm (0.998)`, `SelfCertainty (0.961)`, `SpilledEnergy_marginal_mean (0.936)` |
| 8 | `SpilledEnergy_marginal_mean_noterm` | 0.7575 | `SpilledEnergy_logit_mean_noterm (0.998)`, `SelfCertainty (0.958)`, `SpilledEnergy_marginal_mean (0.934)` |

## Clusters at |rho| >= 0.9

11 distinct signals among 33 estimators.

| # | Size | Best PRR | Members (by PRR) |
|---:|---:|---:|---|
| 1 | 10 | 0.8678 | `SpilledEnergy_marginal_mean`, `MeanTokenEntropy`, `SelfCertainty`, `Pooled_log_likelihood_mean_noterm`, `SpilledEnergy_logit_max_noterm`, `SpilledEnergy_logit_mean_noterm`, `SpilledEnergy_marginal_mean_noterm`, `Pooled_log_likelihood_max_noterm`, `SpilledEnergy_marginal_max`, `SpilledEnergy_marginal_max_noterm` |
| 2 | 3 | 0.8208 | `SpilledEnergy_logit_min`, `SpilledEnergy_logit_min_noterm`, `SpilledEnergy_marginal_min_noterm` |
| 3 | 2 | 0.7190 | `Pooled_log_likelihood_min`, `Pooled_log_likelihood_min_noterm` |
| 4 | 1 | 0.7124 | `SpilledEnergy_logit_mean` |
| 5 | 1 | 0.7051 | `SpilledEnergy_marginal_min` |
| 6 | 5 | 0.4871 | `SpilledEnergy_logit_max`, `MaximumSequenceProbability`, `SpilledEnergy_spilled_max`, `SpilledEnergy_scaled_spilled_max`, `Pooled_log_likelihood_max` |
| 7 | 4 | 0.3719 | `SpilledEnergy_scaled_spilled_min`, `SpilledEnergy_scaled_spilled_min_noterm`, `SpilledEnergy_spilled_min`, `SpilledEnergy_spilled_min_noterm` |
| 8 | 3 | 0.3419 | `SpilledEnergy_spilled_mean`, `SpilledEnergy_scaled_spilled_mean`, `Pooled_log_likelihood_mean` |
| 9 | 1 | 0.2600 | `SpilledEnergy_scaled_spilled_mean_noterm` |
| 10 | 1 | 0.2570 | `SpilledEnergy_spilled_mean_noterm` |
| 11 | 2 | -0.3253 | `SpilledEnergy_spilled_max_noterm`, `SpilledEnergy_scaled_spilled_max_noterm` |

## Selected cross-family pairs

| A | B | rho | tau |
|---|---|---:|---:|
| `SpilledEnergy_marginal_min_noterm` | `SpilledEnergy_logit_min_noterm` | 0.999 | 0.983 |
| `SpilledEnergy_marginal_mean_noterm` | `SpilledEnergy_logit_mean_noterm` | 0.998 | 0.964 |
| `SpilledEnergy_marginal_max_noterm` | `SpilledEnergy_logit_max_noterm` | 0.991 | 0.928 |
| `SpilledEnergy_marginal_mean` | `SelfCertainty` | 0.982 | 0.894 |
| `SpilledEnergy_logit_max` | `SpilledEnergy_scaled_spilled_max` | 0.942 | 0.813 |
| `SpilledEnergy_marginal_mean` | `MeanTokenEntropy` | 0.909 | 0.730 |
| `SpilledEnergy_marginal_max` | `MeanTokenEntropy` | 0.897 | 0.710 |
| `SpilledEnergy_marginal_mean_noterm` | `SpilledEnergy_logit_min_noterm` | 0.886 | 0.710 |
| `SpilledEnergy_marginal_max` | `SelfCertainty` | 0.876 | 0.685 |
| `SpilledEnergy_marginal_min_noterm` | `SpilledEnergy_logit_mean_noterm` | 0.875 | 0.695 |
| `SpilledEnergy_logit_max` | `SpilledEnergy_spilled_max` | 0.853 | 0.743 |
| `SpilledEnergy_marginal_max_noterm` | `Pooled_log_likelihood_max_noterm` | 0.853 | 0.653 |
| `SpilledEnergy_marginal_max_noterm` | `SpilledEnergy_logit_mean_noterm` | 0.851 | 0.648 |
| `SpilledEnergy_marginal_mean_noterm` | `SpilledEnergy_logit_max_noterm` | 0.835 | 0.629 |
| `SpilledEnergy_logit_min_noterm` | `SpilledEnergy_scaled_spilled_min_noterm` | 0.834 | 0.658 |
| `SpilledEnergy_marginal_min_noterm` | `SpilledEnergy_scaled_spilled_min_noterm` | 0.834 | 0.657 |
| `SpilledEnergy_marginal_mean_noterm` | `Pooled_log_likelihood_mean_noterm` | 0.825 | 0.632 |
| `SpilledEnergy_marginal_min_noterm` | `Pooled_log_likelihood_min_noterm` | 0.825 | 0.622 |

## Reading

The top-ranked `SpilledEnergy_marginal_mean` is not a distinct measurement: it correlates at
|rho| >= 0.9 with 4 other estimator(s). Its margin over them is
not evidence that it measures something they do not.
