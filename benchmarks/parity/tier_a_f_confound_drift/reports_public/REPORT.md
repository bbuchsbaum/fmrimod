# Parity Report: tier_a_public_f_confound_drift

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| design | (120, 7) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| effect_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 2.22045e-15 | 6.30789e-16 | 1 | 1 | yes |
| f_conditions_omnibus | (2048,) | allclose+pearson+spearman |  | 1 | 2.27416e-05 | 1.53447e-06 | 1 | 1 | yes |
| t_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 1.72178e-08 | 1.43999e-09 | 1 | 1 | yes |
