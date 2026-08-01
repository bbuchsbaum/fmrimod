# Parity Report: tier_a_public_f_confound_drift

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| design | (120, 7) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| effect_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 1.77636e-15 | 4.24438e-16 | 1 | 1 | yes |
| f_conditions_omnibus | (2048,) | allclose+pearson+spearman |  | 1 | 2.89901e-11 | 1.18437e-12 | 1 | 1 | yes |
| t_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 6.46594e-13 | 3.45054e-14 | 1 | 1 | yes |
