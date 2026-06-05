# Parity Report: tier_a_public_f_confound_drift

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| design | (120, 7) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| effect_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 3.10862e-15 | 6.72531e-16 | 1 | 1 | yes |
| f_conditions_omnibus | (2048,) | allclose+pearson+spearman |  | 1 | 2.57741e-06 | 2.17715e-07 | 1 | 1 | yes |
| t_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 3.58619e-09 | 3.7755e-10 | 1 | 1 | yes |
