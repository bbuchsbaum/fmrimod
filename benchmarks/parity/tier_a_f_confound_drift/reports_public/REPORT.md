# Parity Report: tier_a_public_f_confound_drift

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| design | (120, 7) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| effect_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 2.66454e-15 | 6.18266e-16 | 1 | 1 | yes |
| f_conditions_omnibus | (2048,) | allclose+pearson+spearman |  | 1 | 3.67757e-05 | 3.36755e-06 | 1 | 1 | yes |
| t_condition_a_minus_b | (2048,) | allclose+pearson+spearman |  | 1 | 2.69395e-08 | 3.01531e-09 | 1 | 1 | yes |
