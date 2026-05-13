# Parity Report: tier_c_second_level_synthetic

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| age_effect | (27,) | allclose+pearson+spearman |  | 1 | 6.2215e-09 | 2.36908e-09 | 1 | 1 | yes |
| age_p_signed_one_sided | (27,) | allclose+pearson+spearman |  | 1 | 8.45433e-08 | 4.19823e-09 | 1 | 1 | yes |
| age_t | (27,) | allclose+pearson+spearman |  | 1 | 2.40914e-06 | 4.56303e-07 | 1 | 1 | yes |
| one_sample_effect | (27,) | allclose+pearson+spearman |  | 1 | 5.0351e-09 | 1.92514e-09 | 1 | 1 | yes |
| one_sample_t | (27,) | allclose+pearson+spearman |  | 1 | 2.62921e-06 | 3.50567e-07 | 1 | 1 | yes |
