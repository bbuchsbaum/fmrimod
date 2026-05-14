# Parity Report: tier_a_multirun_concat

Status: `pass`

## Array Deltas

| quantity | shape | gate | caveat | scale | max_abs | mae | pearson_r | spearman_rho | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| design | (200, 10) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| effect_main_A_minus_B | (1024,) | allclose+pearson+spearman |  | 1 | 7.4472e-11 | 6.74834e-11 | 1 | 1 | yes |
| effect_run_diff_A | (1024,) | allclose+pearson+spearman |  | 1 | 9.02389e-12 | 2.46607e-12 | 1 | 1 | yes |
| effect_trial_x_run_interaction | (1024,) | allclose+pearson+spearman |  | 1 | 1.63993e-11 | 4.19121e-12 | 1 | 1 | yes |
| f_task_omnibus | (1024,) | allclose+pearson+spearman |  | 1 | 7.00676e-10 | 5.177e-11 | 1 | 1 | yes |
| rank | (1,) | allclose+pearson+spearman |  | 1 | 0 | 0 | 1 | 1 | yes |
| t_main_A_minus_B | (1024,) | allclose+pearson+spearman |  | 1 | 2.19694e-10 | 2.02142e-11 | 1 | 1 | yes |
| t_run_diff_A | (1024,) | allclose+pearson+spearman |  | 1 | 2.04332e-10 | 1.91042e-11 | 1 | 1 | yes |
| t_trial_x_run_interaction | (1024,) | allclose+pearson+spearman |  | 1 | 1.84138e-10 | 1.93742e-11 | 1 | 1 | yes |
