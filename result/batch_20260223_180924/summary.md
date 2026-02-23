# Batch Evaluation Summary

## Results

| Category | Benchmark | Mode Error | Train TC | Test TC | Max Diff | Mean Diff | Total Time (s) | Status |
|----------|-----------|------------|----------|---------|----------|-----------|-----------------|--------|
| ATVA | ball | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.67 | ok |
| ATVA | cell | 0 | 0.0000 | 0.0100 | 0.0176 | 0.0002 | 18.01 | ok |
| ATVA | oci | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.91 | ok |
| ATVA | tanks | 0 | 0.0000 | 0.0100 | 0.0177 | 0.0007 | 9.54 | ok |
| FaMoS | buck_converter | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.06 | ok |
| FaMoS | complex_tank | 0 | 0.0000 | 0.0950 | 0.2714 | 0.0039 | 18.76 | ok |
| FaMoS | multi_room_heating | 0 | 0.0000 | 0.6550 | 0.2391 | 0.0189 | 10.39 | ok |
| FaMoS | simple_heating_system | 0 | 0.0000 | 0.0200 | 0.0102 | 0.0007 | 3.00 | ok |
| FaMoS | three_state_ha | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.99 | ok |
| FaMoS | two_state_ha | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.70 | ok |
| FaMoS | variable_heating_system | 0 | 0.0000 | 0.0300 | 0.0200 | 0.0005 | 5.96 | ok |
| linear | complex_underdamped_system | 0 | 0.0000 | 0.0100 | 0.0080 | 0.0004 | 12.05 | ok |
| linear | dc_motor_position_PID | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.37 | ok |
| linear | linear_1 | 0 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 6.56 | ok |
| linear | loop | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.69 | ok |
| linear | one_legged_jumper | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.99 | ok |
| linear | two_tank | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 13.12 | ok |
| linear | underdamped_system | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.01 | ok |
| non_linear | duffing | 0 | 0.0000 | 0.0010 | 0.0003 | 0.0000 | 20.04 | ok |
| non_linear | lander | 0 | 0.0000 | 0.0100 | 0.0024 | 0.0000 | 6.95 | ok |
| non_linear | lotkaVolterra | 0 | 0.0000 | 0.0200 | 0.0047 | 0.0006 | 2.63 | ok |
| non_linear | oscillator | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.38 | ok |
| non_linear | simple_non_linear | 0 | 0.0000 | 0.0060 | 0.0367 | 0.0020 | 16.04 | ok |
| non_linear | simple_non_poly | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.34 | ok |
| non_linear | spacecraft | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.11 | ok |
| non_linear | sys_bio | 0 | 0.0000 | 0.0120 | 0.0960 | 0.0081 | 70.97 | ok |

## Category Summary

| Category | Benchmarks | Succeeded | Avg Mode Error | Avg Train TC | Avg Test TC | Avg Max Diff | Avg Mean Diff | Avg Time (s) |
|----------|------------|-----------|----------------|--------------|-------------|--------------|---------------|---------------|
| ATVA | 4 | 4 | 0.00 | 0.0000 | 0.0050 | 0.0088 | 0.0002 | 9.03 |
| FaMoS | 7 | 7 | 0.00 | 0.0000 | 0.1143 | 0.0772 | 0.0034 | 7.27 |
| linear | 7 | 7 | 0.00 | 0.0014 | 0.0014 | 0.0011 | 0.0001 | 9.11 |
| non_linear | 8 | 8 | 0.00 | 0.0000 | 0.0061 | 0.0175 | 0.0013 | 16.93 |

## Timing Breakdown

| Category | Benchmark | Change Points (s) | Clustering (s) | Guard Learning (s) | Total (s) |
|----------|-----------|--------------------|-----------------|--------------------|----------|
| ATVA | ball | 1.03 | 0.44 | 0.34 | 1.86 |
| ATVA | cell | 4.93 | 3.92 | 0.03 | 9.13 |
| ATVA | oci | 1.79 | 0.57 | 0.05 | 2.50 |
| ATVA | tanks | 2.12 | 2.37 | 0.22 | 4.83 |
| FaMoS | buck_converter | 2.01 | 1.43 | 0.03 | 3.58 |
| FaMoS | complex_tank | 3.47 | 5.20 | 0.55 | 9.54 |
| FaMoS | multi_room_heating | 2.30 | 2.61 | 0.16 | 5.31 |
| FaMoS | simple_heating_system | 1.06 | 0.40 | 0.01 | 1.53 |
| FaMoS | three_state_ha | 0.97 | 0.49 | 0.01 | 1.52 |
| FaMoS | two_state_ha | 0.96 | 0.35 | 0.01 | 1.38 |
| FaMoS | variable_heating_system | 1.85 | 1.03 | 0.06 | 3.03 |
| linear | complex_underdamped_system | 2.92 | 2.91 | 0.08 | 6.14 |
| linear | dc_motor_position_PID | 2.90 | 2.07 | 0.06 | 5.34 |
| linear | linear_1 | 1.21 | 1.34 | 0.68 | 3.33 |
| linear | loop | 1.85 | 1.43 | 0.01 | 3.40 |
| linear | one_legged_jumper | 0.87 | 0.30 | 2.31 | 3.52 |
| linear | two_tank | 1.77 | 0.53 | 4.21 | 6.60 |
| linear | underdamped_system | 2.81 | 1.07 | 0.05 | 4.08 |
| non_linear | duffing | 6.43 | 3.24 | 0.09 | 10.27 |
| non_linear | lander | 2.47 | 0.94 | 0.00 | 3.55 |
| non_linear | lotkaVolterra | 0.85 | 0.40 | 0.03 | 1.35 |
| non_linear | oscillator | 1.18 | 0.47 | 0.00 | 1.73 |
| non_linear | simple_non_linear | 5.72 | 2.08 | 0.03 | 8.21 |
| non_linear | simple_non_poly | 3.37 | 1.15 | 0.03 | 4.79 |
| non_linear | spacecraft | 2.21 | 0.76 | 0.02 | 3.12 |
| non_linear | sys_bio | 21.10 | 13.14 | 0.24 | 36.50 |
