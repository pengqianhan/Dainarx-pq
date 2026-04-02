# Batch Evaluation Summary

## Results

| Category | Benchmark | Mode Error | Train TC | Test TC | Max Diff | Mean Diff | Total Time (s) | Status |
|----------|-----------|------------|----------|---------|----------|-----------|-----------------|--------|
| ATVA | ball | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.31 | ok |
| ATVA | cell | 0 | 0.0000 | 0.0100 | 0.0176 | 0.0002 | 22.05 | ok |
| ATVA | oci | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.60 | ok |
| ATVA | tanks | 0 | 0.0000 | 0.0100 | 0.0177 | 0.0007 | 10.73 | ok |
| FaMoS | buck_converter | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.08 | ok |
| FaMoS | complex_tank | 0 | 0.0000 | 0.0950 | 0.2714 | 0.0039 | 21.74 | ok |
| FaMoS | multi_room_heating | 0 | 0.0000 | 0.6550 | 0.2391 | 0.0189 | 11.29 | ok |
| FaMoS | simple_heating_system | 0 | 0.0000 | 0.0200 | 0.0102 | 0.0007 | 3.35 | ok |
| FaMoS | three_state_ha | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.38 | ok |
| FaMoS | two_state_ha | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.15 | ok |
| FaMoS | variable_heating_system | 0 | 0.0000 | 0.0300 | 0.0200 | 0.0005 | 6.91 | ok |
| RHA | Aircraft | 2 | 0.0000 | 0.0000 | 3.3159 | 0.2377 | 3.25 | ok |
| RHA | BouncingBall | 0 | 0.0100 | 0.0000 | 1.4369 | 0.0008 | 0.71 | ok |
| RHA | FillingTanks | 0 | 0.0000 | 0.1150 | 0.2788 | 0.0130 | 11.56 | ok |
| RHA | JetEngine | 10 | 2.7100 | 0.0000 | 17.6952 | 4.6574 | 1.75 | ok |
| RHA | Oscillator | - | - | - | - | - | - | unknown mode: 3 |
| RHA | PiecewisePoly1 | 2 | 0.0000 | 0.0000 | 0.0398 | 0.0060 | 6.76 | ok |
| RHA | PiecewisePoly2 | - | - | - | - | - | - | unknown mode: 1 |
| RHA | TwoRoomHeater | 3 | 0.1100 | 0.9800 | 504.4591 | 7.3129 | 3.00 | ok |
| RHA | VanDerPol | 8 | 1.6600 | 0.0000 | 1.8766 | 0.5782 | 1.65 | ok |
| linear | complex_underdamped_system | 0 | 0.0000 | 0.0100 | 0.0080 | 0.0004 | 13.96 | ok |
| linear | dc_motor_position_PID | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.93 | ok |
| linear | linear_1 | 0 | 0.0100 | 0.0000 | 0.0000 | 0.0000 | 7.52 | ok |
| linear | loop | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.59 | ok |
| linear | one_legged_jumper | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.39 | ok |
| linear | two_tank | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 15.81 | ok |
| linear | underdamped_system | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.62 | ok |
| non_linear | duffing | 0 | 0.0000 | 0.0010 | 0.0003 | 0.0000 | 22.76 | ok |
| non_linear | lander | 0 | 0.0000 | 0.0100 | 0.0024 | 0.0000 | 8.02 | ok |
| non_linear | lotkaVolterra | 0 | 0.0000 | 0.0200 | 0.0047 | 0.0006 | 3.08 | ok |
| non_linear | oscillator | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.12 | ok |
| non_linear | simple_non_linear | 0 | 0.0000 | 0.0060 | 0.0367 | 0.0020 | 19.02 | ok |
| non_linear | simple_non_poly | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.75 | ok |
| non_linear | spacecraft | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.24 | ok |
| non_linear | sys_bio | 0 | 0.0000 | 0.0120 | 0.0960 | 0.0081 | 82.12 | ok |

## Category Summary

| Category | Benchmarks | Succeeded | Avg Mode Error | Avg Train TC | Avg Test TC | Avg Max Diff | Avg Mean Diff | Avg Time (s) |
|----------|------------|-----------|----------------|--------------|-------------|--------------|---------------|---------------|
| ATVA | 4 | 4 | 0.00 | 0.0000 | 0.0050 | 0.0088 | 0.0002 | 10.67 |
| FaMoS | 7 | 7 | 0.00 | 0.0000 | 0.1143 | 0.0772 | 0.0034 | 8.27 |
| RHA | 9 | 7 | 3.57 | 0.6414 | 0.1564 | 75.5860 | 1.8294 | 4.10 |
| linear | 7 | 7 | 0.00 | 0.0014 | 0.0014 | 0.0011 | 0.0001 | 10.55 |
| non_linear | 8 | 8 | 0.00 | 0.0000 | 0.0061 | 0.0175 | 0.0013 | 19.77 |

## Timing Breakdown

| Category | Benchmark | Change Points (s) | Clustering (s) | Guard Learning (s) | Total (s) |
|----------|-----------|--------------------|-----------------|--------------------|----------|
| ATVA | ball | 1.19 | 0.52 | 0.41 | 2.18 |
| ATVA | cell | 5.77 | 5.00 | 0.06 | 11.22 |
| ATVA | oci | 2.03 | 0.65 | 0.05 | 2.87 |
| ATVA | tanks | 2.41 | 2.61 | 0.24 | 5.46 |
| FaMoS | buck_converter | 2.27 | 1.64 | 0.04 | 4.14 |
| FaMoS | complex_tank | 4.05 | 6.04 | 0.61 | 11.05 |
| FaMoS | multi_room_heating | 2.59 | 2.75 | 0.18 | 5.77 |
| FaMoS | simple_heating_system | 1.18 | 0.43 | 0.01 | 1.73 |
| FaMoS | three_state_ha | 1.10 | 0.55 | 0.01 | 1.72 |
| FaMoS | two_state_ha | 1.14 | 0.38 | 0.01 | 1.63 |
| FaMoS | variable_heating_system | 2.09 | 1.24 | 0.06 | 3.52 |
| RHA | Aircraft | 1.09 | 0.50 | 0.00 | 1.66 |
| RHA | BouncingBall | 0.23 | 0.07 | 0.05 | 0.36 |
| RHA | FillingTanks | 3.42 | 1.85 | 0.37 | 5.93 |
| RHA | JetEngine | 0.77 | 0.10 | 0.00 | 0.88 |
| RHA | PiecewisePoly1 | 2.68 | 0.62 | 0.00 | 3.46 |
| RHA | TwoRoomHeater | 0.50 | 0.76 | 0.23 | 1.51 |
| RHA | VanDerPol | 0.77 | 0.05 | 0.00 | 0.83 |
| linear | complex_underdamped_system | 3.13 | 3.62 | 0.09 | 7.12 |
| linear | dc_motor_position_PID | 3.04 | 2.67 | 0.07 | 6.16 |
| linear | linear_1 | 1.39 | 1.52 | 0.76 | 3.84 |
| linear | loop | 2.08 | 1.63 | 0.01 | 3.87 |
| linear | one_legged_jumper | 0.99 | 0.33 | 2.86 | 4.23 |
| linear | two_tank | 2.04 | 0.62 | 5.17 | 7.98 |
| linear | underdamped_system | 2.96 | 1.18 | 0.05 | 4.42 |
| non_linear | duffing | 7.29 | 3.72 | 0.09 | 11.66 |
| non_linear | lander | 2.78 | 1.14 | 0.00 | 4.10 |
| non_linear | lotkaVolterra | 1.02 | 0.44 | 0.04 | 1.58 |
| non_linear | oscillator | 1.35 | 0.66 | 0.01 | 2.10 |
| non_linear | simple_non_linear | 6.58 | 2.68 | 0.04 | 9.72 |
| non_linear | simple_non_poly | 3.96 | 1.77 | 0.03 | 5.99 |
| non_linear | spacecraft | 2.50 | 0.99 | 0.02 | 3.72 |
| non_linear | sys_bio | 23.45 | 16.18 | 0.27 | 42.22 |
