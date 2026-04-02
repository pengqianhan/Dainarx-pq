# Batch Evaluation Summary

## Results

| Category | Benchmark | Mode Error | Train TC | Test TC | Max Diff | Mean Diff | Total Time (s) | Status |
|----------|-----------|------------|----------|---------|----------|-----------|-----------------|--------|
| ATVA | ball | 0 | 4.9300 | 0.0000 | 36171.6409 | 2526.1938 | 2.34 | ok |
| ATVA | cell | - | - | - | - | - | - | unknown mode: 1 |
| ATVA | oci | - | - | - | - | - | - | unknown mode: 1 |
| ATVA | tanks | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | buck_converter | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | complex_tank | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | multi_room_heating | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | simple_heating_system | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | three_state_ha | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | two_state_ha | - | - | - | - | - | - | unknown mode: 1 |
| FaMoS | variable_heating_system | - | - | - | - | - | - | unknown mode: 1 |
| RHA | Aircraft | - | - | - | - | - | - | unknown mode: 1 |
| RHA | BouncingBall | 2 | 0.8800 | 0.0000 | 17.3818 | 2.9177 | 0.47 | ok |
| RHA | FillingTanks | - | - | - | - | - | - | unknown mode: 1 |
| RHA | JetEngine | - | - | - | - | - | - | unknown mode: 1 |
| RHA | Oscillator | - | - | - | - | - | - | unknown mode: 1 |
| RHA | PiecewisePoly1 | - | - | - | - | - | - | unknown mode: 1 |
| RHA | PiecewisePoly2 | - | - | - | - | - | - | unknown mode: 1 |
| RHA | TwoRoomHeater | - | - | - | - | - | - | unknown mode: 1 |
| RHA | VanDerPol | - | - | - | - | - | - | unknown mode: 1 |
| linear | complex_underdamped_system | - | - | - | - | - | - | unknown mode: 1 |
| linear | dc_motor_position_PID | 1 | 9.7300 | 9.7100 | nan | nan | 17.54 | ok |
| linear | linear_1 | - | - | - | - | - | - | unknown mode: 1 |
| linear | loop | - | - | - | - | - | - | unknown mode: 1 |
| linear | one_legged_jumper | - | - | - | - | - | - | unknown mode: 1 |
| linear | two_tank | - | - | - | - | - | - | unknown mode: 1 |
| linear | underdamped_system | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | duffing | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | lander | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | lotkaVolterra | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | oscillator | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | simple_non_linear | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | simple_non_poly | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | spacecraft | - | - | - | - | - | - | unknown mode: 1 |
| non_linear | sys_bio | - | - | - | - | - | - | unknown mode: 1 |

## Category Summary

| Category | Benchmarks | Succeeded | Avg Mode Error | Avg Train TC | Avg Test TC | Avg Max Diff | Avg Mean Diff | Avg Time (s) |
|----------|------------|-----------|----------------|--------------|-------------|--------------|---------------|---------------|
| ATVA | 4 | 1 | 0.00 | 4.9300 | 0.0000 | 36171.6409 | 2526.1938 | 2.34 |
| FaMoS | 7 | 0 | - | - | - | - | - | - |
| RHA | 9 | 1 | 2.00 | 0.8800 | 0.0000 | 17.3818 | 2.9177 | 0.47 |
| linear | 7 | 1 | 1.00 | 9.7300 | 9.7100 | nan | nan | 17.54 |
| non_linear | 8 | 0 | - | - | - | - | - | - |

## Timing Breakdown

| Category | Benchmark | Change Points (s) | Clustering (s) | Guard Learning (s) | Total (s) |
|----------|-----------|--------------------|-----------------|--------------------|----------|
| ATVA | ball | 1.11 | 0.06 | 0.00 | 1.17 |
| RHA | BouncingBall | 0.22 | 0.02 | 0.00 | 0.24 |
| linear | dc_motor_position_PID | 7.96 | 0.81 | 0.00 | 8.77 |
