# JSON Automata Configuration Analysis Report

**扫描目录**: `/home/phan635/HybridAutomata/baseline_ha/Dainarx-pq/automata`

---

## 1. 基本统计

- **总JSON文件数**: 27
- **包含input字段的文件数**: 4
- **不包含input字段的文件数**: 23
- **包含config字段的文件数**: 27

## 2. Input字段分析

### 包含input字段的文件 (4):

- ✓ `ATVA/tanks.json`
- ✓ `linear/complex_underdamped_system.json`
- ✓ `non_linear/duffing.json`
- ✓ `non_linear/duffing_simulate.json`

### 不包含input字段的文件 (23):

- ✗ `ATVA/ball.json`
- ✗ `ATVA/cell.json`
- ✗ `ATVA/oci.json`
- ✗ `FaMoS/buck_converter.json`
- ✗ `FaMoS/complex_tank.json`
- ✗ `FaMoS/multi_room_heating.json`
- ✗ `FaMoS/simple_heating_system.json`
- ✗ `FaMoS/three_state_ha.json`
- ✗ `FaMoS/two_state_ha.json`
- ✗ `FaMoS/variable_heating_system.json`
- ✗ `linear/dc_motor_position_PID.json`
- ✗ `linear/linear_1.json`
- ✗ `linear/loop.json`
- ✗ `linear/one_legged_jumper.json`
- ✗ `linear/two_tank.json`
- ✗ `linear/underdamped_system.json`
- ✗ `non_linear/lander.json`
- ✗ `non_linear/lotkaVolterra.json`
- ✗ `non_linear/oscillator.json`
- ✗ `non_linear/simple_non_linear.json`
- ✗ `non_linear/simple_non_poly.json`
- ✗ `non_linear/spacecraft.json`
- ✗ `non_linear/sys_bio.json`

## 3. Config参数统计

### 3.1 Order参数

| Order值 | 文件数量 | 文件列表 |
|---------|---------|----------|
| 1 | 19 | `ATVA/ball.json`<br>`ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/buck_converter.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/multi_room_heating.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/linear_1.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`non_linear/lander.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/simple_non_poly.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |
| 2 | 7 | `FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`linear/complex_underdamped_system.json`<br>`linear/loop.json`<br>`linear/underdamped_system.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulate.json` |
| 4 | 1 | `linear/dc_motor_position_PID.json` |

### 3.2 Need_reset参数

| Need_reset值 | 文件数量 | 文件列表 |
|--------------|---------|----------|
| True | 10 | `ATVA/ball.json`<br>`FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`linear/complex_underdamped_system.json`<br>`linear/dc_motor_position_PID.json`<br>`linear/loop.json`<br>`linear/underdamped_system.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulate.json`<br>`non_linear/simple_non_poly.json` |
| null | 17 | `ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/buck_converter.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/multi_room_heating.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/linear_1.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`non_linear/lander.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |

### 3.3 Kernel参数

| Kernel值 | 文件数量 | 文件列表 |
|----------|---------|----------|
| rbf | 6 | `FaMoS/buck_converter.json`<br>`FaMoS/multi_room_heating.json`<br>`non_linear/duffing.json`<br>`non_linear/duffing_simulate.json`<br>`non_linear/lotkaVolterra.json`<br>`non_linear/oscillator.json` |
| null | 21 | `ATVA/ball.json`<br>`ATVA/cell.json`<br>`ATVA/oci.json`<br>`ATVA/tanks.json`<br>`FaMoS/complex_tank.json`<br>`FaMoS/simple_heating_system.json`<br>`FaMoS/three_state_ha.json`<br>`FaMoS/two_state_ha.json`<br>`FaMoS/variable_heating_system.json`<br>`linear/complex_underdamped_system.json`<br>`linear/dc_motor_position_PID.json`<br>`linear/linear_1.json`<br>`linear/loop.json`<br>`linear/one_legged_jumper.json`<br>`linear/two_tank.json`<br>`linear/underdamped_system.json`<br>`non_linear/lander.json`<br>`non_linear/simple_non_linear.json`<br>`non_linear/simple_non_poly.json`<br>`non_linear/spacecraft.json`<br>`non_linear/sys_bio.json` |

### 3.4 所有配置项汇总

在所有JSON文件的config字段中，出现过的所有配置项：

- `class_weight`
- `dt`
- `kernel`
- `minus`
- `need_bias`
- `need_reset`
- `order`
- `other_items`
- `self_loop`
- `svm_c`
- `total_time`
- `window_size`

## 4. 详细配置表

| 文件 | Order | Need_reset | Kernel | 其他配置项 |
|------|-------|------------|--------|------------|
| `ATVA/ball.json` | 1 | True | null | `dt`, `total_time`, `other_items`, `self_loop` |
| `ATVA/cell.json` | 1 | null | null | `dt`, `total_time` |
| `ATVA/oci.json` | 1 | null | null | `dt`, `total_time` |
| `ATVA/tanks.json` | 1 | null | null | `dt`, `total_time`, `other_items` |
| `FaMoS/buck_converter.json` | 1 | null | rbf | `dt`, `total_time`, `other_items`, `class_weight` |
| `FaMoS/complex_tank.json` | 1 | null | null | `dt`, `total_time`, `window_size`, `class_weight`, `svm_c` |
| `FaMoS/multi_room_heating.json` | 1 | null | rbf | `dt`, `total_time`, `window_size`, `class_weight` |
| `FaMoS/simple_heating_system.json` | 1 | null | null | `dt`, `total_time`, `class_weight` |
| `FaMoS/three_state_ha.json` | 2 | True | null | `dt`, `total_time`, `need_bias` |
| `FaMoS/two_state_ha.json` | 2 | True | null | `dt`, `total_time`, `need_bias` |
| `FaMoS/variable_heating_system.json` | 1 | null | null | `dt`, `total_time`, `class_weight` |
| `linear/complex_underdamped_system.json` | 2 | True | null | `dt`, `total_time`, `minus` |
| `linear/dc_motor_position_PID.json` | 4 | True | null | `dt`, `total_time`, `minus` |
| `linear/linear_1.json` | 1 | null | null | `dt`, `total_time`, `window_size` |
| `linear/loop.json` | 2 | True | null | `dt`, `total_time` |
| `linear/one_legged_jumper.json` | 1 | null | null | `dt`, `total_time`, `other_items` |
| `linear/two_tank.json` | 1 | null | null | `dt`, `total_time` |
| `linear/underdamped_system.json` | 2 | True | null | `dt`, `total_time`, `minus` |
| `non_linear/duffing.json` | 2 | True | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/duffing_simulate.json` | 2 | True | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/lander.json` | 1 | null | null | `dt`, `total_time`, `minus`, `other_items` |
| `non_linear/lotkaVolterra.json` | 1 | null | rbf | `dt`, `total_time`, `window_size`, `other_items` |
| `non_linear/oscillator.json` | 1 | null | rbf | `dt`, `total_time`, `other_items` |
| `non_linear/simple_non_linear.json` | 1 | null | null | `dt`, `total_time`, `need_bias`, `other_items` |
| `non_linear/simple_non_poly.json` | 1 | True | null | `dt`, `total_time`, `other_items`, `svm_c` |
| `non_linear/spacecraft.json` | 1 | null | null | `dt`, `total_time`, `other_items`, `class_weight` |
| `non_linear/sys_bio.json` | 1 | null | null | `dt`, `total_time`, `need_bias`, `other_items`, `class_weight` |

