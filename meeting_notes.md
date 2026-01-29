Derivative-Agnostic Inference of Nonlinear Hybrid Systems 需要很多先验设定

- order
- need_reset
- self_loop
- kernel: 初始化为rbf是因为guard 是非线性的
- other_items ： 非线性项(而且定义形式很多样，有的不带角标，有的带角标)，这个是我想解决的重点
- total_time: automata\non_linear\sys_bio.json 中为2.0, 但是当total_time 为10.0 时,sample_5 plot 出现不收敛状态。 automata\json_analysis_report.md 中total_time 的统计结果：
  | Total_time值 | 文件数量 | 文件列表 |
|--------------|---------|----------|
| 0.02 | 1 | `FaMoS\buck_converter.json` |
| 1 | 1 | `non_linear\oscillator.json` |
| 2.0 | 1 | `non_linear\sys_bio.json` |
| 5.0 | 1 | `linear\one_legged_jumper.json` |
| 10.0 | 7 | `ATVA\ball.json`<br>`non_linear\duffing.json`<br>`non_linear\duffing_simulate.json`<br>`non_linear\lander.json`<br>`non_linear\lotkaVolterra.json`<br>`non_linear\simple_non_linear.json`<br>`non_linear\spacecraft.json` |
| 13.0 | 1 | `FaMoS\multi_room_heating.json` |
| 20.0 | 12 | `ATVA\oci.json`<br>`ATVA\tanks.json`<br>`FaMoS\complex_tank.json`<br>`FaMoS\three_state_ha.json`<br>`FaMoS\two_state_ha.json`<br>`FaMoS\variable_heating_system.json`<br>`linear\complex_underdamped_system.json`<br>`linear\dc_motor_position_PID.json`<br>`linear\linear_1.json`<br>`linear\loop.json`<br>`linear\two_tank.json`<br>`linear\underdamped_system.json` |
| 21.0 | 1 | `FaMoS\simple_heating_system.json` |
| 60.0 | 1 | `non_linear\simple_non_poly.json` |
| 100.0 | 1 | `ATVA\cell.json` |

思路1： 输入trace plot image，然后直接用LLM 生成 other_items，通过DAINARX 方法计算验证结果，根据验证结果来反馈给LLM 是否正确。LLM 作为策略网络，DAINARX 是环境，验证结果是reward。验证了duffing，基本可以实现

思路2：直接用LLM agent 来完整生成 Hybrid automaton JSON，优点是完全不依赖先验知识，缺点是速度慢，贵，而且LLM 通常倾向于生成一个mode，而且很难推理出edge 也就是transition。试过了Gemini flash， 10次迭代之后没能输出最终结果；Gemini 3 pro 由于价格昂贵，没有试

思路3：输入trace plot image 到LLM， 用DAINARX 中的segmentation& clustering 做tool，来计算出mode 数量，甚至包括可能的eq，以及transition，然后把这些反馈信息给回LLM 来生成最终的hybrid automaton JSON。这个方法听起来像是用DAINARX做工具生成一个差分方程形式的Hybrid automaton ，然后用LLM 来整理成符号化的微分方程