Derivative-Agnostic Inference of Nonlinear Hybrid Systems 需要很多先验设定

- order
- need_reset
- self_loop
- kernel: 初始化为rbf是因为guard 是非线性的
- other_items ： 非线性项(而且定义形式很多样，有的不带角标，有的带角标)，这个是我想解决的重点
- total_time: automata\non_linear\sys_bio.json 中为2.0, 但是当total_time 为10.0 时,sample_5 plot 出现不收敛状态

思路1： 输入trace plot image，然后直接用LLM 生成 other_items，通过DAINARX 方法计算验证结果，根据验证结果来反馈给LLM 是否正确。LLM 作为策略网络，DAINARX 是环境，验证结果是reward。验证了duffing，基本可以实现

思路2：直接用LLM agent 来完整生成 Hybrid automaton JSON，优点是完全不依赖先验知识，缺点是速度慢，贵，而且LLM 通常倾向于生成一个mode，而且很难推理出edge 也就是transition。试过了Gemini flash， 10次迭代之后没能输出最终结果；Gemini 3 pro 由于价格昂贵，没有试

思路3：输入trace plot image 到LLM， 用DAINARX 中的segmentation& clustering 做tool，来计算出mode 数量，甚至包括可能的eq，以及transition，然后把这些反馈信息给回LLM 来生成最终的hybrid automaton JSON。这个方法听起来像是用DAINARX做工具生成一个差分方程形式的Hybrid automaton ，然后用LLM 来整理成符号化的微分方程