## main.py:160 和 HybridAutomata_simulate.py:65 里的调用形式虽然一样，但底层 sys 类型不同：

在 main.py 里，sys 来自 build_system，其 mode_list 元素是 DESystem（src/BuildSystem.py:30），而 DESystem.reset 定义为 reset(self, init_state, input_data, reset_fun=None)（src/DE_System.py:52），因此既需要初始状态也需要输入历史。
在 HybridAutomata_simulate.py 中，sys 由 HybridAutomata.from_json 直接构造，mode_list 是 ODESystem（src/HybridAutomata.py:70），而 ODESystem.reset 只有 reset(self, init_state)（src/ODE_System.py:110）。HybridAutomata.reset 会把传入的额外参数原样传给对应的系统（src/HybridAutomata.py:112），于是 sys.reset(init_state, input_data[:, :config['order']]) 额外多出的参数触发 TypeError: reset() takes 2 positional arguments but 3 were given。
要在 HybridAutomata_simulate.py 中复用 sys.reset，可以仿照 ODESystem 的接口只传入 init_state，或者改写 JSON 初始化出的系统，让它支持输入历史参数。

## 在 automata 文件夹下的 JSON 文件，eq 是用ODE描述的，也就是用ODE_system.py中的ODE类描述的