算法在进行 run, config 是从JSON文件中读取的，

- 所有非线性项（other_items）是提前人为设定好的，
- 是否self_loop 也是提前人为设定好的,

- 是否需要reset也是提前人为设定好的need_reset = config['need_reset'] (in guard_learning in src\GuardLearning.py )