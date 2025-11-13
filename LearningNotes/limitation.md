算法在进行 run, config 是从JSON文件中读取的，也就意味着，算法的设计依赖人类经验来进行一些提前的设置

- 所有非线性项（other_items）是提前人为设定好的，
  在main.py中：
  ```python
  FeatureExtractor(len(data_list[0]), len(input_data[0]),
                                   order=config['order'], dt=config['dt'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'])
  ```
- 是否self_loop 也是提前人为设定好的
  在main.py中：
  ```python
  clustering(slice_data, config['self_loop'])
  ```
  在Clustering.py中：
  ```python
  def clustering(data: list[Slice], self_loop=False):
  ```
  ```python
  if not self_loop:
            last_mode = data[i].mode
  ```

- 是否需要reset也是提前人为设定好的need_reset = config['need_reset']
  在GuardLearning.py中：
  ```python
  def guard_learning(data: list[Slice], get_feature, config):
    positive_sample = {}
    negative_sample = {}
    slice_data = {}
    need_reset = config['need_reset']
  ```