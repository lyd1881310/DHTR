# DHTR

+ 目录说明

  + `./data` 路径下存放预处理后的数据集(划分好cell) `data_set_7.csv`
  + `./saved_models` 路径下存放每个epoch训练之后的模型文件

+ 模型训练

  train_model.py 程序入口处可修改 `n_epochs, batch_size, learning_rate` 参数

  ```pyton
  python train_model.py
  ```

+ 验证

  validate.py 程序入口处可修改要加载的模型文件路径 `path`

  ```
  python validate.py
  ```

  





