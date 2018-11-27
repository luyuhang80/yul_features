简介: 以ENG-wiki 为例，构建 bag of words 特征和 提取 bi-lstm embedding 作为节点特征的示例代码。
作者: 卢宇航 

------

#### build_bow.py

- load_wl(dic)：读取含有词频的字典，并且返回带有索引的单词字典。
- main()：
  - 根据总词表数量，确定bow特征维度
  - 输入每篇文章得到特定的bow特征，保存成npy格式。

#### model.py

- Bi-lstm 训练模型

#### train.py

- 训练Bi-lstm代码
- 训练时默认模型保存文件名"save"

#### utils.py

-  数据读取、预处理代码

#### inference.py

- 读取训练好的模型，生成基于bi-lstm的词向量
- 当前默认读取双向50维词向量，每向25维。

