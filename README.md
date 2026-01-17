# 垃圾邮件检测

## 代码结构

### app.py

应用程序，运行下面命令开始。

```
uvicorn app:app
```

### data.py

将原始数据转换为csv文件，方便后续加载数据，无需重复处理。

### datasets.py

数据处理函数。

### ml_train.py

Naive Bayes、Decision Tree (entropy)、Decision Tree (gini, deeper)、Linear SVM模型训练和测试。

### model.py

CNN、LSTM、Transformer的训练和评估。

### SKDCN.py

设计的网络

### skdcn_train.py

训练SKDCN并评估。