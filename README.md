# SMFIF

SMFIF 是一个集成多个机器学习模型的堆叠分类器。支持训练、保存、加载、预测和指标评估。

## 使用方法

```python
from smfif import SMFIFModel, evaluate_model
model = SMFIFModel()
model.fit(train_df)
preds = model.predict(test_df)
```

也可以保存和加载模型：

```python
model.save("smfif_model.pkl")
model.load("smfif_model.pkl")
```