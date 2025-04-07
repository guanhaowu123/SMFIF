# SMFIF

A Multi-criterion Feature Integration Framework for Accurate Diagnosis of Primary Sjögren's Syndrome Using Laboratory Tests: A Multicenter Retrospective Cohort Study in China

##Usage method

```python
from smfif import SMFIFModel, evaluate_model
model = SMFIFModel()
model.fit(train_df)
preds = model.predict(test_df)
```

You can also save and load models:

```python
model.save("smfif_model.pkl")
model.load("smfif_model.pkl")
```
