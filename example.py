import pandas as pd
from smfif import SMFIFModel, evaluate_model
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv("model_train_data.csv")
model = SMFIFModel()

# select data
X = data[model.selected_features]
y = data['lable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# merge data
train_df = X_train.copy()
train_df['lable'] = y_train
test_df = X_test.copy()
test_df['lable'] = y_test

# train and evaluate model
model.fit(train_df)
y_pred = model.predict(test_df)
y_proba = model.predict_proba(test_df)[:, 1]
metrics = evaluate_model(test_df['lable'], y_pred, y_proba)
print("测试集指标：", metrics)

# save model
model.save("smfif_model.pkl")