import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

file_path = './dataset/ManualTweetClassification.xlsx'
data = pd.read_excel(file_path, skiprows=1)

model_columns = ['bert', 'bertweet', 'distilbert', 'roberta', 'vader']
true_column = 'classification'

label_encoder = LabelEncoder()
data[true_column] = label_encoder.fit_transform(data[true_column])
for model in model_columns:
    data[model] = label_encoder.transform(data[model])

metrics = {model: {} for model in model_columns}

for model in model_columns:
    metrics[model]['precision'] = precision_score(data[true_column], data[model], average='weighted')
    metrics[model]['recall'] = recall_score(data[true_column], data[model], average='weighted')
    metrics[model]['f1_score'] = f1_score(data[true_column], data[model], average='weighted')
    metrics[model]['accuracy'] = accuracy_score(data[true_column], data[model])

metrics_df = pd.DataFrame(metrics).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.show()