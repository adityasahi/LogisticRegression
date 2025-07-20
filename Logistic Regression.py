
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix



df = pd.read_csv('dataset.csv') # loading the dataset
# Data Cleaning 
df = df.drop_duplicates() # removes any duplicates in rows

df = df.replace(0, pd.NA) # replacing zeroes with NaN

df = df.dropna(axis=0, how='all')  # Drop rows that contain all NaN values
df = df.dropna(axis=1, how='all')  # Drop columns containing all NaN values

df_num = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')  # converts all the columns to numeric


number_counts = df_num.count() # counting all non-zero entries 

# print(number_counts)
# statistics 
total_mutations = df_num.count(axis=1)
avg_mutations = total_mutations.mean()
med_mutations = total_mutations.median()
var_mutations = total_mutations.var()

'''
print(total_mutations)
print(avg_mutations)
print(med_mutations)
print(var_mutations)
'''
#Feature Scaling
scaler = MinMaxScaler() # normalizing total mutations
total_mut_normalized = scaler.fit_transform(total_mutations.values.reshape(-1, 1))

df_normalized_count = pd.DataFrame(total_mut_normalized, columns=["Normalized Mutation Count"]) # data frame for normalized count

# df_normalized_count.head(100)
# Feature Selection
selection = VarianceThreshold(threshold=0.003) # variance threshold for  feature selection 
df_new = selection.fit_transform(df_num)
df_new = pd.DataFrame(df_new, columns=df_num.columns[selection.get_support()])


X = df_num.iloc[:, :-1]
y = df_num.iloc[:, -1]
imputer = SimpleImputer(strategy='median') ## handling any missing values
X_imputed = imputer.fit_transform(X)

X_scaled = scaler.fit_transform(X_imputed)

X_selected = selection.fit_transform(X_scaled)
# normalize labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predictions
y_predict = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
print("\nAccuracy:", accuracy_score(y_test, y_predict))

# Confusion Matrix
print("\nConfusion Matrix:\n", cm)

# Precision and recall
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
print("\nPrecision:", precision)
print("\nRecall:", recall)


plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


