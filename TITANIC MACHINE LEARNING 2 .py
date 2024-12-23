#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("C:/Users/PC/MACHINE LEARNING/DATASETS/traintitanic.csv")
df_test = pd.read_csv("C:/Users/PC/MACHINE LEARNING/DATASETS/testtitanic.csv")



# In[4]:


df_train["SibSp"].unique()


# In[5]:


df_train["Parch"].unique()


# In[10]:


missing = df_train["Age"].isnull().sum()


# In[25]:


total = df_train.shape[0]


# In[27]:


#Porcentaje de valores faltantes en la columna SibSp
pmissing = (missing/total)*100

print(pmissing)


# In[31]:


#Porcentaje de valores faltantes en la columna Cabin
missing2 = df_train["Cabin"].isnull().sum()
total2 = df_train.shape[0]
pmissing2 = (missing2/total2)*100

print(pmissing2)


# In[47]:


df_train.head()


# In[44]:


df_test.head()


# In[46]:


df_test.isnull().sum()


# In[61]:


#Dicotomizacion de valores en data train y data test

df_train["Faltantes_Age"] = df_train["Age"].isnull().astype(int)
df_train.head(10)


# In[62]:


df_test["Faltantes_Age"] = df_train["Age"].isnull().astype(int)
df_test.head(10)


# In[64]:


#Eliminar la columna Cabin, dado que tiene más del 50% de datos faltantes, para train and test.
df_train = df_train.drop("Cabin", axis = 1)


# In[65]:


df_test = df_test.drop("Cabin", axis = 1)


# In[61]:


df_train.head(10)


# In[62]:


df_test.head(10)


# In[67]:


#computando los valores faltantes en Age para train y test
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())


# In[68]:


df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())


# In[70]:


sns.boxplot(df_train["Age"])


# In[114]:


outliers = df_train.loc[df_train["Age"]>50]


# In[81]:


outliers["PassengerId"].unique()


# In[82]:


normal = df_train.loc[df_train["Age"]<=50]


# In[83]:


normal["PassengerId"].nunique()


# In[86]:


# Assuming df_train and df_test are your two DataFrames
datasets = {'Train Dataset': df_train, 'Test Dataset': df_test}

for name, df in datasets.items():
    plt.figure(figsize=(8, 6))
    sns.boxplot(df["Fare"])
    plt.title(f"Boxplot of Fare - {name}")
    plt.show()


# In[70]:


# Assuming df_train and df_test are your two DataFrames
datasets = {'Train Dataset': df_train, 'Test Dataset': df_test}

for name, df in datasets.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(x = df.index, y = df["Fare"])
    plt.title(f"Boxplot of Fare - {name}")
    plt.show()


# In[73]:


#Eliminando outliers en dataset 
df_train = df_train[df_train["Fare"]<500]


# In[45]:


df_test = df_test[df_test["Fare"]<500]


# In[71]:


df_train = df_train.drop("Ticket", axis = 1)


# In[72]:


df_test = df_test.drop("Ticket", axis =1)


# In[48]:


numerical_values = df_train.select_dtypes(include = ['number']).drop(columns= ["PassengerId"])
correlation_matrix = numerical_values.corr()
plt.figure(figsize=(10,8))

sns.heatmap(correlation_matrix,annot= True, cmap= "coolwarm", fmt = ".2f")
plt.title("Matriz de correlacion")
plt.show()


# In[14]:


import statsmodels.formula.api as smf

# Drop PassengerId and keep only numerical columns
numerical_data = df_train.select_dtypes(include=['number']).drop(columns=["PassengerId"])

# Create a formula with Survived as the target and the rest as predictors
predictors = ' + '.join([col for col in numerical_data.columns if col != "Survived"])
formula = f"Survived ~ {predictors}"

# Fit the linear regression model
model = smf.ols(formula, data=df_train).fit()

# Display the summary of the model
print(model.summary())


# In[74]:


df_train["Embarked"] = df_train["Embarked"].fillna(df_train["Embarked"].mode().iloc[0])



# In[16]:


df_train.isnull().sum()


# In[75]:


df_train = df_train.drop("Name", axis = 1)


# In[25]:


# Drop PassengerId and keep only numerical columns
df_encoded = pd.get_dummies(df_train, columns = ["Sex"], drop_first = True)
df_encoded = df_encoded.drop(columns = ["PassengerId"])

# Create a formula with Survived as the target and the rest as predictors
predictors = ' + '.join([col for col in df_encoded.columns if col != "Survived"])
formula = f"Survived ~ {predictors}"

# Fit the linear regression model
model = smf.ols(formula, data=df_encoded).fit()

# Display the summary of the model
print(model.summary())


# In[76]:


#Eliminando columnas que no aportan al modelo
df_train = df_train.drop("Embarked",axis = 1)


# In[77]:


#Eliminando columnas que no aportan al modelo
df_test = df_test.drop("Embarked",axis= 1)


# In[78]:


df_test = df_test.drop("Name",axis = 1)


# In[55]:


df_test


# In[79]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preprocessing Train Data
X_train_full = df_train.drop(columns=['PassengerId', 'Survived'])  # Drop irrelevant columns
X_train_full = pd.get_dummies(X_train_full, columns=['Sex'], drop_first=True)  # Encode 'Sex'
y_train_full = df_train['Survived']

# Preprocess Test Data
X_test = df_test.drop(columns=['PassengerId'])
X_test = pd.get_dummies(X_test, columns=['Sex'], drop_first=True)
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

# Split train data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 2. Train the XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 3. Evaluate on Validation Set
y_valid_pred = model.predict(X_valid)
y_valid_pred_proba = model.predict_proba(X_valid)[:, 1]

# Metrics
accuracy = accuracy_score(y_valid, y_valid_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
r2 = r2_score(y_valid, y_valid_pred)
conf_matrix = confusion_matrix(y_valid, y_valid_pred)

print(f"Validation Accuracy: {accuracy:.2f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}\n")

print("Classification Report (Validation):")
print(classification_report(y_valid, y_valid_pred))

# 4. Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# 5. Predict on Test Dataset
y_test_pred = model.predict(X_test)

# Save Test Predictions
submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_test_pred
})
output_file_path = 'E:\Python\practicas kaggel kernel\TITANIC MACHINE LEARNING\submission.csv'  # Replace with your desired output path
submission.to_csv(output_file_path, index=False)

# 6. Feature Importance
plt.figure(figsize=(8, 6))
plt.bar(X_train.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance in XGBoost")
plt.show()


# In[69]:


df_test.shape


# In[ ]:




