#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'loan_dataset.xlsx'
# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)


# In[82]:


# Display the first few rows of the dataset
print(data.head())


# In[83]:


# Get an overview of the data types and missing values
print(data.info())


# In[84]:


#droping the columns as it seems not that important for our analysis
data=data.drop("Pin-code",axis=1)
data=data.drop("Fam members",axis=1)
data=data.drop("Serial",axis=1)


# In[85]:


# Summary statistics
print(data.describe())


# In[86]:


#Shows the no. of null values in dataset according to columns.
data.isnull().sum()


# In[87]:


# Creating a pie chart to visualize the mean mortgage amount for each education level
plt.figure(figsize=(4, 4))
plt.pie(grp, labels=grp.index, autopct='%1.1f%%', startangle=140)
plt.title('Mean Mortgage Amount by Education Level')
plt.show()


# In[88]:


import seaborn as sns
import matplotlib.pyplot as plt

# Creating a count plot for 'Net Banking'
plt.figure(figsize=(6, 4))
sns.countplot(x='Net Banking', data=data, palette='Set2')
plt.title('Number of People Using Net Banking')
plt.show()

# Creating a count plot for 'Loan'
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan', data=data, palette='Set1')
plt.title('Number of People with Existing Loans')
plt.show()


# In[89]:


#Define the age ranges
age_bins = [20, 30, 40, 50, 60, 100]  # Define the age bins
age_labels = ['20-29', '30-39', '40-49', '50-59', '60+']  # Define the labels for the age ranges

# Creating a new column 'Age Range' based on the age bins
data['Age Range'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

# Calculating the mean income for each age range
income_by_age_range = data.groupby('Age Range')['Income'].mean().reset_index()

# Print the mean income for each age range
print(income_by_age_range)


# In[90]:


# Creating a bar plot to visualize the mean income for each age range
plt.figure(figsize=(10, 6))
sns.barplot(x='Age Range', y='Income', data=income_by_age_range, palette='viridis')
plt.title('Mean Income by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Mean Income')
plt.show()


# In[91]:


# Defining the age bins
age_bins = [20, 30, 40, 50, 60, 70]  # Define the age bins
age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69']  # Define the labels for the age ranges

# Creating a new column 'Age Group' based on the age bins
data['Age Group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

# Creating separate count plots for net banking and loans based on age groups
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Age Group', hue='Net Banking', data=data, palette='Set2')
plt.title('Net Banking Usage by Age Group')

plt.subplot(1, 2, 2)
sns.countplot(x='Age Group', hue='Loan', data=data, palette='Set1')
plt.title('Loan Acceptance by Age Group')

plt.tight_layout()
plt.show()


# In[92]:


# Defining the income bins
income_bins = [0, 30000, 50000, 70000, 100000, float('inf')]  # Define your own income bins as per your dataset

# Creating a new column 'income_bin' based on the income bins
data['income_bins'] = pd.cut(data['Income'], bins=income_bins)

# Calculating the number of people having a loan or not within each income bin
loan_status_by_income = data.groupby(['income_bins', 'Loan']).size().unstack(fill_value=0)

print(loan_status_by_income)


# In[ ]:





# In[94]:


# Defining the income bins
income_bins = [60000, 400000, 800000, 1200000, 1500000 ,float('inf')]  # Define your own income bins as per your dataset

# Creating a new column 'income_bin' based on the income bins
data['income_bins'] = pd.cut(data['Income'], bins=income_bins)

# Calculating the number of people having a loan or not within each income bin
loan_status_by_income = data.groupby(['income_bins', 'Loan']).size().unstack(fill_value=0)

# Reseting the index for better visualization
loan_status_by_income = loan_status_by_income.reset_index()

# Plotting the data using seaborn and matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x='income_bins', y='yes', data=loan_status_by_income, color='skyblue', label='Has Loan')
sns.barplot(x='income_bins', y='no', data=loan_status_by_income, color='orange', label='No Loan')
plt.xlabel('Income Bin')
plt.ylabel('Count')
plt.title('Loan Status by Income Bin')
plt.legend()
plt.show()


# In[95]:


# Defining the age bins
age_bins = [20, 30, 40, 50, 60, 70, 80, 90]

# Creating a new column 'age' based on the age bins
data['age_bins'] = pd.cut(data['age'], bins=age_bins)

# Selecting only the columns 'age', 'fixed_deposit', and 'demat'
selected_data = data[['age_bins', 'Fixed Deposit', 'Demat']]

# Filtering the data to include only 'yes' or 'no' values for 'fixed_deposit' and 'demat'
filtered_data = selected_data[(selected_data['Fixed Deposit'].isin(['yes', 'no'])) & (selected_data['Demat'].isin(['yes', 'no']))]

# Plotting the data using seaborn and matplotlib
plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_data, x='age_bins', hue='Fixed Deposit')
plt.title('Count of Fixed Deposits by Age Bin')
plt.xlabel('Age Bin')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_data, x='age_bins', hue='Demat')
plt.title('Count of Demat Accounts by Age Bin')
plt.xlabel('Age Bin')
plt.ylabel('Count')
plt.show()


# In[114]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'loan_dataset26.xlsx'
# Read the Excel file into a DataFrame
data1 = pd.read_excel(file_path)


# In[115]:


#droping the columns as it seems not that important for our analysis
data1=data1.drop("Pin-code",axis=1)
data1=data1.drop("Fam members",axis=1)
data1=data1.drop("Serial",axis=1)
data1=data1.drop("Education",axis=1)
data1=data1.drop("Fixed Deposit",axis=1)
data1=data1.drop("Demat",axis=1)
data1=data1.drop("Net Banking",axis=1)
data1=data1.drop("ID",axis=1)


# In[116]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Spliting the dataset into features (X) and target variable (y)
X = data1.drop('Loan', axis=1)
y = data1['Loan']

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Development
# Choosing a classifier and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
# Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[117]:


# Spliting the data into features and target variable
X = data1.drop('Loan', axis=1)  # Features
y = data1['Loan']  # Target variable

from sklearn.model_selection import train_test_split

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier  # Example model

# Creating and training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[119]:


from sklearn.ensemble import RandomForestClassifier

# Example: Train a random forest classifier to assess feature importance
model = RandomForestClassifier()
model.fit(data1.drop('Loan', axis=1), data1['Loan'])

# Assess feature importance
feature_importance = model.feature_importances_
print(feature_importance)

from sklearn.feature_selection import mutual_info_classif

# Example: Calculating mutual information between each feature and the target variable 'Loan'
feature_importance = mutual_info_classif(data1.drop('Loan', axis=1), data1['Loan'])
print(feature_importance)


# In[121]:


data1=data1.drop("Loan",axis=1)


# In[122]:


plt.figure(figsize=(15,8))
sns.heatmap(data1.corr(),annot=True).set_title('Correlation Heatmap') #square=True
plt.show()


# In[ ]:




