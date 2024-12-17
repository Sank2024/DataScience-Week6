import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#file_path = "/Users/skattel/Downloads/employee.csv"

# Load test data
test_data = pd.read_csv("employee.csv")

# Dropping unnecessary columns
test_data = test_data.drop(columns=['id', 'timestamp', 'country'])

# Check the first few rows
# print(test_data.head())


# Verify the changes
print(test_data.head())



# Identify numeric columns
test_data.loc[test_data['hours_per_week'].isna(), 'hours_per_week'] = test_data['hours_per_week'].median()
test_data.loc[test_data['telecommute_days_per_week'].isna(), 'telecommute_days_per_week'] =test_data['telecommute_days_per_week'].median()


data = test_data.dropna()
data.info()


data_train = data.copy()
data_train.head()

cat_cols = [c for c in data_train.columns if data_train[c].dtype == 'object'
            and c not in ['is_manager', 'certifications']]
cat_data = data_train[cat_cols]
cat_cols


binary_cols = ['is_manager', 'certifications']
for c in binary_cols:
    data_train[c] = data_train[c].replace(to_replace=['Yes'], value=1)
    data_train[c] = data_train[c].replace(to_replace=['No'], value=0)

final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True,dtype=int)
final_data.shape


final_data

y = final_data['salary']
X = final_data.drop(columns=['salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Training Set Dimensions:", X_train.shape)
print("Validation Set Dimensions:", X_test.shape)


final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True,dtype=int)
final_data.shape


final_data.columns


num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
num_cols


# Apply standard scaling on numeric data
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])


X_train

#Fitting a Linear Regression Model
reg=LinearRegression()
reg.fit(X_train, y_train)

reg.coef_

reg.intercept_


mean_squared_error(y_train,reg.predict(X_train))/np.mean(y_train)

y_pred = reg.predict(X_test)


mse = mean_squared_error(y_pred, y_test)/np.mean(y_test)
print("Mean Squared Error:", mse)


# Create a DataFrame for predictions
predictions = pd.DataFrame({'Predicted_Salary': y_pred})

# Save to CSV
predictions.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
