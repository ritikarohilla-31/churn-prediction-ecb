import pandas as pd
import numpy as np

df = pd.read_csv("European_Bank.csv")
model_df = df.drop(columns=['CustomerId', 'Surname'])
if model_df['Year'].nunique() <= 1:
    model_df = model_df.drop(columns=['Year'])

model_df['BalanceToSalaryRatio'] = model_df['Balance'] / model_df['EstimatedSalary'].replace(0, 1)
model_df['ProductDensity'] = model_df['NumOfProducts'] / model_df['Tenure'].replace(0, 1)
model_df['EngagementProductInteraction'] = model_df['IsActiveMember'] * model_df['NumOfProducts']
model_df['AgeTenureInteraction'] = model_df['Age'] * model_df['Tenure']

model_df = pd.get_dummies(model_df, columns=['Geography', 'Gender'], drop_first=True)
model_df.to_csv("model_ready_data.csv", index=False)
print("done", model_df.shape)
