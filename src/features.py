import numpy as np
import pandas as pd 

df = pd.read_csv('Data\\Anonymize_Loan_Default_data.csv', encoding= 'ISO-8859-1', index_col=0)

columns_to_drop = ['member_id', 'id', 'issue_d', 'zip_code', 'addr_state', 'earliest_cr_line', 'mths_since_last_delinq', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']

df.drop(columns=columns_to_drop, inplace=True)

df.dropna(subset=['loan_amnt'], inplace=True)
df = df[df['loan_amnt'] != 0]

emp_length_mapping = {
    '< 1 year': 1,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10,
}

df['emp_length_numeric'] = df['emp_length'].map(emp_length_mapping)

df.loc[df['emp_length_numeric'].isnull(), 'emp_length_numeric'] = 0

term_mapping = {
    '36 months': 36,
    '60 months': 60
}

df['term_numeric'] = df['term'].map(term_mapping)

df['is_verified'] = np.where(df['verification_status'] == 'Not Verified', 0, 1)

df['debt_consol'] = np.where(df['purpose'] == 'debt_consolidation', 1, 0)
df['productive_prps'] = np.where(df['purpose'] == 'small_business', 1, 0)

df['pymnt_progress'] = df['total_pymnt_inv'] / df['total_pymnt']
df[['pymnt_progress', 'total_pymnt', 'total_pymnt_inv']].head()
df.loc[df['pymnt_progress'].isnull(), 'pymnt_progress'] = 0

columns_to_drop2 = ['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'revol_util']

# Drop the specified columns
df.drop(columns=columns_to_drop2, inplace=True)

df = df.loc[~df.isnull().any(axis=1)]

df.info()
df.to_csv("final_data.csv", index=False)