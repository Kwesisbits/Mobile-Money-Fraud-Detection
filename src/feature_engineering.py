"""
These engineered features help undertand strange occurances in transcation that could indicate fraudulence
"""

#Feature Engineering
df['diff_orig_balance'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount'] #Fake deductions or reversals
df['diff_dest_balance'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount'] #Ghost or rerouted deposits
df['balance_change_ratio'] = (df['newbalanceDest'] - df['oldbalanceDest']) / (df['amount'] + 1) #Disproportionate updates or laundering

#this visualization helps to understand how essential these added features are based on their variation accross the 2 classes
import matplotlib.pyplot as plt
features = ['diff_orig_balance', 'diff_dest_balance', 'balance_change_ratio']

for col in features:
    plt.figure(figsize=(7,4))
    sns.kdeplot(data=df, x=col, hue='isFraud', common_norm=False, fill=True)
    plt.title(f'Distribution of {col} by Fraud Class')
    plt.show()
