import pandas as pd

df = pd.read_csv('Boning.csv')
fe = pd.read_csv('Slicing.csv')

df['Class'] = 0
fe['Class'] = 1

df.to_csv('Boning_updated.csv', index=False)
fe.to_csv('Slicing_updated.csv', index=False)


BN = pd.read_csv('Boning_updated.csv')
FE = pd.read_csv('Slicing_updated.csv')

combined_data = pd.concat([BN, FE], ignore_index=True)

combined_data.to_csv('all_data.csv', index=False)

