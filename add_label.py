import pandas as pd

file_name = 'veinData'

df = pd.read_csv(f'matlab_raman_preprocessed/{file_name}.csv', header=None)
df.columns += 800

has_DM2 = pd.read_csv('original_raman_spectroscopy/earLobe.csv', header=None, skiprows=1)
has_DM2 = has_DM2[1][1:].reset_index(drop=True)
df['has_DM2'] = has_DM2
print(df)
print(has_DM2)

df.to_csv(f'matlab_raman_preprocessed copy/{file_name}.csv')