import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as mno
import itertools
import seaborn as sns

# Import read file
# df = pd.read_csv("a1.csv")
# df = pd.read_csv("c1.csv")
# df = pd.read_csv("converted_concrete.csv")

# print("Simple print")
# print(df) # Simple print

# # Summary basics
# print("Head", df.head)
# print("Shape ", df.shape)
# print("DType ", df.dtypes)
# print("Info ", df.info())
# print("Columns ", df.columns)

# 4.1. Duplicates
# print("Duplicated Sum ", df.duplicated().sum)
# duplicates = df.duplicated()
# print("Duplicates ")
# print(df[duplicates])
# print("Delete dups ", df.drop_duplicates(inplace=True))

# 4.2. Drop Duplicates
# # Delete duplicate rows
# df.drop_duplicates(inplace=True)
# # Get the shape of Concrete data
# df.shape

# 4.3. Outlier
# Plot
# df.boxplot(column = ['AT', 'V', 'AP', 'RH', 'PE'], rot=45, figsize = (5,5))
# plt.show()

# 4.4. Working with Outliers: Correcting, Removing
# df_outliers = pd.DataFrame(df.loc[:,])
# # Calculate IQR
# Q1 = df_outliers.quantile(0.25)
# Q3 = df_outliers.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
# # We can use IQR score to filter out the outliers by keeping only valid values
# # Replace every outlier on the upper side by the upper whisker 
# for i, j in zip(np.where(df_outliers > Q3 + 1.5 * IQR)[0], np.where(df_outliers > Q3 + 1.5 * IQR)[1]):
#     whisker = Q3 + 1.5 * IQR
#     df_outliers.iloc[i, j] = whisker[j]
# # Replace every outlier on the lower side by the lower whisker 
# for i, j in zip(np.where(df_outliers < Q1 - 1.5 * IQR)[0], np.where(df_outliers < Q1 - 1.5 * IQR)[1]):
#     whisker = Q1 - 1.5 * IQR
#     df_outliers.iloc[i, j] = whisker[j]
# # Remove outliers columns
# df.drop(columns = df.loc[:,], inplace = True)
# df = pd.concat([df, df_outliers], axis = 1)

# 4.5. Check Outliers after correction
# df.boxplot(column = ['AT', 'V', 'AP', 'RH', 'PE'], rot=45, figsize = (5,5))
# plt.show()

# 4.6. Missing values
# df.isnull().sum()
# # Check the presence of missing values
# df_missval = df.copy()   # Make a copy of the dataframe
# isduplicates = False

# for x in df_missval.columns:
#     df_missval[x] = df_missval[x].astype(str).str.replace(".", "")
#     result = df_missval[x].astype(str).str.isalnum() # Check whether all characters are alphanumeric
#     if False in result.unique():
#         isduplicates = True
#         print('For column "{}" unique values are {}'.format(x, df_missval[x].unique()))
#         print('\n')
# if not isduplicates:
#     print('No duplicates in this dataset')
# # Visualize missing values
# mno.matrix(df, figsize = (5, 5))
# plt.show()
# # Summary statistics
# print("Desc ", df.describe().T)

# 5.1. Variable Identification
# 5.2. Univariate Analysis (for c1.csv)
# cols = [i for i in df.columns if i not in 'strength']
# length = len(cols)
# cs = ["b","r","g","c","m","k","lime","c"]
# fig = plt.figure(figsize=(7,7))
# for i,j,k in itertools.zip_longest(cols,range(length),cs):
#     plt.subplot(4,2,j+1)
#     ax = sns.distplot(df[i],color=k,rug=True)
#     ax.set_facecolor("w")
#     plt.axvline(df[i].mean(),linestyle="dashed",label="mean",color="k")
#     plt.legend(loc="best")
#     plt.title(i,color="navy")
#     plt.xlabel("")
# plt.show()
# # Strength column seems to be uniformly distributed
# plt.figure(figsize=(13,6))
# sns.distplot(df["strength"],color="b",rug=True)
# plt.axvline(df["strength"].mean(), linestyle="dashed",color="k", label='mean',linewidth=2)
# plt.legend(loc="best",prop={"size":14})
# plt.title("Concrete compressivee strength distribution")
# plt.show()

# 5.3. Study Summary Statistics
# # Summary statistics
# print("Desc ", df.describe().T)

# 5.4 Multivariate Analysis
# sns.pairplot(df, diag_kind = 'kde', corner = True)

# 5.5 Study Correlation
# # Check the Correlation
# print("Correlation ", df.corr())
# Pairplot for checking the Correlation
# sns.pairplot(df[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
#        'fineagg', 'age', 'strength']], kind = 'reg', corner = True, figsize = (5,5))
# plt.show()
# # Heatmap for checking the Correlation¶
# corr = abs(df.corr()) # correlation matrix
# lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
# mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

# plt.figure(figsize = (12,10))
# sns.heatmap(lower_triangle, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
#             cbar= True, linewidths= 1, mask = mask)   # Da Heatmap
# plt.show()

# STUDIO 2 ACT 1
# Convert numerical strength to categorical values
def categorize_strength(value):
    if value < 20:
        return 1
    elif 20 <= value < 30:
        return 2
    elif 30 <= value < 40:
        return 3
    elif 40 <= value < 50:
        return 4
    else:
        return 5

# Apply the function to the 'strength' column
df['strength'] = df['strength'].apply(categorize_strength)
# Save the converted DataFrame to a new CSV file
df.to_csv("converted_concrete.csv", index=False)

# Plotting the distribution
df['strength'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Strength Category')
plt.ylabel('Frequency')
plt.title('Distribution of Strength Categories')
plt.show()