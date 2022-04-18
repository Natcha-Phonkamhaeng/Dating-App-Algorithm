import pandas as pd
import numpy as np
import random

# Algorithm for Dating App --> use recommendation system --> Netflix movie recommend

# 1. Generate Dataset for both Male & Female
# 	  - each dataset will have 5 question each question will have 5 answer (a,b,c,d,e)
#     - Matching will be (Y, N, Unseen)

# 2. Find 1 Male or Female that have most unseen profile --> this user is perfectly for our app to recommend the profile
#     - we will use this user as example data to find other similar user

# 3. Find similar user that do the question the same --> we will use correlation to find similar user and use top 10 most similar

# 4. Find the women or men that our similar user like the most then recommend to our example user


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df_m = pd.DataFrame()
df_w = pd.DataFrame()

num = 30

# creating ID columns for both men and women
m_id = []
for i in range(num):
	i = 'm'+str(i)
	m_id.append(i)
df_m['ID'] = m_id
# for women creating by using list comprehension
df_w['ID'] = ['w'+str(i) for i in range(num)]


# creating question and answer for each men and women
qs = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
ans = ['A', 'B', 'C', 'D', 'E']

for q in qs:
	df_m[q] = pd.Categorical(random.choices(ans, k=num), categories=ans)
	df_w[q] = pd.Categorical(random.choices(ans, k=num), categories=ans)

df_m.set_index('ID', inplace=True)
df_w.set_index('ID', inplace=True)
 

# generate matching data for men and women
df_rating = pd.DataFrame(index=df_m.index, columns=df_w.index)
# 0 represents = No
# 1 represents = Yes
# Unseen will be replaced with np.nan
match = [0, 1, 'Unseen']

for i in df_rating.columns:
	df_rating[i] = random.choices(match, k=num)

# sort df_rating male user by Unseen. Rank male user by top unseen womem user
df_m_user = df_rating.apply(pd.value_counts, axis=1).sort_values(by=['Unseen'], ascending=False)

# Top male user with unseen women --> poor guy (>-<)
df_m_user = df_m_user.iloc[0]

# find out what women that he is not yet seen
# using T to transform what used to be index will be column, what used to be column will be index
filt_unseen = df_rating.T[df_m_user.name]=='Unseen'
df_w_not_seen = df_rating.T[filt_unseen].index

# below code is map letter to numerical value but dataframe will be category which we cannot use to find correlation
df_m_numeri = df_m.apply(lambda x: x.map({
	'A':1,
	'B':2,
	'C':3,
	'D':4,
	'E':5
}))

# instead we will use cat.codes to convert letter to numerical value and dataframe will be int so we can find correlation
df_m_num = df_m.apply(lambda x: x.cat.codes)

# find similar user that did the question the same as our df_m_user --> using correlation --> Pearson Correlation
filt_m = df_m_num.T[df_m_user.name]
df_m_cor = df_m_num.T.corrwith(filt_m).sort_values(ascending=False)[1:6]

# creating new dataframe that our similar user interact with women profile
df_cor_rating = df_rating.loc[list(df_m_cor.index)][df_w_not_seen]
# replace Unseen with np.nan
df_cor_rating.replace('Unseen', np.nan, inplace=True)
print(df_cor_rating)
print('---------------------')

# aggregate value 1.Mean 2.Median
df_aggregate = pd.DataFrame()
# 1.Mean Value will calculate between 0 (not match) and 1 (match), np.nan will not include in calculation
df_aggregate['Mean'] = df_cor_rating.mean()
# 2.Median Value will calculate between 0 (not match) and 1 (match), np.nan will not include in calculation
df_aggregate['Median'] = df_cor_rating.median()
print(df_aggregate)







