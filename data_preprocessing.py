######################################################
##                                                  ##
##           TEAM 10 - DATA PREPROCESSING           ##
##                                                  ##
######################################################

import pandas as pd
from datetime import datetime
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# first data cleaning and transforming the ebb_set1
df1 = pd.read_csv("/data/team10/data/ebb_set1.csv")
extra1 = pd.read_csv('/data/team10/data/activations_ebb_set1.csv')

# making 1 hot encoding for the feature manufacturer
manu_encoder = OneHotEncoder()
manu_1hot_df = pd.DataFrame(manu_encoder.fit_transform(df1[['manufacturer']]).toarray())
manu_1hot_df.columns = manu_encoder.get_feature_names_out(['manufacturer'])

df1 = df1.join(manu_1hot_df)
df1.drop('manufacturer', axis=1, inplace=True)

# making 1 hot encoding for the feature operating_system
os_encoder = OneHotEncoder()
df1[['operating_system']] = df1[['operating_system']].replace(r'^Not', np.NaN, regex=True)
df1[['operating_system']] = df1[['operating_system']].replace('NONE', np.NaN, regex=True)
os_1hot_df = pd.DataFrame(os_encoder.fit_transform(df1[['operating_system']]).toarray())
os_1hot_df.columns = os_encoder.get_feature_names_out(['operating_system'])

df1 = df1.join(os_1hot_df)
df1.drop('operating_system', axis=1, inplace=True)

# making 1 hot encoding for the feature state
state_encoder = OneHotEncoder()
state_1hot_df = pd.DataFrame(state_encoder.fit_transform(df1[['state']]).toarray())
state_1hot_df.columns = state_encoder.get_feature_names_out(['state'])
state_column_names = []

df1 = df1.join(state_1hot_df)
df1.drop('state', axis=1, inplace=True)

# transforming date from string into timestamp (numerical data)
for index, row in df1.iterrows():
    # last_redemption_date
    if isinstance(row['last_redemption_date'], str):
        datetime_object = datetime.strptime(row['last_redemption_date'], '%Y-%m-%d')
        df1.loc[index,'last_redemption_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'last_redemption_date'] = math.nan
    
    # first_activation_date
    if isinstance(row['first_activation_date'], str):
        datetime_object = datetime.strptime(row['first_activation_date'], '%Y-%m-%d')
        df1.loc[index,'first_activation_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'first_activation_date'] = math.nan

# activations, data cleaning and transforming
duplicate_counter = 1
new_customer_id = []
number_of_activations = []
previous_customer_id = ''
# calculate the number of activations each customer have
for index, row in extra1.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_activations.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']
    
number_of_activations.append(duplicate_counter)
number_of_activations = number_of_activations[1:]
new_customer_id = extra1['customer_id'].unique()
extra1 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_activations': number_of_activations})
god_set = pd.merge(df1, extra1, on = 'customer_id', how = 'left').fillna(0)

# auto_refill, data cleaning and transforming
extra2 = pd.read_csv('/data/team10/data/auto_refill_ebb_set1.csv')
prev_id = ''
drops = []
first_counter = 0
dup_to_no = False
for index, row in extra2.iterrows():
    if row['customer_id'] == prev_id and first_counter != 0:
        drops.append(index)
        first_counter += 1
    if row['customer_id'] == prev_id and first_counter == 0:
        drops.append(index)
        first_counter += 1
        dup_to_no = True
    if row['customer_id'] != prev_id:
        first_counter = 0
        if dup_to_no == True:
            drops.append(index)
            dup_to_no = False
    prev_id = row['customer_id']

extra2 = extra2.drop(extra2.index[drops])

new_enroll_date = []
new_de_enroll_date = []
# transforming date from string into timestamp (numerical data)
for index, row in extra2.iterrows():
    if isinstance(row['auto_refill_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_enroll_date'], '%Y-%m-%d')
        new_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_enroll_date.append(math.nan)
    
    if isinstance(row['auto_refill_de_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_de_enroll_date'], '%Y-%m-%d')
        new_de_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_de_enroll_date.append(math.nan)
   
new_customer_id = extra2['customer_id'].unique()
extra2 = pd.DataFrame({'customer_id': new_customer_id, 'auto_refill_enroll_date': new_enroll_date, 'auto_refill_de_enroll_date': new_de_enroll_date})
god_set = pd.merge(god_set, extra2, on='customer_id', how='left').fillna(math.nan)

# throttling, data cleaning and transforming
extra14 = pd.read_csv('/data/team10/data/throttling_ebb_set1.csv')
duplicate_counter = 1
new_customer_id = []
number_of_throttling = []
previous_customer_id = ''
# creating a new feature "num_of_throttling"
for index, row in extra14.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_throttling.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']
    
number_of_throttling.append(duplicate_counter)
number_of_throttling = number_of_throttling[1:]
new_customer_id = extra14['customer_id'].unique()
extra14 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_throttling': number_of_throttling})
god_set = pd.merge(god_set, extra14, on = 'customer_id', how = 'left').fillna(0)

# support, data cleaning and transforming
from sklearn.preprocessing import MultiLabelBinarizer

extra12 = pd.read_csv('/data/team10/data/support_ebb_set1.csv')
duplicate_counter = 1
number_of_support = []
previous_customer_id = ''
support_category = []
support_customer = []
# creating new feature "num_of_support"
for index, row in extra12.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_support.append(duplicate_counter)
        duplicate_counter = 1
        support_category.append(support_customer)
        support_customer = []
        support_customer.append(row['case_type'])
    else:
        duplicate_counter = duplicate_counter + 1
        support_customer.append(row['case_type'])
    previous_customer_id = row['customer_id']

number_of_support.append(duplicate_counter)
number_of_support = number_of_support[1:]

support_category.append(support_customer)
support_category = support_category[1:]

new_customer_id = extra12['customer_id'].unique()
extra12 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_support': number_of_support})

# using multi hot encoding to turn support category into numerical value
sup_encoder = MultiLabelBinarizer()
sup_1hot_df = pd.DataFrame(sup_encoder.fit_transform(support_category))
sup_1hot_df.columns = list(sup_encoder.classes_)
extra12 = extra12.join(sup_1hot_df)

god_set = pd.merge(god_set, extra12, on = 'customer_id', how = 'left').fillna(0)

# loyalty_program, data cleaning and transforming
extra6 = pd.read_csv('/data/team10/data/loyalty_program_ebb_set1.csv')

new_date = []
# transforming date from string into timestamp (numerical data)
for index, row in extra6.iterrows():
    if isinstance(row['date'], str):
        datetime_object = datetime.strptime(row['date'], '%Y-%m-%d')
        new_date.append(int(round(datetime_object.timestamp())))
    else:
        new_date.append(math.nan)
# making 1 hot encoding for the feature lrp_enrolled
lrp_encoder = OneHotEncoder()
lrp_1hot_df = pd.DataFrame(lrp_encoder.fit_transform(extra6[['lrp_enrolled']]).toarray())
lrp_column_names = []
for i in range(len(lrp_1hot_df.columns)):
    lrp_column_names.append(f'lrp{i}')
lrp_1hot_df.columns = lrp_column_names
        
extra6 = pd.DataFrame({'customer_id': extra6['customer_id'], 'lrp_date': new_date, 'total_quntity': extra6['total_quantity']})
extra6 = extra6.join(lrp_1hot_df)
god_set = pd.merge(god_set, extra6, on='customer_id', how='left').fillna(math.nan)

god_set.to_csv('midground1.csv')

# suspension, data cleaning and transforming
god_set = pd.read_csv('/data/team10/code/Danver/midground1.csv')
extra13 = pd.read_csv('/data/team10/data/suspensions_ebb_set1.csv')

prev_id = 'e3da0f9e3076cdf64decd7cb79dc174a51f874a3' # first id of the dataset, need to change for different dataset
total_time = 0
sus_tot_time = []
dup_counter = 0
# calculating the number of seconds each customer is suspended from multiple 2 dates
for index, row in extra13.iterrows():
    if index > 0:
        if prev_id != row['customer_id']:
            datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
            datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
            time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
            total_time = total_time + time_diff
            sus_tot_time.append(total_time)
            total_time = 0
        else:
            datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
            datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
            time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
            total_time = total_time + time_diff
        prev_id = row['customer_id']

sus_tot_time.append(total_time)

extra13 = pd.DataFrame({'customer_id': extra13['customer_id'].unique(), 'sus_tot_time': sus_tot_time})
god_set = pd.merge(god_set, extra13, on='customer_id', how='left').fillna(0)

god_set.to_csv('ebb_set1_v2.csv')

df1 = pd.read_csv("/data/team10/data/ebb_set2.csv")
extra1 = pd.read_csv('/data/team10/data/activations_ebb_set2.csv')
# making 1 hot encoding for the feature manufacturer
manu_encoder = OneHotEncoder()
manu_1hot_df = pd.DataFrame(manu_encoder.fit_transform(df1[['manufacturer']]).toarray())
manu_1hot_df.columns = manu_encoder.get_feature_names_out(['manufacturer'])

df1 = df1.join(manu_1hot_df)
df1.drop('manufacturer', axis=1, inplace=True)
# making 1 hot encoding for the feature operating_system
os_encoder = OneHotEncoder()
df1[['operating_system']] = df1[['operating_system']].replace(r'^Not', np.NaN, regex=True)
df1[['operating_system']] = df1[['operating_system']].replace('NONE', np.NaN, regex=True)
os_1hot_df = pd.DataFrame(os_encoder.fit_transform(df1[['operating_system']]).toarray())
os_1hot_df.columns = os_encoder.get_feature_names_out(['operating_system'])

df1 = df1.join(os_1hot_df)
df1.drop('operating_system', axis=1, inplace=True)
# making 1 hot encoding for the feature state
state_encoder = OneHotEncoder()
state_1hot_df = pd.DataFrame(state_encoder.fit_transform(df1[['state']]).toarray())
state_1hot_df.columns = state_encoder.get_feature_names_out(['state'])
state_column_names = []

df1 = df1.join(state_1hot_df)
df1.drop('state', axis=1, inplace=True)
# transforming date from string into timestamp (numerical data)
for index, row in df1.iterrows():
    # last_redemption_date
    if isinstance(row['last_redemption_date'], str):
        datetime_object = datetime.strptime(row['last_redemption_date'], '%Y-%m-%d')
        df1.loc[index,'last_redemption_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'last_redemption_date'] = math.nan
    
    # first_activation_date
    if isinstance(row['first_activation_date'], str):
        datetime_object = datetime.strptime(row['first_activation_date'], '%Y-%m-%d')
        df1.loc[index,'first_activation_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'first_activation_date'] = math.nan

# activations, data cleaning and transforming
duplicate_counter = 1
new_customer_id = []
number_of_activations = []
previous_customer_id = ''
# calculate the number of activations each customer have
# creating new feature "num_of_activations"
for index, row in extra1.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_activations.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']
    
number_of_activations.append(duplicate_counter)
number_of_activations = number_of_activations[1:]
new_customer_id = extra1['customer_id'].unique()
extra1 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_activations': number_of_activations})
god_set = pd.merge(df1, extra1, on = 'customer_id', how = 'left').fillna(0)

# auto_refill, data cleaning and transforming
extra2 = pd.read_csv('/data/team10/data/auto_refill_ebb_set2.csv')
prev_id = ''
drops = []
first_counter = 0
dup_to_no = False
for index, row in extra2.iterrows():
    if row['customer_id'] == prev_id and first_counter != 0:
        drops.append(index)
        first_counter += 1
    if row['customer_id'] == prev_id and first_counter == 0:
        drops.append(index)
        first_counter += 1
        dup_to_no = True
    if row['customer_id'] != prev_id:
        first_counter = 0
        if dup_to_no == True:
            drops.append(index)
            dup_to_no = False
    prev_id = row['customer_id']

extra2 = extra2.drop(extra2.index[drops])

new_enroll_date = []
new_de_enroll_date = []
# transforming date from string into timestamp (numerical data)
for index, row in extra2.iterrows():
    if isinstance(row['auto_refill_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_enroll_date'], '%Y-%m-%d')
        new_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_enroll_date.append(math.nan)
    
    if isinstance(row['auto_refill_de_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_de_enroll_date'], '%Y-%m-%d')
        new_de_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_de_enroll_date.append(math.nan)

new_customer_id = extra2['customer_id'].unique()
extra2 = pd.DataFrame({'customer_id': new_customer_id, 'auto_refill_enroll_date': new_enroll_date, 'auto_refill_de_enroll_date': new_de_enroll_date})
god_set = pd.merge(god_set, extra2, on='customer_id', how='left').fillna(math.nan)

# throttling, data cleaning and transforming
extra14 = pd.read_csv('/data/team10/data/throttling_ebb_set2.csv')
duplicate_counter = 1
new_customer_id = []
number_of_throttling = []
previous_customer_id = ''
# creating new feature "num_of_throttling"
for index, row in extra14.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_throttling.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']

number_of_throttling.append(duplicate_counter)
number_of_throttling = number_of_throttling[1:]
new_customer_id = extra14['customer_id'].unique()
extra14 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_throttling': number_of_throttling})
god_set = pd.merge(god_set, extra14, on = 'customer_id', how = 'left').fillna(0)

# support, data cleaning and transforming
from sklearn.preprocessing import MultiLabelBinarizer

extra12 = pd.read_csv('/data/team10/data/support_ebb_set2.csv')
duplicate_counter = 1
number_of_support = []
previous_customer_id = ''
support_category = []
support_customer = []
# creating new feature of num_of_support
for index, row in extra12.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_support.append(duplicate_counter)
        duplicate_counter = 1
        support_category.append(support_customer)
        support_customer = []
        support_customer.append(row['case_type'])
    else:
        duplicate_counter = duplicate_counter + 1
        support_customer.append(row['case_type'])
    previous_customer_id = row['customer_id']

number_of_support.append(duplicate_counter)
number_of_support = number_of_support[1:]

support_category.append(support_customer)
support_category = support_category[1:]

new_customer_id = extra12['customer_id'].unique()
extra12 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_support': number_of_support})
# using multihot encoding to turn support category into numerical values
sup_encoder = MultiLabelBinarizer()
sup_1hot_df = pd.DataFrame(sup_encoder.fit_transform(support_category))

sup_1hot_df.columns = list(sup_encoder.classes_)
extra12 = extra12.join(sup_1hot_df)

god_set = pd.merge(god_set, extra12, on = 'customer_id', how = 'left').fillna(0)

# loyalty_program, data cleaning and transforming
extra6 = pd.read_csv('/data/team10/data/loyalty_program_ebb_set2.csv')

new_date = []
# transforming date from string into timestamp (numerical data)
for index, row in extra6.iterrows():
    if isinstance(row['date'], str):
        datetime_object = datetime.strptime(row['date'], '%Y-%m-%d')
        new_date.append(int(round(datetime_object.timestamp())))
    else:
        new_date.append(math.nan)
# making 1 hot encoding for the feature lrp_enrolled
lrp_encoder = OneHotEncoder()
lrp_1hot_df = pd.DataFrame(lrp_encoder.fit_transform(extra6[['lrp_enrolled']]).toarray())
lrp_column_names = []
for i in range(len(lrp_1hot_df.columns)):
    lrp_column_names.append(f'lrp{i}')
lrp_1hot_df.columns = lrp_column_names
        
extra6 = pd.DataFrame({'customer_id': extra6['customer_id'], 'lrp_date': new_date, 'total_quntity': extra6['total_quantity']})
extra6 = extra6.join(lrp_1hot_df)
god_set = pd.merge(god_set, extra6, on='customer_id', how='left').fillna(math.nan)

# suspension, data cleaning and transforming
extra13 = pd.read_csv('/data/team10/data/suspensions_ebb_set2.csv')

prev_id = '' 
total_time = 0
sus_tot_time = []
dup_counter = 0
prev_diff = 1209600 
# calculating the number of seconds each customer is suspended from multiple 2 dates
for index, row in extra13.iterrows():
    if index > 0:
        if prev_id != row['customer_id']:
            if dup_counter == 0:
                sus_tot_time.append(prev_diff)
                total_time = 0
                dup_counter = 0
            else:
                sus_tot_time.append(total_time)
                total_time = 0
                dup_counter = 0
        else:
            if dup_counter == 0:
                datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
                datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
                time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
                total_time = total_time + prev_diff + time_diff
                dup_counter = dup_counter + 1
            else:
                datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
                datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
                time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
                total_time = total_time + time_diff
        
        datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
        datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
        prev_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
        prev_id = row['customer_id']
        
sus_tot_time.append(total_time)

extra13 = pd.DataFrame({'customer_id': extra13['customer_id'].unique(), 'sus_tot_time': sus_tot_time})
god_set = pd.merge(god_set, extra13, on='customer_id', how='left').fillna(0)

god_set.to_csv('ebb_set2_v2.csv')

df1 = pd.read_csv("/data/team10/data/eval_set.csv")
extra1 = pd.read_csv('/data/team10/data/activations_eval_set.csv')
# making 1 hot encoding for the feature manufacturer
manu_encoder = OneHotEncoder()
manu_1hot_df = pd.DataFrame(manu_encoder.fit_transform(df1[['manufacturer']]).toarray())
manu_1hot_df.columns = manu_encoder.get_feature_names_out(['manufacturer'])

df1 = df1.join(manu_1hot_df)
df1.drop('manufacturer', axis=1, inplace=True)
# making 1 hot encoding for the feature operating_system
os_encoder = OneHotEncoder()
df1[['operating_system']] = df1[['operating_system']].replace(r'^Not', np.NaN, regex=True)
df1[['operating_system']] = df1[['operating_system']].replace('NONE', np.NaN, regex=True)
os_1hot_df = pd.DataFrame(os_encoder.fit_transform(df1[['operating_system']]).toarray())
os_1hot_df.columns = os_encoder.get_feature_names_out(['operating_system'])

df1 = df1.join(os_1hot_df)
df1.drop('operating_system', axis=1, inplace=True)
# making 1 hot encoding for the feature state
state_encoder = OneHotEncoder()
state_1hot_df = pd.DataFrame(state_encoder.fit_transform(df1[['state']]).toarray())
state_1hot_df.columns = state_encoder.get_feature_names_out(['state'])
state_column_names = []

df1 = df1.join(state_1hot_df)
df1.drop('state', axis=1, inplace=True)

for index, row in df1.iterrows():
    # last_redemption_date
    if isinstance(row['last_redemption_date'], str):
        datetime_object = datetime.strptime(row['last_redemption_date'], '%Y-%m-%d')
        df1.loc[index,'last_redemption_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'last_redemption_date'] = math.nan
    
    # first_activation_date
    if isinstance(row['first_activation_date'], str):
        datetime_object = datetime.strptime(row['first_activation_date'], '%Y-%m-%d')
        df1.loc[index,'first_activation_date'] = int(round(datetime_object.timestamp()))
    else:
        df1.loc[index,'first_activation_date'] = math.nan

# activations, data cleaning and transforming
duplicate_counter = 1
new_customer_id = []
number_of_activations = []
previous_customer_id = ''
for index, row in extra1.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_activations.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']
    
number_of_activations.append(duplicate_counter)
number_of_activations = number_of_activations[1:]
new_customer_id = extra1['customer_id'].unique()
extra1 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_activations': number_of_activations})
god_set = pd.merge(df1, extra1, on = 'customer_id', how = 'left').fillna(0)

# auto_refill, data cleaning and transforming
extra2 = pd.read_csv('/data/team10/data/auto_refill_eval_set.csv')
prev_id = ''
drops = []
first_counter = 0
dup_to_no = False
for index, row in extra2.iterrows():
    if row['customer_id'] == prev_id and first_counter != 0:
        drops.append(index)
        first_counter += 1
    if row['customer_id'] == prev_id and first_counter == 0:
        drops.append(index)
        first_counter += 1
        dup_to_no = True
    if row['customer_id'] != prev_id:
        first_counter = 0
        if dup_to_no == True:
            drops.append(index)
            dup_to_no = False
    prev_id = row['customer_id']

extra2 = extra2.drop(extra2.index[drops])

new_enroll_date = []
new_de_enroll_date = []

for index, row in extra2.iterrows():
    if isinstance(row['auto_refill_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_enroll_date'], '%Y-%m-%d')
        new_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_enroll_date.append(math.nan)
    
    if isinstance(row['auto_refill_de_enroll_date'], str):
        datetime_object = datetime.strptime(row['auto_refill_de_enroll_date'], '%Y-%m-%d')
        new_de_enroll_date.append(int(round(datetime_object.timestamp())))
    else:
        new_de_enroll_date.append(math.nan)

new_customer_id = extra2['customer_id'].unique()
extra2 = pd.DataFrame({'customer_id': new_customer_id, 'auto_refill_enroll_date': new_enroll_date, 'auto_refill_de_enroll_date': new_de_enroll_date})
god_set = pd.merge(god_set, extra2, on='customer_id', how='left').fillna(math.nan)

# throttling, data cleaning and transforming
extra14 = pd.read_csv('/data/team10/data/throttling_eval_set.csv')
duplicate_counter = 1
new_customer_id = []
number_of_throttling = []
previous_customer_id = ''
for index, row in extra14.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_throttling.append(duplicate_counter)
        duplicate_counter = 1
    else:
        duplicate_counter = duplicate_counter + 1
    previous_customer_id = row['customer_id']
    
number_of_throttling.append(duplicate_counter)
number_of_throttling = number_of_throttling[1:]
new_customer_id = extra14['customer_id'].unique()
print(len(number_of_throttling))
print(len(new_customer_id))
extra14 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_throttling': number_of_throttling})
god_set = pd.merge(god_set, extra14, on = 'customer_id', how = 'left').fillna(0)

# support, data cleaning and transforming
from sklearn.preprocessing import MultiLabelBinarizer

extra12 = pd.read_csv('/data/team10/data/support_eval_set.csv')
duplicate_counter = 1
number_of_support = []
previous_customer_id = ''
support_category = []
support_customer = []
for index, row in extra12.iterrows():
    if previous_customer_id != row['customer_id']:
        number_of_support.append(duplicate_counter)
        duplicate_counter = 1
        support_category.append(support_customer)
        support_customer = []
        support_customer.append(row['case_type'])
    else:
        duplicate_counter = duplicate_counter + 1
        support_customer.append(row['case_type'])
    previous_customer_id = row['customer_id']

number_of_support.append(duplicate_counter)
number_of_support = number_of_support[1:]

support_category.append(support_customer)
support_category = support_category[1:]

new_customer_id = extra12['customer_id'].unique()
extra12 = pd.DataFrame({'customer_id': new_customer_id, 'num_of_support': number_of_support})

sup_encoder = MultiLabelBinarizer()
sup_1hot_df = pd.DataFrame(sup_encoder.fit_transform(support_category))
sup_1hot_df.columns = list(sup_encoder.classes_)
extra12 = extra12.join(sup_1hot_df)

god_set = pd.merge(god_set, extra12, on = 'customer_id', how = 'left').fillna(0)

# loyalty_program, data cleaning and transforming
extra6 = pd.read_csv('/data/team10/data/loyalty_program_eval_set.csv')

new_date = []

for index, row in extra6.iterrows():
    if isinstance(row['date'], str):
        datetime_object = datetime.strptime(row['date'], '%Y-%m-%d')
        new_date.append(int(round(datetime_object.timestamp())))
    else:
        new_date.append(math.nan)

lrp_encoder = OneHotEncoder()
lrp_1hot_df = pd.DataFrame(lrp_encoder.fit_transform(extra6[['lrp_enrolled']]).toarray())
lrp_column_names = []
for i in range(len(lrp_1hot_df.columns)):
    lrp_column_names.append(f'lrp{i}')
lrp_1hot_df.columns = lrp_column_names
        
extra6 = pd.DataFrame({'customer_id': extra6['customer_id'], 'lrp_date': new_date, 'total_quntity': extra6['total_quantity']})
extra6 = extra6.join(lrp_1hot_df)
god_set = pd.merge(god_set, extra6, on='customer_id', how='left').fillna(math.nan)

god_set.to_csv('midground1_eval.csv')

# suspension, data cleaning and transforming
god_set = pd.read_csv('/data/team10/code/Danver/midground1_eval.csv')
extra13 = pd.read_csv('/data/team10/data/suspensions_eval_set.csv')
# calculating the number of seconds each customer is suspended from multiple 2 dates
prev_id = 'c5ba520345d97c7e20a6fdb34abd8035f55b2678' 
total_time = 0
sus_tot_time = []
dup_counter = 0
prev_diff = 0
for index, row in extra13.iterrows():
    if index > 0:
        if prev_id != row['customer_id']:
            if dup_counter == 0:
                sus_tot_time.append(prev_diff)
                total_time = 0
                dup_counter = 0
            else:
                sus_tot_time.append(total_time)
                total_time = 0
                dup_counter = 0
        else:
            if dup_counter == 0:
                datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
                datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
                time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
                total_time = total_time + prev_diff + time_diff
                dup_counter = dup_counter + 1
            else:
                datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
                datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
                time_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
                total_time = total_time + time_diff
        
        datetime_object1 = datetime.strptime(row['start_date'], '%Y-%m-%d')
        datetime_object2 = datetime.strptime(row['end_date'], '%Y-%m-%d')
        prev_diff = int(round(datetime_object2.timestamp())) - int(round(datetime_object1.timestamp()))
        prev_id = row['customer_id']

sus_tot_time.append(total_time)

extra13 = pd.DataFrame({'customer_id': extra13['customer_id'].unique(), 'sus_tot_time': sus_tot_time})
god_set = pd.merge(god_set, extra13, on='customer_id', how='left').fillna(0)

god_set.to_csv('eval_set_v2.csv')

#a function which processes network data and then merge to the main dataset
import pandas as pd

def preprocess_network_data_and_merge(file,main_file,new_file_name):
    df = pd.read_csv(file)
    df = df.drop(columns=['date'])
    df = df.groupby('customer_id',as_index=False).agg({'voice_minutes': ['mean'],'total_sms':['mean'],'total_kb':['mean'],'hotspot_kb':['mean']})
    
    bigSet = pd.read_csv(main_file)
    bigSetPulse = pd.merge(left=bigSet, right=df, how='left', left_on='customer_id', right_on='customer_id')
    
    bigSetPulse.to_csv(new_file_name, index = False)

preprocess_network_data_and_merge("/data/team10/data/network_ebb_set1.csv","ebb_set1_v2.csv","ebb_set1_v3.csv")
preprocess_network_data_and_merge("/data/team10/data/network_ebb_set2.csv","ebb_set2_v2.csv","ebb_set2_v3.csv")
preprocess_network_data_and_merge("/data/team10/data/network_eval_set.csv","eval_set_v2.csv","eval_set_v3.csv")

import matplotlib.pyplot as plt
import csv
import os
import re

#a function for cleaning the issue column in the interactions dataset
def clean_issue(x):
    if re.search("Transit",x,re.IGNORECASE):
        return "Meme"
    if re.search('Other',x,re.IGNORECASE):
        return 'Other'
    elif re.search('Agent Released',x,re.IGNORECASE):
        return 'Agent Released'
    elif re.search('O ',x,re.IGNORECASE):
        return 'Other'
    elif re.search('Customer Released',x,re.IGNORECASE):
        return 'Customer Released'
    elif re.search('Unlock Phone',x,re.IGNORECASE):
        return 'Unlock Phone'
    elif re.search('CC Declined',x,re.IGNORECASE):
        return 'CC Declined'
    elif re.search('Device Issue',x,re.IGNORECASE):
        return 'Device Issue'
    elif re.search('Delayed Activation',x,re.IGNORECASE):
        return 'Delayed Activation'
    elif re.search('Authentication Failure',x,re.IGNORECASE):
        return 'Authentication Failure'
    elif re.search('Risk Assessment',x,re.IGNORECASE):
        return 'Risk Assessment'
    elif re.search('No/Poor Coverage',x,re.IGNORECASE):
        return 'No/Poor Coverage'
    elif re.search('New Sim Required',x,re.IGNORECASE):
        return 'New Sim Required'
    elif re.search('SIM Issue',x,re.IGNORECASE):
        return 'SIM Issue'
    elif re.search('SIM Exchange',x,re.IGNORECASE):
        return 'SIM Exchange'
    elif re.search('Order not received',x,re.IGNORECASE):
        return 'Order not received'
    elif re.search('Due Date and Time',x,re.IGNORECASE):
        return 'Due Date and Time'
    elif re.search('Multiple Charges',x,re.IGNORECASE):
        return 'Multiple Charges'
    elif re.search('Resolution Required',x,re.IGNORECASE):
        return 'Resolution Required'
    elif re.search('Inactive Posa',x,re.IGNORECASE):
        return 'Inactive Posa'
    elif re.search('New Sim',x,re.IGNORECASE):
        return 'New SIM Card'
    elif re.search('Throttled Incorrectly',x,re.IGNORECASE):
        return 'Throttled Incorrectly'
    elif re.search('Phone Exchange',x,re.IGNORECASE):
        return 'Phone Exchange'
    elif re.search('Transaction Pending',x,re.IGNORECASE):
        return 'Transaction Pending'
    elif re.search('Not Compatible',x,re.IGNORECASE):
        return 'Not Compatible'
    elif re.search('System',x,re.IGNORECASE):
        return 'System Error'
    elif re.search('Error',x,re.IGNORECASE):
        return 'Error'
    elif re.search('SIM Not Accepted',x,re.IGNORECASE):
        return 'SIM Not Accepted'
    elif re.search('Hotspot',x,re.IGNORECASE):
        return 'Hotspot'
    elif re.search('Phone Locked',x,re.IGNORECASE):
        return 'Phone Locked'
    elif re.search('ARO',x,re.IGNORECASE):
        return 'ARO'
    elif re.search('ONR',x,re.IGNORECASE):
        return 'ONR'
    elif re.search('Accessibility',x,re.IGNORECASE):
        return 'Accessibility'
    elif re.search('Carrier Outage',x,re.IGNORECASE):
        return 'Carrier Outage'
    elif re.search('Recording Issues',x,re.IGNORECASE):
        return 'Recording Issues'
    elif re.search('Error',x,re.IGNORECASE):
        return 'Error'
    elif re.search(r"[X]+",x,re.IGNORECASE):
        return re.sub(r"[X]+","",x)
    elif re.search("Customer Rel",x,re.IGNORECASE):
        return 'Customer Released'
    elif re.search("Carrier Created",x,re.IGNORECASE):
        return 'Carrier Created'
    elif re.search("Delayed Activ",x,re.IGNORECASE):
        return 'Delayed Activation'
    elif re.search("Othe",x,re.IGNORECASE):
        return 'Other'
    elif re.search("Check Application",x,re.IGNORECASE):
        return 'Check Application'
    elif re.search("Enrollment",x,re.IGNORECASE):
        return 'Enrollment'
    elif re.search("Not Qualify",x,re.IGNORECASE):
        return 'Qualified'
    elif re.search("Cust ",x,re.IGNORECASE):
        return re.sub("Cust ","Customer ",x,re.IGNORECASE)
    else:
        return x

#a function for cleaning the disposition column in the interactions dataset
def clean_disposition(x):
    if re.search("Successful",x,re.IGNORECASE):
        return 'Successful'
    elif re.search("Suc",x,re.IGNORECASE):
        return 'Successful'
    elif re.search("Unsucc",x,re.IGNORECASE):
        return 'Unsuccessful'
    elif re.search("No",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search("Call Terminated",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search("Customer Hung",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search(".com",x,re.IGNORECASE):
        return float("NaN")
    elif re.search("mail",x,re.IGNORECASE):
        return float("NaN")
    elif re.search("Created",x,re.IGNORECASE):
        return float("NaN")
    elif re.search("Sale",x,re.IGNORECASE):
        return "Successful"
    elif re.search("Sale Order",x,re.IGNORECASE):
        return "Successful"
    elif re.search("Created",x,re.IGNORECASE):
        return "Successful"
    elif re.search("Transferred",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("Schedule Callback",x,re.IGNORECASE):
        return "Schedule Callback"
    elif re.search("Terminated",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search("Disconnected",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search("Transfer",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("Callback",x,re.IGNORECASE):
        return "Schedule Callback"
    elif re.search("close",x,re.IGNORECASE):
        return "Closed"
    elif re.search("TC",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("TF",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("CHU",x,re.IGNORECASE):
        return "Unsuccessful"
    elif re.search("TC",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("TCS",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("SCB",x,re.IGNORECASE):
        return "Schedule Callback"
    elif re.search("CT",x,re.IGNORECASE):
        return "Transferred"
    elif re.search("USC",x,re.IGNORECASE):
        return "Schedule Callback"
    elif re.search('nan',x,re.IGNORECASE):
        return float('nan')
    elif re.search('Closed',x,re.IGNORECASE):
        return 'Unsuccessful'
    else:
        return x

def clean_reason(x):
    if re.search("Emergency Broadband Benefit Call",x):
        return 1
    else:
        return 0

#further processing of disposition column
def foo(x): 
    m = pd.Series.mode(x)
    return m.values[0] if not m.empty else np.nan

def disposition_freq(x):
    m = pd.Series.mode(x)
    if len(m.values) == 3:
        return 'Successful'
    elif len(m.values) == 2:
        if m.values[0] == 'Transferred' and m.values[1] == 'Successful':
            return 'Successful'
        elif m.values[0] == 'Successful' and m.values[1] == 'Transferred':
            return 'Successful'
        elif m.values[0] == 'Successful' and m.values[1] == 'Schedule Callback':
            return 'Successful'
        elif m.values[0] == 'Schedule Callback' and m.values[1] == 'Successful' :
            return 'Successful'
        elif m.values[0] == 'Unsuccessful' and m.values[1] == 'Successful':
            return 'Unsuccessful'
        elif m.values[0] == 'Successful' and m.values[1] == 'Unsuccessful':
            return 'Unsuccessful'
        elif m.values[0] == 'Unsuccessful' and m.values[1] == 'Schedule Callback':
            return 'Unsuccessful'
        elif m.values[0] == 'Schedule Callback' and m.values[0] == 'Unsuccessful':
            return 'Unsuccessful'
    return m.values[0] if not m.empty else np.nan

# called previous functions to add to dataset
def clean_interaction (file,main_file,new_file_name):
    df = pd.read_csv(file)
    
    df['disposition'] = df['disposition'].apply(str)
    df['disposition'] = df['disposition'].apply(clean_disposition)
    
    df['reason'] = df['reason'].apply(str)
    df['reason'] = df['reason'].apply(clean_reason)
    
    df = df.groupby('customer_id',as_index=False).agg({'reason':foo,'disposition':disposition_freq})
    
    disposition = pd.get_dummies(df.disposition, prefix='disposition')
    df = df.drop(['disposition'],axis=1)
    
    df = pd.concat([df, disposition], axis=1)
        
    bigSet = pd.read_csv(main_file)
    bigSetPulse = pd.merge(left=bigSet, right=df, how='left', left_on='customer_id', right_on='customer_id')
    
    bigSetPulse.to_csv(new_file_name, index = False)

clean_interaction ("/data/team10/data/interactions_ebb_set1.csv", "ebb_set1_v3.csv", "ebb_set1_v4.csv")
clean_interaction ("/data/team10/data/interactions_ebb_set2.csv", "ebb_set2_v3.csv", "ebb_set2_v4.csv")
clean_interaction ("/data/team10/data/interactions_eval_set.csv", "eval_set_v3.csv", "eval_set_v4.csv")    

def processReDeActivation(deactivationfile, reactivationfile, mainFile, newName):
    #combine row with same customber id. Finds average for numerical features
    #For non-numerical features, find mode, then use one hot encoding
    
    mainFile = pd.read_csv(mainFile)
    
    df = pd.read_csv(reactivationfile)
    df = pd.crosstab(index=df['customer_id'], columns='num_of_reactivation')
    df.reset_index(inplace=True)
    
    df1 = pd.read_csv(deactivationfile)
    df1 = pd.crosstab(index=df1['customer_id'], columns='num_of_deactivation')
    df1.reset_index(inplace=True)
    
    mainFilePluse = pd.merge(left=mainFile, right=df, how='left', left_on='customer_id', right_on='customer_id')
    mainFilePluse = pd.merge(left=mainFilePluse, right=df1, how='left', left_on='customer_id', right_on='customer_id')
  
    mainFilePluse = mainFilePluse.fillna(0)
    mainFilePluse = mainFilePluse.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, errors='ignore')
    mainFilePluse.to_csv(newName, index = False)

processReDeActivation("/data/team10/data/deactivations_ebb_set1.csv", "/data/team10/data/reactivations_ebb_set1.csv", "ebb_set1_v4.csv", "ebb_set1_v5.csv")
processReDeActivation("/data/team10/data/deactivations_ebb_set2.csv", "/data/team10/data/reactivations_ebb_set2.csv", "ebb_set2_v4.csv", "ebb_set2_v5.csv")
processReDeActivation("/data/team10/data/deactivations_eval_set.csv", "/data/team10/data/reactivations_eval_set.csv", "eval_set_v4.csv", "eval_set_v5.csv")
