######################################################
##                                                  ##
##              TEAM 10 - ONECLASS SVM              ##
##                                                  ##
######################################################

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

df1 = pd.read_csv("/data/team10/code/ebb_set1_v6.csv")
df1_y = df1[['ebb_eligible']]
df1 = df1[['last_redemption_date','first_activation_date','total_redemptions','tenure','number_upgrades','year',
           'total_revenues_bucket','num_of_activations',
           'num_of_throttling','num_of_support','lrp_date','total_quntity','lrp0','sus_tot_time',
           'num_of_reactivation','num_of_deactivation']]

df2 = pd.read_csv("/data/team10/code/ebb_set2_v6.csv")
df2_y = df2[['ebb_eligible']]
df2 = df2[['last_redemption_date','first_activation_date','total_redemptions','tenure','number_upgrades','year',
           'total_revenues_bucket','num_of_activations',
           'num_of_throttling','num_of_support','lrp_date','total_quntity','lrp0','sus_tot_time',
           'num_of_reactivation','num_of_deactivation']]

df3 = pd.read_csv("/data/team10/code/eval_set_v6.csv")
df3 = df3[['last_redemption_date','first_activation_date','total_redemptions','tenure','number_upgrades','year',
           'total_revenues_bucket','num_of_activations',
           'num_of_throttling','num_of_support','lrp_date','total_quntity','lrp0','sus_tot_time',
           'num_of_reactivation','num_of_deactivation']]

df_final = pd.concat([df1, df2], ignore_index=True)

final_y = df1_y.values.tolist() + df2_y.values.tolist()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_final)
df_final = scaler.transform(df_final)

clf = OneClassSVM(kernel='rbf').fit(df_final, final_y)

scaler2 = StandardScaler()
scaler2.fit(df3)
df3 = scaler.transform(df3)

prediction_y = clf.predict(df3).tolist()
prediction_y = [0 if x==-1 else x for x in prediction_y]

df3 = pd.read_csv("/data/team10/code/eval_set_v2.csv")
answer_df = pd.DataFrame({'customer_id':df3['customer_id'].tolist(),'ebb_eligible':prediction_y})
answer_df.to_csv('2022-04-17-alltrainless.csv', index=False)