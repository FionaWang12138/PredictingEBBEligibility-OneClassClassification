##################################################################
##                                                              ##
##            TEAM 10 - LINEAR DISCRIMINANT ANALYSIS            ##
##                                                              ##
##################################################################


import pandas as pd
import re
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


data_name = ["ebb_set1.csv", "activations_ebb_set1.csv", "auto_refill_ebb_set1.csv", 
        "deactivations_ebb_set1.csv", "interactions_ebb_set1.csv", "ivr_calls_ebb_set1.csv",
       "lease_history_ebb_set1.csv", "loyalty_program_ebb_set1.csv", "network_ebb_set1.csv", 
        "notifying_ebb_set1.csv", "phone_data_ebb_set1.csv", "reactivations_ebb_set1.csv",
       "redemptions_ebb_set1.csv", "support_ebb_set1.csv", "suspensions_ebb_set1.csv", "throttling_ebb_set1.csv"]


#function to preposess dataset and merge them to one pandas table
def preprocessiong_data_new(data_name):
    # get all data
    ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[0]}")
    # # activations_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[1]}")
    # auto_refill_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[2]}")
    deactivations_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[3]}")
    interactions_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[4]}")
    ivr_calls_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[5]}")
    lease_history_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[6]}")
    loyalty_program_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[7]}")
    network_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[8]}")
    notifying_ebb_set1 = pd.read_csv(f"/data/team10/data/{data_name[9]}")
    # phone_data_ebb_set1 = pd.read_csv(f"/data/team10/data/{data_name[10]}")
    reactivations_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[11]}")
    redemptions_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[12]}")
    # support_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[13]}")
    suspensions_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[14]}")
    # throttling_ebb_set1= pd.read_csv(f"/data/team10/data/{data_name[15]}")
    
    
    #''' prepocess all the relevant datasets and select relevant features''''
    #process ebb_set1
    exp = ebb_set1
    if 'ebb_eligible' in exp.columns:
        exp.loc[exp['ebb_eligible'] != 1, 'ebb_eligible'] = 0
    exp['manufacturer'] = exp['manufacturer'].fillna('Empty')
    exp['operating_system'] = exp['operating_system'].fillna('Empty')
    exp['state'] = exp['state'].fillna('Empty')
    exp['tenure'] = exp['tenure'].fillna(0)
    exp['number_upgrades'] = exp['number_upgrades'].fillna(0)
    exp['total_revenues_bucket'] = exp['total_revenues_bucket'].fillna(0)
    if 'ebb_eligible' in exp.columns:
        exp['ebb_eligible'] = exp['ebb_eligible'].fillna(0)
    exp['manufacturer'] = exp['manufacturer'].apply(lambda x: x.split(' ')[0].lower())
    exp['operating_system'] = exp['operating_system'].apply(lambda x: re.split(",| ", x)[0].lower())
    if 'ebb_eligible' in exp.columns:
        exp = exp[["customer_id","total_redemptions", "tenure","number_upgrades", "manufacturer", "operating_system", "state", "total_revenues_bucket", "ebb_eligible"]]
    else:
        exp = exp[["customer_id","total_redemptions", "tenure","number_upgrades", "manufacturer", "operating_system", "state", "total_revenues_bucket"]]
    exp2 = exp
    y = pd.get_dummies(exp2.manufacturer, prefix='man')
    j = pd.get_dummies(exp2.operating_system, prefix='op')
    k = pd.get_dummies(exp2.state, prefix='state')
    w = y.merge(j, left_index=True, right_index=True)
    w = w.merge(k,left_index=True, right_index=True)
    exp2 = exp2.merge(w,left_index=True, right_index=True)
    ebb_set1 = exp2.drop(columns=["operating_system", "manufacturer","state"], axis=1)
    
    
    
    # activations
    # act = activations_ebb_set1
    # act = pd.crosstab(index=act['customer_id'], columns='activation_count')
    # act.reset_index(inplace=True)
    # activations_ebb_set1 = act
    
    #auto_refill
    #not used
    
    #deactivation
    dummy_deac = deactivations_ebb_set1
    dummy_deac = pd.crosstab(index=dummy_deac['customer_id'], columns='deactivation_count')
    dummy_deac.reset_index(inplace=True)
    deactivations_ebb_set1 = dummy_deac
    
    
    #interactions
    dummy_int = interactions_ebb_set1
    def set_interaction(value):
        if value == 'Emergency Broadband Benefit Call':
            value = 1
            return value
        else:
            value = 0
            return value
    dummy_int['reason'] = dummy_int['reason'].apply(set_interaction)
    df_dummy = dummy_int.groupby('customer_id').agg(list)
    df_dummy.reset_index(inplace=True)
    def return_max(value):
        return max(value)
    df_dummy['reason'] = df_dummy['reason'].apply(return_max)
    interactions_ebb_set1 = df_dummy[['customer_id', 'reason']]
    
    #ivr call
    dummy_ivr = ivr_calls_ebb_set1 
    dummy_ivr = pd.crosstab(index=dummy_ivr['customer_id'], columns='ivr_call_count')
    dummy_ivr.reset_index(inplace=True)
    ivr_calls_ebb_set1 = dummy_ivr
    
    #lease history
    dummy_lh = lease_history_ebb_set1
    dummy_lh = pd.crosstab(index=dummy_lh['customer_id'], columns='lease_number')
    dummy_lh.reset_index(inplace=True)
    lease_history_ebb_set1 = dummy_lh
    
    #loyalty
    dummy_loyalty = loyalty_program_ebb_set1
    dummy_loyalty.loc[dummy_loyalty['lrp_enrolled'] == 'Y', 'lrp_enrolled'] = 1
    loyalty_program_ebb_set1 = dummy_loyalty[["customer_id","lrp_enrolled", "total_quantity"]]
    
    #network
    dummy_net = network_ebb_set1[['customer_id', 'voice_minutes', 'total_kb']]
    dummy_net = dummy_net.groupby('customer_id').agg(list)
    dummy_net.reset_index(inplace=True)
    # dummy_net
    def return_sum(value):
        return sum(value)
    
    dummy_net['voice_minutes'] = dummy_net['voice_minutes'].apply(return_sum)
    dummy_net['total_kb'] = dummy_net['total_kb'].apply(return_sum)
    network_ebb_set1 = dummy_net
    
    
    # notifying
    dummy_not = notifying_ebb_set1 
    dummy_not = pd.crosstab(index=dummy_not['customer_id'], columns='num_notification')
    dummy_not.reset_index(inplace=True)
    notifying_ebb_set1 = dummy_not
    
    
    #phone data not useful
    
    
    #reactivation
    dummy_react = reactivations_ebb_set1
    dummy_react = pd.crosstab(index=dummy_react['customer_id'], columns='num_of_reactivation')
    dummy_react.reset_index(inplace=True)
    reactivations_ebb_set1 = dummy_react
    
    
    #redemption
    dummy_red = redemptions_ebb_set1[['customer_id', 'revenues']]
    dummy_red = dummy_red.groupby('customer_id').agg(list)
    dummy_red['revenues'] = dummy_red['revenues'].apply(return_max)
    dummy_red.reset_index(inplace=True)
    redemptions_ebb_set1 = dummy_red
    
    
    # support-not used
    
    
    # Suspension
    dummy_susp = suspensions_ebb_set1
    dummy_susp = pd.crosstab(index=dummy_susp['customer_id'], columns='num_of_suspension')
    dummy_susp.reset_index(inplace=True)
    suspensions_ebb_set1 = dummy_susp
    
    
    #Throttling not used
    #merged with suspension
    sus_red = pd.merge(ebb_set1, suspensions_ebb_set1,left_index=True, right_index=True,  how="outer")
    sus_red['num_of_suspension'] = sus_red['num_of_suspension'].fillna(0)
    sus_red = sus_red.drop(columns=['customer_id_y'])
    
    
    #''' merge all the tables together''''
    
    # ebb_set merge with redemption
    ebb_red = pd.merge(sus_red, redemptions_ebb_set1,left_index=True, right_index=True,  how="outer")
    test = ebb_red
    tt = test.dropna(subset=['customer_id_x'])
    tt['revenues'] = tt['revenues'].fillna(0)
    tt  = tt.drop(columns=['customer_id'])
    tt.rename(columns={'customer_id_x':'customer_id'}, inplace=True)
    ebb_red = tt
    
    # reactivation
    ebb_react = pd.merge(ebb_red, reactivations_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_react['num_of_reactivation'] = ebb_react['num_of_reactivation'].fillna(0)
    ebb_react = ebb_react.drop(columns=['customer_id_y'])
    ebb_react.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
    
    # Notifying
    # ebb_react
    ebb_not = pd.merge(ebb_react, notifying_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_not['num_notification'] = ebb_not['num_notification'].fillna(0)
    ebb_not = ebb_not.drop(columns=['customer_id_y'])
    ebb_not.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
    
    # network
    # ebb_not
    ebb_network = pd.merge(ebb_not, network_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_network = ebb_network.dropna(subset=['customer_id_x'])
    ebb_network['voice_minutes'] = ebb_network['voice_minutes'].fillna(0)
    ebb_network['total_kb'] = ebb_network['total_kb'].fillna(0)
    ebb_network = ebb_network.drop(columns=['customer_id_y'])
    ebb_network.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
    
    
    # add loyalty program
    #ebb_network
    ebb_loyalty = pd.merge(ebb_network, loyalty_program_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_loyalty['lrp_enrolled'] = ebb_loyalty['lrp_enrolled'].fillna(0)
    ebb_loyalty['total_quantity'] = ebb_loyalty['total_quantity'].fillna(0)
    ebb_loyalty = ebb_loyalty.drop(columns=['customer_id_y'])
    ebb_loyalty.rename(columns={'customer_id_x':'customer_id'}, inplace=True)
    
    
    # add leaseHistory

    # ebb_loyalty
    ebb_lease = pd.merge(ebb_loyalty, lease_history_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_lease =ebb_lease.dropna(subset=['customer_id_x'])
    ebb_lease = ebb_lease.drop(columns=['customer_id_y'])
    ebb_lease.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
   
    
    # #add ivr_calls
    # ebb_lease
    ebb_ivr = pd.merge(ebb_lease, ivr_calls_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_ivr['ivr_call_count'] = ebb_ivr['ivr_call_count'].fillna(0)
    ebb_ivr = ebb_ivr.drop(columns=['customer_id_y'])
    ebb_ivr.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 

    #add interaction

    ebb_inter = pd.merge(ebb_ivr, interactions_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_inter['reason'] = ebb_inter['reason'].fillna(0)
    ebb_inter = ebb_inter.drop(columns=['customer_id_y'])
    ebb_inter.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
    

    # #Add deactivate
    ebb_deact = pd.merge(ebb_inter, deactivations_ebb_set1, left_index=True, right_index=True,  how="outer")
    ebb_deact['deactivation_count'] = ebb_deact['deactivation_count'].fillna(0)
    ebb_deact = ebb_deact.drop(columns=['customer_id_y'])
    ebb_deact.rename(columns={'customer_id_x':'customer_id'}, inplace=True) 
   
    return ebb_deact




set2_dataset = ["ebb_set2.csv", "activations_ebb_set2.csv", "auto_refill_ebb_set2.csv", 
        "deactivations_ebb_set2.csv", "interactions_ebb_set2.csv", "ivr_calls_ebb_set2.csv",
       "lease_history_ebb_set2.csv", "loyalty_program_ebb_set2.csv", "network_ebb_set2.csv", 
        "notifying_ebb_set2.csv", "phone_data_ebb_set2.csv", "reactivations_ebb_set2.csv",
       "redemptions_ebb_set2.csv", "support_ebb_set2.csv", "suspensions_ebb_set2.csv", "throttling_ebb_set2.csv"]


eval_dataset = ["eval_set.csv", "activations_eval_set.csv", "auto_refill_eval_set.csv", 
        "deactivations_eval_set.csv", "interactions_eval_set.csv", "ivr_calls_eval_set.csv",
       "lease_history_eval_set.csv", "loyalty_program_eval_set.csv", "network_eval_set.csv", 
        "notifying_eval_set.csv", "phone_data_eval_set.csv", "reactivations_eval_set.csv",
       "redemptions_eval_set.csv", "support_eval_set.csv", "suspensions_eval_set.csv", "throttling_eval_set.csv"]


set1 = preprocessiong_data_new(data_name)
set2 = preprocessiong_data_new(set2_dataset)

eval_data = preprocessiong_data_new(eval_dataset)



#function to make prediction
def predict(train, test, model):
    column_list = (test.append([train])).columns.tolist()
    test = test.reindex(columns=test.columns | train.columns).fillna(0)
    train = train.reindex(columns=train.columns | test.columns).fillna(0)
    
    X_train = train.drop(['customer_id', 'ebb_eligible'], axis=1)
    y_train = train[['ebb_eligible']]


    X_test = test.drop(['customer_id', 'ebb_eligible'], axis=1)
    y_test = test[['ebb_eligible']]
    
    model_fit = model.fit(X_train, y_train)
    
    y_Pred = model_fit.predict(X_test)
    
    
    return y_Pred



model = LinearDiscriminantAnalysis()
predict(set1, set2, model)


def predict_with_eval(train1,train2, test, model):
    
    #merge ebb_set1 and ebb_set2 to form a train dataset
    train1 = train1.reindex(columns=train1.columns | train2.columns).fillna(0)
    train2 = train2.reindex(columns=train2.columns | train1.columns).fillna(0)
    train = pd.concat([train1, train2])

    test = test.reindex(columns=test.columns | train.columns).fillna(0)
    train = train.reindex(columns=train.columns | test.columns).fillna(0)
    
    X_train = train.drop(['customer_id', 'ebb_eligible'], axis=1)
    y_train = train[['ebb_eligible']]


    X_test = test.drop(['customer_id', 'ebb_eligible'], axis=1)
    y_test = test[['ebb_eligible']]
    
    model_fit = model.fit(X_train, y_train)
    
    y_Pred = model_fit.predict(X_test)
    
    table = pd.DataFrame()
    table['customer_id'] = test['customer_id']
    table['ebb_eligible'] = y_Pred
    return table


full = predict_with_eval(set1, set2,eval_data, model)
full.to_csv('2022-4-17-lda-02.csv', index=False)