import pandas as pd
from datetime import datetime
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter
from collections import defaultdict
import re
import matplotlib.pyplot as plt

def activation():
    df1 = pd.read_csv("/data/team10/data/activations_ebb_set1.csv")
    df2 = pd.read_csv("/data/team10/data/activations_ebb_set1.csv")
    df3 = pd.read_csv("/data/team10/data/activations_ebb_set1.csv")
    df_activation = pd.concat([df1,df2,df3])
    rc = df_activation['activation_channel']
    rc = list(rc)
    rc_count = Counter(rc)
    tots = sum(rc_count.values())
    x = []
    y = []
    for k,v in rc_count.items():
        x.append(rc_count[k] / tots * 100)
        y.append(k)

    plt.rcParams['figure.figsize'] = [10,10]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.pie(x,autopct='%1.1f%%')
    plt.legend(y)
    plt.title('Percentage of activation channels')
    plt.show()

def deactivation():
    df1 = pd.read_csv("/data/team10/data/deactivations_ebb_set1.csv")
    df2 = pd.read_csv("/data/team10/data/deactivations_ebb_set1.csv")
    df3 = pd.read_csv("/data/team10/data/deactivations_ebb_set1.csv")
    df_deactivation = pd.concat([df1,df2,df3])
    d = df_deactivation['deactivation_reason']
    deactivation_reasons_list = list(df_deactivation['deactivation_reason'])
    deactivation_counter = Counter(deactivation_reasons_list)
    total = sum(deactivation_counter.values())
    x = []
    y = []
    for k,v in deactivation_counter.items():
        frac = deactivation_counter[k]/total
        x.append(frac*100)
        y.append(k)
    y_l = np.arange(len(y))

    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 10
    plt.barh(y,x,color='blue')
    plt.xlabel('Percentage')
    plt.ylabel('Reason for deactivation')
    plt.title('Percentage of Reasons for deactivation')
    plt.show()

def interactions_cleaned():
    df = pd.read_csv('/data/team10/data/interactions_cleaned.csv')
    df = df[df['reason'] == 'Emergency Broadband Benefit Call']
    disposition  = list(df['disposition'])
    disposition_count = Counter(disposition)
    disposition_names = list(disposition_count.keys())
    disposition_counts = list(disposition_count.values())
    disposition_names = [str(i) for i in disposition_names]
    plt.rcParams['figure.figsize'] = [5,5]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    plt.rcParams['font.size'] = 10
    plt.xticks(rotation=90)
    plt.xlabel('Disposition')
    plt.ylabel("Frequency of EBB Call")
    plt.bar(disposition_names,disposition_counts,color='red')
    plt.show()

def network_ebb():
    plt.rcParams['figure.figsize'] = [100,100]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 30
    df_1 = pd.read_csv("/data/team10/data/network_ebb_set1.csv")
    df_2 = pd.read_csv("/data/team10/data/network_ebb_set2.csv")
    df_3 = pd.read_csv("/data/team10/data/network_eval_set.csv")
    df_network = pd.concat([df_1,df_2,df_3])
    df_network = df_network.set_index(['customer_id','date'])
    df_network.loc['e3da0f9e3076cdf64decd7cb79dc174a51f874a3'].plot.bar(figsize=(50,30),subplots=True,fontsize=20)
    plt.show()

def phone_ebb():
    df_1 = pd.read_csv("/data/team10/data/phone_data_ebb_set1.csv")
    df_2 = pd.read_csv("/data/team10/data/phone_data_ebb_set2.csv")
    df_3 = pd.read_csv("/data/team10/data/phone_data_eval_set.csv")
    df_phone = pd.concat([df_1,df_2,df_3])
    df_phone = df_phone.set_index(['customer_id','timestamp'])
    df_network.loc['db830e033aeaf092be7a8ded20e9d1da2aca2ad8'].plot.bar(title='Time Series Data Analysis of Phone Dataset',figsize=(50,30),subplots=True,fontsize=20)
    plt.show()

if __name__ == '__main__':
    activation()
    deactivation()
    interactions_cleaned()
    network_ebb()
    phone_ebb()