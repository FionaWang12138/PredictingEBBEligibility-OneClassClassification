######################################################
##                                                  ##
##      TEAM 10 - MODIFIED LOGISTIC REGRESSION      ##
##                                                  ##
######################################################

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.preprocessing import normalize
import random

df1 = pd.read_csv("eval_set_v5.csv")
df2 = pd.read_csv("ebb_set1_v5.csv")
df3 = pd.read_csv("ebb_set2_v5.csv")

df = pd.concat([ df1, df2, df3])

df.drop('language_preference', inplace=True, axis=1)

df.fillna(0, inplace=True)

#shuffles
df = df.sample(n=df.shape[0], random_state=1)

#pop out customer id
customer_id = df.pop('customer_id')

#change Nan to 0, and ebb_eligible column = 0 if no label is given
df.loc[df['ebb_eligible'] != 1, 'ebb_eligible'] = 0 

s = df.pop('ebb_eligible')

X = np.array(df)

#scales data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

w = np.ones(X.shape[1]) 
w = np.array(w, dtype=np.float128)
b = 1

s=np.array(s)

alpha = 0.01

l2norms = []

for epoch in range (0, 100): 
    for i in range (0, X.shape[0]):
    
        exp_wx =  np.array( np.exp( -1 * np.dot (w, X[i]) ), dtype=np.float128 )

        part1 = (s[i] - 1)/(b * b  + exp_wx )
        part2 = 1 /(1 + b * b  + exp_wx)


        partial_der_1 = (part1+part2)*exp_wx*X[i]
        part3=(1 - s[i])/(b*b + exp_wx)
        partial_der_2 = (part3 - part2)*2*b


        w = np.add (w,  alpha * partial_der_1)
        b = b + alpha *  partial_der_2

        w = np.around( w , decimals= 100) 
        b = np.around( b , decimals= 100)
    
        if i%1000 == 0:
            l2norms.append(np.linalg.norm(partial_der_1))

    if epoch%5 == 0:
        print(epoch)
        plt.plot(l2norms)
        plt.ylabel("L2-norms (of gradient)")
        plt.xlabel('Epochs')
        plt.title("alpha = 0.01")

        rolling = pd.DataFrame(l2norms).rolling(200).mean()
        plt.plot(rolling)
        plt.show()

        print(w)
        print(b)

#following pseudo code from this https://www.youtube.com/watch?v=uk6SlTzfbUY
plt.plot(l2norms)
plt.ylabel("L2-norms (of gradient)")
plt.xlabel('Epochs')
plt.title("alpha = 0.01")
rolling = pd.DataFrame(l2norms).rolling(200).mean()
plt.plot(rolling)
plt.show()

import sys

s_hat = []
for i in range (0, X.shape[0]):
    
    dot = np.matmul(X[i], w)
    if (dot > 0):
        exp_wx = np.exp (-1 * dot )
        denomenator =1 + b*b + exp_wx 
        s_hat.append(np.divide(1, denomenator))
    else:
        exp_wx = np.exp (dot )
        denomenator =exp_wx + b*b*exp_wx + 1 
        s_hat.append(np.divide(exp_wx, denomenator))

c_hat = 1/(1 + b*b)

y_hat = np.divide(s_hat, c_hat)
y_hat = y_hat.round(decimals= 0)
y_hat = y_hat.astype(int)

result = np.column_stack((customer_id.to_numpy().reshape(1, -1)[0], y_hat))
result_df = pd.DataFrame(result, columns=['customer_id', 'ebb_eligible'])
eval_customers = pd.read_csv("/data/team10/data/eval_set.csv").pop('customer_id').values.tolist()
result_df = result_df.loc[result_df['customer_id'].isin(eval_customers)]
print(result_df)
result_df.to_csv('2022-04-17_final.csv', index = False)

occurrencesPos = np.count_nonzero(result_df.ebb_eligible == 1)
occurrencesNeg = np.count_nonzero(result_df.ebb_eligible == 0)
print(occurrencesNeg/(occurrencesNeg + occurrencesPos) )

a = np.array(pd.read_csv('2022-04-17_final.csv').sort_values('customer_id'))[:,1]
b = np.array(pd.read_csv('2022-04-14.csv').sort_values('customer_id'))[:,1]
np.linalg.norm(a - b, ord=1)
