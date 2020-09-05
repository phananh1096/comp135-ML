import argparse
import sys
import os
import numpy as np
import pandas as pd
import re as re
import sklearn.metrics
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

def calc_avg_y(predict_list, num):
    y_predict = 0
    for predictions in predict_list:
        y_predict = y_predict + predictions
    return y_predict/num

def tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    return words
def bootstrap_sample(x_NF,y_NF, random_state = np.random):
    N = x_NF.shape[0]
    row_ids = random_state.choice(np.arange(N),size = N, replace = True)
    return x_NF[row_ids].copy(), y_NF[row_ids].copy()

x_test_NF = pd.read_csv('data_reviews/x_test.csv')
x_NF = pd.read_csv('data_reviews/x_train.csv')
y_train_df = pd.read_csv('data_reviews/y_train.csv')
w_e = {}

#Reading in GLOVE
glove = open('pretrained_word_embeddings/glove.6B.50d.txt', 'r')
for i in glove:
    i = i.split()
    w_e[i[0]] = np.array([float(s) for s in i[1:]])
glove.close()

FP_list = list()
FN_list = list()

#Y values
y_NF = np.array(y_train_df['is_positive_sentiment'].values.tolist())
tr_text_list = x_NF['text'].values.tolist()
processed_list = list()
test_text_list = x_test_NF['text'].values.tolist()
test_processed_list = list()

#Function to do shorten, may change later
def makelist(text_list, end_list):
    for text in text_list:
        sen_val = np.full((50,),0)
        w_len = 0
        new_t = tokenizer(text)
        sentence = list()
        for i in range(0,len(new_t)):
            #if (new_t[i] in w_e): #For without stop words
            if (new_t[i] in w_e and new_t[i] not in stop_words.ENGLISH_STOP_WORDS): 
                sentence.append(w_e.get(new_t[i], "none"))
                sen_val = sen_val + np.array(w_e.get(new_t[i], "none"))
                w_len = w_len + 1
        if w_len != 0:
            # sen_val = np.hstack([np.hstack([np.amin(sentence,axis=0),(sen_val / w_len)]),np.amax(sentence,axis=0)])
            sen_val = np.hstack([np.hstack([np.hstack([np.amin(sentence,axis=0),(sen_val / w_len)]),np.std(sentence,axis=0)]),np.amax(sentence,axis=0)])
        else:
            sen_val = np.full((200,),0)
        end_list.append(sen_val)

makelist(tr_text_list, processed_list)
makelist(test_text_list, test_processed_list)

#Polynomial feature transform
x_NF = np.array(processed_list)
poly = PolynomialFeatures(2, interaction_only=True)
X_NF = poly.fit_transform(x_NF)
x_test_NF = poly.fit_transform(np.array(test_processed_list))
print(X_NF.shape)
print(x_test_NF.shape)

#BAGGING
range1 = 2
run = 0
x_list = list()
y_list = list()
y_predict_list = list()
y_predict_proba_list = list()
for i in range(0,range1):
    x_i, y_i = bootstrap_sample(X_NF,y_NF)
    x_list.append(x_i)
    y_list.append(y_i)

"""_____Logistic Regression using GridSearchCV_____"""

#max_iter: No tuning needed  bc they all converge
#C: tuned from 0.0001, 0.01, 1, 100, 10000 and 100 was the best.
#solver: newton-cg, sag, lbfg - newton was slightly better 
gnb_params = {'C':[0.0001,0.01,1,100,10000], 'solver':['newton-cg']}

y_predict_on_train = 0
for x_NF,y_NF in zip(x_list,y_list):
    run = run +1
    gnb = LogisticRegression(max_iter = 10000)
    gnb_cv = GridSearchCV(gnb,gnb_params,cv = 3, return_train_score = True)
    gnb_cv.fit(x_NF,np.ravel(y_NF))
    print("Run no:  %d" %run)
    print(1-gnb_cv.cv_results_.get("split0_train_score"))
    print(1-gnb_cv.cv_results_.get("split1_train_score"))
    print(1-gnb_cv.cv_results_.get("split2_train_score"))
    print(1-gnb_cv.cv_results_.get("split0_test_score"))
    print(1-gnb_cv.cv_results_.get("split1_test_score"))
    print(1-gnb_cv.cv_results_.get("split2_test_score"))
    y_predict_proba_list.append(gnb_cv.predict_proba(x_test_NF)[:,1])
    y_predict_list.append(gnb_cv.predict(x_test_NF))
    y_predict_on_train = gnb_cv.predict(x_NF)
    for x,y in zip(y_predict_on_train,y_list[1]):
        if x == y:
            correct = correct + 1
        elif x < y:
            FN_list.append(x)
        elif x > y:
            FP_list.append(x)
        print("FP no.: %d"%len(FP_list))
        print("FN no.: %d"%len(FN_list))
        print("Correct no.: %d"%correct)

yproba1_te_N = calc_avg_y(y_predict_proba_list,range1)
yproba1_te_N_predict = calc_avg_y(y_predict_list,range1)



"""_________To get testing metrics________"""
# gnb = LogisticRegression(max_iter = 10000)
# gnb_cv = GridSearchCV(gnb,gnb_params,cv = 3, return_train_score = True)
# gnb_cv.fit(X_NF,np.ravel(y_NF))
# print("Running metrics LR:")
# metrics=list()
# for i in range(0,5):
#     curr_metrics = dict(
#         split0_train=(1-gnb_cv.cv_results_.get("split0_train_score")[i]),
#         split1_train=(1-gnb_cv.cv_results_.get("split1_train_score")[i]),
#         split2_train=(1-gnb_cv.cv_results_.get("split2_train_score")[i]),  
#         split0_test=(1-gnb_cv.cv_results_.get("split0_test_score")[i]),
#         split1_test=(1-gnb_cv.cv_results_.get("split1_test_score")[i]),
#         split2_test=(1-gnb_cv.cv_results_.get("split2_test_score")[i])
#         )
#     metrics.append(curr_metrics)
# metrics=pd.DataFrame(metrics)
# metrics.to_csv(
#     os.path.join("P2_LR.csv"),
#     index=False,
#     float_format='%.4f',
#     columns=['split0_train','split1_train','split2_train','split0_test','split1_test','split2_test'])

#Obtain FP/FN values

correct = 0


np.savetxt("P2_LR_FP.txt", FP_list, delimiter=",")
np.savetxt("P2_LR_FN.txt", FN_list, delimiter=",")

"""_____Naive Bayes using GridSearchCV_____"""

# gnb_params = {'var_smoothing':[0, 0.1, 1, 10, 100]}
# for x_NF,y_NF in zip(x_list,y_list):
#     run = run +1
#     gnb = GaussianNB()
#     gnb_cv = GridSearchCV(gnb,gnb_params,cv = 3)
#     gnb_cv.fit(x_NF,np.ravel(y_NF))
#     print("Run no:  %d" %run)
#     print(1-gnb_cv.cv_results_.get("split0_test_score"))
#     print(1-gnb_cv.cv_results_.get("split1_test_score"))
#     print(1-gnb_cv.cv_results_.get("split2_test_score"))
#     y_predict_proba_list.append(gnb_cv.predict_proba(x_test_NF)[:,1])
#     y_predict_list.append(gnb_cv.predict(x_test_NF))

# yproba1_te_N = calc_avg_y(y_predict_proba_list,range1)
# yproba1_te_N_predict = calc_avg_y(y_predict_list,range1)


# np.savetxt("P2_yproba1_test_proba.txt", yproba1_te_N, delimiter=",")
# np.savetxt("P2_yproba1_test_predict.txt", yproba1_te_N_predict, delimiter=",")