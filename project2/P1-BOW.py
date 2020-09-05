import argparse
import sys
import os
import numpy as np
import pandas as pd
import re as re
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower()
    return words

x_NF = pd.read_csv('data_reviews/x_train.csv')
x_test_NF = pd.read_csv('data_reviews/x_test.csv')
y_NF = pd.read_csv('data_reviews/y_train.csv')
y_train_df = np.array(y_NF['is_positive_sentiment'].values.tolist())

tr_text_list = x_NF['text'].values.tolist()
processed_list = list()

#Preprocessing to remove digits, punctuations and convert all to lower case
for text in tr_text_list:
	new_t = tokenizer(text)
	processed_list.append(new_t)

#use_idf=False
vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True, smooth_idf=True)
X = vectorizer.fit_transform(processed_list)
x_NF = X.toarray()

"""_____Logistic Regression using GridSearchCV_____"""
print(X)
FP_list = list()
FN_list = list()
gnb_params = {'n_neighbors':[1,3]}

gnb = KNeighborsClassifier()
gnb_cv = GridSearchCV(gnb,gnb_params,cv = 3, return_train_score = True)
gnb_cv.fit(x_NF,np.ravel(y_NF))
print("Run no: ")
print(1-gnb_cv.cv_results_.get("split0_train_score"))
print(1-gnb_cv.cv_results_.get("split1_train_score"))
print(1-gnb_cv.cv_results_.get("split2_train_score"))
print(1-gnb_cv.cv_results_.get("split0_test_score"))
print(1-gnb_cv.cv_results_.get("split1_test_score"))
print(1-gnb_cv.cv_results_.get("split2_test_score"))
# y_predict_proba_list.append(gnb_cv.predict_proba(x_test_NF)[:,1])
# yproba1_te_N_predict=gnb_cv.predict(x_test_NF)
y_predict_on_train = gnb_cv.predict(x_NF)
for i in range(0,len(y_predict_on_train)):
    if y_predict_on_train[i] < y_NF[i]:
        print("Found FN: %s" %tr_text_list[indexes.item(i)]) 
        FN_list.append(tr_text_list[indexes.item(i)])
    if y_predict_on_train[i] > y_NF[i]:
        print("Found FP: %s" %tr_text_list[indexes.item(i)])  
        FP_list.append(tr_text_list[indexes.item(i)])
print("FP no.: %d"%len(FP_list))
print("FN no.: %d"%len(FN_list))
print("Correct no.: %d"%correct)

np.savetxt("P1_KNN_FP.txt", FP_list, delimiter=",",fmt="%s")
np.savetxt("P1_KNN_FN.txt", FN_list, delimiter=",",fmt="%s")


# ## Determine how to allocate contiguous rows to the K folds
# # Try to have as evenly sized folds as possible
# num_folds = 3
# N = y_train_df.size
# n_rows_per_fold = int(np.ceil(N / float(num_folds))) * np.ones(num_folds, dtype=np.int32)
# n_surplus = np.sum(n_rows_per_fold) - N
# if n_surplus > 0:
#     n_rows_per_fold[-n_surplus:] -= 1
# assert np.allclose(np.sum(n_rows_per_fold), N)
# fold_boundaries = np.hstack([0, np.cumsum(n_rows_per_fold)])
# start_per_fold = fold_boundaries[:-1]
# stop_per_fold = fold_boundaries[1:]

# all_score_dict_list = list()
# cur_alpha_score_dict_list = list()

# ## Loop over folds from 1, 2, ... K=num_folds
# for fold_id in range(1, num_folds + 1):
#     fold_start = start_per_fold[fold_id-1]
#     fold_stop = stop_per_fold[fold_id-1]

#     print("fold %d/%d of size %5d | validating on rows %5d-%5d of %5d" % (
#         fold_id, num_folds, fold_stop - fold_start, fold_start, fold_stop, N))

#     # Training data is everything that's not current validation fold
#     x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
#     y_tr_N = np.hstack([y_train_df[:fold_start], y_train_df[fold_stop:]])

#     x_va_NF = x_NF[fold_start:fold_stop].copy()
#     y_va_N = y_train_df[fold_start:fold_stop].copy()

#     # Fit the model on current TRAINING split
#     gnb = ComplementNB()
#     gnb.fit(x_tr_NF, y_tr_N)

#     # Evaluate on current validation fold
#     yproba1_va_N = gnb.predict_proba(x_va_NF)[:,1]
#     va_score_dict = dict(
#         fold_id=fold_id,
#         auroc=sklearn.metrics.roc_auc_score(y_va_N, yproba1_va_N),
#         error_rate=sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5),
#         log_loss=sklearn.metrics.log_loss(y_va_N, yproba1_va_N),
#         )
#     cur_alpha_score_dict_list.append(va_score_dict)
#     all_score_dict_list.append(va_score_dict)

#     ## Write scores to csv file only for current alpha value
#     cur_alpha_cv_scores_df = pd.DataFrame(cur_alpha_score_dict_list)
#     cur_alpha_cv_scores_df.to_csv(
#         os.path.join("results/P1_cv_scores_m1.csv"),
#         index=False,
#         float_format='%.4f',
#         columns=['fold_id', 'error_rate', 'log_loss', 'auroc'])

#     # ## Write scores to csv file for ALL alpha values we've tested so far
#     # all_cv_scores_df = pd.DataFrame(all_score_dict_list)
#     # all_cv_scores_df.to_csv(
#     #     os.path.join(results_path, "all_cv_scores.csv"),
#     #     index=False,
#     #     float_format='%.4f',
#     #     columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

#     ## Write weights to txt file
#     model_txt_path = "fold%02d_weights.txt" % (fold_id)

#     # gnb.write_to_txt_file(os.path.join(results_path, model_txt_path))


# """_________________________LOGISTIC REGRESSION__________________________"""

# # for fold_id in range(1, num_folds + 1):
# #     fold_start = start_per_fold[fold_id-1]
# #     fold_stop = stop_per_fold[fold_id-1]

# #     print("fold %d/%d of size %5d | validating on rows %5d-%5d of %5d" % (
# #         fold_id, num_folds, fold_stop - fold_start, fold_start, fold_stop, N))

# #     # Training data is everything that's not current validation fold
# #     x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
# #     y_tr_N = np.hstack([y_train_df[:fold_start], y_train_df[fold_stop:]])

# #     x_va_NF = x_NF[fold_start:fold_stop].copy()
# #     y_va_N = y_train_df[fold_start:fold_stop].copy()

# #     # Fit the model on current TRAINING split
# #     gnb = LogisticRegression(max_iter=10000, solver='sag')
# #     gnb.fit(x_tr_NF, y_tr_N)

# #     # Evaluate on current validation fold
# #     yproba1_va_N = gnb.predict_proba(x_va_NF)[:,1]
# #     va_score_dict = dict(
# #         fold_id=fold_id,
# #         auroc=sklearn.metrics.roc_auc_score(y_va_N, yproba1_va_N),
# #         error_rate=sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5),
# #         log_loss=sklearn.metrics.log_loss(y_va_N, yproba1_va_N),
# #         )
# #     cur_alpha_score_dict_list.append(va_score_dict)
# #     all_score_dict_list.append(va_score_dict)

# #     ## Write scores to csv file only for current alpha value
# #     cur_alpha_cv_scores_df = pd.DataFrame(cur_alpha_score_dict_list)
# #     cur_alpha_cv_scores_df.to_csv(
# #         os.path.join("results/P1_cv_scores_m2.csv"),
# #         index=False,
# #         float_format='%.4f',
# #         columns=['fold_id', 'error_rate', 'log_loss', 'auroc'])

# #     # ## Write scores to csv file for ALL alpha values we've tested so far
# #     # all_cv_scores_df = pd.DataFrame(all_score_dict_list)
# #     # all_cv_scores_df.to_csv(
# #     #     os.path.join(results_path, "all_cv_scores.csv"),
# #     #     index=False,
# #     #     float_format='%.4f',
# #     #     columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

# #     ## Write weights to txt file
# #     model_txt_path = "fold%02d_weights.txt" % (fold_id)

# #     # gnb.write_to_txt_file(os.path.join(results_path, model_txt_path))

# # """_________________________SVM Linear Classification__________________________"""

# # for fold_id in range(1, num_folds + 1):
# #     fold_start = start_per_fold[fold_id-1]
# #     fold_stop = stop_per_fold[fold_id-1]

# #     print("fold %d/%d of size %5d | validating on rows %5d-%5d of %5d" % (
# #         fold_id, num_folds, fold_stop - fold_start, fold_start, fold_stop, N))

# #     # Training data is everything that's not current validation fold
# #     x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
# #     y_tr_N = np.hstack([y_train_df[:fold_start], y_train_df[fold_stop:]])

# #     x_va_NF = x_NF[fold_start:fold_stop].copy()
# #     y_va_N = y_train_df[fold_start:fold_stop].copy()

# #     # Fit the model on current TRAINING split
# #     gnb = SVC(max_iter=10000)
# #     gnb.fit(x_tr_NF, y_tr_N)

# #     # Evaluate on current validation fold
# #     yproba1_va_N = gnb.predict(x_va_NF)
# #     va_score_dict = dict(
# #         fold_id=fold_id,
# #         auroc=sklearn.metrics.roc_auc_score(y_va_N, yproba1_va_N),
# #         error_rate=sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5),
# #         log_loss=sklearn.metrics.log_loss(y_va_N, yproba1_va_N),
# #         )
# #     cur_alpha_score_dict_list.append(va_score_dict)
# #     all_score_dict_list.append(va_score_dict)

# #     ## Write scores to csv file only for current alpha value
# #     cur_alpha_cv_scores_df = pd.DataFrame(cur_alpha_score_dict_list)
# #     cur_alpha_cv_scores_df.to_csv(
# #         os.path.join("results/P1_cv_scores_m3.csv"),
# #         index=False,
# #         float_format='%.4f',
# #         columns=['fold_id', 'error_rate', 'log_loss', 'auroc'])

# #     # ## Write scores to csv file for ALL alpha values we've tested so far
# #     # all_cv_scores_df = pd.DataFrame(all_score_dict_list)
# #     # all_cv_scores_df.to_csv(
# #     #     os.path.join(results_path, "all_cv_scores.csv"),
# #     #     index=False,
# #     #     float_format='%.4f',
# #     #     columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

# #     ## Write weights to txt file
# #     model_txt_path = "fold%02d_weights.txt" % (fold_id)

# #     # gnb.write_to_txt_file(os.path.join(results_path, model_txt_path))

# """_________________________Test data__________________________"""

test_text_list = x_test_NF['text'].values.tolist()
test_processed_list = list()
for text in test_text_list:
	new_t = tokenizer(text)
	test_processed_list.append(new_t)
X = vectorizer.transform(test_processed_list)
x_NF = X.toarray()
# yproba1_va_N = gnb.predict_proba(x_NF)[:,1]
# np.savetxt("yproba1_test.txt", yproba1_va_N, delimiter=",")
