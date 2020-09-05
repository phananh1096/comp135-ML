#CV
num_folds = 3
N = y_train_df.size
n_rows_per_fold = int(np.ceil(N / float(num_folds))) * np.ones(num_folds, dtype=np.int32)
n_surplus = np.sum(n_rows_per_fold) - N
if n_surplus > 0:
    n_rows_per_fold[-n_surplus:] -= 1
assert np.allclose(np.sum(n_rows_per_fold), N)
fold_boundaries = np.hstack([0, np.cumsum(n_rows_per_fold)])
start_per_fold = fold_boundaries[:-1]
stop_per_fold = fold_boundaries[1:]

all_score_dict_list = list()
cur_alpha_score_dict_list = list()
"""
"""
## Loop over folds from 1, 2, ... K=num_folds
for fold_id in range(1, num_folds + 1):
    fold_start = start_per_fold[fold_id-1]
    fold_stop = stop_per_fold[fold_id-1]

    print("fold %d/%d of size %5d | validating on rows %5d-%5d of %5d" % (
        fold_id, num_folds, fold_stop - fold_start, fold_start, fold_stop, N))

    # Training data is everything that's not current validation fold
    x_tr_NF = np.vstack([x_NF[:fold_start], x_NF[fold_stop:]])
    y_tr_N = np.hstack([y_train_df[:fold_start], y_train_df[fold_stop:]])

    x_va_NF = x_NF[fold_start:fold_stop].copy()
    y_va_N = y_train_df[fold_start:fold_stop].copy()

    # Fit the model on current TRAINING split
    gnb = GaussianNB()
    gnb.fit(x_tr_NF, y_tr_N)

    # Evaluate on current validation fold
    yproba1_va_N = gnb.predict_proba(x_va_NF)[:,1]
    va_score_dict = dict(
        fold_id=fold_id,
        auroc=sklearn.metrics.roc_auc_score(y_va_N, yproba1_va_N),
        error_rate=sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5),
        log_loss=sklearn.metrics.log_loss(y_va_N, yproba1_va_N),
        )
    cur_alpha_score_dict_list.append(va_score_dict)
    all_score_dict_list.append(va_score_dict)

    ## Write scores to csv file only for current alpha value
    cur_alpha_cv_scores_df = pd.DataFrame(cur_alpha_score_dict_list)
    cur_alpha_cv_scores_df.to_csv(
        os.path.join("results/P2_cv_scores_m1.csv"),
        index=False,
        float_format='%.4f',
        columns=['fold_id', 'error_rate', 'log_loss', 'auroc'])

    # ## Write scores to csv file for ALL alpha values we've tested so far
    # all_cv_scores_df = pd.DataFrame(all_score_dict_list)
    # all_cv_scores_df.to_csv(
    #     os.path.join(results_path, "all_cv_scores.csv"),
    #     index=False,
    #     float_format='%.4f',
    #     columns=['alpha', 'fold_id', 'error_rate', 'log_loss', 'auroc', 'step_size', 'did_converge', 'L1_norm_grad'])

    ## Write weights to txt file
    model_txt_path = "fold%02d_weights.txt" % (fold_id)

"""


"""_________________________LOGISTIC REGRESSION__________________________"""

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
#     gnb = LogisticRegression(max_iter=10000, solver='sag')
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
#         os.path.join("results/P2_cv_scores_m2.csv"),
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


"""_________________________SVM__________________________"""

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
#     gnb = SVC(max_iter=10000)
#     gnb.fit(x_tr_NF, y_tr_N)

#     # Evaluate on current validation fold
#     yproba1_va_N = gnb.predict(x_va_NF)
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
#         os.path.join("results/P2_cv_scores_m2.csv"),
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

"""_________________________KNN__________________________"""

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
#     gnb = KNeighborsClassifier(n_neighbors=5)
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
#         os.path.join("results/P2_cv_scores_m2.csv"),
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

"""_________________________Test data__________________________"""

