from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import pandas as pd
from sklearn.impute import SimpleImputer
import multiprocessing as mp
from modeling.modeling_utils import find_optimal_threshold, eval_classification_preds
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from copy import deepcopy


class StratifiedCV(object):
    def __init__(self, cv_folds, classification_model, cpus_to_use=5, random_seed=1984, verbose=1):
        self.cv_folds = cv_folds
        self.classification_model = classification_model
        self.random_seed = random_seed
        self.cpus_to_use = cpus_to_use # this is used only for the modeling part
        self.folds_indices = dict()
        self.verbose = verbose
        self.eval_measures_train_df = None
        self.eval_measures_test_df = None
        self.row_lvl_preds_df = None
        self.feature_importance = None

    def split_data_to_folds(self, x_data, y_data):
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        for i, (train_index, test_index) in enumerate(skf.split(x_data, y_data)):
            self.folds_indices[i] = {'train': train_index, 'test': test_index}
        return self.folds_indices

    def run_model_over_all_folds(self, x_data, y_data):
        # since we run the model over a few iterations, we do it in a mp way. We first create the input for each cpu
        input_for_pool = list()
        for fold_idx, train_test_dict in self.folds_indices.items():
            train_idx = train_test_dict['train']
            test_idx = train_test_dict['test']
            x_train, x_test = x_data.iloc[train_idx].copy(), x_data.iloc[test_idx].copy()
            y_train, y_test = y_data.iloc[train_idx].copy(), y_data.iloc[test_idx].copy()
            # filling the missing values with the mean
            imputer = SimpleImputer(strategy='mean')

            # Fit the imputer on the training data and transform both train and test sets
            x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns).copy()
            x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns).copy()
            input_for_pool.append((fold_idx, x_train_imputed, y_train, x_test_imputed, y_test))
        # now we can run the multiprocess function
        if self.verbose > 0:
            print(f"Multiprocessing of {self.cv_folds} folds starts now. Wish me luck!", flush=True)
        pool = mp.Pool(processes=self.cpus_to_use)
        with pool as pool:
            results = pool.starmap(self._run_model_over_a_single_fold, input_for_pool)
        # merging the list of dicts into one dict
        eval_measures_train = dict()
        eval_measures_test = dict()
        predictions = dict()
        feature_importance = defaultdict(float)
        for fold_idx, train_res, test_res, cur_pred_per_page, cur_feature_importance in results:
            eval_measures_train[fold_idx] = train_res
            eval_measures_test[fold_idx] = test_res
            predictions.update(cur_pred_per_page)
            for cfi, value in cur_feature_importance.items():
                feature_importance[cfi] += value
        # updating the predictions per instance and the feature importance
        row_lvl_preds_df = pd.DataFrame.from_dict(predictions, orient='index')
        # we also add to this df, the
        self.row_lvl_preds_df = row_lvl_preds_df
        #self.feature_importance = feature_importance
        eval_measures_train_df = pd.DataFrame.from_dict(eval_measures_train, orient='index')
        eval_measures_test_df = pd.DataFrame.from_dict(eval_measures_test, orient='index')
        # append the average, std and CI values as a new row to the DataFrame
        eval_measures_train_df.loc['mean'] = eval_measures_train_df.mean()
        eval_measures_train_df.loc['std'] = eval_measures_train_df.std()
        ci_column_train = [stats.t.interval(0.95, self.cv_folds, loc=cur_mean, scale=cur_std)
                           for cur_mean, cur_std in zip(eval_measures_train_df.loc['mean'],
                                                        eval_measures_train_df.loc['std'])]
        # rounding the CI to 2 digits only and adding them as a new row
        eval_measures_train_df.loc['ci'] = [(round(c[0], 3), round(c[1], 3)) for c in ci_column_train]

        eval_measures_test_df.loc['mean'] = eval_measures_test_df.mean()
        eval_measures_test_df.loc['std'] = eval_measures_test_df.std()
        ci_column_test = [stats.t.interval(0.95, self.cv_folds, loc=cur_mean, scale=cur_std)
                          for cur_mean, cur_std in zip(eval_measures_test_df.loc['mean'],
                                                       eval_measures_test_df.loc['std'])]
        # rounding the CI to 2 digits only and adding them as a new row
        eval_measures_test_df.loc['ci'] = [(round(c[0], 3), round(c[1], 3)) for c in ci_column_test]
        self.eval_measures_train_df = eval_measures_train_df
        self.eval_measures_test_df = eval_measures_test_df
        return 0

    def _run_model_over_a_single_fold(self, fold_idx, x_data_train, y_data_train, x_data_test, y_data_test):
        # making a local version of the model to run
        loc_model = deepcopy(self.classification_model)
        loc_model.fit(x_data_train, list(y_data_train))

        # Evaluate the Model
        # finding the optimal threshold
        y_train_pred_proba = loc_model.predict_proba(x_data_train)[:, 1]
        macro_optimal_th = find_optimal_threshold(y_true=y_data_train, predicted_proba=y_train_pred_proba, average='macro')
        binary_optimal_th = find_optimal_threshold(y_true=y_data_train, predicted_proba=y_train_pred_proba, average='binary')

        # Evaluate the Model
        y_test_pred_proba = loc_model.predict_proba(x_data_test)[:, 1]
        y_pred_test = loc_model.predict(x_data_test)
        eval_dict_test = eval_classification_preds(true_values=list(y_data_test), preds=list(y_pred_test),
                                                   preds_proba=y_test_pred_proba, macro_optimal_th=macro_optimal_th,
                                                   binary_optimal_th=binary_optimal_th)
        eval_dict_train = eval_classification_preds(true_values=list(y_data_train), preds=list(loc_model.predict(x_data_train)),
                                                    preds_proba=y_train_pred_proba, macro_optimal_th=macro_optimal_th,
                                                    binary_optimal_th=binary_optimal_th)
        # saving the prediction per page. We now save a lot more information about each page (e.g., true value)
        preds_optimal_th_binary = (y_test_pred_proba > binary_optimal_th).astype(int)
        cur_pred_per_page = {page_id: {'fold': fold_idx, 'pred_proba': pred_proba,
                                       'opt_th_pred':  pred_opt_th, 'true_value': true_value}
                             for page_id, pred_proba, pred_opt_th, true_value in
                             zip(y_data_test.index, y_test_pred_proba, list(preds_optimal_th_binary), list(y_data_test))}
        # feature importance process (will only work for GBT and RF models, not LR for example)
        try:
            cur_feature_importance = {f_name: f_importance for f_name, f_importance in
                                      zip(loc_model.feature_names_in_, loc_model.feature_importances_)}
        except AttributeError:
            cur_feature_importance = None
        if self.verbose > 1:
            print(f"Fold {fold_idx} has ended and returns its evaluation measures.", flush=True)
        return fold_idx, eval_dict_train, eval_dict_test, cur_pred_per_page, cur_feature_importance
