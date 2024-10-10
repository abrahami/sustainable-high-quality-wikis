# the purpose of this code is to model the data in hand using CV, while taking into account time aspects
import pandas as pd
import numpy as np
from os.path import join as opj
import json
from collections import Counter, defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.impute import SimpleImputer
from modeling.modeling_utils import find_optimal_threshold, eval_classification_preds
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from modeling.bootstrapping import Bootstrapping
from modeling.stratified_cv import StratifiedCV

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

model_version = 20.2
target_column = 'is_sustainable_conservative_approach' #is_sustainable'
use_only_meta_features = False
model_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/models'
seed = 1984
classification_model = GradientBoostingClassifier(n_estimators=100)
cv_process = 'expand_dataset_based_promotion_year'#'per_period_train_test' #'per_period_train_test' # 'cv_split_by_time' #'expand_dataset_based_promotion_year'


def split_years_to_cv_buckets(cases_per_year, splits=5):
    # Step 1: Sort the dictionary by years
    sorted_cases = dict(sorted(cases_per_year.items()))

    # Step 2: Calculate the total number of cases
    total_cases = sum(sorted_cases.values())
    target_cases_per_bucket = total_cases / splits  # We want N "splits"

    # Step 3: Distribute years into "splits"
    buckets = defaultdict(dict)
    current_bucket = 0
    current_sum = 0

    for year, cases in sorted_cases.items():
        if current_sum + cases > target_cases_per_bucket * 1.2 and current_bucket < splits-1:
            # Move to the next bucket if adding the current year exceeds the target
            current_bucket += 1
            current_sum = 0  # Reset the sum for the new bucket

        buckets[current_bucket][year] = cases
        current_sum += cases

    # buckets now include the important information. However, we only need the mapping of years to buckets
    year_to_bucket_mapping = dict()
    for bucket_idx, bucket_years in buckets.items():
        for year, count in bucket_years.items():
            year_to_bucket_mapping[year] = bucket_idx
    return year_to_bucket_mapping


def cv_folds_modeling_given_splits(x_df, y_col, cv_split):
    eval_measures_train = dict()
    eval_measures_test = dict()
    predictions = dict()
    cv_unique_splits = sorted(list(set(cv_split)))
    for cv_index, cur_cv_num in enumerate(cv_unique_splits):
        test_flag = [True if cp == cur_cv_num else False for cp in cv_split]
        train_flag = [False if cp == cur_cv_num else True for cp in cv_split]
        x_train, x_test = x_df[train_flag], x_df[test_flag]
        y_train, y_test = y_col[train_flag], y_col[test_flag]

        # filling the missing values with the mean
        imputer = SimpleImputer(strategy='mean')
        # Fit the imputer on the training data and transform both train and test sets
        x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)
        x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_train.columns)

        model = classification_model
        model.fit(x_train_imputed, y_train)
        # finding the optimal threshold
        y_train_pred_proba = model.predict_proba(x_train_imputed)[:, 1]
        macro_optimal_th = find_optimal_threshold(y_true=y_train, predicted_proba=y_train_pred_proba, average='macro')
        binary_optimal_th = find_optimal_threshold(y_true=y_train, predicted_proba=y_train_pred_proba, average='binary')

        # Evaluate the Model (test data)
        y_test_pred_proba = model.predict_proba(x_test_imputed)[:, 1]
        y_pred_test = model.predict(x_test_imputed)
        eval_dict_test = eval_classification_preds(true_values=y_test, preds=y_pred_test,
                                                   preds_proba=y_test_pred_proba, macro_optimal_th=macro_optimal_th,
                                                   binary_optimal_th=binary_optimal_th)
        eval_dict_train = eval_classification_preds(true_values=y_train, preds=model.predict(x_train_imputed),
                                                    preds_proba=y_train_pred_proba, macro_optimal_th=macro_optimal_th,
                                                    binary_optimal_th=binary_optimal_th)
        eval_measures_train[cv_index] = eval_dict_train
        eval_measures_test[cv_index] = eval_dict_test
        cur_pred_per_page = {page_id: pred for page_id, pred in zip(y_test.index, y_test_pred_proba)}
        predictions.update(cur_pred_per_page)
        print(f"Fold {cv_index} has ended.")
    # end of the cv loop
    eval_measures_train_df = pd.DataFrame.from_dict(eval_measures_train, orient='index')
    eval_measures_test_df = pd.DataFrame.from_dict(eval_measures_test, orient='index')
    # append the average values as a new row to the DataFrame
    eval_measures_train_df.loc['mean'] = eval_measures_train_df.mean()
    eval_measures_train_df.loc['std'] = eval_measures_train_df.std()
    eval_measures_test_df.loc['mean'] = eval_measures_test_df.mean()
    eval_measures_test_df.loc['std'] = eval_measures_test_df.std()
    return eval_measures_train_df, eval_measures_test_df, predictions


def train_test_modeling(x_df, y_col, test_perc=0.3):
    # In this function, we split the data into train/test and run a model (stratified). This is used to
    # split the data into years, and then in each year (or multiple sequantial years), run a train-test process
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_col, stratify=y_col, test_size=test_perc)
    # filling the missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    # Fit the imputer on the training data and transform both train and test sets
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x_train.columns)

    model = classification_model
    model.fit(x_train_imputed, y_train)
    # finding the optimal threshold
    y_train_pred_proba = model.predict_proba(x_train_imputed)[:, 1]
    macro_optimal_th = find_optimal_threshold(y_true=y_train, predicted_proba=y_train_pred_proba, average='macro')
    binary_optimal_th = find_optimal_threshold(y_true=y_train, predicted_proba=y_train_pred_proba, average='binary')
    # Evaluate the Model
    y_test_pred_proba = model.predict_proba(x_test_imputed)[:, 1]
    y_pred = model.predict(x_test_imputed)

    eval_dict_test = eval_classification_preds(true_values=y_test, preds=y_pred,
                                               preds_proba=y_test_pred_proba, macro_optimal_th=macro_optimal_th,
                                               binary_optimal_th=binary_optimal_th)
    eval_dict_train = eval_classification_preds(true_values=y_train, preds=model.predict(x_train_imputed),
                                                preds_proba=y_train_pred_proba, macro_optimal_th=macro_optimal_th,
                                                binary_optimal_th=binary_optimal_th)
    cur_pred_per_page = {page_id: pred for page_id, pred in zip(y_test.index, y_test_pred_proba)}
    # end of the cv loop
    # append the average values as a new row to the DataFrame
    return eval_dict_train, eval_dict_test, cur_pred_per_page


def time_sensitive_cv_methods():
    # loading the data for modeling
    data_df = pd.read_csv(opj(model_folder, str(model_version), 'modeling_df.csv'))
    # loading the json file of the model (configurations)
    model_params_f_name = opj(model_folder, str(model_version), 'model_params.json')
    with open(model_params_f_name, 'r', encoding='utf-8') as infile:
        model_info_dict = json.load(infile)
        infile.close()
    usecase = model_info_dict['usecase']
    max_year_to_include = model_info_dict['max_year_to_include']

    # Set the first column as the index
    data_df.set_index(data_df.columns[0], inplace=True)
    data_df[target_column] = data_df[target_column].astype(int)
    # filtering too new articles
    if max_year_to_include is not None:
        data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied. Filtered dataset shape: {data_df.shape}.", flush=True)
    # we now split the data into year buckets
    cnt_per_year = dict(sorted(Counter(data_df['TIME_promotion_year']).items()))
    year_to_bucket_mapping = split_years_to_cv_buckets(cnt_per_year, splits=5)
    # in case we want to split each year to a bucket
    # year_to_bucket_mapping = {year: idx for idx, (year, _) in enumerate(cnt_per_year.items())}
    cv_split = [year_to_bucket_mapping[promotion_year] for promotion_year in data_df['TIME_promotion_year']]
    data_df[target_column] = data_df[target_column].astype(int)
    # all features that start with 'TIME_' should not be included in the modeling df
    x_columns = [c for c in data_df.columns if c != target_column and not c.startswith('TIME_')]
    # if we want to include specific sets of features (e.g., meta)
    if use_only_meta_features:
        x_columns = [c for c in x_columns if c.startswith('META_')]
    x_df = data_df[x_columns].copy()
    y_col = data_df[target_column]
    if cv_process == 'cv_split_by_time':
        eval_measures_df_train, eval_measures_df_test, _ = (
            cv_folds_modeling_given_splits(x_df=x_df, y_col=y_col, cv_split=cv_split))
    elif cv_process == 'per_period_train_test':
        cv_unique_splits = sorted(list(set(cv_split)))
        eval_measures_train = dict()
        eval_measures_test = dict()
        predictions = dict()
        for cv_index, cur_cv_num in enumerate(cv_unique_splits):
            cur_split_flag = [True if cp == cur_cv_num else False for cp in cv_split]
            cur_x_df = x_df[cur_split_flag]
            cur_y_col = y_col[cur_split_flag]
            cur_eval_measures_train, cur_eval_measures_test, cur_pred_per_page = train_test_modeling(x_df=cur_x_df,
                                                                                                     y_col=cur_y_col,
                                                                                                     test_perc=0.3)
            eval_measures_train[cv_index] = cur_eval_measures_train
            eval_measures_test[cv_index] = cur_eval_measures_test
            predictions.update(cur_pred_per_page)
            print(f"Fold {cv_index} has ended.")

        # end of loop, calculating agg results
        eval_measures_df_train = pd.DataFrame.from_dict(eval_measures_train, orient='index')
        eval_measures_df_test = pd.DataFrame.from_dict(eval_measures_test, orient='index')

        eval_measures_df_train.loc['mean'] = eval_measures_df_train.mean()
        eval_measures_df_train.loc['std'] = eval_measures_df_train.std()
        ci_column_train = [stats.t.interval(0.95, len(cv_unique_splits), loc=cur_mean, scale=cur_std) for cur_mean, cur_std in
                           zip(eval_measures_df_train.loc['mean'], eval_measures_df_train.loc['std'])]
        # rounding the CI to 3 digits only and adding them as a new row
        eval_measures_df_train.loc['ci'] = [(round(c[0], 3), round(c[1], 3)) for c in ci_column_train]

        eval_measures_df_test.loc['mean'] = eval_measures_df_test.mean()
        eval_measures_df_test.loc['std'] = eval_measures_df_test.std()
        ci_column_test = [stats.t.interval(0.95, len(cv_unique_splits), loc=cur_mean, scale=cur_std) for cur_mean, cur_std in
                          zip(eval_measures_df_test.loc['mean'], eval_measures_df_test.loc['std'])]
        # rounding the CI to 3 digits only and adding them as a new row
        eval_measures_df_test.loc['ci'] = [(round(c[0], 3), round(c[1], 3)) for c in ci_column_test]
    else:
        raise IOError("Invalid 'cv_process' parameter")
    print(f"TRAIN results:\n {eval_measures_df_train}\n\n")
    print(f"TEST results:\n {eval_measures_df_test}\n\n")


def expand_dataset_based_promotion_year(model_version, years_lag=1, initial_years_to_include=3, use_bootstrap=True):
    # In this function, we expand the dataset in each iteration, allowing the model to learn from a larger set per iter.
    # E.g., first cycle: 2009-2012 (train+test), 2nd cycle: 2009-2014 (train+test) etc..
    # loading the config file of the model
    model_params_f_name = opj(model_folder, str(model_version), 'model_params.json')
    with open(model_params_f_name, 'r', encoding='utf-8') as infile:
        model_info_dict = json.load(infile)
        infile.close()
    max_year_to_include = int(model_info_dict['max_year_to_include'])
    # loading the data for modeling
    data_df = pd.read_csv(opj(model_folder, str(model_version), 'modeling_df.csv'))
    # Set the first column as the index
    data_df.set_index(data_df.columns[0], inplace=True)
    data_df[target_column] = data_df[target_column].astype(int)

    # filtering too "new" articles
    if max_year_to_include is not None:
        data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied over the whole dataset. "
              f"Filtered dataset shape: {data_df.shape}.", flush=True)

    # pulling the list of years we should consider for the analysis
    years_to_consider = list(sorted(set(data_df['TIME_promotion_year'])))
    years_to_consider = years_to_consider[initial_years_to_include-1::years_lag]
    eval_measures_over_years = dict()
    print(f"About to start looping over the following max years: {years_to_consider}.")
    for iteration, max_year_to_include in enumerate(years_to_consider[0:]):
        # filtering "too new" articles
        cur_data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied. Max year in the data: {max_year_to_include}. "
              f"Filtered dataset shape: {cur_data_df.shape}.", flush=True)
        # all features that start with 'TIME_' should not be included in the modeling df
        x_columns = [c for c in data_df.columns if c != target_column and not c.startswith('TIME_')]
        # if we want to include specific sets of features (e.g., meta)
        if use_only_meta_features:
            x_columns = [c for c in x_columns if c.startswith('META_')]
        x_data = cur_data_df[x_columns].copy()
        y_data = cur_data_df[target_column]

        # running the bootstrap/cv method
        if use_bootstrap:
            boostrap_obj = Bootstrapping(bootstrap_folds=100, classification_model=classification_model)
            _ = boostrap_obj.split_data_to_folds(x_data=x_data, y_data=y_data)
            _ = boostrap_obj.run_model_over_all_folds(x_data=x_data, y_data=y_data)
            eval_measures_df_train = boostrap_obj.eval_measures_train_df.copy()
            eval_measures_df_test = boostrap_obj.eval_measures_test_df.copy()
        # Option B - X-fold-CV
        else:
            stratifies_cv_obj = StratifiedCV(cv_folds=5, classification_model=classification_model,
                                             cpus_to_use=5, random_seed=1984, verbose=1)
            _ = stratifies_cv_obj.split_data_to_folds(x_data=x_data, y_data=y_data)
            _ = stratifies_cv_obj.run_model_over_all_folds(x_data=x_data, y_data=y_data)
            eval_measures_df_train = stratifies_cv_obj.eval_measures_train_df.copy()
            eval_measures_df_test = stratifies_cv_obj.eval_measures_test_df.copy()

        minority_class_cnt = min(list(Counter(y_data).values()))
        minority_class_perc = minority_class_cnt / x_data.shape[0]
        # end of the train process, saving results
        eval_measures_over_years[iteration] = {'max_year': max_year_to_include, 'n': cur_data_df.shape[0],
                                               'minority_class_perc': minority_class_perc,
                                               'train_mean_macro_f1': eval_measures_df_train.loc['mean']['macro_f1_opt_th'],
                                               'train_std_macro_f1': eval_measures_df_train.loc['std']['macro_f1_opt_th'],
                                               'train_ci_macro_f1': eval_measures_df_train.loc['ci']['macro_f1_opt_th'],
                                               'train_mean_binary_f1': eval_measures_df_train.loc['mean']['binary_f1'],
                                               'train_std_binary_f1': eval_measures_df_train.loc['std']['binary_f1'],
                                               'train_ci_binary_f1': eval_measures_df_train.loc['ci']['binary_f1'],
                                               'train_mean_auc': eval_measures_df_train.loc['mean']['macro_auc'],
                                               'train_std_auc': eval_measures_df_train.loc['std']['macro_auc'],
                                               'train_ci_auc': eval_measures_df_train.loc['ci']['macro_auc'],
                                               'test_mean_macro_f1': eval_measures_df_test.loc['mean']['macro_f1_opt_th'],
                                               'test_std_macro_f1': eval_measures_df_test.loc['std']['macro_f1_opt_th'],
                                               'test_ci_macro_f1': eval_measures_df_test.loc['ci']['macro_f1_opt_th'],
                                               'test_mean_binary_f1': eval_measures_df_test.loc['mean']['binary_f1'],
                                               'test_std_binary_f1': eval_measures_df_test.loc['std']['binary_f1'],
                                               'test_ci_binary_f1': eval_measures_df_test.loc['ci']['binary_f1'],
                                               'test_mean_auc': eval_measures_df_test.loc['mean']['macro_auc'],
                                               'test_std_auc': eval_measures_df_test.loc['std']['macro_auc'],
                                               'test_ci_auc': eval_measures_df_test.loc['ci']['macro_auc']
                                               }
        print(f"Process for year {max_year_to_include} has ended.\n\n")
    print(f"Process for all years has ended.")
    return pd.DataFrame.from_dict(eval_measures_over_years, orient='index')


def sliding_window_dataset_based_promotion_year(model_version, years_lag=1, initial_years_to_include=3,
                                                move_window=False):
    # loading the config file of the model
    model_params_f_name = opj(model_folder, str(model_version), 'model_params.json')
    with open(model_params_f_name, 'r', encoding='utf-8') as infile:
        model_info_dict = json.load(infile)
        infile.close()
    max_year_to_include = int(model_info_dict['max_year_to_include'])
    # loading the data for modeling
    data_df = pd.read_csv(opj(model_folder, str(model_version), 'modeling_df.csv'))
    # Set the first column as the index
    data_df.set_index(data_df.columns[0], inplace=True)
    data_df[target_column] = data_df[target_column].astype(int)

    # filtering too "new" articles, over all the dataset that we plan to use
    if max_year_to_include is not None:
        data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied over the whole dataset. "
              f"Filtered dataset shape: {data_df.shape}.", flush=True)

    # pulling the list of years we should consider for the analysis
    years_to_consider = list(sorted(set(data_df['TIME_promotion_year'])))
    years_to_consider = years_to_consider[initial_years_to_include-1::years_lag]
    min_year_for_training = min(years_to_consider)
    eval_measures_over_years = dict()
    print(f"About to start looping over the following max years: {years_to_consider}.")
    for iteration, max_year_to_include in enumerate(years_to_consider[0:]):
        # if we apply the sliding window (and not taking years from start to a certain point), we have another filter
        # it does not work as required if years_lag!=1
        if move_window:
            min_year_for_training = max_year_to_include - initial_years_to_include
            cur_train_df = data_df[(data_df['TIME_promotion_year'] <= max_year_to_include) &
                                   (min_year_for_training <= data_df['TIME_promotion_year'])].copy()
        else:
            cur_train_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        cur_test_df = data_df[data_df['TIME_promotion_year'] > max_year_to_include].copy()
        print(f"Max year filter has been applied. Max year in the TRAIN data: {max_year_to_include}. "
              f"Filtered TRAIN dataset shape: {cur_train_df.shape}.", flush=True)
        # the last set of datapoints for test should be empty, as the window moved all the way to the right
        if cur_test_df.shape[0] == 0:
            break
        # all features that start with 'TIME_' should not be included in the modeling df
        x_columns = [c for c in data_df.columns if c != target_column and not c.startswith('TIME_')]
        x_train_data = cur_train_df[x_columns].copy()
        y_train_data = cur_train_df[target_column]
        x_test_data = cur_test_df[x_columns].copy()
        y_test_data = cur_test_df[target_column]

        # filling up the missing values
        # filling the missing values with the mean
        imputer = SimpleImputer(strategy='mean')

        # Fit the imputer on the training data and transform both train and test sets
        x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train_data), columns=x_train_data.columns).copy()
        x_test_imputed = pd.DataFrame(imputer.transform(x_test_data), columns=x_test_data.columns).copy()

        minority_class_cnt_train = min(list(Counter(y_train_data).values()))
        minority_class_perc_train = minority_class_cnt_train / x_train_imputed.shape[0]
        minority_class_cnt_test = min(list(Counter(y_test_data).values()))
        minority_class_perc_test = minority_class_cnt_test / x_test_imputed.shape[0]

        # extreme case of no negative instances in the train/test
        if minority_class_cnt_train == 0 or minority_class_cnt_test == 0:
            print(f"minority class in either train/test is zero. Skipping year {max_year_to_include}.")
        # running the model training part (and testing). I use the StratifiedCV object to do that, although we run
        # the model over a single fold and not really in a CV way
        stratifies_cv_obj = StratifiedCV(cv_folds=1, classification_model=classification_model,
                                         cpus_to_use=1, random_seed=1984, verbose=1)
        try:
            _, eval_dict_train, eval_dict_test, _, _ = (
                stratifies_cv_obj._run_model_over_a_single_fold(fold_idx=0, x_data_train=x_train_imputed,
                                                                y_data_train=y_train_data, x_data_test=x_test_imputed,
                                                                y_data_test=y_test_data))
        except ValueError:
            print(f"minority class in either train/test is zero. Skipping year {max_year_to_include}.")
            continue
        eval_measures_df_train = pd.DataFrame.from_dict(eval_dict_train, orient='index')
        eval_measures_df_test = pd.DataFrame.from_dict(eval_dict_test, orient='index')

        # end of the train process, saving results
        eval_measures_over_years[iteration] = {'min_year': min_year_for_training, 'max_year': max_year_to_include,
                                               'train_n': x_train_imputed.shape[0],
                                               'test_n': x_test_imputed.shape[0],
                                               'minority_class_perc_train': minority_class_perc_train,
                                               'minority_class_perc_test': minority_class_perc_test,
                                               'train_macro_f1': eval_measures_df_train.loc['macro_f1_opt_th'][0],
                                               'train_binary_f1': eval_measures_df_train.loc['binary_f1'][0],
                                               'train_auc': eval_measures_df_train.loc['macro_auc'][0],
                                               'test_macro_f1': eval_measures_df_test.loc['macro_f1_opt_th'][0],
                                               'test_binary_f1': eval_measures_df_test.loc['binary_f1'][0],
                                               'test_auc': eval_measures_df_test.loc['macro_auc'][0],
                                               }
        print(f"Process for year {max_year_to_include} has ended.\n\n")
    print(f"Process for all years has ended.")
    return pd.DataFrame.from_dict(eval_measures_over_years, orient='index')


if __name__ == '__main__':
    if cv_process == 'cv_split_by_time' or cv_process == 'per_period_train_test':
        time_sensitive_cv_methods()
    elif cv_process == 'expand_dataset_based_promotion_year':
        eval_measures_over_years = expand_dataset_based_promotion_year(model_version=model_version, years_lag=1)
        print(eval_measures_over_years)
        # for printing, refer to notebook called 'analyze_performance_of_sust_in_the_next_x_years'
    elif cv_process == 'sliding_window_dataset_based_promotion_year':
        sliding_window_dataset_based_promotion_year(model_version=model_version, years_lag=1, initial_years_to_include=3,
                                                    move_window=True)


