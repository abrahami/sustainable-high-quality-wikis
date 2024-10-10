import pandas as pd
from os.path import join as opj
import json
from modeling.modeling_utils import generate_model_workspace, exclude_unreliable_cases_from_modeling_df
from bootstrapping import Bootstrapping
from stratified_cv import StratifiedCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from catboost import CatBoostClassifier


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# general configuration parameters
base_model_version = 20.2
new_model_version = -1
target_column = 'is_sustainable_conservative_approach'#is_sustainable' # 'is_sustainable_conservative_approach'
debug_mode = True
base_folder = '/shared/3/projects/relationship-aware-networks-wikipedia'
data_folder = opj(base_folder, 'wiki_generated_data')
model_folder = opj(base_folder, 'models')
use_edit_features = False
use_team_composition_features = False
use_network_features = False
use_discussions_features = False
use_topic_features = False
use_user_experience_features = True
# this 'special' feature is currently only the 'used_to_be_ga'
use_special_features = False

specific_column_to_use = None#['EDIT_num_revisions_normalized']# None
classification_model = GradientBoostingClassifier(n_estimators=100)#CatBoostClassifier()##RandomForestClassifier(n_estimators=100) #LogisticRegression()
run_bootstrap_method = True
bootstrap_folds = 100 if run_bootstrap_method else None
run_k_fold_cv = True
filter_unreliable_cases = True

if __name__ == '__main__':
    # loading the json file of the model (configurations)
    model_params_f_name = opj(model_folder, str(base_model_version), 'model_params.json')
    with open(model_params_f_name, 'r', encoding='utf-8') as infile:
        model_info_dict = json.load(infile)
        infile.close()
    usecase = model_info_dict['usecase']
    seed = int(model_info_dict['seed'])
    folds_k = int(model_info_dict['folds_k'])
    max_year_to_include = int(model_info_dict['max_year_to_include'])
    until_promotion = model_info_dict['until_promotion']
    max_revisions_to_use = model_info_dict['max_revisions_to_use']

    # create a new workspace based the information we have
    saving_workspace = (
        generate_model_workspace(model_version=new_model_version,target_column=target_column, usecase=usecase,
                                 output_folder=model_folder, seed=seed, folds_k=folds_k,
                                 use_edit_features=use_edit_features,
                                 use_team_composition_features=use_team_composition_features,
                                 use_network_features=use_network_features,
                                 use_discussions_features=use_discussions_features,
                                 use_topic_features=use_topic_features,
                                 use_user_experience_features=use_user_experience_features,
                                 use_special_features=use_special_features,
                                 max_year_to_include=max_year_to_include,
                                 until_promotion=until_promotion, max_revisions_to_use=max_revisions_to_use,
                                 classification_model=classification_model,
                                 specific_column_to_use=specific_column_to_use,
                                 bootstrap_folds=bootstrap_folds, filter_unreliable_cases=filter_unreliable_cases))
    # loading the data for modeling
    data_df = pd.read_csv(opj(model_folder, str(base_model_version), 'modeling_df.csv'))
    # Set the first column as the index
    data_df.set_index(data_df.columns[0], inplace=True)
    data_df[target_column] = data_df[target_column].astype(int)
    # filtering too new articles
    if max_year_to_include is not None:
        data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied. Filtered dataset shape: {data_df.shape}.", flush=True)
    # before moving forward, we filter out unreliable cases (e.g., time in promotion is too short). There shouldn't be
    # any of those, but just in cases...
    if filter_unreliable_cases:
        data_df = exclude_unreliable_cases_from_modeling_df(data_df=data_df, target_column=target_column)
        print(f"filter_unreliable_cases option has been applied. Modified dataset size: {data_df.shape[0]}", flush=True)
    # all features that start with 'TIME_' should not be included in the modeling df
    x_columns = [c for c in data_df.columns if c != target_column and not c.startswith('TIME_')]
    columns_to_remove = list()
    if specific_column_to_use is not None:
        columns_to_remove.extend([c for c in x_columns if c not in specific_column_to_use])
    if not use_edit_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('EDIT_')])
    if not use_team_composition_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('COMPOSITION_')])
    if not use_network_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('NETWORK')])
    if not use_discussions_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('DISCUSSIONS_')])
    if not use_topic_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('TOPIC_')])
    if not use_user_experience_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('CREDIBILITY_') or c.startswith('EXPERIENCE_')])
    if not use_special_features:
        columns_to_remove.extend([c for c in x_columns if c.startswith('SPECIAL_')])
    x_columns = [c for c in x_columns if c not in columns_to_remove]
    x_data = data_df[x_columns].copy()
    y_data = data_df[target_column].copy()
    print(f"Modeling dataframe shape: {x_data.shape}")
    # saving the modeling df to disk
    data_df.to_csv(opj(saving_workspace, 'modeling_df.csv'))

    # here, we have two options, either a simple 5-fold-cv or stratified method (or both)
    # Option A - boostrap
    if run_bootstrap_method:
        boostrap_obj = Bootstrapping(bootstrap_folds=bootstrap_folds, classification_model=classification_model)
        _ = boostrap_obj.split_data_to_folds(x_data=x_data, y_data=y_data)
        _ = boostrap_obj.run_model_over_all_folds(x_data=x_data, y_data=y_data)
        # saving the results to disk
        boostrap_obj.eval_measures_train_df.to_csv(opj(saving_workspace, "bootstrap_eval_measures_train.csv"))
        boostrap_obj.eval_measures_test_df.to_csv(opj(saving_workspace, "bootstrap_eval_measures_test.csv"))
        print(f"TRAIN results:\n {boostrap_obj.eval_measures_train_df}\n\n")
        print(f"TEST results:\n {boostrap_obj.eval_measures_test_df}\n\n")

    # Option B - X-fold-CV
    if run_k_fold_cv:
        stratifies_cv_obj = StratifiedCV(cv_folds=folds_k, classification_model=classification_model,
                                         cpus_to_use=folds_k, random_seed=1984, verbose=1)
        _ = stratifies_cv_obj.split_data_to_folds(x_data=x_data, y_data=y_data)
        _ = stratifies_cv_obj.run_model_over_all_folds(x_data=x_data, y_data=y_data)
        stratifies_cv_obj.eval_measures_train_df.to_csv(opj(saving_workspace, "eval_measures_train.csv"))
        stratifies_cv_obj.eval_measures_test_df.to_csv(opj(saving_workspace, "eval_measures_test.csv"))
        stratifies_cv_obj.row_lvl_preds_df.to_csv(opj(saving_workspace, "row_lvl_preds.csv"))
        print(f"TRAIN results:\n {stratifies_cv_obj.eval_measures_train_df}\n\n")
        print(f"TEST results:\n {stratifies_cv_obj.eval_measures_test_df}\n\n")
        # TODO: feature importance is not used at all
