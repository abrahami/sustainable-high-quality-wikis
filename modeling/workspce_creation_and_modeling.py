import pandas as pd
from os.path import join as opj
import glob
from modeling.modeling_utils import (meta_and_editing_structural_features_extractor, generate_model_workspace,
                                     find_optimal_threshold, discussions_features_extractor, extract_topic_features,
                                     eval_classification_preds, users_overlap_features_extractor,
                                     exclude_unreliable_cases_from_modeling_df)
from modeling.stratified_cv import StratifiedCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import itertools
from sklearn.impute import SimpleImputer
from datetime import datetime
from collections import Counter
from scipy import stats
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# general configuration parameter
model_version = 20.2
target_column = 'is_sustainable_conservative_approach'#is_sustainable' # 'is_sustainable_conservative_approach'
debug_mode = False
usecase = 'ga' # 'fa' # 'fa' stands for"{{featured article}}" while 'ga' for "{{good article}}"
base_folder = '/shared/3/projects/relationship-aware-networks-wikipedia'
data_folder = opj(base_folder, 'wiki_generated_data')
output_folder = opj(base_folder, 'models')
seed = 1984

# parameters that are associated with the data creation (e.g., whether to create discussion features or not)
max_year_to_include = 2018 if usecase == 'fa' else 2019 #None # None means all years
until_promotion = True # if True, only revisions until promotion time are counted. Promotion date depends on the usecase
max_revisions_to_use = None # if None, we use until_promotion indicator. If both are turned off, we take the whole data
filter_unreliable_cases = True # used to remove cases of demotion date too early, or no demotion date for demoted cases

# these should not be changed (old setup). To examine the value of each feature type, we can run another function later.
create_metadata_features = True
create_editing_structural_features = True
create_community_dynamic_features = True
create_discussion_features = True
create_user_credit_features = True
create_topic_features = True
create_users_overlap_features = True


# modeling parameters
use_topics_in_binary_mode = False
folds_k = 5
classification_model = GradientBoostingClassifier(n_estimators=100)#LogisticRegression(max_iter=100)#GradientBoostingClassifier(n_estimators=10)#LogisticRegression(max_iter=100)# #LogisticRegression(max_iter=200)
specific_column_to_use = None#['politeness_mean']#None #['num_authors_normalized']#['num_revisions_normalized']#['promotion_age']#['num_authors_normalized']#['num_authors_normalized']#['promotion_age'] #['num_revisions'] # ['num_authors'] should be None in regular runs
num_topics_to_use = 'all' # 50 # 100 # 'all' means we will use all ~250. 0 means no feature is used


def run_classifier(data_df):
    # filtering too new articles
    if max_year_to_include is not None:
        data_df = data_df[data_df['TIME_promotion_year'] <= max_year_to_include].copy()
        print(f"Max year filter has been applied. Filtered dataset shape: {data_df.shape}.", flush=True)
    data_df[target_column] = data_df[target_column].astype(int)
    # all features that start with 'TIME_' should not be included in the modeling df
    x_columns = [c for c in data_df.columns if c != target_column and not c.startswith('TIME_')]
    # in case we defined a specific column to be used (good for baselines)
    if specific_column_to_use is not None:
        x_columns = specific_column_to_use
    x_data = data_df[x_columns]
    y_data = data_df[target_column]
    # running 5-fold-cv using the object I have created for that
    stratifies_cv_obj = StratifiedCV(cv_folds=folds_k, classification_model=classification_model, cpus_to_use=folds_k,
                                     random_seed=1984, verbose=1)
    _ = stratifies_cv_obj.split_data_to_folds(x_data=x_data, y_data=y_data)
    _ = stratifies_cv_obj.run_model_over_all_folds(x_data=x_data, y_data=y_data)
    print(f"TRAIN results:\n {stratifies_cv_obj.eval_measures_train_df}\n\n")
    print(f"TEST results:\n {stratifies_cv_obj.eval_measures_test_df}\n\n")
    return stratifies_cv_obj.eval_measures_train_df, stratifies_cv_obj.eval_measures_test_df, stratifies_cv_obj.row_lvl_preds_df


def create_workspace_and_dataset():
    start_time = datetime.now()
    # configurations
    n_cpus = 1 if debug_mode else 100
    metadata_files_folder = opj(data_folder, 'meta_data')
    pickle_files_folder = opj(data_folder, 'article_assessment_objs', 'good_articles') \
        if usecase == 'ga' else opj(data_folder, 'article_assessment_objs', 'featured_articles')
    # in both cases of good/featured, we have to add the 'both' folder
    pickle_files_folder = [pickle_files_folder, opj(data_folder, 'article_assessment_objs', 'both')]
    # extracting the discussion files path
    discussion_files_folder = opj(data_folder, 'talkpage_discussions_with_dl_preds')
    # actual code start here
    saving_workspace = (
        generate_model_workspace(model_version=model_version, target_column=target_column, usecase=usecase,
                                 output_folder=output_folder, seed=seed, folds_k=folds_k,
                                 use_edit_features=True,
                                 use_team_composition_features=True,
                                 use_network_features=True,
                                 use_discussions_features=True,
                                 use_topic_features=True,
                                 use_user_experience_features=True,
                                 use_special_features=True,
                                 max_year_to_include=max_year_to_include,
                                 until_promotion=until_promotion, max_revisions_to_use=max_revisions_to_use,
                                 classification_model=classification_model,
                                 specific_column_to_use=specific_column_to_use,
                                 bootstrap_folds=0, filter_unreliable_cases=filter_unreliable_cases))
    pickle_files_path = sorted(list(itertools.chain(*[glob.glob(opj(pff, '*.p')) for pff in pickle_files_folder])))
    #pickle_files_path = pickle_files_path[0:1000]
    print(f"Data import starts for {len(pickle_files_path)} articles.", flush=True)
    meta_and_editing_structural_df, unreliable_cases = (
        meta_and_editing_structural_features_extractor(files_path=pickle_files_path, target_column=target_column,
                                                       usecase=usecase, cpus_to_use=n_cpus,
                                                       extract_meta=create_metadata_features,
                                                       extract_structural=create_editing_structural_features,
                                                       extract_community_dynamics=create_community_dynamic_features,
                                                       until_promotion=until_promotion,
                                                       max_revisions_to_use=max_revisions_to_use))

    # NOTE!!!! If until_promotion==False, the next two rows are problematic and has to be fixed!!!!!
    meta_and_editing_structural_df['TIME_promotion_year'] = pd.DatetimeIndex(meta_and_editing_structural_df['TIME_promotion_date']).year
    last_revision_timestamp_per_article = meta_and_editing_structural_df['TIME_last_revision_timestamp'].to_dict()
    data_df = meta_and_editing_structural_df.copy()

    # before moving forward, we filter out unreliable cases (e.g., time in promotion is too short
    if filter_unreliable_cases:
        data_df = exclude_unreliable_cases_from_modeling_df(data_df=data_df, target_column=target_column)
        print(f"filter_unreliable_cases option has been applied. Modified dataset size: {data_df.shape[0]}", flush=True)

    # target feature distribution
    target_feature_counters = dict(Counter(data_df[target_column]))
    target_feature_distribution = {k: round(v / data_df.shape[0], 2) for k, v in target_feature_counters.items()}
    print(f"Target feature distribution: {target_feature_distribution}. Counters per class: {target_feature_counters}")

    if create_discussion_features:
        discussion_files_path = sorted(glob.glob(opj(discussion_files_folder, '*jsonl.bz2')))
        discussions_df, problematic_cases = (
            discussions_features_extractor(files_path=discussion_files_path, cpus_to_use=n_cpus,#1,#n_cpus # 1
                                           last_revision_timestamp_per_article=last_revision_timestamp_per_article))
        discussions_df_found_pages = set(discussions_df.index)
        unfound_discussion_pages = [dd_index for dd_index in data_df.index if dd_index not in discussions_df_found_pages]
        unfound_discussion_pages_counter = dict(Counter(data_df.loc[unfound_discussion_pages, target_column]))
        unfound_discussion_pages_distribution = {k: round(v / len(unfound_discussion_pages), 2)
                                                 for k, v in unfound_discussion_pages_counter.items()}
        print(f"We were not able to get discussions of {len(unfound_discussion_pages)} pages. "
              f"The target feature distribution over these pages is: {unfound_discussion_pages_distribution}.")
        # join the modeling_df with the discussion_df
        #data_df = pd.merge(data_df, discussions_df, how="left", left_index=True, right_index=True)
        data_df = pd.concat([data_df, discussions_df], axis=1, join="inner")

    if create_user_credit_features:
        users_credit_features = pd.read_csv(opj(data_folder, 'user_level_data',
                                                usecase + '_usecase_users_credit_features_per_pageid.csv'))
        users_credit_features.set_index('page_id', drop=True, inplace=True)
        users_credit_features = (
            users_credit_features)[['till_promotion_credibility_unweighted', 'till_promotion_credibility_weighted',
                                    'till_promotion_experience_unweighted', 'till_promotion_experience_weighted']]
                                    ##'entire_wikipedia_other_pages_edited', 'entire_wikipedia_num_edits_on_other_pages', 'entire_wikipedia_num_users_edited_other_pages']]
        users_credit_features.rename(columns={'till_promotion_credibility_unweighted': 'CREDIBILITY_unweighted_score',
                                              'till_promotion_credibility_weighted': 'CREDIBILITY_weighted_score',
                                              'till_promotion_experience_unweighted': 'EXPERIENCE_unweighted_score',
                                              'till_promotion_experience_weighted': 'EXPERIENCE_weighted_score'},
                                              ##'entire_wikipedia_other_pages_edited': 'EXPERIENCE_other_pages_edited', 'entire_wikipedia_num_edits_on_other_pages': 'EXPERIENCE_num_edits_on_other_pages', 'entire_wikipedia_num_users_edited_other_pages': 'EXPERIENCE_users_edited_other_pages'},
                                     inplace=True)
        print(f"Users credit feature extraction ended. Created dataset size: "
              f"{users_credit_features.shape[0]}.", flush=True)
        #data_df = pd.merge(data_df, users_credit_features, how="left", left_index=True, right_index=True)
        data_df = pd.concat([data_df, users_credit_features], axis=1, join="inner")

    if create_topic_features:
        topics_info_file_path = opj(base_folder, 'wikipedia_meta_data',
                                    'pages_project_name_importance_only_modeled_articles.csv')
        page_ids_to_pull = set(data_df.index)
        topic_features_df, topic_features_binary_df = extract_topic_features(data_path=topics_info_file_path,
                                                                             required_page_ids=page_ids_to_pull)
        topic_df_to_use = topic_features_binary_df if use_topics_in_binary_mode else topic_features_df
        # filtering the number of features from the topics, in case it is not 'all' which means to use all of them
        if num_topics_to_use != 'all':
            topic_df_to_use = topic_df_to_use.iloc[:, 0:num_topics_to_use]
        #data_df = pd.merge(data_df, topic_df_to_use, how="left", left_index=True, right_index=True)
        data_df = pd.concat([data_df, topic_df_to_use], axis=1, join="inner")

    if create_users_overlap_features:
        discussion_files_path = sorted(glob.glob(opj(discussion_files_folder, '*jsonl.bz2')))
        pickle_files_path = sorted(list(itertools.chain(*[glob.glob(opj(pff, '*.p')) for pff in pickle_files_folder])))
        users_overlap_df, problematic_cases = (
            users_overlap_features_extractor(discussion_files_path=discussion_files_path,
                                             pickle_files_path=pickle_files_path,
                                             last_revision_timestamp_per_article=last_revision_timestamp_per_article,
                                             usecase=usecase, until_promotion=until_promotion,
                                             max_revisions_to_use=max_revisions_to_use, cpus_to_use=n_cpus))
        data_df = pd.concat([data_df, users_overlap_df], axis=1, join="inner")


    # saving the modeling df to disk
    if saving_workspace is not None:
        data_df.to_csv(opj(saving_workspace, 'modeling_df.csv'))
    missing_vals_cnt = data_df.isna().sum().sum()
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"\nAll features creation ended. The modeling dataframe size: {data_df.shape}. "
          f"{len(unreliable_cases)} cases were removed from the initial dataset as they are not reliable enough. "
          f"There are {missing_vals_cnt} missing values in the data. Elapsed time (sec.): {code_duration.seconds}. "
          f"\n\n", flush=True)
    return saving_workspace, data_df


if __name__ == '__main__':
    saving_workspace, modeling_dataset = create_workspace_and_dataset()
    cur_run_eval_measures_train, cur_run_eval_measures_test, row_lvl_preds_df = (
        run_classifier(data_df=modeling_dataset))
    # saving the results to disk
    cur_run_eval_measures_train.to_csv(opj(saving_workspace, "eval_measures_train.csv"))
    cur_run_eval_measures_test.to_csv(opj(saving_workspace, "eval_measures_test.csv"))
    row_lvl_preds_df.to_csv(opj(saving_workspace, "row_lvl_preds.csv"))
    print(f"TRAIN results:\n {cur_run_eval_measures_train}\n\n")
    print(f"TEST results:\n {cur_run_eval_measures_test}\n\n")

