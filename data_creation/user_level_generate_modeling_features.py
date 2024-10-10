# the purpose of this code is to generate features for modeling. The features are generated per page_id.
# The input for the feature creation are the two csv that the user_level_generate_credit_features.py code created.
import sys
sys.path.append("/home/isabrah/sustainable-high-quality-wikis")
import pandas as pd
import numpy as np
import glob
from os.path import join as opj
from datetime import datetime
import multiprocessing as mp
import os
import itertools
import pickle
from collections import Counter
from modeling.modeling_utils import load_and_decompress
import gc

cpus_to_use = 10
usecase = 'fa' #'fa
data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
output_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/user_level_data'
date_format = '%Y-%m-%dT%H:%M:%SZ'


def extract_info_per_edit_from_bz2(bz2_file, required_user_ids, job_idx):
    json_lines = load_and_decompress(bz2_file)
    data_as_df = pd.DataFrame(json_lines)
    # removing None values (these are IP based users)
    data_as_df = data_as_df[~data_as_df['user_id'].isna()].copy()
    # converting the user_id column to int
    data_as_df['user_id'] = data_as_df['user_id'].astype(int)
    # removing None user values
    data_as_df_filtered = data_as_df[data_as_df['user_id'].isin(required_user_ids)].copy()
    # converting the timestamp column to datetime
    data_as_df_filtered['timestamp'] = pd.to_datetime(data_as_df_filtered['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"Job index {job_idx} has ended.", flush=True)
    return data_as_df_filtered


start_time = datetime.now()
user_credit_df = pd.read_csv(opj(data_folder, 'user_level_data', usecase+'_usecase_user_level_info.csv'))
# the next code is not working, probably due to very high memory issues

# loading the info of ALL Wikipedia (this is HUGE!!). We do so only for a subset of users (from the above df)
required_user_ids = set(user_credit_df['user_id'])
# making sure the set contains only int and not Nones
required_user_ids = set({u for u in required_user_ids if type(u) is int})
info_per_edit_path_files = glob.glob(opj(data_folder, 'concise_info_per_edit', '*.bz2'))
input_for_pool = [(file_path, required_user_ids, idx) for idx, file_path in enumerate(info_per_edit_path_files)]
#input_for_pool = input_for_pool[0:100]
pool = mp.Pool(processes=cpus_to_use)
with pool as pool:
    edit_lvl_dfs = pool.starmap(extract_info_per_edit_from_bz2, input_for_pool)
print(f"Finished extracting info in a multi-process way. Created a list of size: {len(edit_lvl_dfs)}\n")
full_edit_level_df = pd.concat(edit_lvl_dfs, ignore_index=True)
end_time = datetime.now()
code_duration = end_time - start_time
print(f"Credit df and edit level df have been loaded in {code_duration.seconds} sec. Data frame sizes: "
      f"{user_credit_df.shape[0]} and {full_edit_level_df.shape[0]}", flush=True)


def create_features_for_page(page_id, pickle_file_path, usecase, job_index):
    # creating empty features, to be returned any way (if domething return empty along the way)
    till_promotion_credibility_unweighted = None
    till_promotion_credibility_weighted = None
    till_promotion_experience_unweighted = None
    till_promotion_experience_weighted = None
    till_demotion_unweighted_score = None
    till_demotion_weighted_score = None
    till_demotion_experience_unweighted = None
    till_demotion_experience_weighted = None
    cur_article_assessment_obj = pickle.load(open(pickle_file_path, "rb"))
    promotion_date = cur_article_assessment_obj.fa_promotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_promotion_date
    # first thing we have to filter the data to remove all revisions after promotion/demotion
    revision_level_info_till_promotion = cur_article_assessment_obj.filter_revision_level_info(usecase=usecase,
                                                                                               until_promotion=True)
    users_contrib_till_promotion = extract_contrib_per_user(revisions_data=revision_level_info_till_promotion)
    contributing_users = set(users_contrib_till_promotion.keys())
    # in case the dict is not empty
    if users_contrib_till_promotion:
        # filtering the credit df to only relevant dates and users
        promotion_date_mask = pd.to_datetime(user_credit_df['date'], format=date_format) <= promotion_date
        user_credit_df_filtered = user_credit_df[(user_credit_df['user_id'].isin(contributing_users))
                                                 & promotion_date_mask
                                                 & (user_credit_df['page_id'] != page_id)].copy()
        till_promotion_credibility_unweighted, till_promotion_credibility_weighted = (
            calc_credit_score(rel_credit_df=user_credit_df_filtered, users_contrib=users_contrib_till_promotion))
        till_promotion_experience_unweighted, till_promotion_experience_weighted = (
            calc_experience_score(rel_credit_df=user_credit_df_filtered, users_contrib=users_contrib_till_promotion))
    # second thing we have to handle demotions (if exist)
    demotion_date = cur_article_assessment_obj.fa_demotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_demotion_date
    if demotion_date is not None:
        max_revisions_mask = pd.to_datetime(cur_article_assessment_obj.revision_level_info['timestamp'],
                                            format=date_format) < demotion_date
        revision_level_info_till_demotion = cur_article_assessment_obj.revision_level_info[max_revisions_mask].copy()
        users_contrib_till_demotion = extract_contrib_per_user(revisions_data=revision_level_info_till_demotion)
        contributing_users = set(users_contrib_till_demotion.keys())
        if users_contrib_till_demotion:
            # filtering the credit df to only relevant dates and users
            demotion_date_mask = pd.to_datetime(user_credit_df['date'], format=date_format) <= demotion_date
            user_credit_df_filtered = user_credit_df[(user_credit_df['user_id'].isin(contributing_users))
                                                     & demotion_date_mask
                                                     & (user_credit_df['page_id'] != page_id)].copy()
            till_demotion_credibility_unweighted, till_demotion_credibility_weighted = (
                calc_credit_score(rel_credit_df=user_credit_df_filtered, users_contrib=users_contrib_till_demotion))
            till_demotion_experience_unweighted, till_demotion_experience_weighted = (
                calc_credit_score(rel_credit_df=user_credit_df_filtered, users_contrib=users_contrib_till_demotion))
    # creating a joint dict to return soon
    dict_to_return = {'page_id': page_id, 'promotion_date': promotion_date, 'demotion_date': demotion_date,
                      'till_promotion_credibility_unweighted': till_promotion_credibility_unweighted,
                      'till_promotion_credibility_weighted': till_promotion_credibility_weighted,
                      'till_promotion_experience_unweighted': till_promotion_experience_unweighted,
                      'till_promotion_experience_weighted': till_promotion_experience_weighted,
                      'till_demotion_credibility_unweighted': till_demotion_unweighted_score,
                      'till_demotion_credibility_weighted': till_demotion_weighted_score,
                      'till_demotion_experience_unweighted': till_demotion_experience_unweighted,
                      'till_demotion_experience_weighted': till_demotion_experience_weighted}
    # another set of features is based on the entire set of wikipedia edits
    entire_wikipedia_edits_measures = calc_measures_based_entire_wikipedia_edits(page_id=page_id,
                                                                                 users_contrib_till_promotion=users_contrib_till_promotion,
                                                                                 promotion_date=promotion_date)
    dict_to_return.update(entire_wikipedia_edits_measures)
    return dict_to_return


def extract_contrib_per_user(revisions_data):
    users_list = list(revisions_data['user'])
    # removing nones and converting to int
    users_list = [int(ul) for ul in users_list if not np.isnan(ul)]
    n = len(users_list)
    contrib_per_user_dict = {user_id: num_contribs / n *1.0 for user_id, num_contribs in Counter(users_list).items()}
    return contrib_per_user_dict


def calc_credit_score(rel_credit_df, users_contrib):
    unweighted_score = rel_credit_df['credit'].sum() / len(users_contrib) * 1.0
    exisitng_users_in_score_df = set(rel_credit_df['user_id'])
    weighted_score = 0
    for user, contribution in users_contrib.items():
        if user in exisitng_users_in_score_df:
            weighted_score += rel_credit_df[rel_credit_df['user_id'] == user]['credit'].sum() * contribution
    return unweighted_score, weighted_score


def calc_experience_score(rel_credit_df, users_contrib):
    unweighted_score = abs(rel_credit_df['credit']).sum() / len(users_contrib) * 1.0
    exisitng_users_in_score_df = set(rel_credit_df['user_id'])
    weighted_score = 0
    for user, contribution in users_contrib.items():
        if user in exisitng_users_in_score_df:
            weighted_score += abs(rel_credit_df[rel_credit_df['user_id'] == user]['credit']).sum() * contribution
    return unweighted_score, weighted_score


def calc_measures_based_entire_wikipedia_edits(page_id, users_contrib_till_promotion, promotion_date):
    # to make thinks simpler, I only calculate things until promotion and not demotion at all
    # taking only the contributing users
    contributing_users = set(users_contrib_till_promotion.keys())
    if len(contributing_users) == 0:
        return {'entire_wikipedia_other_pages_edited': None, 'entire_wikipedia_num_edits_on_other_pages': None,
                'entire_wikipedia_num_users_edited_other_pages': None}
    promotion_date_mask = pd.to_datetime(full_edit_level_df['timestamp'], format=date_format) <= promotion_date
    # apply relevant filters
    full_edit_level_df_users_subset = full_edit_level_df[(full_edit_level_df['user_id'].isin(contributing_users))
                                                         & promotion_date_mask
                                                         & (full_edit_level_df['page_id'] != page_id)].copy()
    num_pages = full_edit_level_df_users_subset['page_id'].nunique()
    num_edits = full_edit_level_df_users_subset.shape[0]
    num_users = full_edit_level_df_users_subset['user_id'].nunique()
    # TODO: normalize measures
    # TODO: calc new measures based on the relative contribution of each user (users_contrib_till_promotion is a dict)
    del full_edit_level_df_users_subset
    gc.collect()
    return {'entire_wikipedia_other_pages_edited': num_pages / len(contributing_users) * 1.0,
            'entire_wikipedia_num_edits_on_other_pages': num_edits / len(contributing_users) * 1.0,
            'entire_wikipedia_num_users_edited_other_pages': num_users / len(contributing_users) * 1.0}


if __name__ == '__main__':
    pickle_files_folder = opj(data_folder, 'article_assessment_objs', 'good_articles') \
        if usecase == 'ga' else opj(data_folder, 'article_assessment_objs', 'featured_articles')
    # in both cases of good/featured, we have to add the 'both' folder
    pickle_files_folder = [pickle_files_folder, opj(data_folder, 'article_assessment_objs', 'both')]
    pickle_files_path = sorted(list(itertools.chain(*[glob.glob(opj(pff, '*.p')) for pff in pickle_files_folder])))
    #pickle_files_path = pickle_files_path[0:100]
    # extracting the page id of each file
    page_id_to_path_mapping = {int(os.path.basename(pfp).split('.p')[0]): pfp for pfp in pickle_files_path}

    print(f"We have found {len(page_id_to_path_mapping)} files to process. It will take some time now, but soon you will "
          f"have the data as a csv file.", flush=True)
    input_for_pool = [(page_id, fp, usecase, idx) for idx, (page_id, fp) in enumerate(page_id_to_path_mapping.items())]
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        user_level_sustainability_data = pool.starmap(create_features_for_page, input_for_pool)
    page_level_sustainability_data_df = pd.DataFrame.from_records(user_level_sustainability_data)
    # saving the df to disk
    page_level_sustainability_data_df.to_csv(opj(output_folder, usecase +
                                                 '_usecase_users_experience_features_per_pageid_whole_wikipedia.csv'), index=False)
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Code ended in {code_duration.total_seconds()} sec. DF of size {page_level_sustainability_data_df.shape} "
          f"saved to {opj(output_folder)}.\n Overall, we handled {len(page_id_to_path_mapping)} files.", flush=True)
