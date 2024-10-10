# the purpose of this code is to generate 2 csv files which are later used for modeling.
# the two files contain edit credit per editor for a specific page_id in time. The two files are later used by the
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

cpus_to_use = 100 #1
usecase = 'ga' #'fa
data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
output_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/user_level_data'
date_format = '%Y-%m-%dT%H:%M:%SZ'


def extract_user_level_sustainability_from_file(page_id, pickle_file_path, usecase, job_index):
    # creating two empty dfs, in case one of them is not created along the way, so we can still return something
    till_promotion_df = pd.DataFrame(columns=['page_id', 'date', 'user_id', 'credit'])
    till_demotion_df = pd.DataFrame(columns=['page_id', 'date', 'user_id', 'credit'])
    cur_article_assessment_obj = pickle.load(open(pickle_file_path, "rb"))
    promotion_date = cur_article_assessment_obj.fa_promotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_promotion_date
    # first thing we have to filter the data to remove all revisions after promotion/demotion
    revision_level_info_till_promotion = cur_article_assessment_obj.filter_revision_level_info(usecase=usecase,
                                                                                               until_promotion=True)
    users_contrib_till_promotion = extract_contrib_per_user(revisions_data=revision_level_info_till_promotion)
    # in case the dict is not empty
    if users_contrib_till_promotion:
        till_promotion_df = pd.DataFrame.from_dict(users_contrib_till_promotion, orient='index')
        till_promotion_df.reset_index(inplace=True)
        till_promotion_df.columns = ['user_id', 'credit']
        # adding two fixed values for the df
        till_promotion_df['date'] = promotion_date.strftime(date_format)
        till_promotion_df['page_id'] = page_id
        # resorting the column names
        till_promotion_df = till_promotion_df[['page_id', 'date', 'user_id', 'credit']]

    # second thing we have to handle demotions (if exist)
    demotion_date = cur_article_assessment_obj.fa_demotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_demotion_date
    if demotion_date is not None:
        max_revisions_mask = pd.to_datetime(cur_article_assessment_obj.revision_level_info['timestamp'], format=date_format) < demotion_date
        revision_level_info_till_demotion = cur_article_assessment_obj.revision_level_info[max_revisions_mask].copy()
        users_contrib_till_demotion = extract_contrib_per_user(revisions_data=revision_level_info_till_demotion)
        if users_contrib_till_demotion:
            till_demotion_df = pd.DataFrame.from_dict(users_contrib_till_demotion, orient='index')
            till_demotion_df.reset_index(inplace=True)
            till_demotion_df.columns = ['user_id', 'credit']
            till_demotion_df['credit'] = till_demotion_df['credit']*-1
            # adding two fixed values for the df
            till_demotion_df['date'] = demotion_date.strftime(date_format)#pd.to_datetime(demotion_date, format=date_format)
            till_demotion_df['page_id'] = page_id
            # resorting the column names
            till_demotion_df = till_demotion_df[['page_id', 'date', 'user_id', 'credit']]
    # returning the two dfs together
    return pd.concat([df for df in [till_promotion_df, till_demotion_df] if not df.empty], ignore_index=True)


def extract_contrib_per_user(revisions_data):
    users_list = list(revisions_data['user'])
    # removing nones and converting to int
    users_list = [int(ul) for ul in users_list if not np.isnan(ul)]
    n = len(users_list)
    contrib_per_user_dict = {user_id: num_contribs / n *1.0 for user_id, num_contribs in Counter(users_list).items()}
    return contrib_per_user_dict


if __name__ == '__main__':
    start_time = datetime.now()
    page_id_to_path = dict()
    pickle_files_folder = opj(data_folder, 'article_assessment_objs', 'good_articles') \
        if usecase == 'ga' else opj(data_folder, 'article_assessment_objs', 'featured_articles')
    # in both cases of good/featured, we have to add the 'both' folder
    pickle_files_folder = [pickle_files_folder, opj(data_folder, 'article_assessment_objs', 'both')]
    pickle_files_path = sorted(list(itertools.chain(*[glob.glob(opj(pff, '*.p')) for pff in pickle_files_folder])))
    #pickle_files_path = pickle_files_path[0:100]
    # extracting the page id of each file
    page_id_to_path_mapping = {int(os.path.basename(pfp).split('.p')[0]): pfp for pfp in pickle_files_path}

    input_for_pool = [(page_id, fp, usecase, idx) for idx, (page_id, fp) in enumerate(page_id_to_path_mapping.items())]
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        user_level_sustainability_data = pool.starmap(extract_user_level_sustainability_from_file, input_for_pool)
    # now looping over the information we got from each file and joining it all together
    user_level_sustainability_df = pd.concat(user_level_sustainability_data)
    # saving the df to disk
    user_level_sustainability_df.to_csv(opj(output_folder, usecase + '_usecase_user_level_info.csv'), index=False)
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Code ended in {code_duration.total_seconds()} sec. DF of size {user_level_sustainability_df.shape} "
          f"saved to {opj(output_folder)}.\n Overall, we handled {len(page_id_to_path_mapping)} files.", flush=True)
