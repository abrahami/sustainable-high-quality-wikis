from article_assessment import *
import pandas as pd
import glob
from os.path import join as opj
import os
import multiprocessing as mp
from datetime import datetime
import pickle
from tqdm import tqdm
from collections import Counter


data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
metadata_files_folder = opj(data_folder, 'meta_data')
override_existing_objs = False#False
processes_amount = 100
promotion_demotion_dates_df = pd.read_csv(opj(metadata_files_folder, 'promotion_demotion_dates.csv'), index_col=0)
# this is a table that contains all the data that I crawled from Wikipedia pages that indicate an explicit promotion
# or demotion of pages. This table is used to create the "conservative" target feature
potential_fa_ga_wikipedia_info = pd.read_csv(opj(metadata_files_folder, 'potential_fa_ga_wikipedia_info.csv'))


def create_article_assessment_obj(article_id, json_path, call_index):
    local_start_time = datetime.now()
    article_assessment_obj = ArticleAssessment(article_id=article_id)
    article_assessment_obj.promotion_demotion_full_info = promotion_demotion_dates_df.loc[article_id].to_dict()
    try:
        article_assessment_obj.extract_info_from_json(json_path=json_path)
    except FileNotFoundError:
        print(f'Article ID {article_id} file does not exist in {json_path}. Skipping it.', flush=True)

    valid_case = article_assessment_obj.validate_dates_and_durations()
    if not valid_case:
        print(f"Error with article ID; {article_id}, json file: {json_path}. Unlogical duration and times.", flush=True)
        return "error"
    # NEW STEP: adding a new label called 'is_sustainable_conservative_approach'
    cur_page_info_from_wiki_lists = potential_fa_ga_wikipedia_info[potential_fa_ga_wikipedia_info['page_id'] == article_id]
    appears_in_which_wiki_lists = list(cur_page_info_from_wiki_lists['type']) if cur_page_info_from_wiki_lists.shape[0] > 0 else None
    sustainability_determined = (
        article_assessment_obj.determine_sustainability_conservative_approach(appears_in=appears_in_which_wiki_lists))
    # saving the object if it is valid. We split it into four cases - featured, good, both, and None
    # None are the cases which were not mapped to any (e.g., not promotion/demotion were found)
    usecase = article_assessment_obj.determine_usecase()
    if usecase == 'fa':
        saving_path = opj(data_folder, 'article_assessment_objs', 'featured_articles', str(article_id) + '.p')
        saved_as = 'fa'
    elif usecase == 'ga':
        saving_path = opj(data_folder, 'article_assessment_objs', 'good_articles', str(article_id) + '.p')
        saved_as = 'ga'
    elif usecase == 'both':
        saving_path = opj(data_folder, 'article_assessment_objs', 'both', str(article_id) + '.p')
        saved_as = 'both'
    else:
        saved_as = 'None'
        saving_path = opj(data_folder, 'article_assessment_objs', str(article_id) + '.p')
    pickle.dump(article_assessment_obj, open(saving_path, "wb"))
    local_end_time = datetime.now()
    local_code_duration = local_end_time - local_start_time
    print(f"Call index {call_index} has ended in {local_code_duration.total_seconds()} sec. "
          f"File saved under the {saved_as} case.\n"
          f"Based on the new conservative approach, the label was set to {sustainability_determined}", flush=True)
    return saved_as


if __name__ == '__main__':
    start_time = datetime.now()
    # extracting the relevant article ids to analyze
    page_ids_to_analyze = list(promotion_demotion_dates_df.index)
    missing_pages = list()
    page_id_to_path = dict()
    for cur_page_id in tqdm(page_ids_to_analyze):
        # we first try to get the json from the sustained_articles folder
        if os.path.isfile(opj(data_folder, 'revision_jsons_sustained_articles', str(cur_page_id) + '.bz2')):
            cur_json_full_path = opj(data_folder, 'revision_jsons_sustained_articles', str(cur_page_id) + '.bz2')
        # in case it is not there, we try to get it from the other folder that we hold data in
        elif os.path.isfile(opj(data_folder, 'revision_jsons', str(cur_page_id) + '.bz2')):
            cur_json_full_path = opj(data_folder, 'revision_jsons', str(cur_page_id) + '.bz2')
        else:
            missing_pages.append(cur_page_id)
            continue
        # if we have found a match (should be in 100% of the cases)
        page_id_to_path[cur_page_id] = cur_json_full_path
    # in case we want to take into account existing pages that have already been saved
    if not override_existing_objs:
        # extracting the relevant article ids to analyze. These come from two different meta data files
        existing_pickle_files = glob.glob(opj(data_folder, 'article_assessment_objs', '**/*.p'), recursive=True)
        existing_page_ids_obj = set([int(epf.split('/')[-1].split('.p')[0]) for epf in existing_pickle_files])
        page_id_to_path = {key: value for key, value in page_id_to_path.items() if key not in existing_page_ids_obj}
    input_for_pool = [(key, value, idx) for idx, (key, value) in enumerate(page_id_to_path.items())]
    print(f"We have found {len(page_id_to_path)} cases to analyze. Starting now!", flush=True)
    pool = mp.Pool(processes=processes_amount)
    with pool as pool:
        results = pool.starmap(create_article_assessment_obj, input_for_pool)
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Code has ended in {code_duration.total_seconds()} seconds. We processed {len(results)} "
          f"files and created objects per each.", flush=True)
    print(f"Distribution of cases: {Counter(results)}")

