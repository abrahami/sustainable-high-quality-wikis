from promotion_demotion_dates_utils import *
import pandas as pd
import glob
from os.path import join as opj
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import os

data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
metadata_files_folder = opj(data_folder, 'meta_data')
saving_folder = opj(data_folder, 'latest_talk_pages_sustained_articles')
overwrite_existing_files = False
crawl_data_from_wikipedia = False   # this should be done only once, and hence we do not need to do it each run
cpus_to_use = 100

if __name__ == '__main__':
    start_time = datetime.now()
    # extracting the relevant article ids to analyze. These come from two different meta data files
    if crawl_data_from_wikipedia:
        crawl_articles_timeline_from_talk_pages(metadata_files_folder=metadata_files_folder,
                                                saving_folder=saving_folder,
                                                overwrite_existing_files=overwrite_existing_files)
    print(f'Articles timeline has been crawled (now or historically) and saved to {saving_folder}.')
    # using the objects we created (now or in prev runs) to extract promotion/demotion time per article
    existing_talk_pages_jsonl = glob.glob(opj(saving_folder, '*.jsonl.bz2'))
    article_timelines_from_talkpages_df, unreached_pages = (
        article_promotion_demotion_dates_extractor_from_talkpages(talk_pages_jsonl=existing_talk_pages_jsonl))
    print(f'We got {article_timelines_from_talkpages_df.shape[0]} articles from the talk pages. '
          f'Number of non-empty cells:\n {article_timelines_from_talkpages_df.count()}.')
    # now we do the same idea, but pulling the information from the edit history
    # extracting the relevant article ids to analyze. These come from two different meta data files
    metadata_files = glob.glob(opj(metadata_files_folder, 'metadata*.csv'))
    existing_pickle_files = glob.glob(opj(data_folder, 'article_assessment_objs', 'good_articles', '*.p'))
    existing_pickle_files2 = glob.glob(opj(data_folder, 'article_assessment_objs', 'featured_articles', '*.p'))
    existing_pickle_files.extend(existing_pickle_files2)
    existing_page_ids_obj = set([int(epf.split('/')[-1].split('.p')[0]) for epf in existing_pickle_files])
    page_id_to_path = dict()
    # looping over the files found and extracting the relevant json files to analyze
    for mf in metadata_files:
        cur_metadata_df = pd.read_csv(mf)
        rel_rows = cur_metadata_df[(cur_metadata_df['is_now_fa_or_ga']) | (cur_metadata_df['ever_was_fa_or_ga'])]
        # by the name of the metadata file, can we infer in which folder the json file is located
        cur_json_folder_name = 'revision_jsons_sustained_articles' if 'sustainability' in mf else 'revision_jsons'
        cur_json_full_path = opj(data_folder, cur_json_folder_name)
        page_id_to_path.update({rr: opj(cur_json_full_path, str(rr) + '.bz2') for rr in list(rel_rows['page_id'])})
    input_for_pool = [(page_id, fp, idx) for idx, (page_id, fp) in enumerate(page_id_to_path.items())]
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        article_timelines_from_dumps_list = pool.starmap(article_timeline_extractor_from_dumps, input_for_pool)
    article_timelines_from_dumps_dict = {}
    for d in article_timelines_from_dumps_list:
        for row_name, columns in d.items():
            article_timelines_from_dumps_dict[row_name] = columns

    # Create a DataFrame
    article_timelines_from_dumps_df = pd.DataFrame.from_dict(article_timelines_from_dumps_dict, orient='index')
    print(f'We got {article_timelines_from_dumps_df.shape[0]} articles from the dump files. '
          f'Number of non-empty cells:\n {article_timelines_from_dumps_df.count()}.')
    # now we can "join" the two dataframes (the one from the talkpages and the one from the dumps)
    # I prefer to use the talkpages information rather than the dumps (more reliable)
    joint_df = pd.concat([article_timelines_from_talkpages_df, article_timelines_from_dumps_df], axis=1)
    # saving the data to a csv file
    joint_df.to_csv(opj(metadata_files_folder, 'promotion_demotion_dates.csv'), index=True)
    end_time = datetime.now()
    code_duration = (end_time - start_time).seconds
    print(f'Code has ended in {code_duration} seconds.')
