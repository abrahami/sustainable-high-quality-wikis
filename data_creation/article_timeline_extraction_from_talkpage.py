"""
The purpose of this script is to extract information from the talkpage of each relevant article.
We extract the information about the article timeline, that contains all the relevant promotions/demotions of a page
This is currently used for the post-modeling analysis (see if predicted demoted pages tend to be over reviewed)
"""
from data_creation.promotion_demotion_dates_utils import *
import pandas as pd
import glob
from os.path import join as opj
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import os

data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
metadata_files_folder = opj(data_folder, 'meta_data')
latest_talkpages_folder = opj(data_folder, 'latest_talk_pages_sustained_articles')
cpus_to_use = 100

if __name__ == '__main__':
    start_time = datetime.now()
    # using the objects we created (now or in prev runs) to extract promotion/demotion time per article
    existing_talk_pages_jsonl = glob.glob(opj(latest_talkpages_folder, '*.jsonl.bz2'))
    article_events_df, unreached_articles = (
        article_events_extractor_from_talkpages(talk_pages_jsonl=existing_talk_pages_jsonl))
    print(f'We extracted {article_events_df.shape[0]} events from the talk pages content. '
          f'{len(unreached_articles)} articles were unreachable and their content is missing (no real worries, most of'
          f'them are not really FAs/GA, they were included in the dataset for an unclear reason.')

    # Saving the df to disk
    article_events_df.to_csv(opj(metadata_files_folder, 'article_events_from_latest_talkpages.csv'), index=True)
    end_time = datetime.now()
    code_duration = (end_time - start_time).seconds
    print(f'Code has ended in {code_duration} seconds. Events csv has been '
          f'saved to {metadata_files_folder} folder', flush=True)
