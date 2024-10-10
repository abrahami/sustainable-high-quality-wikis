# the purpose of this file is to add the predictions we got from the DL models back to the files with all data
# in addition, we add userID to data, as most of the useeIDs are missing. We use a userName -> UserID dict we build here
import pandas as pd
from os.path import join as opj
from modeling.modeling_utils import load_and_decompress, compress_and_save
import glob
import multiprocessing as mp
import os
import pickle


n_cpus = 100
data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
# loading the data with the dl predictions, to be used by all threads
dl_preds = pd.read_csv(opj(data_folder, 'talkpage_discussions_sustained_articles',
                           'all_comments_posted_with_dl_predictions.csv'))
saving_folder = opj(data_folder, 'talkpage_discussions_with_dl_preds')
columns_to_extract = ['sentiment', 'formality', 'politeness', 'toxicity', 'certainty']


def find_matching_row(text_to_find, text_with_dl_preds_subset):
    # the purpose of this function is to find the row which refers to the right text we are looking for
    # if no such row exist, it returns None
    ids_found = [idx for idx, i in enumerate(text_with_dl_preds_subset['text']) if i == text_to_find]
    if len(ids_found) == 0:#or len(ids_found) > 1:
        return None
    # even if the len of this is > 1, we return the first item, as the preds should be the same for same texts!
    else:
        return ids_found[0]


def add_dl_predictions_to_article_data(run_idx, article_id_to_run):
    file_to_load = opj(data_folder, 'talkpage_discussions_sustained_articles', article_id_to_run + '.jsonl.bz2')
    talkpage_data = load_and_decompress(filename=file_to_load)
    dl_preds_subset = dl_preds[dl_preds['article_id'] == int(article_id_to_run)]
    talkpage_data_with_preds = list()
    # now we have to loop over the talkpage and assign the right values in the right place
    dl_pred_index = 0
    error_found = False
    for cur_disc in talkpage_data:
        cur_dis_comments_with_preds = cur_disc.copy()
        comments_with_dl_preds = list()
        cur_dis_comments = cur_disc['comments'].copy()
        # looping over all the list of comments
        for orig_comment in cur_dis_comments:
            # filling the missing userIDs in the comment
            modified_comment = fill_in_user_ids(unfilled_comment=orig_comment)
            if modified_comment['text'] == '' and type(dl_preds_subset.iloc[dl_pred_index]['text']) is not str:
                are_texts_aligned = True
                modified_comment['dl_preds'] = dl_preds_subset.iloc[dl_pred_index][columns_to_extract].to_dict()
            elif modified_comment['text'] == dl_preds_subset.iloc[dl_pred_index]['text']:
                are_texts_aligned = True
                modified_comment['dl_preds'] = dl_preds_subset.iloc[dl_pred_index][columns_to_extract].to_dict()
            # if we get to here, there might be some problem with the data
            else:
                # before we raise a problem flag, we try to find a match
                matching_row = find_matching_row(text_to_find=modified_comment['text'],
                                                 text_with_dl_preds_subset=dl_preds_subset)
                if matching_row is None:
                    are_texts_aligned = False
                    error_found = True
                    print(f"Error with article {article_id_to_run}")
                    modified_comment['dl_preds'] = None
                else:
                    modified_comment['dl_preds'] = dl_preds_subset.iloc[matching_row][columns_to_extract].to_dict()
            comments_with_dl_preds.append(modified_comment)
            dl_pred_index += 1
        # end of the inner loop, need to update the talkpage_data list
        cur_dis_comments_with_preds['comments'] = comments_with_dl_preds
        talkpage_data_with_preds.append(cur_dis_comments_with_preds)
    # end of the big loop, now we need to save things back to disk
    saving_f_name = article_id_to_run + '.jsonl.bz2'
    compress_and_save(talkpage_data_with_preds, opj(saving_folder, saving_f_name))
    print(f"Article {article_id_to_run} has ended and saved", flush=True)
    return error_found


def fill_in_user_ids(unfilled_comment):
    filled_comment = unfilled_comment.copy()
    existing_author_id = unfilled_comment['author_id']
    existing_refers_to_author_id = unfilled_comment['refers_to_author_id']
    existing_root_author_id = unfilled_comment['root_author_id']
    if existing_author_id is None and unfilled_comment['author'] in username_to_userid_dict:
        filled_comment['author_id'] = username_to_userid_dict[unfilled_comment['author']]
    if existing_refers_to_author_id is None and unfilled_comment['refers_to'] in username_to_userid_dict:
        filled_comment['refers_to_author_id'] = username_to_userid_dict[unfilled_comment['refers_to']]
    if existing_root_author_id is None and unfilled_comment['root_author'] in username_to_userid_dict:
        filled_comment['root_author_id'] = username_to_userid_dict[unfilled_comment['root_author']]
    return filled_comment


def build_username_to_userid_mapping(pickle_file_path):
    data = pickle.load(open(pickle_file_path, 'rb'))
    user_name_to_id = dict()
    for d in data:
        for key, value in d.items():
            id_found = value[0]
            if id_found is None:
                continue
            if key in user_name_to_id and user_name_to_id[key] != id_found:
                print(f"found inconsistency for userName {key}, values: {id_found}, {user_name_to_id[key]}")
            else:
                user_name_to_id[key] = id_found
    return user_name_to_id


# building a userName -> UserID dict
pickle_file_path = opj(data_folder, 'meta_data', 'user_name_to_id_mapping_from_talk_pages_only.p')
username_to_userid_dict = build_username_to_userid_mapping(pickle_file_path)
print(f"username_to_userid_dict has been loaded. Dict size: {len(username_to_userid_dict)}")

if __name__ == "__main__":
    # we will have to open each file, and assign the prediction values in the right place
    existing_discussion_files = glob.glob(opj(data_folder, 'talkpage_discussions_sustained_articles', '*.jsonl.bz2'))
    article_ids_to_process = list(os.path.basename(eai).split('.jsonl.bz2')[0] for eai in existing_discussion_files)
    input_for_pool = [(idx, aitp) for idx, aitp in enumerate(article_ids_to_process)]
    pool = mp.Pool(processes=n_cpus)
    with pool as pool:
        results = pool.starmap(add_dl_predictions_to_article_data, input_for_pool)
    errors_found = sum([1 for r in results if r is True])
    print(f"Code ended. There were {errors_found} errors")



