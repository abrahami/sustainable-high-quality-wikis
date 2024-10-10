import pandas as pd
import numpy as np
from collections import defaultdict, ChainMap, Counter
from itertools import chain
import pickle
import networkx as nx
import statistics
import multiprocessing as mp
from datetime import datetime
from os.path import join as opj
import os
import json
import bz2
import jsonlines
import re
from dateutil.parser import parse
from dateutil.parser import ParserError
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

date_format = '%Y-%m-%dT%H:%M:%SZ'


def was_really_demoted(article_obj):
    # simplest case - it was not ever demoted
    if article_obj.latest_assessment_level is not None:
        return False
    elif (article_obj.promotion_demotions_details['days_in_promotion'] < 30
          or article_obj.assessments_counter[article_obj.dominant_assessment['level']] < 5):
        return False
    else:
        return True


def calc_num_authors(article_obj, usecase, max_revisions_to_use=None, until_promotion=True, normlize=True):
    revision_level_info_subset = article_obj.filter_revision_level_info(usecase=usecase,
                                                                        until_promotion=until_promotion,
                                                                        max_revisions_to_use=max_revisions_to_use)
    authors_len = len(set([u for u in revision_level_info_subset['user'] if not np.isnan(u)]))
    # normalization option
    if normlize:
        normalization_factor = len(revision_level_info_subset) # normalizing by the number of revisions
        if max_revisions_to_use is not None:
            normalization_factor = min(normalization_factor, max_revisions_to_use)
    else:
        normalization_factor = 1
    authors_len = 1.0 * (authors_len / max(1, normalization_factor))
    return authors_len


def calc_num_revisions(article_obj, usecase, max_revisions_to_use=None, until_promotion=True, normalize=True):
    revision_level_info_subset = article_obj.filter_revision_level_info(usecase=usecase,
                                                                        until_promotion=until_promotion,
                                                                        max_revisions_to_use=max_revisions_to_use)
    num_revisions = revision_level_info_subset.shape[0]

    # normalization option (by time)
    if normalize:
        try:
            first_revision_date = datetime.strptime(revision_level_info_subset.iloc[0]['timestamp'], date_format)
            last_revision_date = datetime.strptime(revision_level_info_subset.iloc[-1]['timestamp'], date_format)
            days_between_first_and_last_revision_level_info_subset = (last_revision_date - first_revision_date).days
        except IndexError:
            days_between_first_and_last_revision_level_info_subset = 1
        normalization_factor = days_between_first_and_last_revision_level_info_subset
    else:
        normalization_factor = 1
    num_revisions = 1.0 * (num_revisions / max(1, normalization_factor))
    return num_revisions


def editing_structural_features_extractor(article_assessment_obj, usecase, max_revisions_to_use=None, until_promotion=True):
    revision_level_info_subset = (
        article_assessment_obj.filter_revision_level_info(usecase=usecase, max_revisions_to_use=max_revisions_to_use,
                                                          until_promotion=until_promotion))
    social_net_obj = build_social_network_from_revisions_df(revision_level_df=revision_level_info_subset)
    structural_features = extract_structural_features(network_obj=social_net_obj)
    # adding a prefix to each feature, to identify it as an edit feature
    structural_features = {'NETWORK_' + feature: value for feature, value in structural_features.items()}
    return structural_features


def community_dynamics_extractor(article_assessment_obj, usecase, max_revisions_to_use=None, until_promotion=True):
    revision_level_info_subset = (
        article_assessment_obj.filter_revision_level_info(usecase=usecase, max_revisions_to_use=max_revisions_to_use,
                                                          until_promotion=until_promotion))
    valid_users = revision_level_info_subset['user'][~article_assessment_obj.revision_level_info['user'].isna()]
    if len(revision_level_info_subset):
        ip_based_users_revision_perc = 1 - (len(valid_users) / len(revision_level_info_subset)) * 1.0
    else:
        ip_based_users_revision_perc = None
    inequality_index = gini_coefficient(dict(Counter(valid_users)))
    # extracting the % of revisions
    sha1_dict = dict(Counter(revision_level_info_subset['sha1']))
    revert_cases = len([value for key, value in sha1_dict.items() if value > 1])
    revert_sum = sum([value for key, value in sha1_dict.items() if value > 1]) - revert_cases
    reverts_percentage = (revert_sum / len(sha1_dict)) if len(sha1_dict) > 0 else 0
    return {'COMPOSITION_author_gini': inequality_index, 'EDIT_reverts_perc': reverts_percentage,
            'COMPOSITION_ip_based_users_revision_perc': ip_based_users_revision_perc}


def build_social_network_from_edge_list(edge_list, directed=True):
    if directed:
        network_obj = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in edge_list.items())
    else:
        network_obj = nx.Graph((x, y, {'weight': v}) for (x, y), v in edge_list.items())
    return network_obj


def build_social_network_from_revisions_df(revision_level_df):
    edge_list = defaultdict(int)
    historical_authors = set()
    author_prev = np.nan
    author_curr = np.nan
    nan_cases = 0
    for row_idx, row in revision_level_df.iterrows():
        author_curr = row['user']
        cur_author_is_nan = np.isnan(author_curr)
        nan_cases += 1 if cur_author_is_nan else 0
        # in case an edge can be added
        if not cur_author_is_nan and not np.isnan(author_prev):
            edge_list[(int(author_curr), int(author_prev))] += 1
        # if the current user is not None, we can add it to the set of authors
        historical_authors.add(author_curr)
        author_prev = author_curr
    # now we can use networkx to build the network
    network_obj = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in edge_list.items())
    return network_obj


def extract_structural_features(network_obj):
    undirected_graph = nx.Graph(network_obj)
    structural_features = dict()
    structural_features['num_of_edges'] = network_obj.number_of_edges()
    structural_features['num_of_nodes'] = network_obj.number_of_nodes()
    structural_features['num_of_triangles'] = sum(nx.triangles(undirected_graph).values()) / 3
    structural_features['is_biconnected'] = 1 if nx.is_biconnected(undirected_graph) else 0
    structural_features['num_of_nodes_to_cut'] = len(list(nx.articulation_points(undirected_graph)))
    structural_features['density'] = nx.density(network_obj)

    # now using other functions from the code below to get other features
    (in_degrees_stats, out_degrees_stats), _ = extract_degrees(network_obj, return_node_level_values=False)
    centrality_stats = extract_centrality(network_obj, return_node_level_values=False)
    closeness_stats = extract_closeness(network_obj, return_node_level_values=False)
    betweeness_stats = extract_betweeness(network_obj, return_node_level_values=False)
    components_stats = extract_components(network_obj)
    combined_features = ChainMap(*[structural_features, in_degrees_stats, out_degrees_stats, centrality_stats,
                                   closeness_stats, betweeness_stats, components_stats])
    return combined_features


def extract_degrees(network_obj, return_node_level_values=False):
    in_degrees = {}
    out_degrees = {}
    for node in network_obj.nbunch_iter():
        in_degrees[node] = network_obj.in_degree(node)
        out_degrees[node] = network_obj.out_degree(node)
    in_degree = get_agg_stats(list(in_degrees.values()))
    # adding a prefix per key for each item, to recognize it later among the many other features
    in_degree = {'in_deg_'+key: value for key, value in in_degree.items()}
    out_degree = get_agg_stats(list(out_degrees.values()))
    out_degree = {'out_deg_' + key: value for key, value in out_degree.items()}
    if return_node_level_values:
        return {'in_degree': in_degrees, 'out_degrees': out_degrees}
    else:
        return (in_degree, out_degree), {'in_degree': in_degrees, 'out_degrees': out_degrees}


def extract_centrality(network_obj, return_node_level_values=False):
    try:
        centrality = nx.degree_centrality(network_obj)
        centrality_values = list(centrality.values())
        central_dict = get_agg_stats(centrality_values)
        # adding a prefix per key for each item, to recognize it later among the many other features
        central_dict = {'centrality_' + key: value for key, value in central_dict.items()}
    except Exception:
        centrality_values = list()
        central_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0}
    if return_node_level_values:
        return central_dict, centrality_values
    else:
        return central_dict


def extract_betweeness(network_obj, return_node_level_values=False):
    try:
        betweenness = nx.betweenness_centrality(network_obj)
        betweenness_values = list(betweenness.values())
        betweenness_dict = get_agg_stats(betweenness_values)
        # adding a prefix per key for each item, to recognize it later among the many other features
        betweenness_dict = {'betw_' + key: value for key, value in betweenness_dict.items()}
    except Exception:
        betweenness_values = list()
        betweenness_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0}
    if return_node_level_values:
        return betweenness_dict, betweenness_values
    else:
        return betweenness_dict


def extract_closeness(network_obj, return_node_level_values=False):
    try:
        closeness = nx.closeness_centrality(network_obj)
        closeness_values = list(closeness.values())
        closeness_dict = get_agg_stats(closeness_values)
        # adding a prefix per key for each item, to recognize it later among the many other features
        closeness_dict = {'closeness_' + key: value for key, value in closeness_dict.items()}
    except Exception:
        closeness_values = list()
        closeness_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0}
    if return_node_level_values:
        return closeness_dict, closeness_values
    else:
        return closeness_dict


def extract_components(network_obj):
    undirected_graph = nx.Graph(network_obj)
    components_dict = dict()
    components = list(nx.connected_components(undirected_graph))
    components_dict['connected_components'] = len(components)
    components_dict['connected_components_>2'] = len([comp for comp in components if len(comp) > 2])

    strongly_components = list(nx.strongly_connected_components(network_obj))
    components_dict['strongly_connected_components'] = len(strongly_components)
    components_dict['strongly_connected_components_>1'] = len([comp for comp in strongly_components if len(comp) > 1])
    try:
        components_dict['connected_components_max'] = max([len(comp) for comp in components])
        components_dict['strongly_connected_components_max'] = max([len(comp) for comp in strongly_components])
    except ValueError:
        components_dict['strongly_connected_components_max'] = 0
        components_dict['connected_components_max'] = 0
    return components_dict


def get_agg_stats(values):
    # in case the list is empty
    if not len(values):
        return {'avg': None, 'median': None, 'stdev': None}
    return_dict = dict()
    return_dict['avg'] = statistics.mean(values)
    return_dict['median'] = statistics.median(values)
    return_dict['stdev'] = statistics.stdev(values) if len(values) > 1 else 0
    return return_dict


def meta_and_editing_structural_features_extractor(files_path, target_column, usecase, cpus_to_use=80, extract_meta=True,
                                                   extract_creation_date=False, extract_structural=True,
                                                   extract_community_dynamics=True, until_promotion=True,
                                                   max_revisions_to_use=None):
    start_time = datetime.now()
    extracted_features = dict()
    unreliable_unsustained_articles = list()
    # opening each file and extracting the relevant info. We will do it a multiprocess one since it takes a long time
    input_for_pool = [(fp, target_column, usecase, extract_meta, extract_creation_date, extract_structural, extract_community_dynamics, until_promotion, max_revisions_to_use, idx) for idx, fp in enumerate(files_path)]
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        results = pool.starmap(_features_extractor_from_article_objects, input_for_pool)
    # merging the list of dicts into one dict
    for res in results:
        # if the values are a single value of -1, it means that the page is an unreliable one
        if list(res.values())[0] == -1:
            page_id = list(res.keys())[0]
            unreliable_unsustained_articles.append(page_id)
        else:
            extracted_features.update(res)
    end_time = datetime.now()
    code_duration = end_time - start_time
    extracted_features_df = pd.DataFrame.from_dict(extracted_features, orient='index')
    print(f"Meta and editing structural feature extraction ended. Created dataset shape: "
          f"{extracted_features_df.shape}. Elapsed time (sec.): {code_duration.seconds}.", flush=True)
    return extracted_features_df, unreliable_unsustained_articles


def _features_extractor_from_article_objects(file, target_column, usecase, extract_meta=True, extract_creation_date=False,
                                             extract_structural=True, extract_community_dynamics=True,
                                             until_promotion=True, max_revisions_to_use=None, job_idx=-1):
    cur_extracted_features = dict()
    try:
        cur_article_assessment_obj = pickle.load(open(file, "rb"))
    except EOFError:
        return {-1: -1}
    # we validate the data in hand before we process with features extraction
    is_valid = cur_article_assessment_obj.is_obj_valid(usecase)
    if not is_valid:
        return {-1: -1}
    # for each article we extract the (max) date that we use for modeling. This is later used by other functions
    revision_level_info_subset = (
        cur_article_assessment_obj.filter_revision_level_info(usecase=usecase,
                                                              until_promotion=until_promotion,
                                                              max_revisions_to_use=max_revisions_to_use))
    # for each article, we extract two critical timestamps - promotion and demotion
    cur_extracted_features['TIME_promotion_date'] = cur_article_assessment_obj.fa_promotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_promotion_date
    cur_extracted_features['TIME_demotion_date'] = cur_article_assessment_obj.fa_demotion_date if usecase == 'fa' else cur_article_assessment_obj.ga_demotion_date
    cur_extracted_features['TIME_last_revision_timestamp'] = revision_level_info_subset.iloc[-1]['timestamp'] if revision_level_info_subset.shape[0] >0 else None
    # for each article we also extract the gold label: is_sustained or is_sustained_conservative_approach - depends on the input
    is_high_quality = cur_article_assessment_obj.is_sustainable
    is_high_quality_conservative_approach = cur_article_assessment_obj.is_sustainable_conservative_approach
    if target_column == 'is_sustainable':
        cur_extracted_features['is_sustainable'] = True if is_high_quality else False
    elif target_column == 'is_sustainable_conservative_approach':
        cur_extracted_features['is_sustainable_conservative_approach'] = True if is_high_quality_conservative_approach else False
    else:
        raise NotImplementedError(f"The target_column was given as {target_column}. However, only [is_sustained, "
                                  f"is_sustained_conservative_approach] are valid for now. Please fix and run again")
    if extract_meta:
        cur_authors = calc_num_authors(cur_article_assessment_obj, usecase=usecase, until_promotion=until_promotion,
                                       max_revisions_to_use=max_revisions_to_use, normlize=False)
        cur_authors_normalized = calc_num_authors(cur_article_assessment_obj, usecase=usecase, until_promotion=until_promotion,
                                                  max_revisions_to_use=max_revisions_to_use, normlize=True)
        cur_revisions = calc_num_revisions(cur_article_assessment_obj, usecase=usecase, until_promotion=until_promotion,
                                           max_revisions_to_use=max_revisions_to_use, normalize=False)
        cur_revisions_normalized = calc_num_revisions(cur_article_assessment_obj, usecase=usecase, until_promotion=until_promotion,
                                                      max_revisions_to_use=max_revisions_to_use, normalize=True)
        try:
            cur_age_till_promotion = cur_article_assessment_obj.birth_to_fa_promotion if usecase == 'fa' else cur_article_assessment_obj.birth_to_ga_promotion
        except KeyError:
            cur_age_till_promotion = -1
        cur_extracted_features.update({'EDIT_num_authors': cur_authors,
                                       'EDIT_num_authors_normalized': cur_authors_normalized,
                                       'EDIT_num_revisions': cur_revisions,
                                       'EDIT_num_revisions_normalized': cur_revisions_normalized,
                                       'EDIT_time_to_promotion': cur_age_till_promotion,
                                       'SPECIAL_used_to_be_good_article': cur_article_assessment_obj.promoted_to_ga_then_fa})
    if extract_structural:
        cur_structural_features = (
            editing_structural_features_extractor(article_assessment_obj=cur_article_assessment_obj,
                                                  usecase=usecase, max_revisions_to_use=max_revisions_to_use,
                                                  until_promotion=until_promotion))
        cur_extracted_features.update(cur_structural_features)
    if extract_community_dynamics:
        cur_community_dynamics_features = (
            community_dynamics_extractor(article_assessment_obj=cur_article_assessment_obj, usecase=usecase,
                                         max_revisions_to_use=max_revisions_to_use,
                                         until_promotion=until_promotion))
        cur_extracted_features.update(cur_community_dynamics_features)
    if extract_creation_date:
        cur_extracted_features['creation_date'] = cur_article_assessment_obj.creation_date
    # we return a dict of size 1. This dict holds an inner dict with all the features.
    dict_to_return = {cur_article_assessment_obj.article_id: cur_extracted_features}
    #print(f"File {file} ended calculating the required measures.", flush=True)
    return dict_to_return


def discussions_features_extractor(files_path, cpus_to_use, last_revision_timestamp_per_article):
    start_time = datetime.now()
    problematic_cases = list()
    extracted_features = dict()
    file_path_to_page_id_mapping = [(int(os.path.split(fp)[1].split('.')[0]), fp) for fp in files_path]
    input_for_pool = list()
    for idx, (page_id, discussion_file_path) in enumerate(file_path_to_page_id_mapping):
        if page_id in last_revision_timestamp_per_article:
            input_for_pool.append((page_id, discussion_file_path, last_revision_timestamp_per_article[page_id], idx))
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        results = pool.starmap(features_extractor_from_discussions, input_for_pool)
        # merging the list of dicts into one dict
    for res in results:
        # if the values are a single value of -1, it means that the page is an unreliable one
        if list(res.values())[0] == -1:
            page_id = list(res.keys())[0]
            problematic_cases.append(page_id)
        else:
            extracted_features.update(res)
    end_time = datetime.now()
    code_duration = end_time - start_time
    extracted_features_df = pd.DataFrame.from_dict(extracted_features, orient='index')
    print(f"Discussion feature extraction ended. Created dataset shape: {extracted_features_df.shape}. "
          f"Elapsed time (sec.): {code_duration.seconds}. ", flush=True)
    return extracted_features_df, problematic_cases


def features_extractor_from_discussions(page_id, talkpgae_file_path, last_revision_timestamp, job_idx=-1):
    try:
        #load the file
        talkpage_data = load_and_decompress(filename=talkpgae_file_path)
    except EOFError:
        return {-1: -1}
    num_discussions = 0
    uncommneted_discussions = 0
    depth_zero_unreplied_comments_cnt = list()
    depth_zero_replied_comments_cnt = list()
    depth_zero_replied_min_times = list()
    mixed_authors_comments = 0  # these are wired cases, where two authors are sharing a comment (unreliable ones)
    authors_dict = defaultdict(int)
    authors_per_discussion = defaultdict(list)
    direct_user_interactions = defaultdict(int)
    indirect_user_interactions = defaultdict(int)
    sentiment_per_user = defaultdict(list)
    formality_per_user = defaultdict(list)
    politeness_per_user = defaultdict(list)
    toxicity_per_user = defaultdict(list)
    certainty_per_user = defaultdict(list)
    avg_depth = list()
    depth_zero_comments_avg = list()
    # looping over each item in the list and extracting information about the discussion
    for cd in talkpage_data:
        is_relevant = is_discussion_relevant(discussion_info=cd, last_revision_time_to_consider=last_revision_timestamp)
        if not is_relevant:
            continue
        if 'comments' not in cd:
            continue
        # if we got up to here, it means that the discussion is relevant, and we can loop over all comments in it
        # (in case there are, there should be...)
        discussion_id = str(cd['talk_page_id']) + '_' + str(cd['rev_id'])
        num_discussions += 1
        uncommneted_discussions += 1 if len(cd['comments']) == 0 else 0
        cur_discussion_depths = list() # holds the depth of all comments
        # pulling 2 features -- # of unrepaired comments and the avg time when replied
        res_dict = extract_unreplied_and_time_to_reply_comments(comments_list=cd['comments'])
        depth_zero_unreplied_comments_cnt.append(res_dict['unreplied_comments'])
        depth_zero_replied_comments_cnt.append(res_dict['replied_comments'])
        depth_zero_replied_min_times.extend(res_dict['time_to_response_list'])
        for comment in cd['comments']:
            cur_author = comment['author']
            authors_dict[cur_author] += 1
            cur_discussion_depths.append(comment['depth'])
            authors_per_discussion[discussion_id].append(cur_author)
            if 'refers_to' in comment and comment['refers_to'] is not None:
                direct_user_interactions[(cur_author, comment['refers_to'])] += 1
            if 'root_author' in comment and comment['root_author'] is not None:
                indirect_user_interactions[(cur_author, comment['root_author'])] += 1
            num_talk_regex = re.findall("(talk)", comment['text'])
            mixed_authors_comments += 1 if len(num_talk_regex) > 1 else 0
            # we make sure that the predictions are in place and are not None (if sentiment is None, then also others)
            if 'dl_preds' in comment and comment['dl_preds'] is not None:
                sentiment_per_user[cur_author].append(comment['dl_preds']['sentiment'])
                formality_per_user[cur_author].append(comment['dl_preds']['formality'])
                politeness_per_user[cur_author].append(comment['dl_preds']['politeness'])
                toxicity_per_user[cur_author].append(comment['dl_preds']['toxicity'])
                certainty_per_user[cur_author].append(comment['dl_preds']['certainty'])
        cur_discussion_avg_depths = np.nanmean(cur_discussion_depths) if len(cur_discussion_depths) > 0 else None
        cur_discussion_depth_zero_comments_avg = sum([1 for d in cur_discussion_depths if d == 0]) / len(cur_discussion_depths) * 1.0 if len(cur_discussion_depths) > 0 else None
        avg_depth.append(cur_discussion_avg_depths)
        depth_zero_comments_avg.append(cur_discussion_depth_zero_comments_avg)
    # after the big loop, we can average over all discussions
    try:
        depth_zero_unreplied_comments_perc = (sum(depth_zero_unreplied_comments_cnt) /
                                              (sum(depth_zero_replied_comments_cnt) + sum(depth_zero_unreplied_comments_cnt)))
    except ZeroDivisionError:
        depth_zero_unreplied_comments_perc = None
    depth_zero_reply_in_seconds = np.nanmean(depth_zero_replied_min_times) if depth_zero_replied_min_times else None
    avg_depth = np.nanmean(avg_depth) if any(avg_depth) else None
    depth_zero_comments_avg = np.nanmean(depth_zero_comments_avg) if any(depth_zero_comments_avg) else None
    # most of the feature we calculate come from the next function
    features_dict = agg_measures_to_single_value(authors_dict, authors_per_discussion, sentiment_per_user,
                                                 formality_per_user, politeness_per_user, toxicity_per_user,
                                                 certainty_per_user)
    # another set of features are based on a network we build and extract the # of triangles between users
    if direct_user_interactions:
        direct_interactions_graph = build_social_network_from_edge_list(direct_user_interactions, directed=False)
        # Calculate the number of triangles
        direct_interactions_triangles = nx.triangles(direct_interactions_graph)
        # Total number of triangles (each triangle is counted three times, once at each vertex), normalized by # nodes
        direct_interactions_avg_triangles = (sum(direct_interactions_triangles.values()) // 3) / len(direct_interactions_graph.nodes())
    else:
        direct_interactions_avg_triangles = 0
    if indirect_user_interactions:
        indirect_interactions_graph = build_social_network_from_edge_list(indirect_user_interactions, directed=False)
        indirect_interactions_triangles = nx.triangles(indirect_interactions_graph)
        indirect_interactions_avg_triangles = (sum(indirect_interactions_triangles.values()) // 3) / len(indirect_interactions_graph.nodes())
    else:
        indirect_interactions_avg_triangles = 0
    # the other set of features we calculated earlier in the loop and now we add them
    features_dict.update({'DISCUSSIONS_num_discussions': num_discussions,
                          'DISCUSSIONS_mixed_authors_comments': mixed_authors_comments,
                          'DISCUSSIONS_direct_user_interactions_cnt': len(direct_user_interactions),
                          'DISCUSSIONS_indirect_user_interactions_cnt': len(indirect_user_interactions),
                          'DISCUSSIONS_avg_triangles_direct_interactions': direct_interactions_avg_triangles,
                          'DISCUSSIONS_avg_triangles_indirect_interactions': indirect_interactions_avg_triangles,
                          'DISCUSSIONS_avg_depth': avg_depth, 'DISCUSSIONS_depth_zero_comments_avg': depth_zero_comments_avg,
                          'DISCUSSIONS_lvl_zero_unreplied_comments_perc': depth_zero_unreplied_comments_perc,
                          'DISCUSSIONS_lvl_zero_reply_in_seconds': depth_zero_reply_in_seconds
                          })
    # we return a dict of size 1. This dict holds an inner dict with all the features.
    dict_to_return = {page_id: features_dict}
    # print(f"File {file} ended calculating the required measures.", flush=True)
    return dict_to_return


def extract_unreplied_and_time_to_reply_comments(comments_list):
    depth_zero_comments_time = dict()
    depth_zero_replied_min_time = dict()
    for cur_comment in comments_list:
        # maybe what is here is too naive
        comment_time = cur_comment['time'] if 'UTC' not in cur_comment['time'] else cur_comment['time'].split('(UTC)')[0]
        try:
            comment_time_parsed = parse(comment_time)
        except ParserError:
            continue
            #print("ERROR HERE WITH THE PARSING")
        cur_comment_id = cur_comment['comment_id']
        # saving information that helps us understand the % of replied comments
        if cur_comment['depth'] == 0:
            depth_zero_comments_time[cur_comment_id] = comment_time_parsed
        else:
            replies_to = cur_comment['root_comment_id']
            existing_min_time = depth_zero_replied_min_time[replies_to] if replies_to in depth_zero_replied_min_time else -1
            if (existing_min_time == -1 or comment_time_parsed < existing_min_time) and replies_to in depth_zero_comments_time:
                depth_zero_replied_min_time[replies_to] = comment_time_parsed
    # calculating the time it took to reply (in cases the comment was replied)
    time_to_response_list = list()
    for com_id, reply_time in depth_zero_replied_min_time.items():
        time_to_response_list.append((reply_time - depth_zero_comments_time[com_id]).seconds)
    unreplied_comments = sum([1 for com_id, _ in depth_zero_comments_time.items() if com_id not in depth_zero_replied_min_time])
    replied_min_times = list(depth_zero_replied_min_time.values())
    return {'unreplied_comments': unreplied_comments, 'replied_comments': len(depth_zero_replied_min_time),
            'time_to_response_list': time_to_response_list}


def agg_measures_to_single_value(authors_dict, authors_per_discussion, sentiment_per_user, formality_per_user,
                                 politeness_per_user, toxicity_per_user, certainty_per_user):
    gini_measure = gini_coefficient(authors_dict) if authors_dict else None
    num_authors = len(authors_dict)
    num_comments = sum(authors_dict.values())
    if authors_per_discussion:
        avg_authors_per_discussion = np.mean(
            [len(set(authors_list)) for discussion_id, authors_list in authors_per_discussion.items()])
        median_authors_per_discussion = np.median(
            [len(set(authors_list)) for discussion_id, authors_list in authors_per_discussion.items()])
    else:
        avg_authors_per_discussion = None
        median_authors_per_discussion = None
    # for each of the textual features (e.g., sentiment) we calc two measures - overall one and per user one.

    # A. Overall measures - average over all values (allowing heavy users to bias this measure)
    sentiment = list(chain.from_iterable(sentiment_per_user.values()))
    formality = list(chain.from_iterable(formality_per_user.values()))
    politeness = list(chain.from_iterable(politeness_per_user.values()))
    toxicity = list(chain.from_iterable(toxicity_per_user.values()))
    certainty = list(chain.from_iterable(certainty_per_user.values()))
    # sentiment is in the [0,1] range
    sentiment_mean = np.nanmean(sentiment) if any(sentiment) else None
    neg_sentiment_perc = (sum([1 for s in sentiment if s < 0.2]) / len(sentiment)) if sentiment else None
    pos_sentiment_perc = (sum([1 for s in sentiment if s > 0.8]) / len(sentiment)) if sentiment else None
    # formality is in the [0,1] range
    formality_mean = np.nanmean(formality) if any(formality) else None
    high_formality_perc = (sum([1 for s in formality if s > 0.8]) / len(formality)) if formality else None
    # politeness is in the [1,5] range
    politeness_mean = np.nanmean(politeness) if any(politeness) else None
    low_politeness_perc = (sum([1 for s in politeness if s < 3]) / len(politeness)) if politeness else None
    # toxicity is in the [0,1] range
    toxicity_mean = np.nanmean(toxicity) if any(toxicity) else None
    toxic_perc = (sum([1 for s in toxicity if s > 0.2]) / len(toxicity)) if toxicity else None
    # certainty is in the [1,5] range
    certainty_mean = np.nanmean(certainty) if any(certainty) else None
    high_certainty_perc = (sum([1 for s in certainty if s > 4.75]) / len(certainty)) if certainty else None

    # B. Mean user measures - average per user and then average over values (NOT allowing heavy users to bias)
    sentiment_mean_per_user = [np.nanmean(values) for user, values in sentiment_per_user.items()]
    user_agg_sentiment_mean = np.nanmean(sentiment_mean_per_user) if sentiment_mean_per_user else None
    user_agg_sentiment_median = np.nanmedian(sentiment_mean_per_user) if sentiment_mean_per_user else None

    formality_mean_per_user = [np.nanmean(values) for user, values in formality_per_user.items()]
    user_agg_formality_mean = np.nanmean(formality_mean_per_user) if formality_mean_per_user else None
    user_agg_formality_median = np.nanmedian(formality_mean_per_user) if formality_mean_per_user else None

    politeness_mean_per_user = [np.nanmean(values) for user, values in politeness_per_user.items()]
    user_agg_politeness_mean = np.nanmean(politeness_mean_per_user) if politeness_mean_per_user else None
    user_agg_politeness_median = np.nanmedian(politeness_mean_per_user) if politeness_mean_per_user else None

    toxicity_mean_per_user = [np.nanmean(values) for user, values in toxicity_per_user.items()]
    user_agg_toxicity_mean = np.nanmean(toxicity_mean_per_user) if toxicity_mean_per_user else None
    user_agg_toxicity_median = np.nanmedian(toxicity_mean_per_user) if toxicity_mean_per_user else None

    certainty_mean_per_user = [np.nanmean(values) for user, values in certainty_per_user.items()]
    user_agg_certainty_mean = np.nanmean(certainty_mean_per_user) if toxicity_mean_per_user else None
    user_agg_certainty_median = np.nanmedian(certainty_mean_per_user) if toxicity_mean_per_user else None

    features_dict = {'DISCUSSIONS_num_comments': num_comments, 'DISCUSSIONS_num_authors': num_authors,
                     'DISCUSSIONS_gini_measure': gini_measure,
                     'DISCUSSIONS_avg_authors': avg_authors_per_discussion,
                     'DISCUSSIONS_median_authors': median_authors_per_discussion,
                     'DISCUSSIONS_sentiment_mean': sentiment_mean, 'DISCUSSIONS_neg_sentiment_perc': neg_sentiment_perc,
                     'DISCUSSIONS_pos_sentiment_perc': pos_sentiment_perc,
                     'DISCUSSIONS_user_agg_sentiment_mean': user_agg_sentiment_mean,
                     'DISCUSSIONS_user_agg_sentiment_median': user_agg_sentiment_median,
                     'DISCUSSIONS_formality_mean': formality_mean,
                     'DISCUSSIONS_high_formality_perc': high_formality_perc,
                     'DISCUSSIONS_user_agg_formality_mean': user_agg_formality_mean,
                     'DISCUSSIONS_user_agg_formality_median': user_agg_formality_median,
                     'DISCUSSIONS_politeness_mean': politeness_mean,
                     'DISCUSSIONS_low_politeness_perc': low_politeness_perc,
                     'DISCUSSIONS_user_agg_politeness_mean': user_agg_politeness_mean,
                     'DISCUSSIONS_user_agg_politeness_median': user_agg_politeness_median,
                     'DISCUSSIONS_toxicity_mean': toxicity_mean, 'DISCUSSIONS_toxic_perc': toxic_perc,
                     'DISCUSSIONS_user_agg_toxicity_mean': user_agg_toxicity_mean,
                     'DISCUSSIONS_user_agg_toxicity_median': user_agg_toxicity_median,
                     'DISCUSSIONS_certainty_mean': certainty_mean,
                     'DISCUSSIONS_high_certainty_perc': high_certainty_perc,
                     'DISCUSSIONS_user_agg_certainty_mean': user_agg_certainty_mean,
                     'DISCUSSIONS_user_agg_certainty_median': user_agg_certainty_median,
                     }
    return features_dict


def is_discussion_relevant(discussion_info, last_revision_time_to_consider):
    last_revision_time_to_consider_as_datetime = datetime.strptime(last_revision_time_to_consider, date_format)
    cur_time = discussion_info['time'].replace(",", "")
    time_formats = ['%H:%M %d %b %Y (%Z)', '%H:%M %b %d %Y (%Z)', '%H:%M %d %B %Y (%Z)', '%H:%M:%S %Y-%m-%d (%Z)', '%H:%M %B %d %Y (%Z)']
    time_format_testing_idx = 0
    while True:
        try:
            discussion_datetime = datetime.strptime(cur_time, time_formats[time_format_testing_idx])
            break
        except ValueError as e:
            time_format_testing_idx += 1
            if time_format_testing_idx == len(time_formats):
                #print(cur_time) # printing for debugging if needed
                return False
    if discussion_datetime < last_revision_time_to_consider_as_datetime:
        return True
    # if we did not return True by now, we return False by default
    return False


def users_overlap_features_extractor(discussion_files_path, pickle_files_path, last_revision_timestamp_per_article,
                                     usecase, until_promotion, max_revisions_to_use, cpus_to_use=100):
    start_time = datetime.now()
    # mapping of all files into their correspinding pageID
    discussion_file_path_to_page_id_dict = {int(os.path.split(fp)[1].split('.')[0]): fp for fp in discussion_files_path}
    pickle_file_path_to_page_id_dict = {int(os.path.split(fp)[1].split('.')[0]): fp for fp in pickle_files_path}
    # we do not really all the pages, but rather only those that appear in the last_revision_timestamp_per_article
    discussion_file_path_to_page_id_dict = {page_id: path for page_id, path in discussion_file_path_to_page_id_dict.items() if page_id in last_revision_timestamp_per_article}
    pickle_file_path_to_page_id_dict = {page_id: path for page_id, path in pickle_file_path_to_page_id_dict.items() if page_id in last_revision_timestamp_per_article}
    input_for_pool = list()
    for page_id, cur_pickle_file_path in pickle_file_path_to_page_id_dict.items():
        try:
            cur_discussion_file_path = discussion_file_path_to_page_id_dict[page_id]
            cur_last_revisions_timestamp = last_revision_timestamp_per_article[page_id]
        except KeyError:
            continue

        input_for_pool.append((page_id, cur_pickle_file_path, cur_discussion_file_path, usecase, until_promotion,
                               max_revisions_to_use, cur_last_revisions_timestamp))

    # starting the mp
    pool = mp.Pool(processes=cpus_to_use)
    with pool as pool:
        results = pool.starmap(features_extractor_of_users_overlap, input_for_pool)
        # merging the list of dicts into one dict
    extracted_features = dict()
    problematic_cases = list()
    for res in results:
        # if the values are a single value of -1, it means that the page is an unreliable one
        if list(res.values())[0] == -1:
            page_id = list(res.keys())[0]
            problematic_cases.append(page_id)
        else:
            extracted_features.update(res)
    extracted_features_df = pd.DataFrame.from_dict(extracted_features, orient='index')
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Users_overlap feature extraction ended. Created dataset shape: {extracted_features_df.shape}. "
          f"Elapsed time (sec.): {code_duration.seconds} seconds.", flush=True)
    return extracted_features_df, problematic_cases


def features_extractor_of_users_overlap(page_id, pickle_file_path, discussion_file_path, usecase,
                                        until_promotion, max_revisions_to_use, last_revision_timestamp):
    try:
        cur_article_assessment_obj = pickle.load(open(pickle_file_path, "rb"))
    except EOFError:
        return {-1: -1}
    # we validate the data in hand before we process with features extraction
    is_valid = cur_article_assessment_obj.is_obj_valid(usecase)
    if not is_valid:
        return {-1: -1}
    # for each article we extract the (max) date that we use for modeling. This is later used by other functions
    revision_level_info_subset = cur_article_assessment_obj.filter_revision_level_info(usecase=usecase,
                                                                                       until_promotion=until_promotion,
                                                                                       max_revisions_to_use=max_revisions_to_use)
    # calculating the distribution of edits per users that participated in editing
    edit_users_list = [int(u) for u in revision_level_info_subset['user'] if not np.isnan(u)]
    valid_edits_n = len(edit_users_list)
    edit_users_dist = defaultdict(float, {user: cnt/valid_edits_n*1.0 for user, cnt in dict(Counter(edit_users_list)).items()})
    # load the discussions file
    try:
        talkpage_data = load_and_decompress(filename=discussion_file_path)
    except EOFError:
        return {-1: -1}
    # looping over all the discussions
    talkers_dict = defaultdict(int)
    for cd in talkpage_data:
        is_relevant = is_discussion_relevant(discussion_info=cd, last_revision_time_to_consider=last_revision_timestamp)
        if not is_relevant or 'comments' not in cd:
            continue
        for comment in cd['comments']:
            cur_author_id = comment['author_id']
            if cur_author_id is not None:
                talkers_dict[cur_author_id] += 1
    valid_discussions_n = sum(talkers_dict.values())
    discussions_users_dist = defaultdict(float, {user: cnt/valid_discussions_n*1.0 for user, cnt in dict(Counter(talkers_dict)).items()})
    overlap_n = len(set(edit_users_dist.keys()).intersection(set(discussions_users_dist.keys())))
    overlap_div_by_editors = overlap_n / len(edit_users_dist) if edit_users_dist else None
    overlap_div_by_talkers = overlap_n / len(discussions_users_dist) if discussions_users_dist else None
    # a more sophisticated feature (taking into account the contribution of each user)
    all_users_found = set(list(edit_users_dist.keys()) + list(discussions_users_dist.keys()))
    dist_per_user = {u: (discussions_users_dist[u] - edit_users_dist[u])**2 for u in all_users_found}
    distributions_diff = sum(dist_per_user.values()) / len(dist_per_user) if dist_per_user else None
    return {page_id: {'COMPOSITION_users_overlap_div_by_editors': overlap_div_by_editors,
                      'COMPOSITION_users_overlap_div_by_talkers': overlap_div_by_talkers,
                      'COMPOSITION_distributions_diff': distributions_diff}
            }


def generate_model_workspace(model_version, target_column, usecase, output_folder, seed, folds_k,
                             use_edit_features, use_team_composition_features, use_network_features,
                             use_discussions_features, use_topic_features, use_user_experience_features,
                             use_special_features, max_revisions_to_use, max_year_to_include, until_promotion,
                             classification_model, specific_column_to_use, bootstrap_folds, filter_unreliable_cases):
    # checking if the folder of the model already existing, we cannot run the code and return an error
    saving_folder = opj(output_folder, str(model_version))
    if os.path.isdir(saving_folder):
        raise TypeError(f"Folder {saving_folder} exists already. Either delete the folder and rerun or "
                        f"provide a different model version.")
    else:
        os.makedirs(saving_folder)
    model_info_dict = {'model_version': str(model_version), 'target_column': target_column, 'usecase': usecase,
                       'output_folder': output_folder, 'seed': seed, 'folds_k': folds_k,
                       'use_edit_features': use_edit_features, 'use_team_composition_features': use_team_composition_features,
                       'use_network_features': use_network_features, 'use_discussions_features': use_discussions_features,
                       'use_topic_features': use_topic_features, 'use_user_experience_features': use_user_experience_features,
                       'use_special_features': use_special_features, 'max_year_to_include': max_year_to_include,
                       'until_promotion': until_promotion, 'max_revisions_to_use': max_revisions_to_use,
                       'classification_model': str(classification_model),
                       'specific_column_to_use': specific_column_to_use, 'bootstrap_folds': bootstrap_folds,
                       'filter_unreliable_cases': filter_unreliable_cases}
    print(f"Current model metadata:\n{model_info_dict}", flush=True)
    model_params_f_name = opj(saving_folder, 'model_params.json')
    with open(model_params_f_name, 'w', encoding='utf-8') as outfile:
        json.dump(model_info_dict, outfile, ensure_ascii=False, indent=4)
    return saving_folder


def find_optimal_threshold(y_true, predicted_proba, average='macro'):
    # Assuming y_true are the true labels and y_pred_probs are the predicted probabilities from the classifier
    # Adjust the threshold values as needed
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if type(y_true) is not list:
        y_true = list(y_true)
    # Compute F1 scores for each threshold. We split it into 2 options, since when running with binary average,
    # the label has to be given, and in our case it is the minority class (0)
    if average == 'binary':
        f1_scores = [f1_score(y_true, (predicted_proba > threshold).astype(int), pos_label=0, average=average)
                     for threshold in thresholds]
    else:
        f1_scores = [f1_score(y_true, (predicted_proba > threshold).astype(int), average=average)
                     for threshold in thresholds]

    # Find the index of the threshold that maximizes the F1 score
    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_index]
    return optimal_threshold


def compress_and_save(json_lines, filename):
    with bz2.open(filename, 'wt') as f:
        writer = jsonlines.Writer(f)
        try:
            writer.write_all(json_lines)
        except UnicodeEncodeError:
            return -1
    return 0


def load_and_decompress(filename):
    with bz2.open(filename, 'rt') as f:
        reader = jsonlines.Reader(f)
        return list(reader)


def gini_coefficient(contributions_dict):
    x = np.array(list(contributions_dict.values()))
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def extract_topic_features(data_path, required_page_ids):
    start_time = datetime.now()
    # these are the values that we will assign to the string values. None is a special case where importance exist but
    # not specific value is set
    importance_mapping = {'low': 1, 'mid': 2, 'None': 3, 'high': 4, 'top': 5}
    data_per_page = pd.read_csv(data_path)
    data_per_page['importance'] = data_per_page['importance'].replace({np.nan: 'None'})
    # taking only the page required
    data_per_page_filtered = data_per_page[data_per_page['page_id'].isin(required_page_ids)].copy()
    projects_counter = Counter(data_per_page_filtered['project_name']).most_common() # most_common is for sorting
    topics_data_as_list = list()
    column_names = list()
    # looping over each topic and extracting info per page_id (notice! topics are sorted by frequency which is great)
    for idx, (p_name, _) in enumerate(dict(projects_counter).items()):
        # first we make an "empty" dict, which later will be filled with values
        cur_project_data_mapped_to_ints = dict.fromkeys(required_page_ids, 0)
        existing_data = data_per_page_filtered[data_per_page_filtered['project_name'] == p_name].copy()
        # looping over each row and filling a dict with values
        existing_data.set_index('page_id', inplace=True)
        existing_data_as_dict = existing_data['importance'].to_dict()
        # looping over each page that the importance of the project we focus on does exist (for others it will be 0)
        for page_id, importance in existing_data_as_dict.items():
            cur_project_data_mapped_to_ints[page_id] = importance_mapping[importance]
        topics_data_as_list.append(cur_project_data_mapped_to_ints)
        # we add the column name, after lowercase and adding '_' between spaces. We also add a prefix of 'topic_X_'
        p_name_modified = 'TOPIC_' + str(idx) + '_' + p_name.lower().replace(' ', '_')
        column_names.append(p_name_modified)
    # now we can create a big df out of all data created
    topics_df = pd.DataFrame(topics_data_as_list).T
    topics_df.columns = column_names
    # Convert to binary DataFrame
    binary_topics_df = (topics_df != 0).astype(int)
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Topic feature extraction ended. Created dataset shape: {topics_df.shape}. "
          f"Elapsed time: {code_duration.seconds} seconds.", flush=True)
    return topics_df, binary_topics_df


def eval_classification_preds(true_values, preds, preds_proba, macro_optimal_th=0.5, binary_optimal_th=0.5):
    # first set of measures - macro with a threshold of 0.5 (default)
    macro_accuracy = accuracy_score(true_values, preds)
    macro_precision = precision_score(true_values, preds, average='macro')
    macro_recall = recall_score(true_values, preds, average='macro')
    macro_f1 = f1_score(true_values, preds, average='macro')
    macro_ap_score = average_precision_score(true_values, preds, average='macro')

    # 2nd set of measures - binary (in our case these are the unsustainable articles) with a threshold of 0.5 (default)
    binary_precision = precision_score(true_values, preds, average='binary', pos_label=0)
    binary_recall = recall_score(true_values, preds, average='binary', pos_label=0)
    binary_f1 = f1_score(true_values, preds, average='binary', pos_label=0)
    precision_at_2_perc = precision_at_k(y_true=true_values, y_scores=preds_proba, k=2, target_class=0)
    precision_at_5_perc = precision_at_k(y_true=true_values, y_scores=preds_proba, k=5, target_class=0)

    # 3rd set of measures - macro with an optimal threshold for macro
    preds_optimal_th_macro = (preds_proba > macro_optimal_th).astype(int)
    macro_accuracy_optimal_th = accuracy_score(true_values, preds_optimal_th_macro)
    macro_precision_optimal_th = precision_score(true_values, preds_optimal_th_macro, average='macro')
    macro_recall_optimal_th = recall_score(true_values, preds_optimal_th_macro, average='macro')
    macro_f1_optimal_th = f1_score(true_values, preds_optimal_th_macro, average='macro')

    # 4th set of measures -- binary (in our case these are the unsustainable articles) with an optimal threshold of
    preds_optimal_th_binary = (preds_proba > binary_optimal_th).astype(int)
    binary_accuracy_optimal_th = accuracy_score(true_values, preds_optimal_th_binary)
    binary_precision_optimal_th = precision_score(true_values, preds_optimal_th_binary, average='binary', pos_label=0)
    binary_recall_optimal_th = recall_score(true_values, preds_optimal_th_binary, average='binary', pos_label=0)
    binary_f1_optimal_th = f1_score(true_values, preds_optimal_th_binary, average='binary', pos_label=0)

    auc = roc_auc_score(true_values, preds_proba)
    eval_measures = {'n': len(preds), 'macro_optimal_th': macro_optimal_th, 'binary_optimal_th': binary_optimal_th,
                     'macro_accuracy': macro_accuracy, 'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f1': macro_f1,
                     'binary_precision': binary_precision, 'binary_recall': binary_recall, 'binary_f1': binary_f1,
                     'binary_prec_at_2_perc': precision_at_2_perc, 'binary_prec_at_10_perc': precision_at_5_perc,
                     'macro_accuracy_opt_th': macro_accuracy_optimal_th,
                     'macro_precision_opt_th': macro_precision_optimal_th,
                     'macro_recall_opt_th': macro_recall_optimal_th,
                     'macro_f1_opt_th': macro_f1_optimal_th,
                     'binary_accuracy_optimal_th': binary_accuracy_optimal_th,
                     'binary_precision_optimal_th': binary_precision_optimal_th,
                     'binary_recall_optimal_th': binary_recall_optimal_th,
                     'binary_f1_optimal_th': binary_f1_optimal_th,
                     'macro_auc': auc, 'macro_ap_score': macro_ap_score}
    return eval_measures


def precision_at_k(y_true, y_scores, k, target_class=1):
    """
    Calculates Precision@k for a specified target class (0 or 1). This function was written by ChatGPT

    Parameters:
    - y_true: array-like, shape (n_samples,)
        True binary labels (0 or 1).

    - y_scores: array-like, shape (n_samples,)
        Predicted scores or probabilities for the positive class.

    - k: float
        Percentage (between 0 and 100) representing the top k% of predictions to evaluate.

    - target_class: int
        The class for which you want to calculate precision. Should be either 0 or 1.

    Returns:
    - precision_at_k: float
        The precision at the top k% for the specified class.

    Example
    --------
    >>> y_true = np.array([     1,      0,      1,      1,      0,      0,      1,      0,      0,      1])  # True labels
    >>> y_scores = np.array([   0.9,    0.2,    0.75,   0.6,    0.1,    0.3,    0.8,    0.5,    0.4,    0.95])  # Predicted probabilities

    >>> k = 70  # Calculate Precision@30% (top 30% of predictions)
    >>> target_class = 1  # Precision for class 0 (negatives)

    >>> precision_k = precision_at_k(y_true, y_scores, k, target_class)
    >>> print(f"Precision@{k}% for class {target_class}: {precision_k:.2f}")
    """
    # Ensure k is a percentage between 0 and 100
    if k <= 0 or k > 100:
        raise ValueError("k should be a percentage between 0 and 100")

    # Ensure the target_class is either 0 or 1
    if target_class not in [0, 1]:
        raise ValueError("target_class should be either 0 or 1")

    # Number of predictions to consider
    cutoff = int(len(y_scores) * (k / 100.0))

    # Create a DataFrame with true labels and predicted scores
    df = pd.DataFrame({'true_label': y_true, 'score': y_scores})

    # Sort by predicted scores in descending order
    ascending = True if target_class == 0 else False
    df = df.sort_values('score', ascending=ascending)

    # Select the top k% predictions
    top_k = df.head(cutoff)

    # Calculate precision for the target class
    precision_at_k = (top_k['true_label'] == target_class).sum() / len(top_k)

    return precision_at_k


def exclude_unreliable_cases_from_modeling_df(data_df, target_column):
    # first issue - no demotion date, but sustained=0
    demotion_and_label_aligned = [False if demotion_is_na and gold_label == 0 else True for
                                  demotion_is_na, gold_label in
                                  zip(data_df['TIME_demotion_date'].isna(), data_df[target_column])]
    promotion_column = pd.to_datetime(data_df['TIME_promotion_date'])
    demotion_column = pd.to_datetime(data_df['TIME_demotion_date'])
    # Calculate the difference in days
    time_in_promotion = (demotion_column - promotion_column).dt.days
    reasonable_time_in_promotion = [False if t <= 30 else True for t in time_in_promotion]
    # now we'll have to get rid of both... this will be implemented in the production code!
    mask1 = pd.Series(reasonable_time_in_promotion, index=data_df.index)
    mask2 = pd.Series(demotion_and_label_aligned, index=data_df.index)
    # Combine the masks using bitwise AND (&) or OR (|) depending on your filtering logic
    combined_mask = mask1 & mask2  # Use | if you want to keep rows where either condition is True

    # Apply the mask to the DataFrame
    data_df_clean = data_df[combined_mask].copy()
    return data_df_clean
