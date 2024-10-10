import pandas as pd
from os.path import join as opj
from modeling.modeling_utils import load_and_decompress
import glob
import os
from tqdm import tqdm
from collections import Counter
import re
import mwparserfromhell
from datetime import datetime

base_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/'
talkpages_folder = opj(base_folder, 'wiki_generated_data', 'latest_talk_pages_sustained_articles')
output_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wikipedia_meta_data'


def wikiprojects_extractor_from_talkpages(talk_page_files):
    wikiprojects_per_page = list()
    unreached_pages = list()
    for cur_file in tqdm(talk_page_files):
        cur_page_id = int(os.path.split(cur_file)[-1].split('.jsonl.bz2')[0])
        cur_content = load_and_decompress(cur_file)
        # each file is a single line, hence we take place 0 which is the first row
        try:
            projects_found = extract_wiki_projects_from_text(cur_content[0]['content'])
        except IndexError:
            continue
        if len(projects_found) > 0:
            dicts_to_add = [{'page_id': cur_page_id, 'project_name': project_name, 'importance': importance}
                            for (project_name, importance) in projects_found]
            wikiprojects_per_page.extend(dicts_to_add)
    wikiprojects_per_page_df = pd.DataFrame(wikiprojects_per_page)
    return wikiprojects_per_page_df


def extract_wiki_projects_from_text(text):
    # Regular expression to match the entire template section for WikiProject
    wikicode = mwparserfromhell.parse(text)
    templates = wikicode.filter_templates()
    wikiproject_templates = [t for t in templates if 'WikiProject' in t and 'banner shell' not in t]

    pattern = re.compile(
        r'\{\{\s*WikiProject\s+([^\|\}\n]+?)\s*(?:\|([^}]*))?\s*\}\}',
        re.IGNORECASE | re.DOTALL
    )

    projects = []
    for wt in wikiproject_templates:
        matches = pattern.findall(str(wt))
        for match in matches:
            project_name = match[0].strip()
            # this will extarct the importance with all text following this string, we next will take the impo. part
            string_with_potential_importance = match[1].strip() if match[1] else None
            if string_with_potential_importance is not None:
                # here we extract only the 'Low', 'Mid' etc
                importance = [swpi.split('=')[1] for swpi in string_with_potential_importance.split('|') if 'importance' in swpi]
                # if found, it should include list with a single value (there shouldn't be more than a single importance
                importance = importance[0] if len(importance) > 0 else None
            else:
                importance = None
            importance_normalized = normalize_importance_string(importance)
            projects.append((project_name, importance_normalized))
    return projects


def normalize_importance_string(importance_string):
    valid_importance_strings = {'low', 'mid', 'high', 'top'}
    if importance_string is None:
        return importance_string
    # first, lower case
    importance_string_lc = importance_string.lower()
    # removing leading and trailing spaces
    importance_string_lc = importance_string_lc.strip()
    # we only allow 4 values of importance values. Any other value is probably an error in the parsing process
    if importance_string_lc not in valid_importance_strings:
        return None
    else:
        return importance_string_lc


if __name__ == "__main__":
    start_time = datetime.now()
    meta_data_per_article = pd.read_csv(opj(base_folder, 'wiki_generated_data', 'meta_data', 'sustainability_research_meta_info_per_article.csv'))
    page_ids_set = set(meta_data_per_article['page_id'].tolist())

    # the first way to get projects is by reading the talk pages and extracting from it the relevant info
    existing_talk_pages_jsonl = glob.glob(opj(talkpages_folder, '*.jsonl.bz2'))
    data_based_talk_pages = wikiprojects_extractor_from_talkpages(talk_page_files=existing_talk_pages_jsonl)#[0:1000])
    # last step is filtering low frequent projects. We do so since these are later converted to indicator features
    projects_counter = Counter(data_based_talk_pages['project_name'])
    most_common_projects = set([proj for (proj, freq) in Counter.most_common(projects_counter, n=250)])
    # filter unlogical projects (see below a few examples). This is based on a human analysis of all topics we observed
    most_common_projects_filtered = set([mcp for mcp in most_common_projects if mcp not in {'Articles for creation', 'Spoken Wikipedia', 'Guild of Copy Editors'}])
    data_based_talk_pages_filtered = data_based_talk_pages[data_based_talk_pages['project_name'].isin(most_common_projects_filtered)].copy()
    # sorting and saving to disk
    data_based_talk_pages_filtered.sort_values(by=['page_id'], axis=0, inplace=True)
    data_based_talk_pages_filtered.to_csv(opj(output_folder, 'pages_project_name_importance_only_modeled_articles.csv'),
                                          index=False)
    end_time = datetime.now()
    code_duration = end_time - start_time
    print(f"Code ended in {code_duration.total_seconds()} sec. DF of size {data_based_talk_pages_filtered.shape} "
          f"saved to {opj(output_folder)}.\n Overall, we mapped {len(set(data_based_talk_pages_filtered['page_id']))} "
          f"pages to their projects and importance for each of these pages.", flush=True)

    # that's the way to read it
    #data = pd.read_csv('/shared/3/projects/relationship-aware-networks-wikipedia/wikipedia_meta_data/pages_project_name_importance_only_modeled_articles.csv')
    ###### LEFTOVERS ############
    # the second way to get the projects, is using local information we have about projects and pages
    # loading the information we have about the projects which each page is associated with
    # this second method did not yield interesting enough results (on top of the first method). So neglecting it for now
    """
    wikiprojects_info_files = glob.glob(opj(base_folder, 'wikipedia_meta_data', 'wikiprojects_to_wikipages_data', '*.csv'))
    # An extra file that shlould be excluded appear in this folder, removing it
    wikiprojects_info_files = [wif for wif in wikiprojects_info_files if not wif.endswith('wikiprojects_ids_to_names.csv')]
    # looping each of those files and extracting the relevant info
    wikiprojects_data_filtered_list = list()
    for wif in tqdm(wikiprojects_info_files):
        cur_wikiprojects_data = pd.read_csv(opj(base_folder, 'wikipedia_meta_data',
                                                'wikiprojects_to_wikipages_data', wif), encoding='utf-8')

        cur_wikiprojects_data_filtered = cur_wikiprojects_data[cur_wikiprojects_data['pa_page_id'].isin(page_ids_set)].copy()
        cur_wikiprojects_data_filtered = cur_wikiprojects_data_filtered[['pa_page_id', 'pa_project_id', 'pa_importance']].copy()
        cur_wikiprojects_data_filtered['pa_importance'] = [importance[2:len(importance) - 1] for importance in cur_wikiprojects_data_filtered['pa_importance']]
        wikiprojects_data_filtered_list.append(cur_wikiprojects_data_filtered)
    wikiprojects_data_filtered = pd.concat(wikiprojects_data_filtered_list, axis=0, ignore_index=True)
    print(f'There are {len(page_ids_set)} unique pages that we are interested in. We found {wikiprojects_data_filtered.shape[0]} '
          f'associated projects matching (each page can be associated with more than one project)')
    # the last thing we need is to map the projectID with the project name. For this we have a file with partial mapping
    wikiprojects_ids_to_names = pd.read_csv(opj(base_folder, 'wikipedia_meta_data',
                                                'wikiprojects_to_wikipages_data', 'wikiprojects_ids_to_names.csv'))
    # removing projects that have '/' in their name. These are subproject of big ones
    filter_flag = [False if '/' in title else True for title in wikiprojects_ids_to_names['pap_project_title']]
    wikiprojects_ids_to_names = wikiprojects_ids_to_names[filter_flag].copy()
    # join between the two tables
    wikiprojects_data_filtered_with_proj_names = wikiprojects_data_filtered.merge(wikiprojects_ids_to_names,
                                                                                  left_on='pa_project_id',
                                                                                  right_on='pap_project_id',
                                                                                  how='inner')
    # converting the df column names to be similar to the first method + dropiing unused columns
    wikiprojects_data_filtered_with_proj_names.rename(columns={'pa_page_id': 'page_id',
                                                               'pap_project_title': 'project_name',
                                                               'pa_importance': 'importance'}, inplace=True)
    wikiprojects_data_filtered_with_proj_names.drop(['pa_project_id', 'pap_project_id', 'pap_parent_id'],
                                                    axis=1, inplace=True)
    data_based_external_info = wikiprojects_data_filtered_with_proj_names.copy()
    """
