# the purpose of this code is to generate a csv with mera information about each article
import pandas as pd
import glob
from os.path import join as opj
import requests

def login_to_wikipedia(username='Avrahami-isr', password='Givemeusap2'):
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"

    # Step 1: Get a login token
    params = {
        "action": "query",
        "meta": "tokens",
        "type": "login",
        "format": "json"
    }
    response = session.get(url, params=params)
    data = response.json()
    login_token = data['query']['tokens']['logintoken']

    # Step 2: Log in
    params = {
        "action": "login",
        "lgname": username,
        "lgpassword": password,
        "lgtoken": login_token,
        "format": "json"
    }
    response = session.post(url, data=params)
    login_result = response.json()

    if login_result['login']['result'] != 'Success':
        raise Exception("Login failed: {}".format(login_result['login']['reason']))

    return session


def get_titles_from_pageids(session, pageids, batch_size=50):
    """
    Given a list of Wikipedia pageIDs, fetches the corresponding titles in batches.
    """
    endpoint = "https://en.wikipedia.org/w/api.php"
    id_to_title_dict = dict()

    for i in range(0, len(pageids), batch_size):
        batch = pageids[i:i + batch_size]
        params = {
            "action": "query",
            "pageids": '|'.join(map(str, batch)),
            "format": "json"
        }

        response = session.get(endpoint, params=params)
        data = response.json()

        if 'query' in data and 'pages' in data['query']:
            pages = data['query']['pages']
            for pageid in batch:
                if str(pageid) in pages and 'title' in pages[str(pageid)]:
                    title = pages[str(pageid)]['title']
                    id_to_title_dict[pageid] = title
    return id_to_title_dict

metadata_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/meta_data/'

if __name__ == "__main__":
    metadata_files = glob.glob(opj(metadata_folder, 'metadata*.csv'))
    meta_data_all_req_pages = list()

    for mf in metadata_files:
        cur_metadata_df = pd.read_csv(mf)
        rel_rows = cur_metadata_df[(cur_metadata_df['is_now_fa_or_ga']) | (cur_metadata_df['ever_was_fa_or_ga'])]
        meta_data_all_req_pages.append(rel_rows)
        #page_ids_needed.extend(rel_rows['page_id'].unique())
    meta_data_all_req_pages_df = pd.concat(meta_data_all_req_pages)
    # joining this table with the meta information we have regrading the promotion/demotion dates
    # I checked, and these two have the same page_id values! 46304 such ones
    promotion_demotion_info = pd.read_csv(opj(metadata_folder, 'promotion_demotion_dates.csv'))
    promotion_demotion_info.rename(columns={'Unnamed: 0': 'page_id'}, inplace=True)
    meta_data_joined = pd.merge(meta_data_all_req_pages_df, promotion_demotion_info, on='page_id')
    page_ids_to_look_for = list(meta_data_joined['page_id'])
    session = login_to_wikipedia()
    page_id_to_name_mapping = get_titles_from_pageids(session=session, pageids=page_ids_to_look_for, batch_size=50)
    # seems like that we were able to map almost all cases (46303 out of 46304). The missing one does not actually exist
    meta_data_joined['page_title'] = meta_data_joined['page_id'].map(page_id_to_name_mapping)
    # saving the joined df to disk
    meta_data_joined.to_csv(opj(metadata_folder, 'sustainability_research_meta_info_per_article.csv'), index=False)