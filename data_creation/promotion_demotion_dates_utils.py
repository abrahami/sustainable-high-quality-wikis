import re
import requests
import bz2
import jsonlines
import pandas as pd
import glob
from os.path import join as opj
import os
import time
from tqdm import tqdm
from datetime import datetime
import numpy as np
from modeling.modeling_utils import compress_and_save, load_and_decompress
date_formats = [
        '%Y-%m-%d',  # YYYY-MM-DD
        '%d %B %Y', # 14 January 2008
        '%d %B, %Y',  # 14 January, 2008
        '%m/%d/%Y',  # MM/DD/YYYY
        '%-m/%-d/%y',  # M/D/YY (with optional leading zero suppression)
        '%b %d, %Y',  # Month D, YYYY (e.g., Mar 5, 2024)
        '%B %d, %Y', # Month D, YYYY (e.g., Mar 5, 2024), not sure how this differs from the previou one
        '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
        '%Y-%m-%d, %H:%M:%S', # YYYY-MM-DD, HH:MM:SS
        "%Y-%m-%dT%H:%M:%SZ", # YYYY-MM-DDTHH:MM:SS
        '%Y.%m.%d',  # YYYY.MM.DD
        '%d-%b-%Y',  # DD-Mon-YYYY (e.g., 15-Jan-2023)
        '%H:%M, %d %B %Y',  # HH:MM, DD Month YYYY (e.g., 23:29, 12 August 2006)
        '%H:%M %d %B %Y',  # HH:MM, DD Month YYYY (e.g., 23:29 12 August 2006)
        '%Y-%m-%d, %H:%M:%S',  # YYYY-MM-DD, HH:MM:SS (e.g., 2005-08-10, 03:26:13)
        '%H:%M, %d %b %Y', # 04:39, 18 Oct 2004
        '%B %d %Y',  # August 13 2005
        '%B %d, %Y', #August 13, 2005
        '%H:%M, %B %d, %Y', #19:44, October 11, 2003
        '%B %d %Y',  # August 4 2004
        '%B %d, %Y', # August 4, 2004
        '%d %b %Y', #26 Feb 2011
        '%H:%M:%S %d %B %Y',  # 01:02:56 20 April 2004
        '%H:%M:%S, %d %B %Y', #01:02:56, 20 April 2004
        '%H:%M %d %b %Y', #08:04 19 Jul 2003
        '%H:%M %B %d, %Y', #19:39 June 19, 2008
        '%d %b %Y', # 6 Sept 2006
        '%d %b, %Y',  # 6 Sept, 2006
        '%Y-%m-%d %H:%M',  # 2007-04-28 21:24
        '%Y-%m-%d, %H:%M', #2007-04-28, 21:24
        '%H:%M, %B %d %Y', #'20:02, May 20 2005'
        '%H:%M %B %d %Y',  # '20:02, May 20 2005'
        '%H:%M:%S, %B %d, %Y', #'13:57:19, July 20, 2004'
        '%H:%M:%S %B %d, %Y',  # '13:57:19 July 20, 2004'
        '%H:%M:%S %B %d %Y',  # '13:57:19 July 20 2004'
        '%Y-%m-%d, %H:%M', #'2015-08-12, 08:03'
        '%Y-%m-%d %H:%M',  # '2015-08-12 08:03'
        '%H:%M, %d %B, %Y', # '01:59, 24 July, 2007'
        '%H:%M, %d %B %Y',  # '01:59, 24 July 2007'
        '%H:%M %d %B, %Y',  # '01:59 24 July, 2007'
        '%H:%M %d %B %Y',  # '01:59 24 July 2007'
        '%H:%M, %d %B %Y (UTC)'  # HH:MM, DD Month YYYY (UTC) (e.g., 14:03, 23 August 2014 (UTC))
        '%H:%M %d %B %Y (UTC)'  # HH:MM, DD Month YYYY (UTC) (e.g., 14:03, 23 August 2014 (UTC))
    ]
new_date_format = "%Y-%m-%dT%H:%M:%SZ"


def crawl_articles_timeline_from_talk_pages(metadata_files_folder, saving_folder, overwrite_existing_files=False):
    metadata_files = glob.glob(opj(metadata_files_folder, 'metadata*.csv'))
    page_ids_to_extract = list()
    for mf in metadata_files:
        cur_metadata_df = pd.read_csv(mf)
        rel_rows = cur_metadata_df[(cur_metadata_df['is_now_fa_or_ga']) | (cur_metadata_df['ever_was_fa_or_ga'])]
        # by the name of the metadata file, can we infer in which folder the json file is located
        page_ids_to_extract.extend(rel_rows['page_id'])
    # making sure we don't have duplications
    page_ids_to_extract = list(set(page_ids_to_extract))
    # checking which files already exist (in case we wish not to overwrite those)
    if not overwrite_existing_files:
        existing_files = glob.glob(opj(saving_folder, '*.jsonl.bz2'))
        existing_page_ids = set([int(os.path.split(ef)[-1].split('.jsonl.bz2')[0]) for ef in existing_files])
        page_ids_to_extract = [pite for pite in page_ids_to_extract if pite not in existing_page_ids]
        print(f'Found {len(existing_page_ids)} existing saved objects, '
              f'so processing only {len(page_ids_to_extract)} pages', flush=True)
    session = login_to_wikipedia()
    results = []
    successful_savings = list()
    unsuccessful_savings = list()
    # looping over each
    for page_id in tqdm(page_ids_to_extract):
        try:
            content = get_talk_page_content(session, page_id)
            results.append((page_id, content))
            saving_obj = [{'page_id': page_id, 'content': content}]
            # saving the content to a json file
            saving_output = compress_and_save(json_lines=saving_obj,
                                              filename=opj(saving_folder, str(page_id) + '.jsonl.bz2'))
            if saving_output == 0:
                successful_savings.append(page_id)
            else:
                unsuccessful_savings.append(page_id)
        # too many requests error
        except requests.exceptions.HTTPError as e:
            # Sleep for a long time (50 min) to respect rate limits (5K per hour for registered users)
            print(f"Got a rate limited error, going to sleep for a while and then return for more crawling", flush=True)
            time.sleep(3000)  # Example: 1 second delay between requests
        # any other error - we'll break and rerun later
        except Exception as e:
            unsuccessful_savings.append(page_id)
            print(f"We encountered an error: {e} while processing page {page_id}.", flush=True)
            break
    print(f'unsuccessful_savings: {len(unsuccessful_savings)}, successful_savings: {len(successful_savings)}')


def article_promotion_demotion_dates_extractor_from_talkpages(talk_pages_jsonl):
    """
    Given a json talkpages (multiple!), this function extract the ga/fa promotion dates of each article
    :param talk_pages_jsonl: jsonlines object
        the jsonlines object of the relevant article
    :return:
    """
    promotion_demotion_details = dict()
    unreached_pages = list()
    for cur_file in tqdm(talk_pages_jsonl):
        cur_page_id = int(os.path.split(cur_file)[-1].split('.jsonl.bz2')[0])
        cur_content = load_and_decompress(cur_file)
        # each file is a single line, hence we take place 0 which is the first row
        try:
            history_sections = extract_history_template_sections(cur_content[0]['content'])
        except TypeError:
            continue
        if len(history_sections) > 1:
            print(f"Potential bug with page id {cur_page_id} (file name: {cur_file} -- Multiple history sections")
            history_sections = history_sections[0]
        # if no history section is found, we can still try to extract the info from the very basic structure of the page
        if len(history_sections) == 0:
            try:
                promoted_to, promotion_time = extract_promotion_date_from_simple_template(
                    cur_content[0]['content'].lower())
                ga_promotion_time = promotion_time if promoted_to == 'ga' else None
                ga_demotion_time = None
                fa_demotion_time = None
                fa_promotion_time = promotion_time if promoted_to == 'fa' else None
                current_status = promoted_to
            except IOError:
                unreached_pages.append(cur_page_id)
                continue
        else:
            history_section = history_sections[0]
            # adding a new line before any pipe, in case it does not exist. It occurs in some cases (e.g., "ArticleHistory|action1=GAN\n|action1date=15:34, 6 March 2011\n|"...)
            history_section = re.sub(r'([^\n])\|', r'\1\n|', history_section)
            parsed_article_history = parse_article_history(history_section)
            fa_promotion_time = extract_action_time(parsed_article_history, action_type='fac', result={'promoted'},
                                                    sort='first')
            ga_promotion_time = extract_action_time(parsed_article_history, action_type='gan',
                                                    result={'listed', 'passed'}, sort='first')
            fa_demotion_time = extract_action_time(parsed_article_history, action_type='far', result={'demoted'},
                                                   sort='first')
            ga_demotion_time = extract_action_time(parsed_article_history, action_type='gar', result={'delisted'},
                                                   sort='first')
            current_status = parsed_article_history[-1]['currentstatus'].lower() if 'currentstatus' in \
                                                                                    parsed_article_history[-1] else None
            # in case we were not able to get the current status, we make another try to pull it in another way
            if current_status is None:
                current_status = extract_article_status(text=cur_content[0]['content'])
        # end of the loop, we update the dict
        promotion_demotion_details[cur_page_id] = {'talkpage_ga_promotion': ga_promotion_time,
                                                   'talkpage_ga_demotion': ga_demotion_time,
                                                   'talkpage_fa_promotion': fa_promotion_time,
                                                   'talkpage_fa_demotion': fa_demotion_time,
                                                   'talkpage_current_status': current_status}
    # converting the dict to df
    promotion_demotion_df = pd.DataFrame.from_dict(promotion_demotion_details, orient='index')
    # converting the dates column to standard pandas format
    columns_to_convert = ['talkpage_fa_promotion', 'talkpage_ga_promotion', 'talkpage_fa_demotion', 'talkpage_ga_demotion']
    for ctc in columns_to_convert:
        new_column = convert_dates(promotion_demotion_df[ctc], date_formats)
        new_column_in_format = [nc.strftime(new_date_format) if not pd.isna(nc) else None for nc in new_column]
        #print(f"Over the {ctc} column, here are the numbers: used to be {sum(promotion_demotion_df[ctc].isna())} nulls,"
        #      f" now after date conversion there are {sum(new_column.isna())}.")
        promotion_demotion_df[ctc] = new_column_in_format
    return promotion_demotion_df, unreached_pages


def article_events_extractor_from_talkpages(talk_pages_jsonl):
    """
    Given a json talkpages (multiple!), this function extract all the dates and the events associated with each article
    Note! we extract here also event that are not promotion/demotion
    :param talk_pages_jsonl: jsonlines object
        the jsonlines object of the relevant article
    :return:
    """
    unreached_pages = list()
    events_found = list()
    for cur_file in tqdm(talk_pages_jsonl):
        cur_page_id = int(os.path.split(cur_file)[-1].split('.jsonl.bz2')[0])
        cur_content = load_and_decompress(cur_file)
        # each file is a single line, hence we take place 0 which is the first row
        try:
            history_sections = extract_history_template_sections(cur_content[0]['content'])
        except TypeError:
            unreached_pages.append(cur_page_id)
            continue
        # in case more than a single history section found. Not a big deal, taking the first one
        if len(history_sections) > 1:
            print(f"Potential bug with page id {cur_page_id} (file name: {cur_file} -- Multiple history sections")
            history_sections = history_sections[0]
        # standard case of a single history section. Processing it
        if len(history_sections) == 1:
            history_section = history_sections[0]
            # adding a new line before any pipe, in case it does not exist. It occurs in some cases (e.g., "ArticleHistory|action1=GAN\n|action1date=15:34, 6 March 2011\n|"...)
            history_section = re.sub(r'([^\n])\|', r'\1\n|', history_section)
            parsed_article_history = parse_article_history(history_section)
            relevant_fields_to_record = {'action', 'date', 'link', 'result'}
            # loop over the list of actions, and extracting each to a fixed dict which will be converted to df later
            for idx, cur_article_history in enumerate(parsed_article_history):
                cur_prefix = 'action' + str(idx + 1)
                # looping over the dict in the current item
                item_to_add = {'page_id': cur_page_id}
                for event_key, event_value in cur_article_history.items():
                    try:
                        suffix_found = event_key.split(cur_prefix)[1]
                    except IndexError:
                        continue
                    suffix_found = suffix_found if len(suffix_found) > 0 else 'action'
                    if suffix_found in relevant_fields_to_record:
                        item_to_add[suffix_found] = event_value
                # end of the inner loop
                if len(item_to_add) > 1:
                    events_found.append(item_to_add)

        # if no history section is missing, we can still try to extract the info from the basic structure of the page
        elif len(history_sections) == 0:
            try:
                promoted_to, promotion_time = extract_promotion_date_from_simple_template(cur_content[0]['content'].lower())
                item_to_add = {'page_id': cur_page_id, 'action': 'FAC' if promoted_to == 'fa' else 'GAN',
                               'date': promotion_time, 'link': None,
                               'result': 'promoted' if promoted_to == 'fa' else 'listed'}
                events_found.append(item_to_add)
            except IOError:
                unreached_pages.append(cur_page_id)
                continue
    # end of the big loop
    print(len(events_found))
    # I have found 16 pages that went over 25 reviews, out ouf the top20 predicted demoted by the model (1.56 reviewed)
    # In the whole dataset, there were 2276 FAR done to 1912 FAs (1.19 times reviewed)
    # in the whole dataset, there are 7685 promoted articles. Meaning the changes of an article to be reviewed is 0.249
    # In the top20 articles that have been demoted (we ranked them with the lowest prob to be sust.) we find the following:
    # of course all pages have been reviewed, and went through 26 reviews overall

    # converting the dict to df
    events_df = pd.DataFrame(events_found)
    # converting the dates column to standard pandas format
    columns_to_convert = ['date']
    for ctc in columns_to_convert:
        new_column = convert_dates(events_df[ctc], date_formats, verbose=True)
        new_column_in_format = [nc.strftime(new_date_format) if not pd.isna(nc) else None for nc in new_column]
        events_df[ctc] = new_column_in_format
    # last action - converting the action+result columns to lower case
    events_df['action'] = events_df['action'].str.lower()
    events_df['result'] = events_df['result'].str.lower()
    return events_df, unreached_pages


def convert_dates(column_data_as_df, date_formats, verbose=False):
    """
    Convert a column in a DataFrame to datetime using multiple date formats.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to convert.
        column (str): The name of the column to convert.
        date_formats (list): A list of date formats to try for conversion.

    Returns:
        pd.Series: The converted datetime column.
    """
    new_column = list()
    for idx, value in enumerate(column_data_as_df):
        cur_parsing_output = try_parsing_date(text=value, date_formats=date_formats, adjust_text=True, verbose=verbose)
        new_column.append(cur_parsing_output)
        if verbose and cur_parsing_output is None:
            print(f"A None value returned for row {idx}. Original value tried to parse: {value}")
    return pd.Series(new_column, index=column_data_as_df.index)


def try_parsing_date(text, date_formats, adjust_text=True, verbose=False):
    """
    Try to parse a date from the given text using a list of formats.
    Returns the first successfully parsed date or None if no formats match.
    """
    # in some cases, the text ends with (UTC) - we'll remove this
    orig_text = text
    if adjust_text:
        if type(text) is str and text.lower().startswith('date='):
            text = text.replace('Date=', '')
            text = text.replace('date=', '')
        if type(text) is str:
            text = text.title()
            text = text.replace(' (Utc)', '') if text.endswith(' (Utc)') else text
            text = text.replace(' (Utc', '') if text.endswith(' (Utc') else text
            text = text.replace('(Utc)', '') if text.endswith('(Utc)') else text
            text = text.replace(' utc', '') if text.endswith('utc') else text
            text = text.replace(' Utc', '') if text.endswith('Utc') else text
        # in some cases, the string provided doesn't have space after the comma (god hell!). We can try to place a space
        if type(text) is str and ',' in text:
            text = text.replace(',', ', ')
        # in some cases, the string ends with a dot
        if type(text) is str and text.endswith('.'):
            text = text[0:len(text) - 1]
        # in some cases there are multiple spaces instead of a single one
        if type(text) is str and ' ' in text:
            text = re.sub(r'\s+', ' ', text.strip())
        # some dates are written as Jun or Jul
        if type(text) is str and 'Jun ' in text:
            text = text.replace('Jun ', 'June ')
        if type(text) is str and 'Jul ' in text:
            text = text.replace('Jul ', 'July ')
        if type(text) is str and 'Sept ' in text:
            text = text.replace('Sept ', 'September ')
    if pd.isnull(text) or isinstance(text, pd._libs.tslibs.nattype.NaTType):
        return None
    for fmt in date_formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
        except (TypeError, re.error) as e:
            continue
    # if we got up to here, meaning that we were not able to find the right date format (not good :()
    if verbose:
        print(f"Orig text: {orig_text}, processed text: {text}")
    return None  # Return None if no format matches


def article_timeline_extractor_from_dumps(page_id, json_path, job_idx=-1):
    start_time = datetime.now()
    # we will now see if the dict needs to be modified or not. First loading the data
    cur_json_lines = load_and_decompress(filename=json_path)
    creation_time = datetime.strptime(cur_json_lines[0]['timestamp'], "%Y-%m-%dT%H:%M:%SZ")

    # looping over each revision and extracting relevant info
    prev_assessment_level = None
    prev_revision_date = creation_time
    revisions_in_prev_assessment_level = 1
    days_in_prev_assessment_level = 0
    assessment_changes = dict()
    for rev_idx, cjl in enumerate(cur_json_lines):
        cur_assessment_level = cjl['assessment_found']
        # in case we identified a change, we update the assessment_changes dictionary
        if cur_assessment_level != prev_assessment_level:

            cur_id = cjl['id']
            assessment_changes[cur_id] = {'timestamp': cjl['timestamp'], 'from': prev_assessment_level,
                                          'to': cur_assessment_level,
                                          'revisions_in_from': revisions_in_prev_assessment_level,
                                          'days_in_from': days_in_prev_assessment_level}
            # if the dict is at least of size 1, we can update it
            if len(assessment_changes) >= 2:
                prev_id = list(assessment_changes.keys())[-2]
                revisions_in_to = assessment_changes[cur_id]['revisions_in_from']
                days_in_to = assessment_changes[cur_id]['days_in_from']
                significant_change = is_change_significant(revisions_in_to, days_in_to)
                assessment_changes[prev_id].update({'revisions_in_to': revisions_in_to, 'days_in_to': days_in_to,
                                                    'significant_time_in_to': significant_change})
            revisions_in_prev_assessment_level = 1
            days_in_prev_assessment_level = 0
        # if the current revision is in the same state as the prev one, we will update two var with this information
        else:
            revisions_in_prev_assessment_level += 1
            days_in_prev_assessment_level += calc_days_since_prev_revision(prev_revision_time=prev_revision_date,
                                                                           cur_revision_time=cjl['timestamp'])
        prev_assessment_level = cur_assessment_level
        prev_revision_date = cjl['timestamp']
    # by the end of the loop, we do have to update the latest object in the assessment_changes (if it exists) and update the latest status of the page
    if assessment_changes:
        revisions_in_to = revisions_in_prev_assessment_level
        days_in_to = days_in_prev_assessment_level
        significant_change = is_change_significant(revisions_in_to, days_in_to)
        assessment_changes[list(assessment_changes.keys())[-1]].update({'revisions_in_to': revisions_in_to,
                                                                        'days_in_to': days_in_to,
                                                                        'significant_time_in_to': significant_change})
        current_status = assessment_changes[list(assessment_changes.keys())[-1]]['to']
    else:
        current_status = None
    # we now can loop over the changes, and take only the significant ones, to identify the FIRST promotions/demotions
    ga_promotion_time = None
    ga_demotion_time = None
    fa_promotion_time = None
    fa_demotion_time = None
    significant_assessment_changes = {revision_id: change_info for revision_id, change_info in assessment_changes.items() if change_info['significant_time_in_to']}
    for revision_id, change_info in significant_assessment_changes.items():
        if change_info['from'] is None and change_info['to'] == '{{good article}}' and ga_promotion_time is None:
            ga_promotion_time = change_info['timestamp']
            continue
        if change_info['to'] == '{{featured article}}' and fa_promotion_time is None:
            fa_promotion_time = change_info['timestamp']
            continue
        if change_info['from'] == '{{featured article}}' and fa_demotion_time is None:
            fa_demotion_time = change_info['timestamp']
            # in case the page went from GA -> FA -> None
            ga_demotion_time = change_info['timestamp'] if ga_promotion_time is not None else None
            continue
        if change_info['from'] == '{{good article}}' and change_info['to'] is None and ga_demotion_time is None:
            ga_demotion_time = change_info['timestamp']
            continue

    end_time = datetime.now()
    code_duration = (end_time - start_time).seconds
    print(f"Article ID {page_id} processed in {code_duration} seconds.", flush=True)
    return {page_id: {'dumps_creation_time': creation_time, 'dumps_last_revision_time': prev_revision_date,
                      'dumps_ga_promotion': ga_promotion_time, 'dumps_ga_demotion': ga_demotion_time,
                      'dumps_fa_promotion': fa_promotion_time, 'dumps_fa_demotion': fa_demotion_time,
                      'dumps_current_status': current_status}
            }


def is_change_significant(revisions_in_prev_assessment_level, days_in_prev_assessment_level):
    if revisions_in_prev_assessment_level >= 5 or days_in_prev_assessment_level >= 30:
        return True
    else:
        return False


def calc_days_since_prev_revision(prev_revision_time, cur_revision_time):
    if type(prev_revision_time) is str:
        prev_revision_time = datetime.strptime(prev_revision_time, "%Y-%m-%dT%H:%M:%SZ")
    cur_revision_datetime = datetime.strptime(cur_revision_time, "%Y-%m-%dT%H:%M:%SZ")
    cur_revision_datetime = datetime.strptime(cur_revision_time, "%Y-%m-%dT%H:%M:%SZ")
    days_diff = (cur_revision_datetime - prev_revision_time).days
    return max(0, days_diff)


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


def get_talk_page_content(session, page_id):
    def get_talk_page_id(page_id):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "pageids": page_id,
            "prop": "info",
            "inprop": "talkid",
            "format": "json",
            "formatversion": 2
        }
        response = session.get(url, params=params)
        data = response.json()
        pages = data['query']['pages']
        page = pages[0]
        if 'talkid' in page:
            return page['talkid']
        else:
            return None

    def get_latest_talk_page_content(talk_page_id):
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "revisions",
            "pageids": talk_page_id,
            "rvprop": "content",
            "rvlimit": 1,
            "format": "json",
            "formatversion": 2
        }
        response = session.get(url, params=params)
        data = response.json()
        pages = data['query']['pages']
        page = pages[0]

        if 'revisions' in page:
            revision = page['revisions'][0]
            if 'slots' in revision and 'main' in revision['slots']:
                return revision['slots']['main']['content']
            elif '*' in revision:
                return revision['*']
            elif 'content' in revision:
                return revision['content']
        return None

    talk_page_id = get_talk_page_id(page_id)
    if not talk_page_id:
        return None
    content = get_latest_talk_page_content(talk_page_id)
    return content


def extract_history_template_sections(talk_page_content):
    # Pattern to match sections enclosed in {{ }} that contain the word "history"
    pattern = r'{{.*?history.*?}}'

    # Search for the pattern in the talk page content
    matches = re.findall(pattern, talk_page_content, re.DOTALL | re.IGNORECASE)

    if matches:
        # Return all matches found
        list_to_return = [match.strip() for match in matches]
        # our history sections most include phrases like action1 and action2
        return [ltr for ltr in list_to_return if 'action1' in ltr]
    else:
        return []


def parse_article_history(template_content):
    # Remove the enclosing {{ and }} and any leading/trailing whitespace
    content = template_content.strip('{}').strip()

    # Split the content into lines
    lines = content.split('\n')

    # Create a dictionary to hold the key-value pairs
    data = {}

    # Regex pattern to capture key-value pairs
    pattern = r'\|\s*(\w+)\s*=\s*(.*)'

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            key, value = match.groups()
            data[key] = value.strip()
    # instead of returning the dict as a long dict without any order, we break it into parts that make sense (all ones
    # that start with 'action1' for example)
    actions_data = {key: value for key, value in data.items() if key.startswith('action')}
    non_actions_data = {key: value for key, value in data.items() if not key.startswith('action')}
    data_as_list = list()
    cur_action_dict = dict()
    cur_action_id = list(actions_data.keys())[0] if len(actions_data) else -1 # this should be action1
    for key, value in actions_data.items():
        if key.startswith(cur_action_id):
            cur_action_dict[key] = value
        # if it does not start with the cur_action_id, we moved to the next one
        else:
            cur_action_id = key
            data_as_list.append(cur_action_dict) # adding to the list of dicts the current one
            cur_action_dict = {key: value} # resetting the dict with new value (and later to add others)
    # by the end of the loop we anyway need to add the latest object (if not empty)
    if cur_action_dict:
        data_as_list.append(cur_action_dict)
    # end of the list we add the non-action items
    data_as_list.append(non_actions_data)
    return data_as_list


def extract_action_time(parsed_article_history, action_type, result, sort='first'):
    dates_found = list()
    for pah in parsed_article_history:
        cur_action = list(pah.values())[0] if len(pah) else ''
        if cur_action.lower() == action_type:
            cur_date = [value for key, value in pah.items() if 'date' in key]
            cur_decision = [value for key, value in pah.items() if 'result' in key]
            if len(cur_decision) == 1 and cur_decision[0].lower() in result and len(cur_date) == 1:
                dates_found.append(cur_date[0])
    dates_found = dates_found[0] if dates_found else None
    # in some cases, the string still holds the 'Date=...'
    if type(dates_found) is str and dates_found.lower().startswith('date='):
        dates_found = dates_found.replace('Date=', '')
        dates_found = dates_found.replace('date=', '')
    return dates_found


def extract_article_status(text):
    pattern = re.compile(r'currentstatus=.*?(?=\s|$)', re.DOTALL)
    matches = pattern.findall(text)
    if len(matches) == 0:
        current_status = None
    elif len(matches) == 1:
        try:
            current_status = matches[0].split('=')[1].lower()
        except KeyError:
            current_status = None
    else:
        raise ValueError("Somthing went wrong with the 'extract_article_status' function. Too many regex matches.")
    return current_status


def extract_promotion_date_from_simple_template(text):
    pattern = r'\{\{(ga|fa)\|([^}]*)\}\}'
    # Find all matches in the text
    matches = re.findall(pattern, text)
    if matches:
        promoted_to = matches[0][0].lower()
        promotion_date = matches[0][1].split('|')[0]
        return promoted_to, promotion_date
    else:
        raise IOError("Parsing failed.")


def determine_promotion_demotion_dates(instance_info, rely_on='talkpage'):
    if rely_on == 'talkpage':
        ga_promotion_time = instance_info['talkpage_ga_promotion']
        ga_demotion_time = instance_info['talkpage_ga_demotion']
        # in case some unlogical cases found (demotion without promotion)
        if type(ga_promotion_time) is not str and np.isnan(ga_promotion_time) and type(ga_demotion_time) is str:
            ga_demotion_time = None
        # now doing the same for the fa
        fa_promotion_time = instance_info['talkpage_fa_promotion']
        fa_demotion_time = instance_info['talkpage_fa_demotion']
        # in case some unlogical cases found (demotion without promotion)
        if type(fa_promotion_time) is not str and np.isnan(fa_promotion_time) and type(fa_demotion_time) is str:
            fa_demotion_time = None
    elif rely_on == 'both':
        ga_promotion_time = instance_info['talkpage_ga_promotion']
        # if no ga promotion was found in the talk page
        if type(ga_promotion_time) is not str and np.isnan(ga_promotion_time):
            ga_promotion_time = instance_info['dumps_ga_promotion']
        ga_demotion_time = instance_info['talkpage_ga_demotion']
        # if no ga demotion was found in the talk page
        if type(ga_demotion_time) is not str and np.isnan(ga_demotion_time):
            ga_demotion_time = instance_info['dumps_ga_demotion']

        # in case some unlogical cases found (demotion without promotion)
        if type(ga_promotion_time) is not str and np.isnan(ga_promotion_time) and type(ga_demotion_time) is str:
            ga_demotion_time = None

        # now doing the same for the fa
        fa_promotion_time = instance_info['talkpage_fa_promotion']
        # if no ga promotion was found in the talk page
        if type(fa_promotion_time) is not str and np.isnan(fa_promotion_time):
            fa_promotion_time = instance_info['dumps_fa_promotion']
        fa_demotion_time = instance_info['talkpage_fa_demotion']
        # if no fa demotion was found in the talk page
        if type(fa_demotion_time) is not str and np.isnan(fa_demotion_time):
            fa_demotion_time = instance_info['dumps_fa_demotion']

        # in case some unlogical cases found (demotion without promotion)
        if type(fa_promotion_time) is not str and np.isnan(fa_promotion_time) and type(fa_demotion_time) is str:
            fa_demotion_time = None
    else:
        raise IOError("Unknonw parameters. Please use either rely_on='talkpage' or rely_on='both'")
    return {'ga_promotion_time': ga_promotion_time, 'ga_demotion_time': ga_demotion_time,
            'fa_promotion_time': fa_promotion_time, 'fa_demotion_time': fa_demotion_time}
