import pandas as pd
import bz2
import jsonlines
from datetime import datetime
from collections import defaultdict, Counter

date_format = '%Y-%m-%dT%H:%M:%SZ'


class ArticleAssessment(object):
    def __init__(self, article_id):
        self.article_id = article_id
        self.creation_date = None
        self.age_in_days = None
        self.latest_assessment_level = None
        self.revision_level_info = None
        self.assessment_changes = dict()
        self.assessments_counter = defaultdict(int)
        self.is_sustainable = True
        self.is_sustainable_conservative_approach = None
        self.promotion_demotion_full_info = None
        self.fa_promotion_date = None
        self.fa_demotion_date = None
        self.ga_promotion_date = None
        self.ga_demotion_date = None
        self.information_source = dict()
        self.birth_to_fa_promotion = None
        self.birth_to_ga_promotion = None
        self.fa_promotion_to_demotion = None
        self.ga_promotion_to_demotion = None
        self.promoted_to_ga_then_fa = False


    @staticmethod
    def _load_and_decompress(filename):
        with bz2.open(filename, 'rt') as f:
            reader = jsonlines.Reader(f)
            return list(reader)

    @staticmethod
    def is_change_significant(revisions_in_prev_assessment_level, days_in_prev_assessment_level):
        if revisions_in_prev_assessment_level >= 5 or days_in_prev_assessment_level >= 30:
            return True
        else:
            return False

    def calc_days_since_prev_revision(self, info_per_revision, cur_revision_time):
        try:
            latest_revision_id = list(info_per_revision)[-1]
        # case when the dict is empty
        except IndexError:
            return 0
        latest_revision_time = info_per_revision[latest_revision_id]['timestamp']
        latest_revision_datetime = datetime.strptime(latest_revision_time, date_format)
        cur_revision_datetime = datetime.strptime(cur_revision_time, date_format)
        days_diff = (cur_revision_datetime - latest_revision_datetime).days
        # if days_diff < 0:
        #     print(f"calc_days_since_prev_revision function found a negative value for article ID {self.article_id}. "
        #           f"Returning 0", flush=True)
        return max(0, days_diff)

    def determine_promotion_demotion_time_and_duration(self):
        fa_details_output = self.extract_fa_details()
        # converting the extracted strings into datetime and setting to None if needed (some cases it is np.nan)
        self.fa_promotion_date = datetime.strptime(self.fa_promotion_date, date_format) if type(self.fa_promotion_date) is str else None
        self.fa_demotion_date = datetime.strptime(self.fa_demotion_date, date_format) if type(self.fa_demotion_date) is str else None

        # converting the extracted strings into datetime
        ga_details_output = self.extract_ga_details()
        self.ga_promotion_date = datetime.strptime(self.ga_promotion_date, date_format) if type(self.ga_promotion_date) is str else None
        self.ga_demotion_date = datetime.strptime(self.ga_demotion_date, date_format) if type(self.ga_demotion_date) is str else None
        # with the ga, there might be special cases when a demotion occurs after converting the fa - we handle it here
        self.validate_ga_demotion_date()
        self.calc_durations_between_changes()
        return 0

    def extract_fa_details(self):
        promotion_from_talkpage = True if type(self.promotion_demotion_full_info['talkpage_fa_promotion']) is str else False
        demotion_from_talkpage = True if type(self.promotion_demotion_full_info['talkpage_fa_demotion']) is str else False
        promotion_from_dumps = True if type(self.promotion_demotion_full_info['dumps_fa_promotion']) is str else False
        demotion_from_dumps = True if type(self.promotion_demotion_full_info['dumps_fa_demotion']) is str else False

        promotion_from_talkpage, demotion_from_talkpage, promotion_from_dumps, demotion_from_dumps = (
            self.validate_promotion_demotion_indicators(usecase='fa', promotion_from_talkpage=promotion_from_talkpage,
                                                        demotion_from_talkpage=demotion_from_talkpage,
                                                        promotion_from_dumps=promotion_from_dumps,
                                                        demotion_from_dumps=demotion_from_dumps))
        # over all there are 16 options. We have to handle all. We start with considering the info in the talkpages
        # easiest case - promotion and demotion both appear in the talkpage
        if promotion_from_talkpage and demotion_from_talkpage:
            self.fa_promotion_date = self.promotion_demotion_full_info['talkpage_fa_promotion']
            self.fa_demotion_date = self.promotion_demotion_full_info['talkpage_fa_demotion']
            self.information_source['fa'] = 'talkpage'
            return 2
        # a case only promotion was identified in the talk page
        if promotion_from_talkpage and not demotion_from_talkpage:
            # no demotion even from the dumps
            if not demotion_from_dumps:
                self.fa_promotion_date = self.promotion_demotion_full_info['talkpage_fa_promotion']
                self.fa_demotion_date = None
                self.information_source['fa'] = 'talkpage'
                return 1
            # indication of a demotion from the dumps
            else:
                should_dump_info_considered = self.do_talkpage_and_dumps_aligned(talkpage_date=self.promotion_demotion_full_info['talkpage_fa_promotion'],
                                                                                 dumps_date=self.promotion_demotion_full_info['dumps_fa_promotion'],
                                                                                 grace_time=30)
                if should_dump_info_considered:
                    self.fa_promotion_date = self.promotion_demotion_full_info['talkpage_fa_promotion']
                    self.fa_demotion_date = self.promotion_demotion_full_info['dumps_fa_demotion']
                    self.information_source['fa'] = 'both'
                    return 2
                # if we cannot trust the dumps info, we might still determine that there was only promotion of the page
                else:
                    self.fa_promotion_date = self.promotion_demotion_full_info['talkpage_fa_promotion']
                    self.information_source['fa'] = 'talkpage'
                    return 1
        # a case *demotion* was identified from the talkpage (and not promotion) - not so logical, but might be
        if demotion_from_talkpage and not promotion_from_talkpage:
            should_dump_info_considered = self.do_talkpage_and_dumps_aligned(talkpage_date=self.promotion_demotion_full_info['talkpage_fa_demotion'],
                                                                             dumps_date=self.promotion_demotion_full_info['dumps_fa_demotion'],
                                                                             grace_time=30)
            if should_dump_info_considered:
                self.fa_promotion_date = self.promotion_demotion_full_info['dumps_fa_promotion']
                self.fa_demotion_date = self.promotion_demotion_full_info['talkpage_fa_demotion']
                self.information_source['fa'] = 'both'
                return 2
            # if we cannot trust the dumps info, we can do nothing (cant be a demotion w/o promotion)
            else:
                return 0
        # all other cases rely on the information from dump files
        if promotion_from_dumps and demotion_from_dumps:
            self.fa_promotion_date = self.promotion_demotion_full_info['dumps_fa_promotion']
            self.fa_demotion_date = self.promotion_demotion_full_info['dumps_fa_demotion']
            self.information_source['fa'] = 'dumps'
            return 2
        # only promotion case
        if promotion_from_dumps and not demotion_from_dumps:
            self.fa_promotion_date = self.promotion_demotion_full_info['dumps_fa_promotion']
            self.information_source['fa'] = 'dumps'
            return 1
        # unrealistic case of a demotion without promotion
        if demotion_from_dumps and not promotion_from_dumps:
            return 0

    def extract_ga_details(self):
        promotion_from_talkpage = True if type(self.promotion_demotion_full_info['talkpage_ga_promotion']) is str else False
        demotion_from_talkpage = True if type(self.promotion_demotion_full_info['talkpage_ga_demotion']) is str else False
        promotion_from_dumps = True if type(self.promotion_demotion_full_info['dumps_ga_promotion']) is str else False
        demotion_from_dumps = True if type(self.promotion_demotion_full_info['dumps_ga_demotion']) is str else False
        promotion_from_talkpage, demotion_from_talkpage, promotion_from_dumps, demotion_from_dumps = (
            self.validate_promotion_demotion_indicators(usecase='ga', promotion_from_talkpage=promotion_from_talkpage,
                                                        demotion_from_talkpage=demotion_from_talkpage,
                                                        promotion_from_dumps=promotion_from_dumps,
                                                        demotion_from_dumps=demotion_from_dumps))
        # we now go into many if cases, of promotion/demotion use casess
        # over all there are 16 options. We have to handle all. We start with considering the info in the talkpages
        # easiest case - promotion and demotion both appear in the talkpage
        if promotion_from_talkpage and demotion_from_talkpage:
            self.ga_promotion_date = self.promotion_demotion_full_info['talkpage_ga_promotion']
            self.ga_demotion_date = self.promotion_demotion_full_info['talkpage_ga_demotion']
            self.information_source['ga'] = 'talkpage'
            return 2
        # a case only promotion was identified, no demotion even from the dumps
        if promotion_from_talkpage and not demotion_from_talkpage:
            if not demotion_from_dumps:
                self.ga_promotion_date = self.promotion_demotion_full_info['talkpage_ga_promotion']
                self.ga_demotion_date = None
                self.information_source['ga'] = 'talkpage'
                return 1
            else:
                should_dump_info_considered = self.do_talkpage_and_dumps_aligned(talkpage_date=self.promotion_demotion_full_info['talkpage_ga_promotion'],
                                                                                 dumps_date=self.promotion_demotion_full_info['dumps_ga_promotion'],
                                                                                 grace_time=30)
                if should_dump_info_considered:
                    self.ga_promotion_date = self.promotion_demotion_full_info['talkpage_ga_promotion']
                    self.ga_demotion_date = self.promotion_demotion_full_info['dumps_ga_demotion']
                    self.information_source['ga'] = 'both'
                    return 2
                # if we cannot trust the dumps info, we might still determine that there was only promotion of the page
                else:
                    self.ga_promotion_date = self.promotion_demotion_full_info['talkpage_ga_promotion']
                    self.information_source['ga'] = 'talkpage'
                    return 1
        # a case demotion was identified from the talkpage (and not promotion)
        if demotion_from_talkpage and not promotion_from_talkpage:
            should_dump_info_considered = self.do_talkpage_and_dumps_aligned(talkpage_date=self.promotion_demotion_full_info['talkpage_ga_demotion'],
                                                                             dumps_date=self.promotion_demotion_full_info['dumps_ga_demotion'],
                                                                             grace_time=30)
            if should_dump_info_considered:
                self.ga_promotion_date = self.promotion_demotion_full_info['dumps_ga_promotion']
                self.ga_demotion_date = self.promotion_demotion_full_info['talkpage_ga_demotion']
                self.information_source['ga'] = 'both'
                return 2
            # if we cannot trust the dumps info, we can do nothing (cant be a demotion w/o promotion)
            else:
                return 0
        # all other cases rely on the information from dump files
        if promotion_from_dumps and demotion_from_dumps:
            self.ga_promotion_date = self.promotion_demotion_full_info['dumps_ga_promotion']
            self.ga_demotion_date = self.promotion_demotion_full_info['dumps_ga_demotion']
            self.information_source['ga'] = 'dumps'
            return 2
        if promotion_from_dumps and not demotion_from_dumps:
            self.ga_promotion_date = self.promotion_demotion_full_info['dumps_ga_promotion']
            self.information_source['ga'] = 'dumps'
            return 1
        # unrealistic case of a demotion without promotion
        if demotion_from_dumps and not promotion_from_dumps:
            return 0

    def validate_promotion_demotion_indicators(self, usecase, promotion_from_talkpage, demotion_from_talkpage,
                                               promotion_from_dumps, demotion_from_dumps):
        """
        a function to validate the four indicators we use for each promotion/demotion of a use case. These indicators
        are used to determine if the information from a source (e.g., dumps) are valid to be used. This functions
        ensures that this is indeed the case, by comparing the information in hand with the general info we have (e.g.,
        the creation time of the page)
        :param usecase: str
            'fa' or 'ga' only, for others it will not work
        :param promotion_from_talkpage: bool
            whether the information from the talkpage about promotion exist and needs to be checked
        :param demotion_from_talkpage: bool
            whether the information from the talkpage about demotion exist and needs to be checked
        :param promotion_from_dumps: bool
            whether the information from the dumps about promotion exist and needs to be checked
        :param demotion_from_dumps: bool
            whether the information from the dumps about demotion exist and needs to be checked
        :return: tuple
            tuple of size 4, with the original indicators modified. In best cases, these 4 will not be changed at all
        """
        # first, validating the talkpage info
        if promotion_from_talkpage:
            later_than_creation_time = datetime.strptime(self.promotion_demotion_full_info['talkpage_' + usecase + '_promotion'], date_format) > self.creation_date
            promotion_from_talkpage = True if later_than_creation_time else False
        if demotion_from_talkpage:
            later_than_creation_time = datetime.strptime(self.promotion_demotion_full_info['talkpage_' + usecase + '_demotion'], date_format) > self.creation_date
            demotion_from_talkpage = True if later_than_creation_time else False
        if promotion_from_talkpage and demotion_from_talkpage:
            logical_order = datetime.strptime(self.promotion_demotion_full_info['talkpage_' + usecase + '_demotion'], date_format) > datetime.strptime(self.promotion_demotion_full_info['talkpage_' + usecase + '_promotion'], date_format)
            promotion_from_talkpage = True if logical_order else False
            demotion_from_talkpage = True if logical_order else False

        # the fact that the data exist does not mean it is logical, we first validate it
        # now doing the same for the talkpages
        if promotion_from_dumps:
            later_than_creation_time = datetime.strptime(self.promotion_demotion_full_info['dumps_' + usecase + '_promotion'], date_format) > self.creation_date
            promotion_from_dumps = True if later_than_creation_time else False
        if demotion_from_dumps:
            later_than_creation_time = datetime.strptime(self.promotion_demotion_full_info['dumps_' + usecase + '_demotion'], date_format) > self.creation_date
            demotion_from_dumps = True if later_than_creation_time else False
        if promotion_from_dumps and demotion_from_dumps:
            logical_order = datetime.strptime(self.promotion_demotion_full_info['dumps_' + usecase + '_demotion'], date_format) > datetime.strptime(self.promotion_demotion_full_info['dumps_' + usecase + '_promotion'], date_format)
            promotion_from_dumps = True if logical_order else False
            demotion_from_dumps = True if logical_order else False
        return promotion_from_talkpage, demotion_from_talkpage, promotion_from_dumps, demotion_from_dumps

    def do_talkpage_and_dumps_aligned(self, talkpage_date, dumps_date, grace_time=30):
        if not type(talkpage_date) is str or not type(dumps_date) is str:
            return False
        talkpage_datetime = datetime.strptime(talkpage_date, date_format)
        dumps_datetime = datetime.strptime(dumps_date, date_format)
        time_diff = (talkpage_datetime - dumps_datetime).days
        return False if abs(time_diff) > grace_time else True

    def calc_durations_between_changes(self):
        # first calc the time between birth to fa_promotion
        if self.fa_promotion_date is not None:
            self.birth_to_fa_promotion = (self.fa_promotion_date - self.creation_date).days
        if self.ga_promotion_date is not None:
            self.birth_to_ga_promotion = (self.ga_promotion_date - self.creation_date).days
        if self.fa_demotion_date is not None:
            try:
                self.fa_promotion_to_demotion = (self.fa_demotion_date - self.fa_promotion_date).days
            except TypeError:
                self.fa_promotion_to_demotion = None
                print(f"Problematic case with page {self.article_id}. Looks like a FA demotion w/o promotion")
        if self.ga_demotion_date is not None:
            # the ga_demotion can occur if the process was GA -> FA -> None, and then no
            try:
                self.ga_promotion_to_demotion = (self.ga_demotion_date - self.ga_promotion_date).days
            except TypeError:
                self.ga_promotion_to_demotion = None
                print(f"Problematic case with page {self.article_id}. Looks like a GA demotion w/o promotion")

    def validate_ga_demotion_date(self):
        # in case there was a ga promotion before a fa promotion
        if self.ga_promotion_date is not None and self.fa_promotion_date is not None and self.ga_promotion_date < self.fa_promotion_date:
            self.promoted_to_ga_then_fa = True
            # GA -> FA -> FFA
            if self.fa_demotion_date is not None and self.ga_demotion_date is None:
                self.ga_demotion_date = self.fa_demotion_date
        return 0

    def extract_info_from_json(self, json_path):
        start_time = datetime.now()
        # we will now see if the dict needs to be modified or not. First loading the data
        cur_json_lines = self._load_and_decompress(filename=json_path)
        self.creation_date = datetime.strptime(cur_json_lines[0]['timestamp'], date_format)
        self.age_in_days = (datetime.strptime(cur_json_lines[-1]['timestamp'], date_format) - self.creation_date).days

        # looping over each revision and extracting relevant info
        info_per_revision = dict()
        prev_assessment_level = None
        has_been_promoted = False
        has_been_demoted = False
        revisions_in_prev_assessment_level = 1
        days_in_prev_assessment_level = 0
        for rev_idx, cjl in enumerate(cur_json_lines):
            # in case we the user added the text is a valid user and not an IP
            if 'user' in cjl and 'id' in cjl['user']:
                cur_user = int(cjl['user']['id'])
            else:
                cur_user = None
            try:
                cur_text_len = len(cjl['text'])
            except KeyError:
                cur_text_len = 0
            cur_assessment_level = cjl['assessment_found']
            # adding the assessment value found to the dict of assessment values
            self.assessments_counter[cur_assessment_level] += 1
            # in case we identified a change, we update the assessment_changes dictionary
            if cur_assessment_level != prev_assessment_level:
                significant_change = self.is_change_significant(revisions_in_prev_assessment_level,
                                                                days_in_prev_assessment_level)
                # these two would be updated once (the first time) and then will stay True for the rest of the loops
                if significant_change and cur_assessment_level is not None:
                    has_been_promoted = True
                if significant_change and cur_assessment_level is None:
                    has_been_demoted = True
                self.assessment_changes[cjl['id']] = \
                    {'timestamp': cjl['timestamp'], 'from': prev_assessment_level, 'to': cur_assessment_level,
                     'revisions_in_prev': revisions_in_prev_assessment_level,
                     'days_in_prev': days_in_prev_assessment_level, 'significant_change': significant_change}
                revisions_in_prev_assessment_level = 1
                days_in_prev_assessment_level = 0
            # if the current revision is in the same state as the prev one, we will update two var with this information
            else:
                revisions_in_prev_assessment_level += 1
                days_in_prev_assessment_level += self.calc_days_since_prev_revision(info_per_revision=info_per_revision,
                                                                                    cur_revision_time=cjl['timestamp'])
            # if 'sha1' not in cjl:
            #     print(f"SHA1 has not been found in revision {rev_idx} of page {json_path}.", flush=True)
            info_per_revision[cjl['id']] = {'user': cur_user, 'timestamp': cjl['timestamp'], 'text_len': cur_text_len,
                                            'sha1': cjl['sha1'] if 'sha1' in cjl else None,
                                            'been_promoted_yet': has_been_promoted,
                                            'been_demoted_yet': has_been_demoted,
                                            'assessment_level': cur_assessment_level}
            prev_assessment_level = cur_assessment_level

        self.revision_level_info = pd.DataFrame.from_dict(info_per_revision, orient='index')
        # setting the latest assessment value found
        self.latest_assessment_level = cur_json_lines[-1]['assessment_found']
        # saving the indicator whether the article is sustainable or not
        promotion_demotion_time_output = self.determine_promotion_demotion_time_and_duration()
        self.is_sustainable = False if self.fa_demotion_date is not None or self.ga_demotion_date is not None else True
        end_time = datetime.now()
        code_duration = end_time - start_time

    def validate_dates_and_durations(self):
        valid = True
        # validating the fa cases
        if self.fa_promotion_date is not None:
            if self.fa_promotion_date < self.creation_date:
                valid = False
            if self.fa_demotion_date is not None and self.fa_demotion_date < self.fa_promotion_date:
                valid = False
        # validating the ga cases
        if self.ga_promotion_date is not None:
            if self.ga_promotion_date < self.creation_date:
                valid = False
            if self.ga_demotion_date is not None and self.ga_demotion_date < self.ga_promotion_date:
                valid = False
        # validating the durations
        if self.birth_to_fa_promotion is not None and self.birth_to_fa_promotion < 0:
            valid = False
        if self.birth_to_ga_promotion is not None and self.birth_to_ga_promotion < 0:
            valid = False
        if self.fa_promotion_to_demotion is not None and self.fa_promotion_to_demotion < 0:
            valid = False
        if self.ga_promotion_to_demotion is not None and self.ga_promotion_to_demotion < 0:
            valid = False
        if self.age_in_days is not None and self.age_in_days < 0:
            valid = False
        return valid

    def determine_usecase(self):
        usecase = list()
        if self.fa_promotion_date is not None:
            usecase.append('fa')
        if self.ga_promotion_date is not None:
            usecase.append('ga')
        # if the set of cases is > 2, will return both. Otherwise only the found one
        if len(usecase) > 1:
            return 'both'
        elif len(usecase) == 1:
            return usecase[0]
        else:
            return None

    def filter_revision_level_info(self, usecase, until_promotion=True, max_revisions_to_use=None):
        if until_promotion:
            promotion_date = self.fa_promotion_date if usecase == 'fa' else self.ga_promotion_date
            if type(promotion_date) is not datetime:
                return IOError(f"Invalid promotion date for article {self.article_id}")
            max_revisions_mask = pd.to_datetime(self.revision_level_info['timestamp'], format=date_format) < promotion_date
            if max_revisions_to_use is not None:
                max_revisions_mask[max_revisions_to_use:] = False
            revision_level_info_subset = self.revision_level_info[max_revisions_mask].copy()
        else:
            # if the until_promotion is off, but we do want to take a subset of the revisions onlu
            if max_revisions_to_use is not None:
                revision_level_info_subset = self.revision_level_info[0:max_revisions_to_use].copy()
            # if both indicators are off, we take the whole data in hand
            else:
                revision_level_info_subset = self.revision_level_info.copy()
        return revision_level_info_subset

    def is_obj_valid(self, usecase):
        # if the article was FA promoted and GA promoted
        if usecase == 'ga' and self.ga_promotion_date is not None and self.fa_promotion_date is not None:
            # if the FA promotion came before the GA promotion - we do not handle such case due to complexity
            if self.fa_promotion_date < self.ga_promotion_date:
                return False
        # in all other cases, return True
        return True

    def determine_sustainability_conservative_approach(self, appears_in):
        """
        This function is used to determine whether the article should be considered as sustainability.
        We apply here a conservative approach -- if a page appeared in the list of demoted pages - it is not sus.
        If the page appears in the list of existing FAs/GAs - we assign it as sus., BUT only if the demotion date of
        the object is NOT from the talk pages!

        :param appears_in: list
            list of all lists that the page appeared in. Can also be None
        :return: bool
            the decision. It is also saved into the object itself
        """
        # first we check if the page appears in any "former lists" - clean indication that it was demoted at some point
        appears_in_former_list = [True if ai.startswith('former') else False for ai in appears_in] if appears_in is not None else []
        appears_in_former_list = any(appears_in_former_list)
        if appears_in_former_list:
            self.is_sustainable_conservative_approach = False
            return False
        # if we got up to here, it means that the page was not found in the "former lists". Now it is up to us to decide
        can_fa_be_trusted = False if 'fa' in self.information_source and self.information_source['fa'] != 'talkpage' else True
        can_ga_be_trusted = False if 'ga' in self.information_source and self.information_source['ga'] != 'talkpage' else True
        if can_fa_be_trusted and self.fa_demotion_date is not None:
            self.is_sustainable_conservative_approach = False
            return False
        if can_ga_be_trusted and self.ga_demotion_date is not None:
            self.is_sustainable_conservative_approach = False
            return False
        # if none of the above were captured, we assign True value (no real evidence that the page was demoted)
        self.is_sustainable_conservative_approach = True
        return True
