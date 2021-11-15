import re
import os
import time


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc, NamesExtractor, PER


# Analysis params
MAX_PERCENT_LATIN_LETTER_IN_REVIEW = 0.25           # Max percent latin letters in non-spelling review
MIN_LEN_REVIEW = 150                                # Min length of non-spelling review
DUPLICATES_UNIQUENESS = 0.6                         # Percent uniqueness for detect duplicate
NAMES_DICT = ['профи', 'Ваш репетитор', 'Preply']   # Black words

# Debug params
PACKET_SIZE = 250                                   # Google sheets packet size
DEBUG = False                                       # Debug mode on/off
DEBUG_DATA_SIZE = 100                               # Reviews count in debug mode

# Russian alphabet
alphabet = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", " ",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]

# Words without sense
STOP_WORDS = get_stop_words('russian') + stopwords.words('russian')

# Natasha parsers
natasha_emb = NewsEmbedding()
natasha_ner_tagger = NewsNERTagger(natasha_emb)
natasha_segmenter = Segmenter()
natasha_morph_vocab = MorphVocab()
natasha_morph_tagger = NewsMorphTagger(natasha_emb)
natasha_names_extractor = NamesExtractor(natasha_morph_vocab)


class ReviewAnalysis:
    def __init__(self):
        self.data = pd.DataFrame({'review': [], 'sectionId': [], 'type_page': [], 'type_model': []})

        """Stats"""
        self.amount_unique_words = 0
        self.amount_words = 0

        """Settings"""
        self.duplicates_uniqueness = DUPLICATES_UNIQUENESS

    # Add data to ReviewAnalysis
    def add_data(self, data: list):
        # Getting data
        for raw in data:
            reviews = raw[0].split('\n')
            for review in reviews:
                review = review.replace('–', '-')
                self.data = self.data.append({'review': review, 'sectionId': raw[1], 'type_page': raw[2],
                                              'type_model': raw[3]}, ignore_index=True)
        if DEBUG:
            self.data = self.data.head(DEBUG_DATA_SIZE)

    # Mark spelling in data
    def mark_spelling(self):
        self.data['spelling'] = [True if self.is_spoiling(review) else False for review in self.data['review']]

    # Mark spelling in data
    def correct_data(self):
        review_correct = [self.correct_review(review) for review in self.data['review']]
        self.data['review_original'] = self.data['review']
        self.data['review'] = review_correct
        self.delete_names_in_start()

    # Return duplicate matrix of lemmatized_reviews
    def get_duplicat_matrix(self, lemmatized_reviews):
        vectorizer = CountVectorizer().fit_transform(lemmatized_reviews)
        vectors = vectorizer.toarray()
        csim = cosine_similarity(vectors)

        self.amount_unique_words = len(vectors[0])
        self.amount_words = 0
        for vec in vectors:
            self.amount_words += sum(vec)

        return vectors, csim

    # Mark duplicates in data
    def mark_duplicates(self):
        self.data['duble_good'] = True
        self.data['duble_class'] = 0

        # Get cosine_similarity of reviews_good
        reviews_good = self.data[self.data.spelling == False]['review']

        # Exit if reviews_good is empty
        if len(reviews_good.values) == 0:
            print(len(reviews_good.values))
            return

        # Lemmatization reviews
        cleaned_reviews = list(map(self.clean_review, reviews_good.values))
        lemmatized_reviews = list(map(self.lemmatization_review, cleaned_reviews))
        vectors, csim = self.get_duplicat_matrix(lemmatized_reviews)

        # Find uniqueness
        start = time.time()
        self.mark_duplicates_by_csim(reviews_good, csim, vectors)
        goods = len(self.data[(self.data.duble_good == True) & (self.data.spelling == False)]['review'].index)
        print('Time: ', time.time() - start, ' Goods: ', goods)

    # Mark duplicates with file in data
    def mark_file_duplicates(self, csv_file: str):
        self.data['duble_file'] = False

        reviews_good = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]['review']
        reviews_good_count = len(reviews_good.values)

        if not os.path.exists(csv_file):
            print('Can\'t find file: ' + csv_file)
            return

        reviews_file = list(pd.read_csv(csv_file, sep='\t')['review'].values)
        print(reviews_file)
        cleaned_reviews = list(map(self.clean_review, list(reviews_good.values)+reviews_file))
        lemmatized_reviews = list(map(self.lemmatization_review, cleaned_reviews))

        vectors, csim = self.get_duplicat_matrix(lemmatized_reviews)
        duplicates_pairs = self.get_duble_pairs(csim)

        # Find uniqueness
        for pair in duplicates_pairs:
            if pair[0] < reviews_good_count <= pair[1]:
                self.data.at[reviews_good.index[pair[0]], 'duble_file'] = True
                self.data.at[reviews_good.index[pair[0]], 'duble_good'] = False
                self.data.at[reviews_good.index[pair[0]], 'duble_class'] = -1

    # Mark duplicates by duplicate matrix (csim) and vectors of reviews (vectors)
    def mark_duplicates_by_csim(self, reviews_good: pd.Series, csim: np.ndarray, vectors: np.ndarray):
        unique_words_counts = list(np.count_nonzero(np.array(vec)) for vec in vectors)
        duplicate_classes = [-1 for _ in range(len(vectors))]

        # [[2, 11], [2, 91], [30, 70], [40, 64]]
        for i in range(len(csim)):
            if duplicate_classes[i] == -1:
                max_id = i
                max_unique = unique_words_counts[i]
                duplicates = []
                for j in range(len(csim[i])):
                    if csim[i][j] >= self.duplicates_uniqueness and duplicate_classes[j] == -1:
                        duplicates.append(j)
                        if unique_words_counts[j] > max_unique:
                            max_unique = unique_words_counts[j]
                            max_id = j

                if i != max_id:
                    duplicates = []

                for duplicate_id in duplicates:
                    if duplicate_classes[duplicate_id] == -1:
                        duplicate_classes[duplicate_id] = max_id

        for i in range(len(duplicate_classes)):
            self.data.at[reviews_good.index[i], 'duble_class'] = duplicate_classes[i] + 1

            if i == duplicate_classes[i]:
                self.data.at[reviews_good.index[i], 'duble_good'] = True
            else:
                self.data.at[reviews_good.index[i], 'duble_good'] = False

    # Return duplicate pairs from duplicate matrix csim
    def get_duble_pairs(self, csim: np.ndarray):
        duplicates_pairs = []
        for i in range(len(csim)-1):
            for j in range(i+1, len(csim[i])):
                if csim[i][j] >= self.duplicates_uniqueness:
                    duplicates_pairs.append([i, j])
        return duplicates_pairs

    # Mark name entities in data
    def mark_name_entity(self):
        self.data['name_entity'] = [True if self.has_name_entity(review) else False for review in self.data['review']]

    # Return ful name by names dict
    @staticmethod
    def merge_names(names: list):
        full_name = {}
        has_not_middle = True
        for name in names:
            if 'first' in name.keys() and ('first' not in full_name.keys() or len(full_name['first']) < 2):
                full_name['first'] = name['first']

            if 'middle' in name.keys() and ('middle' not in full_name.keys() or len(full_name['middle']) < 2):
                full_name['middle'] = name['middle']
                has_not_middle = False

            if 'last' in name.keys():
                if ('middle' in full_name.keys() and len(full_name['middle'])>1) or 'last' not in full_name.keys():
                    full_name['last'] = name['last']
                elif 'last' in full_name.keys():
                    if (len(full_name['last']) > 1 and name['last'] != full_name['last']) or\
                            (len(full_name['last']) == 1 and name['last'][0] != full_name['last'][0]):
                        full_name['middle'] = full_name['last']
                        full_name['last'] = name['last']
                    else:
                        full_name['last'] = name['last']

        if has_not_middle and 'middle' in full_name.keys() and 'last' in full_name.keys() and\
                full_name['last'] > full_name['middle']:
            full_name['last'], full_name['middle'] = full_name['middle'],  full_name['last']

        return full_name

    # TODO mark_name_entity_details
    # Mark names and genders
    def mark_name_entity_details(self):
        self.data['initials_workers'] = ''
        self.data['gender'] = ''

        goods = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]
        entity_details = list(map(self.get_name_entity_details, goods['review']))

        for i, index in enumerate(goods.index):
            self.data.at[index, 'initials_workers'] = entity_details[i][0]
            self.data.at[index, 'gender'] = entity_details[i][1]

    # Return true if name a and b is equal
    @staticmethod
    def equal_names(a: list, b: list):
        if 'first' in a[0].keys() and 'first' in b[0].keys():
            if len(a[0]['first']) > 1 and len(b[0]['first']) > 1:
                if a[0]['first'] != b[0]['first']:
                    return False
            else:
                if a[0]['first'][0] != b[0]['first'][0]:
                    return False
        if 'last' in a[0].keys() and 'last' in b[0].keys():
            if 'middle' in a[0].keys() and 'middle' not in b[0].keys():
                if len(a[0]['last']) > 0 and len(b[0]['last']) > 0:
                    if a[0]['last'] == b[0]['last']:
                        return True
                    elif len(a[0]['middle']) > 1 and a[0]['middle'] == b[0]['last']:
                        return True
                    elif a[0]['middle'][0] == b[0]['last'][0]:
                        return True
                    else:
                        return False
                else:
                    if a[0]['last'][0] == b[0]['last'][0]:
                        return True
                    elif len(a[0]['middle']) > 1 and a[0]['middle'][0] == b[0]['last'][0]:
                        return True
                    elif a[0]['middle'][0] == b[0]['last'][0]:
                        return True
                    else:
                        return False
            elif 'middle' in b[0].keys() and 'middle' not in a[0].keys():
                if len(b[0]['last']) > 0 and len(a[0]['last']) > 0:
                    if b[0]['last'] == a[0]['last']:
                        return True
                    elif len(b[0]['middle']) > 1 and b[0]['middle'] == a[0]['last']:
                        return True
                    elif b[0]['middle'][0] == a[0]['last'][0]:
                        return True
                    else:
                        return False
                else:
                    if b[0]['last'][0] == a[0]['last'][0]:
                        return True
                    elif len(b[0]['middle']) > 1 and b[0]['middle'][0] == a[0]['last'][0]:
                        return True
                    elif b[0]['middle'][0] == a[0]['last'][0]:
                        return True
                    else:
                        return False
            elif 'middle' in a[0].keys() and 'middle' in b[0].keys():
                if len(a[0]['last']) > 1 and len(b[0]['last']) > 1:
                    if a[0]['last'] != b[0]['last']:
                        return False
                else:
                    if a[0]['last'][0] != b[0]['last'][0]:
                        return False
                if len(a[0]['middle']) > 1 and len(b[0]['middle']) > 1:
                    if a[0]['middle'] != b[0]['middle']:
                        return False
                else:
                    if a[0]['middle'][0] != b[0]['middle'][0]:
                        return False
        return True

    # Return true if names contains equal names
    @staticmethod
    def one_name_in_dict(names: dict):
        keys = list(names.keys())
        len_ = len(keys)
        for i in range(len_-1):
            for j in range(i+1, len_):
                if not ReviewAnalysis.equal_names(names[keys[i]], names[keys[j]]):
                    return False
        return True

    # Return name and gender of review if in review one entity, and '', '' in another case.
    @staticmethod
    def get_name_entity_details(review: str):
        names = ReviewAnalysis.get_names(review)
        if names != {} and ReviewAnalysis.one_name_in_dict(names):
            gender = 'Masc'
            for name in names.keys():
                if names[name][1] == 'Fem':
                    gender = 'Fem'

            names = [_[0] for _ in names.values()]
            full_name = ReviewAnalysis.merge_names(names)
            name_str = ''
            if 'last' in full_name.keys():
                name_str = full_name['last']
            if 'first' in full_name.keys():
                name_str += ' ' + full_name['first']
            if 'middle' in full_name.keys():
                name_str += ' ' + full_name['middle']

            return name_str, gender
        else:
            return '', ''

    # Delete names in start of review in data
    def delete_names_in_start(self):
        self.data['spelling_capital'] = [True if self.has_name_in_start(review) else False
                                     for review in self.data['review']]

        self.data['review'] = self.data['review'].map(lambda x: self.delete_name_in_start(x))

    # Save backup self.data to backup_file
    def save_backup(self, backup_file: str):
        self.data.to_csv(backup_file, sep='\t', header=True, index=False)

    # Load backup self.data from backup_file
    def load_backup(self, backup_file: str):
        if os.path.exists(backup_file):
            self.data = pd.read_csv(backup_file, sep='\t')
        else:
            print('File not found!')

    # Send not spelling data in google sheets
    def report_to_sheet_output(self, sheets_api, table_id: str, list_name: str):
        # self.buf_data = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]
        buf_data = self.data[(self.data.spelling == False)].sort_values(by=['duble_class', 'duble_good'])

        # Put header
        sheets_api.put_row_to_sheets(table_id, list_name, 1, 'A', [
            'review',
            'sectionId',
            'type_page',
            'type_model',
            'initials_workers',
            'gender',
            'name_entity',
            'comment'
        ])

        # Put data
        shift = 2
        data_list = buf_data['review'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'A', shift, data_list, PACKET_SIZE)
        data_list = buf_data['sectionId'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'B', shift, data_list, PACKET_SIZE)
        data_list = buf_data['type_page'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'C', shift, data_list, PACKET_SIZE)
        data_list = buf_data['type_model'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'D', shift, data_list, PACKET_SIZE)

        # Put name_entitry
        has_names_list = buf_data['initials_workers'].to_list()
        # has_names_list = ['initials_workers' if has_names_list[i] else '' for i in range(len(has_names_list))]
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'E', shift, has_names_list, PACKET_SIZE)

        # Put name_entitry
        has_names_list = buf_data['gender'].to_list()
        # has_names_list = ['gender' if has_names_list[i] else '' for i in range(len(has_names_list))]
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'F', shift, has_names_list, PACKET_SIZE)

        # Put name_entitry
        has_names_list = buf_data['name_entity'].to_list()
        has_names_list = ['name_entity' if has_names_list[i] else '' for i in range(len(has_names_list))]
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'G', shift, has_names_list, PACKET_SIZE)

    # Send statistic google sheets
    def report_to_sheet_output_compare(self, sheets_api, table_id: str, list_name: str):
        # Put stats
        sheets_api.put_column_to_sheets(table_id, list_name, 'I', 1, [
            'all_review',
            'amount_Duble',
            'amount_Duble_file',
            'amount_Duble_good',
            'amount_Spelling',
            'amount_name_entity',
            'amount_unique_words',
            'all_words',
            'amount_unique_words / all_words'
        ])
        # Put stats
        reviews_valid = len(self.data[self.data.spelling == False]['review'].index)
        amount_duble = len(self.data.loc[(self.data.duble_good == False) &
                                         (self.data.spelling == False)]['review'].index)
        ammount_named = len(self.data.loc[(self.data.name_entity == True) &
                                          (self.data.duble_good == True) &
                                          (self.data.spelling == False)]['review'].index)

        sheets_api.put_column_to_sheets(table_id, list_name, 'J', 1, [
            str(len(self.data.index)),
            str(amount_duble),
            str(len(self.data[self.data.duble_file == True]['review'].index)),
            str(reviews_valid - amount_duble),
            str(len(self.data[self.data.spelling == True]['review'].index)),
            str(ammount_named),
            str(self.amount_unique_words),
            str(self.amount_words),
            str(self.amount_unique_words / self.amount_words * 100)
        ])

    # Download goods review to csv_file_name
    @staticmethod
    def download_goods(google_api, table_id: str, list_name: str, csv_file_name: str):
        data = google_api.get_data_from_sheets(table_id, list_name, 'A2',
                                        'D' + str(google_api.get_list_size(table_id, list_name)[1]), 'ROWS')
        comment = google_api.get_data_from_sheets(table_id, list_name, 'F2',
                                        'F' + str(google_api.get_list_size(table_id, list_name)[1]), 'ROWS')
        if len(comment) == 0:
            print('Reviews with mark "хороший" has not found, check F column.')

        buf_data = pd.DataFrame({'review': [], 'sectionId': [], 'type_page': [], 'type_model': []})
        for i in range(len(comment)):
            if len(comment[i]) > 0 and comment[i][0] == 'хороший':
                buf_data = buf_data.append({'review': data[i][0], 'sectionId': data[i][1], 'type_review': data[i][2],
                                            'type_model': data[i][3], 'used': False}, ignore_index=True)

        if os.path.exists(csv_file_name):
            buf_data.to_csv(csv_file_name, sep='\t', encoding='utf-8', mode='a', index=False, header=False)
        else:
            buf_data.to_csv(csv_file_name, sep='\t', encoding='utf-8', index=False, header=True)

    # Check true end of sentence (. or !)
    @staticmethod
    def check_end_of_sentence(sentence: str):
        if len(sentence) == 0:
            return False

        end_symbol = ''
        skip_phase = True
        count_end_symbols = 0

        for i in range(len(sentence)):
            if skip_phase:
                if sentence[-i - 1] == ' ':
                    continue
                elif sentence[-i - 1] == '.' or sentence[-i - 1] == '!':
                    skip_phase = False
                    end_symbol = sentence[-i - 1]
                else:
                    return False
            else:
                if sentence[-i - 1] == end_symbol:
                    if count_end_symbols < 3:
                        count_end_symbols += 1
                    else:
                        return False
                else:
                    if sentence[-i - 1] != '.' and sentence[-i - 1] != '!':
                        return True
                    else:
                        return False
        return False

    # Return true if review not spelling
    @staticmethod
    def is_spoiling(review: str):
        if len(review) < MIN_LEN_REVIEW:
            return True

        russian_cymbols = len(re.findall(r'[А-ЯЁа-яё]', review))
        english_cymbols = len(re.findall(r'[A-Za-z]', review))

        is_not_english = russian_cymbols > 0 and\
                         english_cymbols/(russian_cymbols+english_cymbols) <= MAX_PERCENT_LATIN_LETTER_IN_REVIEW

        is_ended_sentence = ReviewAnalysis.check_end_of_sentence(review)

        # Black words
        black_words = False
        for special_name in NAMES_DICT:
            black_words = review.lower().find(special_name.lower()) != -1 or black_words

        return not (is_not_english and is_ended_sentence) or black_words

    # Return corrected review
    @staticmethod
    def correct_review(review: str):
        # # Delete english letter
        # review = re.sub(r'[A-Za-z]', '', review)

        # Correct end
        review = review.strip()

        if len(review) < 1:
            return review

        if review[-1] != '.' and review[-1] != '!':
            sents = review.split('.')
            sents = sents[:len(sents)-1]
            review = '.'.join(sents) + '.'

        return review

    # Return review without stop-words and non-letter symbols
    @staticmethod
    def clean_review(review: str):
        review = review.lower()
        review = ''.join([letter if letter in alphabet else ' ' for letter in review])
        review = ' '.join([word for word in review.split() if word not in STOP_WORDS])
        return review

    # Return lemma of review
    @staticmethod
    def lemmatization_review(review: str):
        doc = Doc(review)
        doc.segment(natasha_segmenter)
        doc.tag_morph(natasha_morph_tagger)

        for token in doc.tokens:
            token.lemmatize(natasha_morph_vocab)

        lemma_review = ' '.join([_.lemma for _ in doc.tokens])
        return lemma_review

    # Return names of review
    @staticmethod
    def get_names(review: str):
        # Detect russian names
        ru_doc = Doc(review)
        ru_doc.segment(natasha_segmenter)
        ru_doc.tag_morph(natasha_morph_tagger)
        ru_doc.tag_ner(natasha_ner_tagger)

        russian_names = {}
        for span in ru_doc.spans:
            if span.type == PER:
                span.normalize(natasha_morph_vocab)
                span.extract_fact(natasha_names_extractor)
                if span.fact is not None:
                    russian_names[span.normal] = [span.fact.as_dict]
                    gender = 'Masc'
                    for token in span.tokens:
                        if 'Gender' in token.feats.keys() and token.feats['Gender'] == 'Fem':
                            gender = 'Fem'
                    russian_names[span.normal].append(gender)
                # else:
                #     print('None fact: ', span.text)

        return russian_names

    # Return true if review has name entity
    @staticmethod
    def has_name_entity(review: str):
        return len(ReviewAnalysis.get_names(review).keys()) != 0

    # Clear data and statistic
    def clear_data(self):
        self.data = pd.DataFrame({'review': [], 'sectionId': [], 'type_page': [], 'type_model': []})
        self.amount_unique_words = 0
        self.amount_words = 0

    # Return true if name in start
    @staticmethod
    def has_name_in_start(review: str):
        has_english_name = len(re.findall(r'^([A-Z][a-z]{0,}\s){1,}-\s{1,}', review)) != 0
        has_russian_name = len(re.findall(r'^([А-ЯЁ][а-яё]{0,}\s){1,}-\s{1,}', review)) != 0

        return has_english_name or has_russian_name

    # Return review without name in the start
    @staticmethod
    def delete_name_in_start(review: str):
        if not ReviewAnalysis.has_name_in_start(review):
            return review

        review = review[review.find('-')+1:].lstrip()
        review = review[0].upper() + review[1:len(review)]
        return review
