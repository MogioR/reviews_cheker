from google_sheets_api import GoogleSheetsApi
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

from natasha import Segmenter
from natasha import MorphVocab
from natasha import NewsEmbedding
from natasha import NewsMorphTagger
from natasha import NewsNERTagger
from natasha import Doc

import enchant
from enchant.checker import SpellChecker

import re

"""Max num of reviews"""
MAX_REVIEWS_COUNT = 10000
"""Words without sense"""
# nltk.download('stopwords')
STOP_WORDS = get_stop_words('russian') + stopwords.words('russian')
"""Spelling params8."""
MAX_PERCENT_LATIN_LETTER_IN_REVIEW = 0.25
MIN_LEN_REVIEW = 200
"""File with google service auth token."""
DUPLICATES_UNIQUENESS = 0.4
"""Russian alphabet with space"""
alphabet = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", " ",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]
"""Colors"""
INCLUDING_NAMES_COLOR = [0.0, 1.0, 0.0]
INCLUDING_MISTAKES_COLOR = [1.0, 1.0, 0.0]


class ReviewAnalysis:
    def __init__(self):
        self.data = pd.DataFrame({'review': [], 'skill': [], 'type_review': []})

        """Natasha parser initialization"""
        self.natasha_emb = NewsEmbedding()
        self.natasha_ner_tagger = NewsNERTagger(self.natasha_emb)
        self.natasha_segmenter = Segmenter()
        self.natasha_morph_vocab = MorphVocab()
        self.natasha_morph_tagger = NewsMorphTagger(self.natasha_emb)

        """Enchant parser initialization"""
        self.enchant_dict_ru = enchant.Dict("ru_RU")
        self.enchant_dict_en = enchant.Dict("en_US")

        """Stats"""
        self.amount_unique_words = 0
        self.amount_words = 0

    def add_data(self, data):
        # Getting data
        for raw in data:
            reviews = raw[0].split('\n')
            for review in reviews:
                self.data = self.data.append({'review': review, 'skill': raw[1], 'type_review': raw[2],
                                              'type_model': raw[3]}, ignore_index=True)

    def mark_spelling(self):
        self.data['spelling'] = [True if self.is_spoiling(review) else False for review in self.data['review']]

    def get_duplicat_matrix_v1(self, reviews_good):
        # Get cosine_similarity of reviews_good
        cleaned_reviews = list(map(self.clean_review, reviews_good))
        lemmatized_reviews = list(map(self.lemmatization_review, cleaned_reviews))

        vectorizer = CountVectorizer().fit_transform(lemmatized_reviews)
        vectors = vectorizer.toarray()
        csim = cosine_similarity(vectors)

        self.amount_unique_words = len(vectors[0])
        self.amount_words = 0
        for vec in vectors:
            self.amount_words += sum(vec)

        return vectors, csim

    def get_duplicat_matrix_v2(self, reviews_good):
        # Get cosine_similarity of reviews_good
        cleaned_reviews = list(map(self.clean_review, reviews_good))
        lemmatized_reviews = list(map(self.lemmatization_review, cleaned_reviews))

        vectorizer = CountVectorizer().fit_transform(lemmatized_reviews)
        vectors = vectorizer.toarray()

        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(lemmatized_reviews)
        pairwise_similarity = tfidf * tfidf.T
        csim = pairwise_similarity.toarray()

        self.amount_unique_words = tfidf.shape[1]
        self.amount_words = 0
        for vec in vectors:
            self.amount_words += sum(vec)

        return vectors, csim

    def mark_duplicates(self):
        self.data['duble_good'] = False
        self.data['duble_class'] = 0

        reviews_good = self.data[self.data.spelling == False]['review']

        # Exit if reviews_good is empty
        if len(reviews_good.values) == 0:
            print(len(reviews_good.values))
            return

        vectors, csim = self.get_duplicat_matrix_v1(reviews_good.values)

        # Find duplicates and count uniqueness words
        duplicates = []
        for i in range(len(csim)):
            duplicates_buf = []
            for j in range(len(csim[i])):
                if i != j and csim[i][j] >= DUPLICATES_UNIQUENESS:
                    duplicates_buf.append([j, np.count_nonzero(np.array(vectors[j]))])
            duplicates.append(duplicates_buf)

        # Find uniqueness
        duble_class = 1
        for i in range(len(duplicates)):
            if duplicates[i] != []:
                max_uniquen = 0
                max_id = 0
                for j in duplicates[i]:
                    if j[1] > max_uniquen:
                        max_uniquen = j[1]
                        max_id = j[0]

                    self.data.at[reviews_good.index[j[0]], 'duble_class'] = duble_class
                    self.data.at[reviews_good.index[j[0]], 'duble_good'] = False
                self.data.at[reviews_good.index[i], 'duble_class'] = duble_class
                self.data.at[reviews_good.index[i], 'duble_good'] = False

                if max_uniquen > np.count_nonzero(np.array(vectors[i])):
                    self.data.at[reviews_good.index[max_id], 'duble_good'] = True
                    duble_class += 1
                else:
                    self.data.at[reviews_good.index[i], 'duble_good'] = True
                    duble_class += 1
            else:
                self.data.at[reviews_good.index[i], 'duble_good'] = True
                self.data.at[reviews_good.index[i], 'duble_class'] = duble_class
                duble_class += 1

        self.data['duble'] = [True if (not review['duble_good'] and not review['spelling'])
                              else False for i, review in self.data.iterrows()]

    def mark_name_entity(self):
        self.data['name_entity'] = [True if self.has_name_entity(review) else False for review in self.data['review']]

    def report_to_sheet_output(self, sheets_api, table_id, list_name):
        # Clear data
        sheets_api.clear_sheet(table_id, list_name)

        # Put header
        sheets_api.put_row_to_sheets(table_id, list_name, 1, 'A', 'I', [
            'review',
            'sectionId',
            'type_page',
            'type_model',
            'duble',
            'duble_good',
            'name_entity',
            'duble_class',
            'comment'
        ])

        self.buf_data = self.data[self.data['spelling'] == False].sort_values(by=['duble_class', 'duble_good'])
        print(self.buf_data)
        # Put data
        shift = 2
        data_list = self.buf_data['review'].to_list()
        sheets_api.put_column_to_sheets(table_id, list_name, 'A', shift, len(data_list) + shift, data_list)
        data_list = self.buf_data['skill'].to_list()
        sheets_api.put_column_to_sheets(table_id, list_name, 'B', shift, len(data_list) + shift, data_list)
        data_list = self.buf_data['type_review'].to_list()
        sheets_api.put_column_to_sheets(table_id, list_name, 'C', shift, len(data_list) + shift, data_list)
        data_list = self.buf_data['type_model'].to_list()
        sheets_api.put_column_to_sheets(table_id, list_name, 'D', shift, len(data_list) + shift, data_list)

        # Put duplicates
        dubles_list = self.buf_data['duble'].to_list()
        dubles_list = ['duble' if duble else '' for duble in dubles_list]
        sheets_api.put_column_to_sheets(table_id, list_name, 'E', shift, len(data_list) + shift, dubles_list)

        # Put goods
        good_list = self.buf_data['duble_good'].to_list()
        good_list = ['duble_good' if good else '' for good in good_list]
        sheets_api.put_column_to_sheets(table_id, list_name, 'F', shift, len(data_list) + shift, good_list)

        # Put name_entitry
        has_names_list = self.buf_data['name_entity'].to_list()
        has_names_list = ['name_entity' if has_names_list[i] else '' for i in range(len(has_names_list))]
        sheets_api.put_column_to_sheets(table_id, list_name, 'G', shift, len(data_list) + shift, has_names_list)

        # Put name_entitry
        good_classes_list = self.buf_data['duble_class'].to_list()
        good_classes_list = [str(good_class) for good_class in good_classes_list]
        sheets_api.put_column_to_sheets(table_id, list_name, 'H', shift, len(data_list) + shift, good_classes_list)


    def report_to_sheet_output_compare(self, sheets_api, table_id, list_name):
        # Put stats
        sheets_api.put_column_to_sheets(table_id, list_name, 'J', 1, 8, [
            'all_review',
            'amount_Spelling',
            'amount_Duble',
            'amount_Duble_good',
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
                                          (self.data.spelling == False)]['review'].index)
        sheets_api.put_column_to_sheets(table_id, list_name, 'K', 1, 8, [
            str(len(self.data.index)),
            str(len(self.data[self.data.spelling == True]['review'].index)),
            str(amount_duble),
            str(reviews_valid - amount_duble),
            str(ammount_named),
            str(self.amount_unique_words),
            str(self.amount_words),
            str(self.amount_unique_words / self.amount_words)
        ])

    def check_end_of_sentence(self, sentence):
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

    def is_spoiling(self, review):
        if len(review) < MIN_LEN_REVIEW:
            return True

        russian_cymbols = len(re.findall(r'[А-ЯЁа-яё]', review))
        english_cymbols = len(re.findall(r'[A-Za-z]', review))
        is_not_english = russian_cymbols > 0 and\
                         english_cymbols/(russian_cymbols+english_cymbols) <= MAX_PERCENT_LATIN_LETTER_IN_REVIEW

        is_ended_sentence = self.check_end_of_sentence(review)

        if is_not_english and is_ended_sentence:
            return False
        else:
            return True

    """Function clear review string"""
    """Input: review(str)"""
    """Output: review(str) """
    def clean_review(self, review):
        review = review.lower()
        review = ''.join([letter if letter in alphabet else ' ' for letter in review])
        review = ' '.join([word for word in review.split() if word not in STOP_WORDS])
        return review

    """Function lemmatization review string"""
    """Input: review(str)"""
    """Output: review(str) """
    def lemmatization_review(self, review):
        doc = Doc(review)
        doc.segment(self.natasha_segmenter)
        doc.tag_morph(self.natasha_morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.natasha_morph_vocab)

        return ' '.join([_.lemma for _ in doc.tokens])

    """Function check including names in review"""
    """Input: review(str)"""
    """Output: Answer(bool)"""
    def has_name_entity(self, review):
        # Detect russian names
        ru_doc = Doc(review)
        ru_doc.segment(self.natasha_segmenter)
        ru_doc.tag_ner(self.natasha_ner_tagger)

        # Detect english
        english_letters = len(re.findall(r'[A-Za-z]', review))

        if len(ru_doc.spans) == 0 and english_letters == 0:
            return False
        else:
            return True

    def clear_data(self):
        self.data = pd.DataFrame({'review': [], 'skill': [], 'type_review': []})
        self.amount_unique_words = 0
        self.amount_words = 0

    def has_name_in_start(self, review):
        has_english_name = len(re.findall(r'^([A-Z][a-z]{0,}\s){1,}-\s{1,}', review)) != 0
        has_russian_name = len(re.findall(r'^([А-ЯЁ][а-яё]{0,}\s){1,}-\s{1,}', review)) != 0

        return has_english_name or has_russian_name

    def delete_name_in_start(self, review):
        review = review[review.find('-')+1:].lstrip
        review[0] = review[0].upper()
        return review