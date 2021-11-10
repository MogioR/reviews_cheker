import re
import os
import math
import time
from threading import Thread

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc, NamesExtractor, PER
from tqdm import tqdm

"""Words without sense"""
STOP_WORDS = get_stop_words('russian') + stopwords.words('russian')
"""Spelling params8."""
MAX_PERCENT_LATIN_LETTER_IN_REVIEW = 0.25
MIN_LEN_REVIEW = 150
"""File with google service auth token."""
DUPLICATES_UNIQUENESS = 0.6
"""Russian alphabet with space"""
alphabet = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", " ",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]
"""Colors"""
INCLUDING_NAMES_COLOR = [0.0, 1.0, 0.0]
INCLUDING_MISTAKES_COLOR = [1.0, 1.0, 0.0]
"""Special name entity"""
NAMES_DICT = ['профи', 'Ваш репетитор', 'Preply']

NUM_THREADS = 8         # Threads num
PACKET_SIZE = 250       # Google sheets packet size
DEBUG = False           # Debug mode on/off
DEBUG_DATA_SIZE = 500   # Reviews count in debug mode


class ReviewAnalysis:
    def __init__(self):
        self.data = pd.DataFrame({'review': [], 'sectionId': [], 'type_page': [], 'type_model': []})

        """Natasha parser initialization"""
        self.natasha_emb = NewsEmbedding()
        self.natasha_ner_tagger = NewsNERTagger(self.natasha_emb)
        self.natasha_segmenter = Segmenter()
        self.natasha_morph_vocab = MorphVocab()
        self.natasha_morph_tagger = NewsMorphTagger(self.natasha_emb)
        self.natasha_names_extractor = NamesExtractor(self.natasha_morph_vocab)
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
        self.data['review_correct'] = [self.correct_review(review) for review in self.data['review']]

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

        # Find duplicates and count uniqueness words
        duplicates_pairs = self.get_duble_pairs(csim)

        # Find uniqueness
        # self.mark_duplicates_by_pairs(reviews_good, vectors, duplicates_pairs)
        # dubles = list([reviews_good.index[i[0]], reviews_good.index[i[1]]] for i in duplicates_pairs)
        # print('Dubles: ', len(duplicates_pairs))
        # print(dubles)
        #
        # start = time.time()
        self.fast_mark_duplicates_by_pairs(reviews_good, vectors, duplicates_pairs)
        # goods = len(self.data[(self.data.duble_good == True) & (self.data.spelling == False)]['review'].index)
        # print('Time: ', time.time()-start, ' Goods: ', goods)

        # reviews_good1 = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]['review']
        # cleaned_reviews1 = list(map(self.clean_review, reviews_good1.values))
        # lemmatized_reviews1 = list(map(self.lemmatization_review, cleaned_reviews1))
        # vectors, csim = self.get_duplicat_matrix(lemmatized_reviews1)
        # duplicates_pairs = self.get_duble_pairs(csim)
        #
        # dubles = list([reviews_good1.index[i[0]], reviews_good1.index[i[1]]] for i in duplicates_pairs)
        # print('Dubles: ', len(duplicates_pairs))
        # print(dubles)

    # Mark duplicates with file in data
    def mark_file_duplicates(self, csv_file: str):
        self.data['duble_file'] = False

        reviews_good = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]['review']
        reviews_good_count = len(reviews_good.values)

        if not os.path.exists(csv_file):
            print('Can\'t find file: '+csv_file)
            return

        reviews_file = list(pd.read_csv(csv_file, sep='\t')['review'].values)
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

    # Mark duplicates by list pairs in data (threading - start)
    def fast_mark_duplicates_by_pairs(self, reviews_good, vectors, duplicates_pairs: list):
        self.data['duble_good'] = True
        self.data['duble_class'] = 0

        buffer_double_good = list(self.data['duble_good'].values)
        buffer_double_class = list(self.data['duble_class'].values)
        originals_list = []

        threads = []

        len_ = math.ceil(len(duplicates_pairs)/NUM_THREADS)

        def func_chunks_generators(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i: i + n]
        duplicates_pairs_parts = list(func_chunks_generators(duplicates_pairs, len_))

        for i in range(NUM_THREADS):
            if i == 0:
                threads.append(Thread(target=self.fast_mark_duplicates_py_pairs_thread_tqdm,
                                      args=(buffer_double_good, buffer_double_class, originals_list, reviews_good,
                                            vectors, duplicates_pairs_parts[i])))
            else:
                threads.append(Thread(target=self.fast_mark_duplicates_py_pairs_thread,
                                      args=(buffer_double_good, buffer_double_class, originals_list, reviews_good,
                                            vectors, duplicates_pairs_parts[i])))
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.data['duble_good'] = buffer_double_good
        self.data['duble_class'] = buffer_double_class

    # Mark duplicates by list pairs in data (threading - thread function)
    def fast_mark_duplicates_py_pairs_thread(self, buffer_double_good, buffer_double_class, originals_list,
                                             reviews_good, vectors, duplicates_pairs):
        for pair in duplicates_pairs:
            # Both in pair hasn't class
            if buffer_double_class[reviews_good.index[pair[0]]] == 0 and \
                    buffer_double_class[reviews_good.index[pair[1]]] == 0:

                buffer_double_class[reviews_good.index[pair[0]]] = len(originals_list) + 1
                buffer_double_class[reviews_good.index[pair[1]]] = len(originals_list) + 1

                if np.count_nonzero(np.array(vectors[pair[0]])) > np.count_nonzero(np.array(vectors[pair[1]])):
                    buffer_double_good[reviews_good.index[pair[1]]] = False
                    originals_list.append(pair[0])
                else:
                    buffer_double_good[reviews_good.index[pair[0]]] = False
                    originals_list.append(pair[1])

            # One in pair hasn't class
            elif buffer_double_class[reviews_good.index[pair[0]]] == 0 and \
                    buffer_double_class[reviews_good.index[pair[1]]] != 0:

                duble_class_1 = buffer_double_class[reviews_good.index[pair[1]]]
                buffer_double_class[reviews_good.index[pair[0]]] = duble_class_1

                # Pair[0] is straight duble originals_list[duble_class_1 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[0], originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, pair[0]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1 - 1]):

                        buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = False
                        originals_list[duble_class_1 - 1] = pair[0]
                    else:
                        buffer_double_good[reviews_good.index[pair[0]]] = False
                # Else create new class
                else:
                    buffer_double_class[reviews_good.index[pair[0]]] = len(originals_list) + 1
                    originals_list.append(pair[0])

            elif buffer_double_class[reviews_good.index[pair[0]]] != 0 and\
                    buffer_double_class[reviews_good.index[pair[1]]] == 0:

                duble_class_0 = buffer_double_class[reviews_good.index[pair[0]]]
                buffer_double_class[reviews_good.index[pair[1]]] = duble_class_0

                # Pair[1] is straight duble originals_list[duble_class_0 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[1], originals_list[duble_class_0 - 1]):
                    if self.count_uniqueness_words(vectors, pair[1]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_0 - 1]):

                        buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = False
                        originals_list[duble_class_0 - 1] = pair[1]
                    else:
                        buffer_double_good[reviews_good.index[pair[1]]] = False
                # Else create new class
                else:
                    buffer_double_class[reviews_good.index[pair[1]]] = len(originals_list) + 1
                    originals_list.append(pair[1])

            # Both in pair has unequal class
            elif buffer_double_class[reviews_good.index[pair[0]]] != buffer_double_class[reviews_good.index[pair[1]]]:
                duble_class_0 = buffer_double_class[reviews_good.index[pair[0]]]
                duble_class_1 = buffer_double_class[reviews_good.index[pair[1]]]

                if self.check_straight_duble(duplicates_pairs, originals_list[duble_class_0 - 1],
                                             originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, originals_list[duble_class_0 - 1]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1 - 1]):
                        # Element mark duble_class in class 0
                        buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = False
                        buffer_double_class[reviews_good.index[originals_list[duble_class_1 - 1]]] = \
                            duble_class_0

                        # Search new max in class 1
                        class_indexes = []
                        for i in range(len(buffer_double_class)):
                            if buffer_double_class[i] == duble_class_1:
                                class_indexes.append(i)

                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_1 - 1,
                                                              originals_list)
                            buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = True
                    else:
                        # Element mark duble_class in class 1
                        buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = False
                        buffer_double_class[reviews_good.index[originals_list[duble_class_0 - 1]]] = \
                            duble_class_1

                        # Search new max in class 0
                        class_indexes = []
                        for i in range(len(buffer_double_class)):
                            if buffer_double_class[i] == duble_class_0:
                                class_indexes.append(i)

                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_0 - 1,
                                                              originals_list)
                            buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = True
                else:
                    pass

            # Both in pair has equal class
            else:
                pass

    # Mark duplicates by list pairs in data (threading - thread function with progressbar)
    def fast_mark_duplicates_py_pairs_thread_tqdm(self, buffer_double_good, buffer_double_class, originals_list,
                                                  reviews_good, vectors, duplicates_pairs):
        for pair in tqdm(duplicates_pairs):
            # Both in pair hasn't class
            if buffer_double_class[reviews_good.index[pair[0]]] == 0 and \
                    buffer_double_class[reviews_good.index[pair[1]]] == 0:

                buffer_double_class[reviews_good.index[pair[0]]] = len(originals_list) + 1
                buffer_double_class[reviews_good.index[pair[1]]] = len(originals_list) + 1

                if np.count_nonzero(np.array(vectors[pair[0]])) > np.count_nonzero(np.array(vectors[pair[1]])):
                    buffer_double_good[reviews_good.index[pair[1]]] = False
                    originals_list.append(pair[0])
                else:
                    buffer_double_good[reviews_good.index[pair[0]]] = False
                    originals_list.append(pair[1])

            # One in pair hasn't class
            elif buffer_double_class[reviews_good.index[pair[0]]] == 0 and \
                    buffer_double_class[reviews_good.index[pair[1]]] != 0:

                duble_class_1 = buffer_double_class[reviews_good.index[pair[1]]]
                buffer_double_class[reviews_good.index[pair[0]]] = duble_class_1

                # Pair[0] is straight duble originals_list[duble_class_1 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[0], originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, pair[0]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1 - 1]):

                        buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = False
                        originals_list[duble_class_1 - 1] = pair[0]
                    else:
                        buffer_double_good[reviews_good.index[pair[0]]] = False
                # Else create new class
                else:
                    buffer_double_class[reviews_good.index[pair[0]]] = len(originals_list) + 1
                    originals_list.append(pair[0])

            elif buffer_double_class[reviews_good.index[pair[0]]] != 0 and\
                    buffer_double_class[reviews_good.index[pair[1]]] == 0:

                duble_class_0 = buffer_double_class[reviews_good.index[pair[0]]]
                buffer_double_class[reviews_good.index[pair[1]]] = duble_class_0

                # Pair[1] is straight duble originals_list[duble_class_0 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[1], originals_list[duble_class_0 - 1]):
                    if self.count_uniqueness_words(vectors, pair[1]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_0 - 1]):

                        buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = False
                        originals_list[duble_class_0 - 1] = pair[1]
                    else:
                        buffer_double_good[reviews_good.index[pair[1]]] = False
                # Else create new class
                else:
                    buffer_double_class[reviews_good.index[pair[1]]] = len(originals_list) + 1
                    originals_list.append(pair[1])

            # Both in pair has unequal class
            elif buffer_double_class[reviews_good.index[pair[0]]] != buffer_double_class[reviews_good.index[pair[1]]]:
                duble_class_0 = buffer_double_class[reviews_good.index[pair[0]]]
                duble_class_1 = buffer_double_class[reviews_good.index[pair[1]]]

                if self.check_straight_duble(duplicates_pairs, originals_list[duble_class_0 - 1],
                                             originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, originals_list[duble_class_0 - 1]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1 - 1]):
                        # Element mark duble_class in class 0
                        buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = False
                        buffer_double_class[reviews_good.index[originals_list[duble_class_1 - 1]]] = \
                            duble_class_0

                        # Search new max in class 1
                        class_indexes = []
                        for i in range(len(buffer_double_class)):
                            if buffer_double_class[i] == duble_class_1:
                                class_indexes.append(i)

                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_1 - 1,
                                                              originals_list)
                            buffer_double_good[reviews_good.index[originals_list[duble_class_1 - 1]]] = True
                    else:
                        # Element mark duble_class in class 1
                        buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = False
                        buffer_double_class[reviews_good.index[originals_list[duble_class_0 - 1]]] = \
                            duble_class_1

                        # Search new max in class 0
                        class_indexes = []
                        for i in range(len(buffer_double_class)):
                            if buffer_double_class[i] == duble_class_0:
                                class_indexes.append(i)

                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_0 - 1,
                                                              originals_list)
                            buffer_double_good[reviews_good.index[originals_list[duble_class_0 - 1]]] = True
                else:
                    pass

            # Both in pair has equal class
            else:
                pass

    # Mark duplicates by list pairs in data (one - thread)
    def mark_duplicates_by_pairs(self, reviews_good, vectors, duplicates_pairs):
        self.data['duble_good'] = True
        self.data['duble_class'] = 0

        duble_class = 1
        originals_list = []

        for pair in tqdm(duplicates_pairs):
            # Both in pair hasn't class
            if self.get_duble_class(reviews_good, pair[0]) == 0 and self.get_duble_class(reviews_good, pair[1]) == 0:
                self.data.at[reviews_good.index[pair[0]], 'duble_class'] = duble_class
                self.data.at[reviews_good.index[pair[1]], 'duble_class'] = duble_class

                duble_class = duble_class + 1

                if np.count_nonzero(np.array(vectors[pair[0]])) > np.count_nonzero(np.array(vectors[pair[1]])):
                    self.data.at[reviews_good.index[pair[1]], 'duble_good'] = False
                    originals_list.append(pair[0])
                else:
                    self.data.at[reviews_good.index[pair[0]], 'duble_good'] = False
                    originals_list.append(pair[1])

            # One in pair hasn't class
            elif self.get_duble_class(reviews_good, pair[0]) == 0 and self.get_duble_class(reviews_good, pair[1]) != 0:
                duble_class_1 = self.data.at[reviews_good.index[pair[1]], 'duble_class']
                self.data.at[reviews_good.index[pair[0]], 'duble_class'] = duble_class_1

                # Pair[0] is straight duble originals_list[duble_class_1 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[0], originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, pair[0]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1 - 1]):

                        self.data.at[reviews_good.index[originals_list[duble_class_1 - 1]], 'duble_good'] = False
                        originals_list[duble_class_1 - 1] = pair[0]
                    else:
                        self.data.at[reviews_good.index[pair[0]], 'duble_good'] = False
                # Else create new class
                else:
                    self.data.at[reviews_good.index[pair[0]], 'duble_class'] = duble_class
                    originals_list.append(pair[0])
                    duble_class = duble_class + 1

            elif self.get_duble_class(reviews_good, pair[0]) != 0 and self.get_duble_class(reviews_good, pair[1]) == 0:
                duble_class_0 = self.get_duble_class(reviews_good, pair[0])
                self.data.at[reviews_good.index[pair[1]], 'duble_class'] = duble_class_0

                # Pair[1] is straight duble originals_list[duble_class_0 - 1]
                if self.check_straight_duble(duplicates_pairs, pair[1], originals_list[duble_class_0 - 1]):
                    if self.count_uniqueness_words(vectors, pair[1]) >\
                            self.count_uniqueness_words(vectors, originals_list[duble_class_0-1]):

                        self.data.at[reviews_good.index[originals_list[duble_class_0-1]], 'duble_good'] = False
                        originals_list[duble_class_0 - 1] = pair[1]
                    else:
                        self.data.at[reviews_good.index[pair[1]], 'duble_good'] = False
                # Else create new class
                else:
                    self.data.at[reviews_good.index[pair[1]], 'duble_class'] = duble_class
                    originals_list.append(pair[1])
                    duble_class = duble_class + 1

            # Both in pair has unequal class
            elif self.get_duble_class(reviews_good, pair[0]) != self.get_duble_class(reviews_good, pair[1]):
                duble_class_0 = self.get_duble_class(reviews_good, pair[0])
                duble_class_1 = self.get_duble_class(reviews_good, pair[1])

                if self.check_straight_duble(duplicates_pairs, originals_list[duble_class_0 - 1],
                                             originals_list[duble_class_1 - 1]):
                    if self.count_uniqueness_words(vectors, originals_list[duble_class_0-1]) > \
                            self.count_uniqueness_words(vectors, originals_list[duble_class_1-1]):
                        # Element mark duble_class in class 0
                        self.data.at[reviews_good.index[originals_list[duble_class_1 - 1]], 'duble_good'] = False
                        self.data.at[reviews_good.index[originals_list[duble_class_1 - 1]], 'duble_class'] = \
                            duble_class_0

                        # Search new max in class 1
                        class_indexes = self.data.loc[self.data['duble_class'] == duble_class_1]['duble_class'].index\
                            .to_list()
                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_1 - 1,
                                                              originals_list)
                            self.data.at[reviews_good.index[originals_list[duble_class_1 - 1]], 'duble_good'] = True
                    else:
                        # Element mark duble_class in class 1
                        self.data.at[reviews_good.index[originals_list[duble_class_0 - 1]], 'duble_good'] = False
                        self.data.at[reviews_good.index[originals_list[duble_class_0 - 1]], 'duble_class'] = \
                            duble_class_1

                        # Search new max in class 0
                        class_indexes = self.data.loc[self.data['duble_class'] == duble_class_0]['duble_class'].index \
                            .to_list()
                        if len(class_indexes) > 0:
                            class_indexes_local = list(
                                map(lambda x: self.get_local_index(reviews_good.index.to_list(), x), class_indexes))
                            self.find_max_uniqueness_in_class(vectors, class_indexes_local, duble_class_0 - 1,
                                                              originals_list)
                            self.data.at[reviews_good.index[originals_list[duble_class_0 - 1]], 'duble_good'] = True
                else:
                    pass

            # Both in pair has equal class
            else:
                pass

    # Return duplicate pairs from duplicate matrix csim
    def get_duble_pairs(self, csim):
        duplicates_pairs = []
        for i in range(len(csim)-1):
            for j in range(i+1, len(csim[i])):
                if csim[i][j] >= self.duplicates_uniqueness:
                    duplicates_pairs.append([i, j])
        return duplicates_pairs

    # Return duplicate class by id
    def get_duble_class(self, reviews_good, id):
        return self.data.at[reviews_good.index[id], 'duble_class']

    # Return count uniqueness words by id
    @staticmethod
    def count_uniqueness_words(vectors, id):
        return np.count_nonzero(np.array(vectors[id]))

    # Return local index by global index
    @staticmethod
    def get_local_index(reviews_good_indexes, global_index):
        for local_index in range(len(reviews_good_indexes)):
            if reviews_good_indexes[local_index] == global_index:
                return local_index
        return -1

    # Find max uniqueness review in class, and save in originals_list
    def find_max_uniqueness_in_class(self, vectors, class_indexes_local, class_id, originals_list: list):
        max_id = 0
        max_value = self.count_uniqueness_words(vectors, class_indexes_local[max_id])

        for i in range(len(class_indexes_local)):
            if self.count_uniqueness_words(vectors, class_indexes_local[i]) > max_value:
                max_value = self.count_uniqueness_words(vectors, class_indexes_local[i])
                max_id = i

        originals_list[class_id] = class_indexes_local[max_id]

    # Return true if reviews i and j is duplicates
    @staticmethod
    def check_straight_duble(duplicates_pairs: list, i: int, j: int):
        return [min(i, j), max(i, j)] in duplicates_pairs

    # Mark name entities in data
    def mark_name_entity(self):
        self.data['name_entity'] = [True if self.has_name_entity(review) else False for review in self.data['review']]

    # Delete names in start of review in data
    def delete_names_in_start(self):
        self.data['spelling_capital'] = [True if self.has_name_in_start(review) else False
                                     for review in self.data['review']]

        self.data['review'] = self.data['review'].map(lambda x: self.delete_name_in_start(x))

    def save_backup(self, backup_file: str):
        self.data.to_csv(backup_file, sep='\t', header=True, index=False)

    def load_backup(self, backup_file: str):
        if os.path.exists(backup_file):
            self.data = pd.read_csv(backup_file, sep='\t')
        else:
            print('File not found!')

    # Send not spelling data in google sheets
    def report_to_sheet_output(self, sheets_api, table_id: str, list_name: str):
        self.buf_data = self.data[(self.data.duble_good == True) & (self.data.spelling == False)]

        # Put header
        sheets_api.put_row_to_sheets(table_id, list_name, 1, 'A', [
            'review',
            'sectionId',
            'type_page',
            'type_model',
            'name_entity',
            'comment'
        ])

        # Put data
        shift = 2
        packet_size = PACKET_SIZE
        data_list = self.buf_data['review'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'A', shift, data_list, packet_size)
        data_list = self.buf_data['sectionId'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'B', shift, data_list, packet_size)
        data_list = self.buf_data['type_page'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'C', shift, data_list, packet_size)
        data_list = self.buf_data['type_model'].to_list()
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'D', shift, data_list, packet_size)

        # Put name_entitry
        has_names_list = self.buf_data['name_entity'].to_list()
        has_names_list = ['name_entity' if has_names_list[i] else '' for i in range(len(has_names_list))]
        sheets_api.put_column_to_sheets_packets(table_id, list_name, 'E', shift, has_names_list, packet_size)

    # Send statistic google sheets
    def report_to_sheet_output_compare(self, sheets_api, table_id, list_name):
        # Put stats
        sheets_api.put_column_to_sheets(table_id, list_name, 'G', 1, [
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
        sheets_api.put_column_to_sheets(table_id, list_name, 'H', 1, [
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
    def download_goods(google_api, table_id, list_name, csv_file_name):
        data = google_api.get_data_from_sheets(table_id, list_name, 'A2',
                                        'D' + str(google_api.get_list_size(table_id, list_name)[1]), 'ROWS')
        comment = google_api.get_data_from_sheets(table_id, list_name, 'F2',
                                        'F' + str(google_api.get_list_size(table_id, list_name)[1]), 'ROWS')

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
    def check_end_of_sentence(sentence):
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
    def is_spoiling(review):
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
            black_words = review.find(special_name) != -1 or black_words

        return not (is_not_english and is_ended_sentence) or black_words

    # Return corrected review
    @staticmethod
    def correct_review(review):
        # Delete english letter
        review = re.sub(r'[A-Za-z]', '', review)
        # Correct end
        review = review.strip()
        if review[-1] != '.' and review[-1] != '!':
            sents = review.split('.')
            sents = sents[:len(sents)-1]
            review = '.'.join(sents) + '.'

        return review

    # Return review without stop-words and non-letter symbols
    @staticmethod
    def clean_review(review):
        review = review.lower()
        review = ''.join([letter if letter in alphabet else ' ' for letter in review])
        review = ' '.join([word for word in review.split() if word not in STOP_WORDS])
        return review

    # Return lemma of review
    def lemmatization_review(self, review):
        doc = Doc(review)
        doc.segment(self.natasha_segmenter)
        doc.tag_morph(self.natasha_morph_tagger)

        for token in doc.tokens:
            token.lemmatize(self.natasha_morph_vocab)

        lemma_review = ' '.join([_.lemma for _ in doc.tokens])
        return lemma_review

    # Return names of review
    def get_names(self, review):
        # Detect russian names
        ru_doc = Doc(review)
        ru_doc.segment(self.natasha_segmenter)
        ru_doc.tag_morph(self.natasha_morph_tagger)
        ru_doc.tag_ner(self.natasha_ner_tagger)

        # # Dictionary
        # dictionary_match = False
        # for special_name in NAMES_DICT:
        #     dictionary_match = review.find(special_name) != -1 or dictionary_match

        russian_names = {}
        for span in ru_doc.spans:
            if span.type == PER:
                span.normalize(self.natasha_morph_vocab)
                span.extract_fact(self.natasha_names_extractor)
                if span.fact is not None:
                    russian_names[span.normal] = [span.fact.as_dict]
                    gender = 'Masc'
                    for token in span.tokens:
                        if 'Gender' in token.feats.keys() and token.feats['Gender'] == 'Fem':
                            gender = 'Fem'
                    russian_names[span.normal].append(gender)
                # else:
                #     print('None fact: ', span.text)

        russian_names_count = len(russian_names.keys())

        return russian_names

    # # Return normalised names list
    # @staticmethod
    # def normalise_names(names: dict):
    #     len_ = len(names.keys())
    #     keys = list(names.keys())
    #     for i in range(len_):
    #         for j in range(i+1, len_):
    #             print(ReviewAnalysis.check_equal_of_names(names[keys[i]][0], names[keys[j]][0]))
    #
    #     return 'TOk'

    # Return true if review has name entity
    def has_name_entity(self, review):
        return len(self.get_names(review).keys()) != 0

    # Clear data and statistic
    def clear_data(self):
        self.data = pd.DataFrame({'review': [], 'sectionId': [], 'type_page': [], 'type_model': []})
        self.amount_unique_words = 0
        self.amount_words = 0

    # Return true if name in start
    @staticmethod
    def has_name_in_start(review):
        has_english_name = len(re.findall(r'^([A-Z][a-z]{0,}\s){1,}-\s{1,}', review)) != 0
        has_russian_name = len(re.findall(r'^([А-ЯЁ][а-яё]{0,}\s){1,}-\s{1,}', review)) != 0

        return has_english_name or has_russian_name

    # Return review without name in the start
    @staticmethod
    def delete_name_in_start(review):
        if not ReviewAnalysis.has_name_in_start(review):
            return review

        review = review[review.find('-')+1:].lstrip()
        review = review[0].upper() + review[1:len(review)]
        return review
