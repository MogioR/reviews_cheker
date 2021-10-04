import unittest
from review_analysis import ReviewAnalysis
import numpy as np
import pandas as pd

class TestReviewAnalysis(unittest.TestCase):
    def setUp(self):
        self.a = ReviewAnalysis()

    def test_is_spoiling(self):
        # English review
        self.assertTrue(self.a.is_spoiling('Maria is a great tutor. She can hear everything and speak very '
                                           'well with her students, which means that she will always be clear '
                                           'well with her students, which means that she will always be clear '
                                           'well with her students, which means that she will always be clear '
                                           'well with her students, which means that she will always be clear '
                                           'well with her students, which means that she will always be clear '
                                           'of your wrongs in English language!'))

        # Short review
        self.assertTrue(self.a.is_spoiling('В общем рекомендую этого преподавателя всем кто хочет подтянуть свой '
                                           'уровен'))

        # Wrong end of review
        self.assertTrue(self.a.is_spoiling('Елена Владимировна - приятный в общении человек. Педагог помогает мне '
                                           'повысить уровень владения английским языком, использует много различных '
                                           'учебных и методических материалов на занятиях с ней по подготовке к '
                                           'ЕГЭ IELTS TOEFL c упором именно как раз-таки конкретно разговорной речи '
                                           '(чтение). Занятия проходят очень продуктивно! Упор идет больше не только '
                                           'лишь собственно непосредственно on line-, но также комбинированно '
                                           'используется аудиоуроками весь материал для занятий подобран '
                                           'квалифицированным специалистом весьма грамотно согласно моему запросу :)'))

        # Normal end, some english words
        self.assertFalse(self.a.is_spoiling('Khan Faisal - отличный преподаватель, мне с ним очень комфортно работать! '
                                            'Он помогает подтянуть уровень владения английским языком и в целом у меня '
                                            'положительный отзыв о его работе. У репетитора есть своя методика '
                                            'обучения английскому языку на основе бизнес-лексики по учебникам '
                                            'McDonalds или Tivari Rupa suite. В основном мы занимаемся разговорной '
                                            'практикой для расширения кругозора, но также много времени уделяем '
                                            'грамматике языка (грамматика нам не нужна). Если возникают какие либо '
                                            'вопросы связанные со здоровьем то я всегда обращаюсь к нему за помощью!'))

        # Russian review
        self.assertFalse(self.a.is_spoiling('Мне очень нравится заниматься с Еленой Игоревной. Она хорошо преподает '
                                            'английский язык, доступно и понятно объясняет материал по предмету! Я уже '
                                            'значительно лучше стала знать грамматику языка благодаря занятиям со '
                                            'специалистом-носителем английского. Елена - доброжелательный человек, '
                                            'который всегда идет навстречу моим пожеланиям в плане графика '
                                            'проведения занятий или их продолжительности (для меня это имеет '
                                            'значение).'))

    def test_check_end_of_sentence(self):
        # True end of sentence
        self.assertTrue(self.a.check_end_of_sentence('Sentence.'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence..'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence...'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence....'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence!'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence!!'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence!!!'))
        self.assertTrue(self.a.check_end_of_sentence('Sentence!!!!'))

        # False end of sentence
        self.assertFalse(self.a.check_end_of_sentence('Sentence'))
        self.assertFalse(self.a.check_end_of_sentence('Sentence!.'))
        self.assertFalse(self.a.check_end_of_sentence('Sentence.!'))
        self.assertFalse(self.a.check_end_of_sentence('Sentence.....'))
        self.assertFalse(self.a.check_end_of_sentence('Sentence!.!!'))
        self.assertFalse(self.a.check_end_of_sentence('Sentence!..'))

    def test_name_entity(self):
        # Russian names entity
        self.assertTrue(self.a.has_name_entity('Елена Владимировна нам подошла полностью как со стороны '
                                               'профессиональных качеств так личностных характеристик преподавателя '
                                               'английского языка Елены Владимировны Багаевой к ребенку 5 класса '
                                               'начальной школы города Москвы.'))

        # English names entity
        self.assertTrue(self.a.has_name_entity('John Alexander - отличный преподаватель английского языка. С ним '
                                               'интересно заниматься, он дает много информации по предмету и грамотно '
                                               'строит уроки в зависимости от уровня ученика (грамматика или '
                                               'аудирование). Уроки проходят легко для восприятия на слух без '
                                               'лишних напоминаний о правилах грамматики!'))

        # Dictionary
        self.assertTrue(self.a.has_name_entity('Хороший педагог с хорошим уровнем знаний английского языка как у '
                                               'носителя; приятный в общении человек со стабильным доходом благодаря '
                                               'работе на сайте компании "Ваш репетитор". Репетитором мы очень '
                                               'довольны, продолжаем заниматься.'))

        # Just review
        self.assertFalse(self.a.has_name_entity('Репетитор мне нравится. Она отлично обучает английскому языку, '
                                                'материал доступно объясняет и занятия проводит очень интересно! На '
                                                'уроках мы работаем над лексикой английского языка (грамматика + '
                                                'аудирование), грамматиками на английском языке в аспектах лексики, '
                                                'произношение - все аспекты изучаем с нуля. Педагог всегда дает много '
                                                'полезной информации для работы или самостоятельного изучения '
                                                'материала по предмету!'))

    def test_has_name_in_start(self):
        # Russian name
        self.assertTrue(self.a.has_name_in_start('Елена Владимировна - подошла полностью как со стороны '
                                                 'профессиональных качеств так личностных характеристик преподавателя '
                                                 'английского языка Елены Владимировны Багаевой к ребенку 5 класса '
                                                 'начальной школы города Москвы.'))

        # English name
        self.assertTrue(self.a.has_name_in_start('John Alexander - отличный преподаватель английского языка. С ним '
                                                 'интересно заниматься, он дает много информации по предмету грамотно '
                                                 'строит уроки в зависимости от уровня ученика (грамматика или '
                                                 'аудирование). Уроки проходят легко для восприятия на слух без '
                                                 'лишних напоминаний о правилах грамматики!'))
        # Nothing
        self.assertFalse(self.a.has_name_in_start('Мне очень нравится заниматься с Еленой Игоревной. Она хорошо '
                                                  'преподает английский язык, доступно и понятно объясняет материал по '
                                                  'предмету! Я уже значительно лучше стала знать грамматику языка '
                                                  'благодаря занятиям со специалистом-носителем английского. Елена - '
                                                  'доброжелательный человек, который всегда идет навстречу моим '
                                                  'пожеланиям в плане графика проведения занятий или их '
                                                  'продолжительности (для меня это имеет значение).'))

    def test_delete_name_in_start(self):
        # Russian name
        self.assertEqual(self.a.delete_name_in_start(
            'Елена Владимировна - подошла полностью как со стороны профессиональных качеств так личностных '
            'характеристик преподавателя английского языка Елены Владимировны Багаевой к ребенку 5 класса начальной '
            'школы города Москвы.'),
            'Подошла полностью как со стороны профессиональных качеств так личностных характеристик преподавателя '
            'английского языка Елены Владимировны Багаевой к ребенку 5 класса начальной школы города Москвы.')

        # English name
        self.assertEqual(self.a.delete_name_in_start(
            'John Alexander - отличный преподаватель английского языка. С ним '
            'интересно заниматься, он дает много информации по предмету грамотно '
            'строит уроки в зависимости от уровня ученика (грамматика или '
            'аудирование). Уроки проходят легко для восприятия на слух без '
            'лишних напоминаний о правилах грамматики!'),
            'Отличный преподаватель английского языка. С ним интересно заниматься, он дает много информации по '
            'предмету грамотно строит уроки в зависимости от уровня ученика (грамматика или аудирование). Уроки '
            'проходят легко для восприятия на слух без лишних напоминаний о правилах грамматики!')

    def test_clean_review(self):
        # Russian review
        self.assertEqual(self.a.clean_review(
            'Елена Владимировна - подошла полностью как со стороны профессиональных качеств так личностных '
            'характеристик преподавателя английского языка Елены Владимировны Багаевой к ребенку 5 класса '
            'начальной школы города Москвы.'),
            'елена владимировна подошла полностью стороны профессиональных качеств личностных характеристик '
            'преподавателя английского языка елены владимировны багаевой ребенку класса начальной школы города москвы')

        # With english review
        self.assertEqual(self.a.clean_review(
            'Преподаватель мне очень нравится. Она быстро нашла подход к ребенку, она внимательная и ответственная во '
            'всем! У нее есть раздаточный материал по английскому языку - это карточки для маленьких детей с надписями '
            'на английском языке или специальные рисунки-картинки в виде животных (наподобие гусей), различные '
            'наклейщики от Elementary до Intermediate проходят регулярно уроки английского языка у Марии '
            'Александровны. Впечатление о ее работе только положительное могу сказать потому что дочка сейчас начала '
            'получать оценки "4", а ранее была тройка!'),
            'преподаватель нравится быстро нашла подход ребенку внимательная ответственная раздаточный материал '
            'английскому языку карточки маленьких детей надписями английском языке специальные рисунки картинки виде '
            'животных наподобие гусей различные наклейщики проходят регулярно уроки английского языка марии '
            'александровны впечатление работе положительное могу дочка получать оценки ранее тройка')

    def test_lemmatization_review(self):
        # Russian review
        self.assertEqual(self.a.lemmatization_review(
            'елена владимировна подошла полностью стороны профессиональных качеств личностных характеристик '
            'преподавателя английского языка елены владимировны багаевой ребенку класса начальной школы города москвы'),
            'елена владимирович подойти полностью сторона профессиональный качество личностный характеристика '
            'преподаватель английский язык елена владимирович багаев ребенок класс начальный школа город москва')

        # With english review
        self.assertEqual(self.a.lemmatization_review(
            'преподаватель нравится быстро нашла подход ребенку внимательная ответственная раздаточный материал '
            'английскому языку карточки маленьких детей надписями английском языке специальные рисунки картинки виде '
            'животных наподобие гусей различные наклейщики проходят регулярно уроки английского языка марии '
            'александровны впечатление работе положительное могу дочка получать оценки ранее тройка'),
            'преподаватель нравиться быстро найти подход ребенок внимательный ответственный раздаточный материал '
            'английский язык карточка маленький ребенок надпись английский язык специальный рисунок картинка вид '
            'животное наподобие гусь различный наклейщик проходить регулярно урок английский язык мария александрович '
            'впечатление работа положительный мочь дочка получать оценка ранее тройка')

    def test_get_duble_pairs(self):
        self.a.duplicates_uniqueness = 0.5
        self.assertEqual(self.a.get_duble_pairs([
            [1, 0.3, 0, 0.5],
            [0.3, 1, 0.6, 0],
            [0, 0.6, 1, 0.7],
            [0.5, 0, 0.7, 1],
        ]),
            [
                [0, 3], [1, 2], [2, 3]
            ])

        self.a.duplicates_uniqueness = 0.3
        self.assertEqual(self.a.get_duble_pairs([
            [1, 0.3, 0, 0.5],
            [0.3, 1, 0.6, 0],
            [0, 0.6, 1, 0.7],
            [0.5, 0, 0.7, 1],
        ]),
            [
                [0, 1], [0, 3], [1, 2], [2, 3]
            ])

    def test_get_duplicat_matrix(self):
        lemmatized_reviews = [
            "абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос",
            "абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос",
            "тыква тыква тыква абрикос абрикос абрикос абрикос абрикос абрикос абрикос",
            "тыква тыква тыква абрикос абрикос абрикос абрикос абрикос арбуз арбуз",
            "арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз",
            "кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек"
        ]

        vectors, csim = self.a.get_duplicat_matrix(lemmatized_reviews)

        self.maxDiff = 1
        self.assertEqual(vectors.tolist(), [
            [10, 0, 0, 0],
            [10, 0, 0, 0],
            [7, 0, 0, 3],
            [5, 2, 0, 3],
            [0, 10, 0, 0],
            [0, 0, 10, 0]])

        csim = csim.round(1)
        self.assertEqual(csim.tolist(), [
            [1.0, 1.0, 0.9, 0.8, 0.0, 0.0],
            [1.0, 1.0, 0.9, 0.8, 0.0, 0.0],
            [0.9, 0.9, 1.0, 0.9, 0.0, 0.0],
            [0.8, 0.8, 0.9, 1.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.3, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        self.assertEqual(self.a.amount_unique_words, 4)
        self.assertEqual(self.a.amount_words, 60)

    def test_add_data(self):
        self.a.clear_data()
        data = [
            ["Ревью 1\n###\nРевью 2\n\n", "1", "AI1", "reviews_0_1"],
            ["Review 2\n###\nРевью 2 #%^🤗\n\t\t\\\n", "2", "AI2", "reviews_0_2"],
        ]
        self.a.add_data(data)
        review_list = ["Ревью 1", "###", "Ревью 2", "", "", "Review 2", "###", "Ревью 2 #%^🤗", "\t\t\\", ""]
        sectionId_list = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
        type_page_list = ["AI1", "AI1", "AI1", "AI1", "AI1", "AI2", "AI2", "AI2", "AI2", "AI2"]
        type_model_list = ["reviews_0_1", "reviews_0_1", "reviews_0_1", "reviews_0_1", "reviews_0_1",
                           "reviews_0_2", "reviews_0_2", "reviews_0_2", "reviews_0_2", "reviews_0_2"]

        self.assertEqual(self.a.data['review'].values.tolist(), review_list)
        self.assertEqual(self.a.data['sectionId'].values.tolist(), sectionId_list)
        self.assertEqual(self.a.data['type_page'].values.tolist(), type_page_list)
        self.assertEqual(self.a.data['type_model'].values.tolist(), type_model_list)

    def test_clear_data(self):
        data = [
            ["Ревью 1\n###\nРевью 2\n\n", "1", "AI1", "reviews_0_1"],
            ["Review 2\n###\nРевью 2 #%^🤗\n\t\t\\\n", "2", "AI2", "reviews_0_2"],
        ]
        self.a.add_data(data)
        self.a.clear_data()
        self.assertEqual(self.a.amount_unique_words, 0)
        self.assertEqual(self.a.amount_words, 0)
        self.assertEqual(len(self.a.data.index), 0)

    def test_mark_name_entity(self):
        self.a.clear_data()
        data = [
            ["абрикос абрикос Алина абрикос абрикос абрикос абрикос абрикос абрикос абрикос", "", "", ""],
            ["абрикос абрикос абрикос абрикос картофель абрикос. Абрикос абрикос абрикос абрикос", "", "", ""],
            ["тыква тыква тыква абрикос абрикос. Печень абрикос абрикос абрикос абрикос", "", "", ""],
            ["тыква тыква тыква абрикос абрикос абрикос абрикос абрикос арбуз арбуз", "", "", ""],
            ["арбуз арбуз арбуз арбуз, Арбуз арбуз арбуз Аркадий арбуз арбуз", "", "", ""],
            ["кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек", "", "", ""],
        ]
        self.a.add_data(data)
        self.a.mark_name_entity()
        self.assertEqual(self.a.data['name_entity'].values.tolist(), [True, False, False, False, True, False])

    def test_mark_spelling(self):
        data = [
            [
                'Maria is a great tutor. She can hear everything and speak very '
                'well with her students, which means that she will always be clear '
                'well with her students, which means that she will always be clear '
                'well with her students, which means that she will always be clear '
                'well with her students, which means that she will always be clear '
                'well with her students, which means that she will always be clear '
                'of your wrongs in English language!', '', '', ''],
            ['В общем рекомендую этого преподавателя всем кто хочет подтянуть свой уровен', '', '', ''],
            [
                'Елена Владимировна - приятный в общении человек.Педагог помогает мне '
                'повысить уровень владения английским языком, использует много различных '
                'учебных и методических материалов на занятиях с ней по подготовке к '
                'ЕГЭ IELTS TOEFL c упором именно как раз-таки конкретно разговорной речи '
                '(чтение). Занятия проходят очень продуктивно! Упор идет больше не только '
                'лишь собственно непосредственно on line-, но также комбинированно '
                'используется аудиоуроками весь материал для занятий подобран '
                'квалифицированным специалистом весьма грамотно согласно моему запросу :)', '', '', ''],
            [
                'Khan Faisal - отличный преподаватель, мне с ним очень комфортно работать! '
                'Он помогает подтянуть уровень владения английским языком и в целом у меня '
                'положительный отзыв о его работе. У репетитора есть своя методика '
                'обучения английскому языку на основе бизнес-лексики по учебникам '
                'McDonalds или Tivari Rupa suite. В основном мы занимаемся разговорной '
                'практикой для расширения кругозора, но также много времени уделяем '
                'грамматике языка (грамматика нам не нужна). Если возникают какие либо '
                'вопросы связанные со здоровьем то я всегда обращаюсь к нему за помощью!', '', '', ''],
            [
                'Мне очень нравится заниматься с Еленой Игоревной. Она хорошо преподает '
                'английский язык, доступно и понятно объясняет материал по предмету! Я уже '
                'значительно лучше стала знать грамматику языка благодаря занятиям со '
                'специалистом-носителем английского. Елена - доброжелательный человек, '
                'который всегда идет навстречу моим пожеланиям в плане графика '
                'проведения занятий или их продолжительности (для меня это имеет '
                'значение).', '', '', '']
        ]
        self.a.clear_data()
        self.a.add_data(data)
        self.a.mark_spelling()
        spelling_list = [True, True, True, False, False]
        self.assertEqual(self.a.data['spelling'].values.tolist(), spelling_list)

    def test_get_duble_class(self):
        self.a.clear_data()
        data = [
            ["абрикос абрикос Алина абрикос абрикос абрикос абрикос абрикос абрикос абрикос", "", "", ""],
            ["абрикос абрикос абрикос абрикос картофель абрикос. Абрикос абрикос абрикос абрикос", "", "", ""],
            ["тыква тыква тыква абрикос абрикос. Печень абрикос абрикос абрикос абрикос", "a", "", ""],
            ["тыква тыква тыква абрикос абрикос абрикос абрикос абрикос арбуз арбуз", "", "", ""],
            ["арбуз арбуз арбуз арбуз, Арбуз арбуз арбуз Аркадий арбуз арбуз", "", "", ""],
            ["кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек", "", "", ""],
        ]
        self.a.add_data(data)
        self.a.data['duble_class'] = 0
        self.a.data.at[3, 'duble_class'] = 5

        reviews_good = self.a.data['review']
        self.assertEqual(self.a.get_duble_class(reviews_good, 0), 0)
        self.assertEqual(self.a.get_duble_class(reviews_good, 3), 5)

    def test_count_uniqueness_words(self):
        vectors = [
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ]
        self.assertEqual(self.a.count_uniqueness_words(vectors, 0), 3)
        self.assertEqual(self.a.count_uniqueness_words(vectors, 1), 6)
        self.assertEqual(self.a.count_uniqueness_words(vectors, 2), 0)

    def test_get_local_index(self):
        global_df = pd.DataFrame({'review': []})
        global_df = global_df.append({'review': 'one'}, ignore_index=True)
        global_df = global_df.append({'review': 'two'}, ignore_index=True)
        global_df = global_df.append({'review': 'two'}, ignore_index=True)

        local_df_indexes = global_df[global_df.review == 'two'].index

        self.assertEqual(self.a.get_local_index(local_df_indexes, 0), -1)
        self.assertEqual(self.a.get_local_index(local_df_indexes, 1), 0)
        self.assertEqual(self.a.get_local_index(local_df_indexes, 2), 1)

    def test_find_max_uniqueness_in_class(self):
        vectors = [
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ]
        originals_list = [0, 0, 0]
        class_indexes_local = [0, 1, 2]
        self.a.find_max_uniqueness_in_class(vectors, class_indexes_local, 1, originals_list)

        self.assertEqual(originals_list, [0, 1, 0])

    def test_check_straight_duble(self):
        pairs = [[1, 2], [2, 3]]
        self.assertEqual(self.a.check_straight_duble(pairs, 1, 2), True)
        self.assertEqual(self.a.check_straight_duble(pairs, 2, 3), True)
        self.assertEqual(self.a.check_straight_duble(pairs, 1, 3), False)
        self.assertEqual(self.a.check_straight_duble(pairs, 0, 4), False)
        self.assertEqual(self.a.check_straight_duble(pairs, 2, 1), True)
        self.assertEqual(self.a.check_straight_duble(pairs, 3, 2), True)

    def test_mark_duplicates_by_pairs(self):
        self.a.clear_data()
        data = [
            ["абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос", "", "", ""],
            ["абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос абрикос", "", "", ""],
            ["тыква тыква тыква абрикос абрикос абрикос абрикос абрикос абрикос абрикос", "", "", ""],
            ["тыква тыква тыква абрикос абрикос абрикос абрикос абрикос арбуз арбуз", "", "", ""],
            ["арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз арбуз", "", "", ""],
            ["кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек кабочек", "", "", ""],
        ]
        self.a.add_data(data)
        self.a.duplicates_uniqueness = 0.9

        reviews_good = self.a.data['review']
        cleaned_reviews = list(map(self.a.clean_review, reviews_good.values))
        lemmatized_reviews = list(map(self.a.lemmatization_review, cleaned_reviews))

        vectors, csim = self.a.get_duplicat_matrix(lemmatized_reviews)
        duplicates_pairs = self.a.get_duble_pairs(csim)
        self.a.mark_duplicates_by_pairs(reviews_good, vectors, duplicates_pairs)

        duble_good = [False, False, False, True, True, True]
        self.assertEqual(self.a.data['duble_good'].values.tolist(), duble_good)
