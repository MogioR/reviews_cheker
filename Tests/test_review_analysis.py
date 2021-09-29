import unittest
from review_analysis import ReviewAnalysis

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

        # Russian name
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