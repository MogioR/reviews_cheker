import string

from google_sheets_api import GoogleSheetsApi
from review_analysis import ReviewAnalysis

"""File with google service auth token."""
TOKEN_FILE = 'token.json'
"""ID Google Sheets document"""
TABLE_ID = '1pucp3M5mV0bQHNI8KjdLcwzU8ry3YyAYRXc2xQp7RQs'
TABLE_TEST_ID = '18CSD7sNaJWQ4DDOv6omd0J2jSYuT7xjlKCyAxSdz-QQ'
"""Max num of reviews"""
MAX_REVIEWS_COUNT = 10000


sheets = GoogleSheetsApi(TOKEN_FILE)
raw_data = sheets.get_data_from_sheets(TABLE_ID, 'Reviews_download', 'A2', 'C'+str(MAX_REVIEWS_COUNT))
analysis = ReviewAnalysis()
print('Calculating: ')
print('\tAdd data')
analysis.add_data(raw_data)
# # Calculate data
#analysis.mark_spelling()
print('\tDuplicates')
analysis.mark_duplicates()
print('\tName enity')
analysis.mark_name_entity()
#analysis.mark_deep_spelling()
print('Sending report')
print('\tFirst')
analysis.report_to_sheet_output(sheets, TABLE_TEST_ID, 'Data_output')
print('\tSecond')
analysis.report_to_sheet_output_compare(sheets, TABLE_TEST_ID, 'Data_output')
print('Done')
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords
# from stop_words import get_stop_words
# from sklearn.feature_extraction.text import TfidfVectorizer

# texts = [
#     "Jeffrey Rodzianko - замечательный преподаватель! Он очень быстро нашел общий язык с ребенком, сумел заинтересовать занятиями. "
#     "Материал репетитор объяснял в простой и доступной форме благодаря чему у ребенка хорошо усвоились грамматика английского языка за "
#     "несколько месяцев обучения. Мы остались довольны работой преподавателя!",
#     "Khan Faisal - очень хороший преподаватель! Он быстро нашел контакт с учеником, сумел увлечь его занятиями. Материал объяснял в "
#     "простой и понятной форме на занятиях всегда присутствовал доброжелательный настрой преподавателя к ученику. Благодаря педагогу сын "
#     "хорошо подтянулся по английскому языку! Сейчас у нас пауза но вскоре мы планируем возобновить уроки английского языка!!!",
#     "Ирина Викторовна преподает мне английский язык. Она очень пунктуальная, никогда не опаздывает и всегда идет навстречу по "
#     "моим просьбам в плане проведения уроков (всякие интересные занятия). Если у меня возникают какие-то вопросы - мы их прорабатываем "
#     "на опережение школьной программы или отрабатывают тему из предыдущего урока для того чтобы ребенок лучше ее усвоил), я всем довольна!"
# ]

# STOP_WORDS2 = get_stop_words('russian')
# print(STOP_WORDS2)
# cleaned_reviews = list(map(analysis.clean_review, texts))
# lemmatized_reviews = list(map(analysis.lemmatization_review, cleaned_reviews))
# print(lemmatized_reviews[0])
# print(lemmatized_reviews[1])
# print(lemmatized_reviews[2])
#
# vect = TfidfVectorizer(min_df=1)
# tfidf = vect.fit_transform(lemmatized_reviews)
# pairwise_similarity = tfidf * tfidf.T
# csim = pairwise_similarity.toarray()
#
# # vectorizer = CountVectorizer().fit_transform(lemmatized_reviews)
# # vectors = vectorizer.toarray()
# # print(vectors)
# # csim = cosine_similarity(vectors)
#
# print(csim)