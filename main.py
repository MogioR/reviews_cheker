import string

from google_sheets_api import GoogleSheetsApi
from review_analysis import ReviewAnalysis

"""File with google service auth token."""
TOKEN_FILE = 'token.json'
"""ID Google Sheets document"""

"""Max num of reviews"""
MAX_REVIEWS_COUNT = 10000

"""Options"""
TABLE_ID = '1pucp3M5mV0bQHNI8KjdLcwzU8ry3YyAYRXc2xQp7RQs'   # Table id (in url)
REPORT_LIST = 'Data_output'                                 # Name of list for upload report
DATA_LIST = 'Reviews_download'                              # Name of list for download data
GOODS_LIST = 'Работаем (порог 0.8...)'                      # Name of list for download goods
GOODS_FILE = 'goods.csv'                                    # Path/name.csv of goods file
MAKE_REPORT = False                                         # True if need make report
DOWNLOAD_GOODS = True                                       # True if need download goods


sheets = GoogleSheetsApi(TOKEN_FILE)
analysis = ReviewAnalysis()

if DOWNLOAD_GOODS:
    analysis.download_goods(sheets, TABLE_ID, GOODS_LIST, GOODS_FILE)

if MAKE_REPORT:
    raw_data = sheets.get_data_from_sheets(TABLE_ID, DATA_LIST, 'A2',
                                           'D' + str(sheets.get_list_size(TABLE_ID, DATA_LIST)[1]), 'ROWS')
    print('Calculating: ')
    print('\tAdd data')
    analysis.add_data(raw_data)
    print('\tMark spelling')
    # Calculate data
    analysis.mark_spelling()
    print('\tDuplicates')
    analysis.mark_duplicates()
    print('\tDuplicates file')
    analysis.mark_file_duplicates('goods.csv')
    print('\tDel names')
    analysis.delete_names_in_start()
    print('\tMark name entity')
    analysis.mark_name_entity()
    print('Sending report')
    print('\tFirst')
    analysis.report_to_sheet_output(sheets, TABLE_ID, REPORT_LIST)
    print('\tSecond')
    analysis.report_to_sheet_output_compare(sheets, TABLE_ID, REPORT_LIST)
    print('Done')
