import time

from Modules.google_sheets_api import GoogleSheetsApi
from Modules.review_analysis import ReviewAnalysis

"""Options"""
TOKEN_FILE = 'Environment/google_token.json'  # File with google service auth token.
TABLE_ID = '1pucp3M5mV0bQHNI8KjdLcwzU8ry3YyAYRXc2xQp7RQs'   # ID Google Sheets document
REPORT_LIST = 'Data_output'                                 # Name of list for upload report
DATA_LIST = 'Reviews_download'                              # Name of list for download data
GOODS_LIST = 'Работаем (порог 0.8...)'                      # Name of list for download goods
GOODS_FILE = 'goods.csv'                                    # Path/name.csv of goods file
MAKE_REPORT = True                                          # True if need make report
DOWNLOAD_GOODS = False                                      # True if need download goods


sheets = GoogleSheetsApi(TOKEN_FILE)
analysis = ReviewAnalysis()
time.sleep(360)

if DOWNLOAD_GOODS:
    analysis.download_goods(sheets, TABLE_ID, GOODS_LIST, GOODS_FILE)

if MAKE_REPORT:
    raw_data = sheets.get_data_from_sheets(TABLE_ID, DATA_LIST, 'A2',
                                           'D' + str(sheets.get_list_size(TABLE_ID, DATA_LIST)[1]), 'ROWS')
    sheets = GoogleSheetsApi(TOKEN_FILE)
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
    if sheets.request_count > 50:
        time.sleep(sheets.request_sleep)
    sheets = GoogleSheetsApi(TOKEN_FILE)
    print('\tSecond')
    analysis.report_to_sheet_output_compare(sheets, TABLE_ID, REPORT_LIST)
    print('Done')

