import time

from Modules.google_sheets_api import GoogleSheetsApi
from Modules.review_analysis import ReviewAnalysis

"""Options"""
TOKEN_FILE = 'Environment/google_token.json'                # File with google service auth token.
TABLE_ID = '18CSD7sNaJWQ4DDOv6omd0J2jSYuT7xjlKCyAxSdz-QQ'   # ID Google Sheets document
REPORT_LIST = 'Data_output'                                 # Name of list for upload report
DATA_LIST = 'Reviews_download'                              # Name of list for download data
GOODS_LIST = 'Работаем (порог 0.8...)'                      # Name of list for download goods
GOODS_FILE = 'goods.tsv'                                    # Path/name.tsv of goods file
BACKUP_NAME = 'backup.tsv'                                  # Backup name
MAKE_REPORT = False                                         # True if need make report
DOWNLOAD_GOODS = False                                      # True if need download goods
MAKE_REPORT_BY_BACKUP = False                               # True if need make report by backup

analysis = ReviewAnalysis()
print(analysis)
if MAKE_REPORT_BY_BACKUP:
    MAKE_REPORT = False
    sheets = GoogleSheetsApi(TOKEN_FILE)
    print('Load backup')
    analysis.load_backup(BACKUP_NAME)
    print('\tData')
    analysis.report_to_sheet_output(sheets, TABLE_ID, REPORT_LIST)
    print('Done')

if DOWNLOAD_GOODS:
    sheets = GoogleSheetsApi(TOKEN_FILE)
    analysis.download_goods(sheets, TABLE_ID, GOODS_LIST, GOODS_FILE)

if MAKE_REPORT:
    sheets = GoogleSheetsApi(TOKEN_FILE)
    raw_data = sheets.get_data_from_sheets(TABLE_ID, DATA_LIST, 'A2',
                                           'D' + str(sheets.get_list_size(TABLE_ID, DATA_LIST)[1]), 'ROWS')
    sheets = GoogleSheetsApi(TOKEN_FILE)
    print('Calculating: ')
    print('\tAdd data')
    analysis.add_data(raw_data)
    print('\tCorrect data')
    analysis.correct_data()
    print('\tMark spelling')
    analysis.mark_spelling()
    print('\tDuplicates')
    analysis.mark_duplicates()
    print('\tDuplicates file')
    analysis.mark_file_duplicates(GOODS_FILE)
    print('\tMark name entity')
    analysis.mark_name_entity()
    print('Save backup')
    analysis.save_backup(BACKUP_NAME)
    print('Sending report')
    # Clear data
    sheets.clear_sheet(TABLE_ID, REPORT_LIST)
    print('\tStatistic')
    analysis.report_to_sheet_output_compare(sheets, TABLE_ID, REPORT_LIST)
    print('\tData')
    # time.sleep(sheets.request_sleep)
    sheets = GoogleSheetsApi(TOKEN_FILE)
    analysis.report_to_sheet_output(sheets, TABLE_ID, REPORT_LIST)
    print('Done')

