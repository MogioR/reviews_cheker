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
raw_data = sheets.get_data_from_sheets(TABLE_ID, 'Reviews_download', 'A2',
                                       'D'+str(sheets.get_list_size(TABLE_ID, 'Reviews_download')[1]), 'ROWS')
analysis = ReviewAnalysis()
analysis.download_goods(sheets, TABLE_ID, 'Data_output', 'goods.csv')

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
analysis.report_to_sheet_output(sheets, TABLE_TEST_ID, 'Data_output')
print('\tSecond')
analysis.report_to_sheet_output_compare(sheets, TABLE_TEST_ID, 'Data_output')
print('Done')
