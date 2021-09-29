import httplib2
import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials

class GoogleSheetsApi:
    def __init__(self, token):
        self.auth_service = self.get_table_service(token)

    """Authorisation function"""
    """Input: Google auth token"""
    """Output: Google api service"""
    def get_table_service(self, token):
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            token,
            ['https://www.googleapis.com/auth/spreadsheets'])
        http_auth = credentials.authorize(httplib2.Http())
        return apiclient.discovery.build('sheets', 'v4', http=http_auth)

    """Function for get  data from range(start_range_point, end_range_point) from Google docs sheets"""
    """Input: table_id(str), list_name(str), start_range_point(str), end_range_point(str), majorDimension(str)"""
    """Output: [[[str]]]"""
    def get_data_from_sheets(self, table_id, list_name, start_range_point, end_range_point, majorDimension):
        values = self.auth_service.spreadsheets().values().get(
            spreadsheetId=table_id,
            range="'{0}'!{1}:{2}".format(list_name, start_range_point, end_range_point),
            majorDimension=majorDimension
        ).execute()

        return values['values']

    """Function put data([[]]) in the google sheets table_id(str) in 
                list_name(str)!start_range_point(str):end_range_point(str)"""
    """Input: table_id(str), list_name(str), start_range_point(str), end_range_point(str)), 
              majorDimension ROWS/COLUMNS, data([[]])"""
    """Output: """
    """influence: writes in table table_id"""
    def put_data_to_sheets(self, table_id, list_name, start_range_point, end_range_point,
                           majorDimension, data):
        values = self.auth_service.spreadsheets().values().batchUpdate(
            spreadsheetId=table_id,
            body={
                "valueInputOption": "USER_ENTERED",
                "data": [{
                    "range": ("{0}!{1}:{2}".format(list_name, start_range_point, end_range_point)),
                    "majorDimension": majorDimension,
                    "values": data
                }]
            }).execute()

    """Function put data([]) in the google sheets table_id(str) in 
                list_name(str)!column(str)start_row(int):column(str)end_row(int)"""
    """Input: table_id(str), list_name(str), start_range_point(str), end_range_point(str)), 
               data([])"""
    """Output: """
    """influence: writes in table table_id"""
    def put_column_to_sheets(self, table_id, list_name, column, start_row, end_row, data):
        values = [[i] for i in data]
        self.put_data_to_sheets(table_id, list_name, column + str(start_row),
                           column + str(end_row), 'ROWS', values)

    """Function put data([]) in the google sheets table_id(str) in 
                list_name(str)!start_column(str)row(int):end_column(str)row(int)"""
    """Input: table_id(str), list_name(str), start_range_point(str), end_range_point(str)), 
               data([])"""
    """Output: """
    """influence: writes in table table_id"""
    def put_row_to_sheets(self, table_id, list_name, row, start_column, end_column, data):
        values = [[i] for i in data]
        self.put_data_to_sheets(table_id, list_name, start_column + str(row),
                           end_column + str(row), 'COLUMNS', values)

    """Function get sheet_id of list_name(str) in table table_id(str)"""
    """Input: table_id(str), list_name(str)"""
    """Output: sheet_id(str)"""
    def get_sheet_id(self, table_id, list_name):
        spreadsheet = self.auth_service.spreadsheets().get(spreadsheetId=table_id).execute()
        sheet_id = None
        for _sheet in spreadsheet['sheets']:
            if _sheet['properties']['title'] == list_name:
                sheet_id = _sheet['properties']['sheetId']
        return sheet_id

    """Function generate colorizing range in table request"""
    """Input: Google auth service, table_id(str), list_name(str) tart_column(int), start_row(int), end_column(int), 
              end_row(int)"""
    """Output: request(dict)"""
    """influence: change color in range"""
    def gen_colorizing_range_in_sheets_request(self, table_id, list_name, start_column, start_row, end_column,
                                               end_row, color):
        return {
            "repeatCell": {
                "range": {
                    "sheetId": self.get_sheet_id(table_id, list_name),
                    "startRowIndex": start_row - 1,
                    "endRowIndex": end_row,
                    "startColumnIndex": start_column - 1,
                    "endColumnIndex": end_column
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {
                            "red": color[0],
                            "green": color[1],
                            "blue": color[2]
                        }
                    }
                },
                "fields": "userEnteredFormat.backgroundColor"
            }
        }

    """Function apply colorizing requests"""
    """Input: Google auth service, table_id(str), requests([dict])"""
    def apply_colorizing_range_in_sheets_requests(self, table_id, requests):
        self.auth_service.spreadsheets().batchUpdate(
            spreadsheetId=table_id,
            body={"requests": [requests]}).execute()

    """Function clear list of sheet"""
    """Input: uth_service(str), table_id(str), list_name(str)"""
    """Output: """
    """influence: clear list"""
    def clear_sheet(self, table_id, list_name):
        rangeAll = '{0}!A1:Z'.format(list_name)
        body = {}
        self.auth_service.spreadsheets().values().clear(spreadsheetId=table_id, range=rangeAll, body=body).execute()

    """Function return sizes of list"""
    """Input: table_id(str), list_name(str)"""
    """Output: [columns_count, rows_count]"""
    def get_list_size(self, table_id, list_name):
        request = self.auth_service.spreadsheets().get(spreadsheetId=table_id, ranges=list_name).execute()
        return [request['sheets'][0]['properties']['gridProperties']['columnCount'],
                request['sheets'][0]['properties']['gridProperties']['rowCount']]
