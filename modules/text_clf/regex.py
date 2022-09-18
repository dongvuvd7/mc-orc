from dateutil.parser import parse, ParserError #dateutil là một thư viện của python dùng để xử lý ngày tháng, parse là một hàm của thư viện dateutil dùng để chuyển đổi một chuỗi thành một đối tượng ngày tháng
from calendar import IllegalMonthError #calender là một thư viện của python dùng để xử lý ngày tháng, IllegalMonthError là một lỗi của thư viện calender dùng để báo lỗi khi tháng không hợp lệ
import re

#Hàm date_finder dùng để tìm ngày tháng trong văn bản
def date_finder(string): #đầu vào là một chuỗi
    regexr = re.search(r'\b((?:\d\d[-/\.:])+\d\d(\d\d)?)[\s-]?((?:\d\d[\.:])+\d\d)?\b', string) #tìm kiếm chuỗi có dạng ngày tháng
#     print(regexr)
    if regexr is None: return None #nếu không tìm thấy thì trả về None
#     try:
#         parse(regexr.group(1) or '' + ' ' + regexr.group(2) or '')
#     except:
#         return None
    return regexr.group(0), regexr.start(0), regexr.end(0) #trả về ngày tháng tìm được, vị trí bắt đầu và vị trí kết thúc, nếu khớp regex ngày tháng thì trả về ngày tháng, regexr.start(0) là vị trí bắt đầu của ngày tháng trong chuỗi, regexr.end(0) là vị trí kết thúc của ngày tháng trong chuỗi, regexr.group(0) là ngày tháng