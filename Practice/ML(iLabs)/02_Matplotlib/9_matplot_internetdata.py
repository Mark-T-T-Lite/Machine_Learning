import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import matplotlib.dates as mdates

def bytespdate2num(fmt,encoding='utf-8'):
    '''
        Handles the date format and encoding
    '''
    strconverter  =mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter


def graph_stock_data(stock):
    '''
        Acquires stock data from given url and graphs it
    '''

    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'

    source_code = urllib.request.urlopen(stock_price_url).read().decode()

    stock_data = []
    split_source = source_code.split('\n')
    csv_file = open(stock,'w')

    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6:
            if 'values' not in line and 'labels' not in line:
                stock_data.append(line)
                csv_file.write(line)
                print('written in file')
                

    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True,
                                                          # %Y = full year. 2015
                                                          # %y = num year. 15
                                                          # %m = num month
                                                          # %d = num day
                                                          # %H = hours
                                                          # %M = minutes
                                                          # %S = seconds
                                                          # 12-09-2020. %Y-%m-%d
                                                          converters={0: bytespdate2num('%Y%m%d')})

    plt.plot_date(date,closep)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('CSV Values')
    plt.legend()
    plt.show()

        

graph_stock_data('TSLA')     
