from WindPy import w
import numpy as np
import pandas as pd
############配置开始##############
# 股票型基金列表
# stock_fund_test.txt 调试文件
# stock_fund_all.txt 真实文件
stock_fund_file_name = "./res/stock_fund_all.txt"

# 因子
factors = "risk_stdevyearly,risk_maxdownside,risk_sharpe,risk_treynor,return_1m,return_1y,return_3y,periodreturnranking_1m,periodreturnranking_1y,periodreturnranking_3y"

# 日期
startDate = '20180430'
endDate = '20210430'
currentDate = '20210505'
params = 'startDate=' + startDate + ';endDate=' + endDate + \
    ";period=2;returnType=1;riskFreeRate=1;index=000001.SH;annualized=0;fundType=1"

# 单次跑多少条数据
splitCount = 2000

############配置结束##############

# 读取股票型基金列表
f = open(stock_fund_file_name, "r")  # 设置文件对象
string = f.read()  # 将txt文件的所有内容读入到字符串str中
f.close()  # 将文件关闭

# 启动
w.start()


def saveData(stock_fund, file_index, need_y = True):

    # 获取x
    # 获取过去36个月的因子数据
    error_code, wsd_data = w.wss(stock_fund, factors, params, usedf=True)

    if error_code == 0:
        # 排名百分位字符串转成数字
        for key in ['PERIODRETURNRANKING_1M', 'PERIODRETURNRANKING_1Y', "PERIODRETURNRANKING_3Y"]:
            for (index, value) in enumerate(wsd_data[key]):
                if value and isinstance(value, str):
                    temp = value.split('/')
                    s = ''
                    if len(temp) == 2 and int(temp[1]) > 1:
                        s = int(temp[0]) / int(temp[1])
                    else:
                        continue
                    wsd_data[key][index] = s
        wsd_data.to_csv("./data/x_"+str(file_index)+".csv")
    else:
        print("Error Code:", error_code)

    if need_y :
        # 获取y 即近一个月回报
        error_code, wsd_data = w.wsd(stock_fund, "return_1m", endDate, currentDate,
                                        "annualized=0;Period=Y;Fill=Previous;PriceAdj=F", usedf=True)
        if error_code == 0:
            wsd_data.to_csv("./data/y_"+str(file_index)+".csv")
        else:
            print("Error Code:", error_code)


fundIds = string.split(',')
count = (len(fundIds) // splitCount) + 1
for file_index in range(count):
    ids = fundIds[file_index*splitCount : (file_index+1)*splitCount]
    if ids:
        # saveData(','.join(ids), file_index)
        saveData(','.join(ids), file_index, False)
