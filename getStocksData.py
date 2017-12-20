#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:09:03 2017

@author: hanmufu
"""

import tushare as ts
import pandas as pd
import numpy as np
import datetime


#********************************************************

#清洗数据函数
#实现了把带有数值0的行都删除 因为数值为0说明这支股票缺失了这个数据
#输入:df 数据大表
#输出:df 清洗好的数据大表
def cleanDF(df):
    dfData = df.values
    #下面三行是把含中文的列去除
    #dfData = np.delete(dfData, 1, 1)
    #dfData = np.delete(dfData, 1, 1)
    #dfData = np.delete(dfData, 1, 1)
    index = -1
    for rows in dfData:
        index = index + 1
        for items in rows:
            if items == 0.00:
                #在df.values中删除这一行
                dfData = np.delete(dfData, index, 0)
                #在dataframe中删除这一行
                df.drop(df.index[index], inplace = True)
                #回退一个index
                index = index - 1
                break
    #df.drop('300431', inplace = True)
    #df.drop('300320', inplace = True)
    #df.drop('300590', inplace = True)
    
    return df

#建立dataframe大表函数
#建立一张所有股票的大表，index是股票代码，columns是各交易指标
def createDF_basic():
    #从网上下载基本面数据
    df = ts.get_stock_basics()
    
    #从本机读取基本面数据
    #df = pd.read_csv('/Users/hanmufu/Desktop/大三冬季课程/模式识别/大作业/tk.csv')
    
    return df

#抓取股票交易指标函数
#从网上抓取某一股票的所有交易指标，并添加到dataframe大表后
#抓取的交易指标有：
#   open：开盘价
#   high：最高价
#   close：收盘价
#   low：最低价
#   volume：成交量
#   price_change：价格变动
#   p_change：涨跌幅
#   ma5：5日均价
#   ma10：10日均价
#   ma20:20日均价
#   v_ma5:5日均量
#   v_ma10:10日均量
#   v_ma20:20日均量
#   turnover:换手率
#输入：df DF数据表， stockCode 股票代码， startTime 数据时间, duration 数据时间跨度
def tagLabel_nextDayProfit(df, startTime, duration):
    df['myProfit'] = 0.0
    #把str类型的时间转成Python内置的datetime时间，然后根据给定的duration算出结束时间
    date_time = datetime.datetime.strptime(startTime,'%Y-%m-%d')
    date_time = date_time + datetime.timedelta(days = duration - 1)
    endTime = date_time.strftime('%Y-%m-%d')
    for stockCode in df.index:
        #从Tushare上抓取交易数据
        priceDF = ts.get_hist_data(stockCode, startTime, endTime)
        #收盘价减开盘价得到利润值
        if len(priceDF.index) != 0:
            profit = priceDF['close'][0] - priceDF['open'][0]
            df['myProfit'][stockCode] = profit
    df = df.sort_values(by=('myProfit'), ascending=False)
    df = tagLabel_rank(df, 0.33)
    return df

#打标签函数
#盈利排在前n*D个股票标记为1，D为全部股票个数
#输入:df 数据大表, n 决定label阈值，介于0-1之间
def tagLabel_rank(df, n):
    df['label1'] = False
    #设定前百分之多少的股票打标签
    ratio = len(df.index) * n
    ratio = int(ratio)
    count = 0 #计数器
    for stockCode in df.index:
        df['label1'][stockCode] = True
        count = count + 1
        if count == ratio:
            break
    return df

#加入某一日历史数据
    #open：开盘价
    #high：最高价
    #close：收盘价
    #low：最低价
    #volume：成交量
    #price_change：价格变动
    #p_change：涨跌幅
    #ma5：5日均价
    #ma10：10日均价
    #ma20:20日均价
    #v_ma5:5日均量
    #v_ma10:10日均量
    #v_ma20:20日均量
    #turnover:换手率[注：指数无此项]
def add_histdataDF(df, startTime):
    df['open'] = 0.0
    df['high'] = 0.0
    df['close'] = 0.0
    df['low'] = 0.0
    df['volume'] = 0.0
    df['price_change'] = 0.0
    df['p_change'] = 0.0
    df['ma5'] = 0.0
    df['ma10'] = 0.0
    df['ma20'] = 0.0
    df['v_ma5'] = 0.0
    df['v_ma10'] = 0.0
    df['v_ma20'] = 0.0
    df['turnover'] = 0.0
    for stockCode in df.index:
        priceDF = ts.get_hist_data(stockCode, startTime, startTime)
        if len(priceDF.index) != 0:
            df['open'][stockCode] = priceDF['open']
            df['high'][stockCode] = priceDF['high']
            df['close'][stockCode] = priceDF['close']
            df['low'][stockCode] = priceDF['low']
            df['volume'][stockCode] = priceDF['volume']
            df['price_change'][stockCode] = priceDF['price_change']
            df['p_change'][stockCode] = priceDF['p_change']
            df['ma5'][stockCode] = priceDF['ma5']
            df['ma10'][stockCode] = priceDF['ma10']
            df['ma20'][stockCode] = priceDF['ma20']
            df['v_ma5'][stockCode] = priceDF['v_ma5']
            df['v_ma10'][stockCode] = priceDF['v_ma10']
            df['v_ma20'][stockCode] = priceDF['v_ma20']
            df['turnover'][stockCode] = priceDF['turnover']
    return df



#********************************************************

#主程序
if __name__=="__main__":
    df = createDF_basic()
    df = cleanDF(df)
    df = add_histdataDF(df, '2017-11-29')
    df = cleanDF(df)
    df = tagLabel_nextDayProfit(df, '2017-11-30', 1)
    df.to_csv('/Users/hanmufu/Desktop/大三冬季课程/模式识别/大作业/taggedData.csv', sep=',', header=True, index=True)
    






















