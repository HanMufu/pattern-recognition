import datetime
import time

import pandas as pd
from numpy import linalg as la

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import numpy as np
import random


# 当前最优参数
#  f = 3 #调仓频率
#     g.monthtake = 10 #每月第几天开始测试
#     g.delatday =120 #用几天前的因子进行模型训练（也等于用几天前到当天的个股涨幅做结果训练模型）
#     g.window = 70  #涨幅阈值
# 399006.XSHE创业板/000300.XSHG沪深/000001.XSHG上证
# ---------------------------------------------
# f = 3 #调仓频率
# g.monthtake = 15 #每月第几天开始测试
# g.delatday =30 #用几天前的因子进行模型训练（也等于用几天前到当天的个股涨幅做结果训练模型）
# g.window = 80  #涨幅阈值
# g.zs = 0.07#止损函数判断的大盘跌幅阈值
# g.n =2 #n日均线判断，或者用n天的大盘跌幅做阈值
# g.kernel = 2#第几种止损方案
def initialize(context):
    # 设定指数
    f = 3  # 调仓频率
    g.monthtake = 10  # 每月第几天开始测试
    g.delatday = 120  # 用几天前的因子进行模型训练（也等于用几天前到当天的个股涨幅做结果训练模型）
    g.window = 70  # 涨幅阈值
    g.zs = 0.1  # 止损函数判断的大盘跌幅阈值
    g.n = 3  # n日均线判断，或者用n天的大盘跌幅做阈值
    g.kernel = 2  # 第几种止损方案
    # 股票池

    g.stockindex = '000001.XSHG'
    # 设定沪深300作为基准
    set_benchmark('000001.XSHG')
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True)
    # 设定成交量比例
    set_option('order_volume_ratio', 1)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, \
                             open_commission=0.0003, close_commission=0.0003, \
                             close_today_commission=0, min_commission=5), type='stock')
    # 最大持仓数量
    g.stocknum = 20

    ## 自动设定调仓月份（如需使用自动，注销下段）
    # f = 4  # 调仓频率
    log.info(range(1, 13, 12 / f))
    g.Transfer_date = range(1, 13, 12 / f)

    ## 手动设定调仓月份（如需使用手动，注释掉上段）
    # g.Transfer_date = (3,9)

    # 根据大盘止损，如不想加入大盘止损，注释下句即可
    run_daily(dapan_stoploss, time='open')

    ## 按月调用程序
    run_monthly(trade, monthday=g.monthtake, time='open')


## 交易函数
def trade(context):
    # 获取当前月份

    months = context.current_dt.month

    # 如果当前月为交易月
    if months in g.Transfer_date:
        ## 获得Buylist

        Buylist = check_stocks(context)
        random.shuffle(Buylist)  # 打乱购买列表的顺序，随机化
        Buylist = Buylist[:g.stocknum + 10]  # 选出靠前的序列

        ## 卖出
        if len(context.portfolio.positions) > 0:
            for stock in context.portfolio.positions.keys():
                if stock not in Buylist:
                    order_target(stock, 0)

        ## 分配资金
        if len(context.portfolio.positions) < g.stocknum:
            Num = g.stocknum - len(context.portfolio.positions)
            Cash = context.portfolio.cash / Num
        else:
            Cash = 0

        ## 买入
        if len(Buylist) > 0:
            for stock in Buylist:
                if stock not in context.portfolio.positions.keys():
                    order_value(stock, Cash)
    else:
        return


## 选股函数
def check_stocks(context):
    # 获取当前时间
    end1 = context.current_dt.strftime("%Y-%m-%d")
    # 格式化时间
    end1 = datetime.datetime.strptime(end1, "%Y-%m-%d")
    # 减去delatday得到训练集开始时间
    delat1 = datetime.timedelta(days=g.delatday)
    delat3 = datetime.timedelta(days=1)
    start1 = end1 - delat1
    # end2 = end1 - delat3
    # end1 =end2
    predicttion, test = predict(start1, end1)
    test['predicttion'] = predicttion
    Codes = test[test['predicttion'] == 1].index
    return list(Codes)


## 根据局大盘止损，具体用法详见dp_stoploss函数说明
def dapan_stoploss(context):
    stoploss = dp_stoploss(kernel=g.kernel, n=g.n, zs=g.zs)
    if stoploss:
        if len(context.portfolio.positions) > 0:
            for stock in list(context.portfolio.positions.keys()):
                order_target(stock, 0)


def predict(startime, endtime):
    train = dataset(startime, endtime)
    train = dealdata(train)

    delat2 = datetime.timedelta(days=2)
    # print('2',endtime-delat2,endtime)
    test = dataset(endtime - delat2, endtime)
    # print (endtime-delat2,endtime)
    test = dealdata(test)
    col = [c for c in train.columns if c not in ['Return', 'Unnamed: 0', 'Y_bin']]

    rfc = RandomForestClassifier()
    rfc.fit(train[col], train['Y_bin'])
    t1 = rfc.predict(test[col])  # test[col]为交易日前两天的股票基本面信息，t1为预测的结果涨幅能在前百分之几的股票列表，顺序没变
    return t1, test


def dealdata(data):
    data = data.drop(
        ['capital_reserve_fund', 'operating_revenue', 'gross_profit_margin', 'inc_total_revenue_year_on_year'], axis=1)
    col = [c for c in data.columns if c not in ['Return', 'Unnamed: 0', 'Y_bin']]

    for i in col:
        data[i].fillna(data[i].mean(), inplace=True)
    data['Y_bin'].fillna(-1, inplace=True)
    data['Return'].fillna(-1, inplace=True)
    return data


def dataset(startdate, enddate):
    fdate = startdate
    stock_set = get_index_stocks(g.stockindex, fdate)
    q = query(
        valuation.code,
        valuation.circulating_market_cap,
        valuation.pe_ratio,
        valuation.pb_ratio,
        valuation.pcf_ratio,
        balance.capital_reserve_fund,
        income.operating_revenue,
        indicator.inc_total_revenue_year_on_year,
        indicator.roe,
        indicator.gross_profit_margin,
        indicator.operation_profit_to_total_revenue,
        indicator.inc_revenue_year_on_year,
        indicator.inc_revenue_annual
    ).filter(
        valuation.code.in_(stock_set),
    )
    fdf = get_fundamentals(q, date=fdate)
    fdf.index = fdf['code']
    fdf.pop('code');

    current_date = startdate
    forcast_date = enddate
    current_close = get_price(stock_set, fields=['close'], end_date=current_date, count=1)['close'].T
    forcast_close = get_price(stock_set, fields=['close'], end_date=forcast_date, count=1)['close'].T
    # print(current_close[(current_close.index != forcast_close.index)].shape[0])
    grow = (forcast_close.iloc[:, 0] - current_close.iloc[:, 0]) / current_close.iloc[:, 0]
    grow = pd.DataFrame(grow, columns=['Return'])

    df = pd.merge(fdf, grow, left_index=True, right_index=True)
    bound = np.nanpercentile(df['Return'], g.window)
    df.loc[(df['Return'] >= bound), 'Y_bin'] = 1
    df.loc[(df['Return'] < bound), 'Y_bin'] = -1

    return df


## 大盘止损函数w
def dp_stoploss(kernel=2, n=10, zs=0.005):
    '''
    方法1：当大盘N日均线(默认60日)与昨日收盘价构成“死叉”，则发出True信号
    方法2：当大盘N日内跌幅超过zs，则发出True信号
    '''
    # 止损方法1：根据大盘指数N日均线进行止损
    if kernel == 1:
        t = n + 2
        hist = attribute_history('000001.XSHG', t, '1d', 'close', df=False)
        temp1 = sum(hist['close'][1:-1]) / float(n)
        temp2 = sum(hist['close'][0:-2]) / float(n)
        close1 = hist['close'][-1]
        close2 = hist['close'][-2]
        if (close2 > temp2) and (close1 < temp1):
            return True
        else:
            return False
    # 止损方法2：根据大盘指数跌幅进行止损
    elif kernel == 2:
        hist1 = attribute_history('000001.XSHG', n, '1d', 'close', df=False)
        if ((1 - float(hist1['close'][-1] / hist1['close'][0])) >= zs):
            return True
        else:
            return False