# pattern-recognition
We are trying to build a model that uses pattern recognition algorithm to predict stock prices.  
尝试用模式识别算法找出股票数据中的规律

```
"jionquant.py":
  样本空间：上证所有股票
  label1：涨幅较大
  label0：涨幅较小以及在跌的股票
  特征选择：固定选了几个特征
  分类算法：二分类，随机森林算法
  在输出空间中随机抽20支股票买入，测试算法准确度
written by 黄枭(Huang Xiao)
```
```
"getStocksDtat.py":
  从Tushare上下载股票交易数据
  可以获取基本面数据以及某一日的交易数据
  可以对数据清洗，排序
  written by 韩沐芾(Han Mufu)
```
```
"featureSelection_RF.py":
  根据相关性选择特征
  用随机森林算法训练数据
  written by 姚育堃(Yao Yukun)
```
这个算法目前做出来效果很差  
我认为股票交易算法更加看重TP/(TP+FP)  
因为买股票会从Positive的pool中挑选  
如果TP占Positive很少，就没有意义了  
之于错分的FN，可以不用在意  
```
后续会加入：
  更好的特征选择算法
  更好的分类算法，以及尝试使用回归算法
  更精细的样本空间，对不同板块定制不同算法
  选择股票时考虑相关性算法
  以及对输出空间的数据更好的使用，对买入卖出时间的控制
  有效的止损算法
```
