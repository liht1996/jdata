# JDATA - China Big Data Algorithm Competition - Prediction of Users' Purchase Dates 

# 中国大数据算法大赛 - 如期而至：用户购买时间预测

## Basic Info:

### Competition Details: https://jdata.jd.com/html/detail.html?id=2

### Team: STAR_BIGDATA

### Rank: 

Stage A: 33/739
      
Stage B: 17/137

### Language：
Python(3.6.3)

### Libraries:

Numpy(1.13.3), Pandas(0.22.0), Scikit-learn(0.19.1), LightGBM(2.1.1), XGBoost(0.71)

Please notice that different versions of libraries may lead to different results

### Algorithm: 
GBDT

### Main Contributor: 

@Francis1986(JData ID: 长离未离) https://github.com/Francis1986
                  
@liht1996(JData ID: Euphoric0x0) https://github.com/liht1996

## How to run: 

1. Create New folder: data, fea_imp_sub, fea_imp_train, feature, submit

2. Put original data into data folder

3. Open main_final.py:

      Change fea_exist to 0
      
      Offline test: tran = 1
      
      Generate online test file: sub = 1

4. Run main_final.py

5. grid.py is used for grid search
