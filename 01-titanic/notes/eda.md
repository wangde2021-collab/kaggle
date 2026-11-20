# EDA 结论（2026-11-19）

train 891 行，test 418 行；目标生还率 0.3838（342/891）。

## 缺失率
             train缺失  test缺失
Age           0.1987  0.2057
Cabin         0.7710  0.7823
Embarked      0.0022  0.0000
Fare          0.0000  0.0024
Name          0.0000  0.0000
Parch         0.0000  0.0000
PassengerId   0.0000  0.0000
Pclass        0.0000  0.0000
Sex           0.0000  0.0000
SibSp         0.0000  0.0000
Survived      0.0000     NaN
Ticket        0.0000  0.0000


## 分组生还率（train）

### Sex
          mean  count
Sex                  
female  0.7420    314
male    0.1889    577

### Pclass
          mean  count
Pclass               
1       0.6296    216
2       0.4728    184
3       0.2424    491

### Embarked
            mean  count
Embarked               
C         0.5536    168
Q         0.3896     77
S         0.3370    644

### SibSp
         mean  count
SibSp               
0      0.3454    608
1      0.5359    209
2      0.4643     28
3      0.2500     16
4      0.1667     18
5      0.0000      5
8      0.0000      7

### Parch
         mean  count
Parch               
0      0.3437    678
1      0.5508    118
2      0.5000     80
3      0.6000      5
4      0.0000      4
5      0.2000      5
6      0.0000      1

### Title
                mean  count
Title                      
Master        0.5750     40
Miss          0.7027    185
Mr            0.1567    517
Mrs           0.7937    126
Rare          0.3182     22
the Countess  1.0000      1

### FamilyBucket
                mean  count
FamilyBucket               
1             0.3035    537
2-4           0.5788    292
5+            0.1613     62

### Deck
        mean  count
Deck               
A     0.4667     15
B     0.7447     47
C     0.5932     59
D     0.7576     33
E     0.7500     32
F     0.6154     13
G     0.5000      4
U     0.2994    688

### TicketPrefix
                mean  count
TicketPrefix               
A             0.0714     28
C             0.4000      5
CA            0.3415     41
FCC           0.8000      5
NUM           0.3843    661
OTHER         0.3200     25
PC            0.6500     60
SCAH          0.6667      3
SCPARIS       0.4545     11
SOC           0.1667      6
SOPP          0.0000      3
SOTONOQ       0.1333     15
STONO         0.4444     18
WC            0.1000     10


## Age / Fare 分箱生还率

Age:
             mean  count
AgeBin                  
(0, 10]    0.5938     64
(10, 18]   0.4267     75
(18, 30]   0.3556    270
(30, 45]   0.4257    202
(45, 60]   0.4074     81
(60, 100]  0.2273     22

Fare:
                  mean  count
FareBin                      
(-1.0, 7.9]     0.1973    223
(7.9, 14.5]     0.2991    234
(14.5, 31.0]    0.4670    212
(31.0, 100.0]   0.5325    169
(100.0, 600.0]  0.7358     53


## Ticket 重复次数分布（train+test 合并）

TicketFreq
1     481
2     181
3     101
4      44
5      21
6      19
7      24
8      13
11      7


## train vs test 分布对比

            train均值  test均值  train女比例
Age           29.70   30.27       NaN
Fare          32.20   35.56       NaN
SibSp          0.52    0.45       NaN
Parch          0.38    0.39       NaN
FamilySize     1.90    1.84       NaN


train 女性比例 0.3524，test 女性比例 0.3636

train Pclass 分布 {3: 0.551, 1: 0.242, 2: 0.207}

test Pclass 分布 {3: 0.522, 1: 0.256, 2: 0.222}


## 对抗验证（第 2 阶段）

LightGBM 5 折 AUC 均值 0.5017 ≈ 0.5，train/test 分布一致，CV 可信。最重要特征为 Fare/Age（均为数值本身，非分布偏移信号）。


## 分组泄漏检查（第 2 阶段）

分组 CV（Ticket，LR，F0）均值 0.7957，与普通 25 折 LR 0.7926 接近，F0 层面无明显组内泄漏。G8 仍需单独做分组 CV 决策。