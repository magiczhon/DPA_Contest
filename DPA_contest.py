# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:13:42 2018

@author: PC
"""

import os
import math
import scipy.stats
import pandas as pd
import numpy as np
import scipy 
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from sklearn import preprocessing
import scipy.stats as stats
from tqdm import tqdm


import seaborn as sns

f = open('D:\\Trace\\test_and_valid_set\\diplom\\df[0].txt', 'w')

for index in range(3253):
    f.write(str(df[index]) + '\n')

f.close()

#TRAINSET

file_list = os.listdir('D:\Trace\DPA_public_base')
np.random.shuffle(file_list)

keys = []
index_k = file_list[0].find('k=') + 2
index_m = file_list[0].find('_m=') + 3
dfX = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
keys.append('0x' + file_list[0][index_k:index_m-33])
for i in tqdm(range(100)):
    index_k = file_list[i].find('k=') + 2
    index_m = file_list[i].find('_m=') + 3
    keys.append('0x' + file_list[i][index_k:index_m-33])
    df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    dfX = pd.concat([dfX, df_temp], axis=1)
dfX = dfX.transpose()
dfY = pd.DataFrame(keys)

dfX = dfX.transpose() 

dfX = preprocessing.scale(dfX)

dfX = pd.DataFrame(dfX)
df = dfX.iloc[0]



#Проверка распределения выборки
df00 = dfX[0:5000]
def graph(df, i):
    n, bins, patches = plt.hist(df.iloc[i], 100, normed=1)
    mu = np.mean(df.iloc[i])
    sigma = np.std(df.iloc[i])
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 10, 10
    fig = plt.plot(bins, mlab.normpdf(bins, mu, sigma))

graph(df00, 1864)

#график с нужной легендой
lstY = []
for i in range(0, 10000, 50):
    lstY.append(dfY.iloc[i, 0])
    ax = dfX[i].plot(kind='density', figsize=(15, 10)) #строит график плотности(работает с датафреймами)
ax.legend(lstY)
    

for i in range(0, 10000, 500):
    x = dfX[i]
    shapiro_results = scipy.stats.shapiro(x)
    print('len DF = ' , len(x) - 1 , '\nTest Statistic = ' , shapiro_results[0] ,  '\np-value = ' , shapiro_results[1], '\n')

#проверка нормальности критерием Шапира-Уилка
x = dfX[5029]
shapiro_results = scipy.stats.shapiro(x)
print('len DF = ' , len(x) - 1 , '\nTest Statistic = ' , shapiro_results[0] ,  '\np-value = ' , shapiro_results[1])
#проверка нормальности критерием Колмогорова-Смирнова
ks_results = scipy.stats.kstest(x, cdf='norm')
print('len DF = ' , len(x) - 1 , '\nTest Statistic = ' , ks_results[0] ,  '\np-value = ' , ks_results[1]) 
#проверка нормальности критерием Андерсона-Дарлинга
anderson_results = scipy.stats.anderson(x)
print('len DF = ' , len(x) - 1 , '\nTest Statistic = ' , anderson_results[0] ,  '\np-value = ' ,anderson_results[1][2])


x = dfX[0]
n, bins, patches = plt.hist(x, normed=1)
mu = np.mean(x)
sigma = np.std(x)
plt.plot(bins, mlab.normpdf(bins, mu, sigma))





x = dfX[0]
def descriptive1d(x):
    _x = x  # Для возможности предобработки данных (например, исключения нечисловых значений) 
    result = []
    result.append(len(x)) # Чисо элементов выборки
    result.append(np.mean(_x))
    result.append((np.min(_x), np.max(_x)))
    result.append(np.std(_x))
    result.append(result[0]/result[-1])
    result.append((np.percentile(_x, 25), np.percentile(_x, 50), np.percentile(_x, 75)))
    result.append(st.mode(_x)) 
    result.append(st.skew(_x))  # асимметрия 
    result.append(st.kurtosis(_x))  # эксцесс
    _range = np.linspace(0.9 * np.min(_x), 1.1 * np.max(_x), 100)
    result.append(st.gaussian_kde(_x)(_range))  # оценка плотности распределения
    return tuple(result)
# Вычисление важных показателей
n, m, minmax, s, cv, perct, mode, skew, kurt, kde = descriptive1d(x)

print('Число элементов выборки: {0:d}'.format(n))
print('Среднее значение: {0:.4f}'.format(m))
print('Минимальное и максимальное значения: ({0:.4f}, {1:.4f})'.format(*minmax))
print('Стандартное отклонение: {0:.4f}'.format(s))
print('Коэффициент вариации (Пирсона): {0:.4f}'.format(cv))
print('Квартили: (25%) = {0:.4f}, (50%) = {1:.4f}, (75%) = {2:.4f}'.format(*perct))
print('Коэффициент асимметрии: {0:.4f}'.format(skew))
print('Коэффициент эксцесса: {0:.4f}'.format(kurt))


#сравнение средних
#сравнение дисперсий
mu1 = 0
mu2 = 0
std1 = 0
std2 = 0
df1 = []
df2 = []
for i in range(0, 5000, 100):
    std1 = std1 + np.std(dfX[i])
    std2 = std2 + np.std(dfX[5000 + i])
    mu1 = mu1 + np.mean(dfX[i])
    mu2 = mu2 + np.mean(dfX[5000 + i])
    for el in dfX[i]:
        df1.append(el)
    for el in dfX[5000 + i]:
        df2.append(el)
  
s = pd.Series(df1)
s.plot.kde(figsize=(15, 10))
p = pd.Series(df2)
p.plot.kde(figsize=(15, 10))

mu1 = mu1 / 5000
mu2 = mu2 / 5000
print(mu1, mu2)

std1 = std1 / 5000
std2 = std2 / 5000
print(std1, std2)


x = dfX[0]



count = [_ for _ in range(3253)]
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(count, dfX[0], color='red')
plt.plot(count, dfX[1], color='blue')

fig = plt.figure()
new_df = np.abs(dfX[0] - dfX[1])
fig.set_size_inches(40, 30)
plt.plot(count, new_df, color='green')
plt.show()

s = pd.Series(new_df)
s.plot.kde(figsize=(15, 10))

#CHI_SQUARE CRITERIUM
def standarting_value(x):
    mu = np.mean(x)
    sigma = np.std(x)    
    lst = []
    for i in range(len(x)):
        lst.append((x[i] - mu) / sigma)
    return lst

def norm_test(a, free_dementions):
    s, _ = stats.skewtest(a, 0)
    k, _ = stats.kurtosistest(a, 0)
    k2 = s*s + k*k
    print('\nTest Statistic = ', k2, '\np-value = ' ,stats.distributions.chi2.sf(k2, free_dementions))

def norm_statistics(x):
    #проверка нормальности критерием Шапира-Уилка   
    shapiro_results = stats.shapiro(x)
    print('Шапира-Уилка ' , '\nTest Statistic = ' , shapiro_results[0] ,  '\np-value = ' , shapiro_results[1], '\n')         
    #проверка нормальности критерием Колмогорова-Смирнова
    ks_results = stats.kstest(x, cdf='norm')
    print('Колмогорова-Смирнова' ,  '\nTest Statistic = ' , ks_results[0] ,  '\np-value = ' , ks_results[1], '\n')     
    #проверка нормальности критерием Андерсона-Дарлинга
    anderson_results = stats.anderson(x)
    print('Андерсона-Дарлинга' , '\nTest Statistic = ' , anderson_results[0] ,  '\np-value = ' ,anderson_results[1][2], '\n')
    print('normality =', stats.normaltest(x), '\n')
    
    
def hist(x):   
    n, bins, patches = plt.hist(x, normed=0)
    mu = np.mean(x)
    sigma = np.std(x)
    #plt.plot(bins, mlab.normpdf(bins, mu, sigma))
    #plt.hist()
    print("mu =", mu, " sigma = ", sigma)


def graph_pdf(x):
    tmp = pd.DataFrame(x)
    tmp.plot(kind='density', figsize=(15, 10))



def statist(x, loc=0, scale=1):
    return stats.distributions.norm(loc, scale).cdf(x)


def chi2_stat (x, count_int): 
    #lst_int_div = [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]    
    lst_freq = [0] * count_int
    mu = np.mean(x)
    sigma = np.std(x)
    #print('mu=%s, sigma=%s'%(mu,sigma))
    for el in x:
        cdf = statist(el, mu, sigma)
        lst_freq[int(cdf * count_int)]  += 1   
    print(lst_freq)
    #lst_freq = [100, 100, 100, 100, 100, 100, 100, 100, 96, 104]
    chi_sq = 0
    for freq in lst_freq:
        chi_sq += ((freq - len(x)/count_int)**2) / (len(x)/count_int)        
    return chi_sq


dfX = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfX_bit_0x00-0x13.csv')
dfY = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfY_bit_0x00-0x13.csv')

dfX = dfX.transpose() 

dfX = preprocessing.scale(dfX)

dfX = pd.DataFrame(dfX)
df = dfX.iloc[0]

df00 = dfX['0'][0:5000]
x = df00
print("CHI_SQUARE = ", chi2_stat(x,13))
scipy.stats.distributions.chi2.ppf(chi2_stat(x,13))
print(norm_statistics(x))

y = np.random.normal(loc = 19, scale = 20, size = 100)
print("CHI_SQUARE = ", chi2_stat(y))
print(norm_statistics(y))
#-------------------------------------------------------------------------------
#MVN
x = dfX.iloc[0]
from scipy.stats.tests import test_multivariate
test_multivariate.TestMultivariateNormal(x)
test_multivariate.TestMultivariateNormal.subTest(x)



import statsmodels.stats.stattools
statsmodels.stats.stattools.omni_normtest(x)

import statsmodels.multivariate.tests.test_multivariate_ols
statsmodels.multivariate.tests.test_multivariate_ols.test_affine_hypothesis()
    
statsmodels.multivariate.multivariate_ols.MultivariateTestResults(x)


from scipy import stats
pts = 1000
np.random.seed(17)
a = np.random.normal(0, 1, size=pts)
b = np.random.normal(2, 1, size=pts)
x = np.concatenate((a, b))

i = [30,100,1000,5000]
j = [5,7,10,15]

for k in range(4):
    print('_____________________\ni=' + str(i[k]))
    x = dfX.iloc[984][:i[k]]
    
    k2, p = stats.normaltest(x)
    #alpha = 1e-3
    alpha = 0.05
    print("CHI_SQUARE = ", chi2_stat(x,j[k]))
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        print("нулевая гипотеза может быть отклонена")
    else:
        print("нулевая гипотеза не может быть отклонена")

res = (1000,0)
for k in range(3253):
    x = dfX.iloc[k][:1000]
    chi = chi2_stat(x,10)
    if chi < res[0]:
        res = (chi,k)
        print('\n\n')
        print(res)
        print('\n\n')

print("WITH PREPROCESSING SCALE")
x = preprocessing.scale(dfX.iloc[0][:30])
xk2, p = stats.normaltest(x)
#alpha = 1e-3
alpha = 0.05
print("CHI_SQUARE = ", chi2_stat(x,5))
print("p = {:g}".format(p))
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("нулевая гипотеза может быть отклонена")
else:
    print("нулевая гипотеза не может быть отклонена")




def scalar_mult(x, y):
    res = 0
    for i in range(len(x)):
        res = res + x[i] * y[i]
    return res



def multivariable_normal_mardias_test(X, n):
    vec_mean = [0] * n   
    for j in range(n):
        vec_mean = np.mean(X.iloc[j])
        
    SIGMA = 0  
    for j in range(n):    
        SIGMA = SIGMA + scalar_mult((X.iloc[i] - vec_mean), (X.iloc[i] - vec_mean))
    SIGMA = SIGMA / n

    for i in range(n): 
        for j in range(n): 


























# СКАЧИВАНИЕ ПАКЕТОВ R ____________________
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages('MVN')
#__________________________________________

def file_write(path, text): 
    file_res = open(path, 'w+')
    for el in text:
        file_res.write(str(el))
    file_res.close()

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
# Calling libraries in R 
r('library("MVN")')

dfX = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfX_bit_0x00-0x13.csv')
dfY = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfY_bit_0x00-0x13.csv')


    x = dfX.iloc[0:1000]
    #x = preprocessing.scale(dfX.iloc[200:300])
    #Converting the dataframe from python to R
    
    # firt take the values of the dataframe to numpy
    resi1=np.array(x, dtype=float)
    
    # Taking the variable from Python to R
    r_resi = numpy2ri(resi1)
    
    # Creating this variable in R (from python)
    r.assign("resi", r_resi)
    
    # Calling a function in R (from python)
    r('res <- mvn(resi, mvnTest = "royston")')
    
    # Retrieving information from R to Python
    r_result = r("res")
    
    r_res1 = r('res$multivariateNormality')    
    r_res2 = r('res$univariateNormality[5]')
    is_yes(r_res2)
    print(r_res1)
   
    file_write('D:\Trace\ResultMVN.txt', r_result)
    
# Printing the output in python
print(r_result)
    

def is_yes(res_tmp):
    res = np.array(res_tmp)
    y = 0
    for el in res[0]:
        if el == '   YES   ':
            y = y + 1
    print(y/len(res[0]))


def shapiro_test(x):
    n = len(x)
    m = int(n/2)
    a = [0] * m
    a[0] = (0.899/((n-2.4)**0.4162)) - 0.02
    for j in range(1, m):
        z = (n-2*j+1)/(n-0.5)
        a[j] = a[0] * (z + 1483/((3-z)**10.845) + (71.6 * 10**(-10))/((1.1 - z)**8.26))
    
    S=0
    for j in range(0, m):
        S = S + a[j]*(x[n-j-1] - x[j]) 
    
    B = S**2   
    s = 0
    mu = 0
    for j in range(0, n):
        
        mu = mu + x[j]
    mu = mu/n
    for j in range(0, n):
        s = s + (x[j] - mu)**2
    W1 = (1 - 0.6695/(n**0.6518)) * (s) / B
    print(W1)


x = dfX.iloc[:][:1000]
shapiro_test(x)

x = dfX.iloc[10][:1000]
shapiro_test(x)


y = np.random.poisson(size = 100)
shapiro_test(y)

#______________________________________________________________________________
#АНАЛИЗ МАТРИЦЫ КОВАРИАЦИИ
def file_write_pd(path, text): 
    file_res = open(path, 'w+')
    for el in text:
        file_res.write(str(el))x
    file_res.close()


    
dfX_preprocessing = pd.DataFrame(preprocessing.scale(dfX))
df = dfX_preprocessing.iloc[0]
df_cov = df   
for i in range(3252):
   df_cov = pd.concat([df_cov, df], axis=1)
        
df = dfX.iloc[0]
df_cov1 = df   
for i in range(3252):
   df_cov1 = pd.concat([df_cov1, df], axis=1) 
   
   
a = df_cov.cov()
b = df_cov1.cov()

a.to_csv('D:\Trace\covar_matrix.txt', header=None, index=None, sep=' ', mode='w+')

b.to_csv('D:\Trace\covar_matrix.txt', header=None, index=None, sep=' ', mode='w+')


df = dfX.iloc[1]
df_cov2 = df   
for i in range(3):
   df_cov2 = pd.concat([df_cov2, df], axis=1) 
c = pd.DataFrame(df_cov2.cov())

a.to_csv('D:\Trace\covar_matrix.txt', header=None, index=None, sep=' ', mode='w+')

print(c)
       
d = dfX[:10][:10]
dt = np.transpose(d)
d = dt[:10]
dt = np.transpose(d)

mean = [0] * len(dt)
for i in range(len(dt)):
    print(i)
    mean[i] = np.mean(dt.iloc[i])

sigma = []
x = d[0]
for i in range(len(dt)):
    sigma.append([])
    for j in range(len(dt)):
        sigma[i].append((x[i] - mean[i]) * (x[j] - mean[j]))
 
    
    
d = dfX[:35]
def covar_matrix(A,B): 
    res = np.dot(A,B)/35
    return  pd.DataFrame(res)

res = covar_matrix(np.transpose(d), d)
res.to_csv('D:\Trace\covar_matrix.txt', header=None, index=None, sep=' ', mode='w+')
   
print(res)
#______________________________________________________________________________

    
    
    
#______________________________________________________________________________
#НЕЙМАН_ПИРСОН    
    

import os
import math
import scipy.stats
import pandas as pd
import numpy as np
import scipy 
from sklearn import preprocessing
import scipy.stats as stats

file_list = os.listdir('D:\Trace\DPA_public_base')
np.random.shuffle(file_list)

N = 50
keys = []
index_k = file_list[0].find('k=') + 2
index_m = file_list[0].find('_m=') + 3
dfX = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
keys.append('0x' + file_list[0][index_k:index_m-33])
for i in range(N):
    index_k = file_list[i].find('k=') + 2
    index_m = file_list[i].find('_m=') + 3
    keys.append('0x' + file_list[i][index_k:index_m-33])
    df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    dfX = pd.concat([dfX, df_temp], axis=1)
dfX = dfX.transpose()
dfY = pd.DataFrame(keys)


# dfX = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfX_bit_0x00-0x13.csv')
# dfY = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfY_bit_0x00-0x13.csv')




bit0 = pd.DataFrame()
bit1 = pd.DataFrame()
for i in range(N):
    if int(dfY.iloc[i][0], 16) % 2 == 0:
        bit0 =  pd.concat([bit0, dfX.iloc[i]], axis=1)
    else:
        bit1 =  pd.concat([bit1, dfX.iloc[i]], axis=1)
    
bit0 = bit0.transpose()
bit1 = bit1.transpose()
    
mu_mass0 = []
mu_mass1 = []
for trace in range(len(bit0)):
    mu_mass0.append(bit0.iloc[trace].mean())
for trace in range(len(bit1)):
    mu_mass1.append(bit1.iloc[trace].mean())
             

    
mu0 = np.mean(mu_mass0)
mu1 = np.mean(mu_mass1)

print("mu0 = %s\nmu1 = %s \nmu1 > mu0:%s"%(mu0, mu1, mu1 > mu0))




#СЛУЧАЙНО ВИБИРАЕМ СЛЕД ДЛЯ ПРОВЕРКИ КРИТЕРИЯ
file_list_NP = os.listdir('D:\Trace\DPA_public_base')
np.random.shuffle(file_list_NP)

index_k = file_list_NP[0].find('k=') + 2
index_m = file_list_NP[0].find('_m=') + 3
tmp_trace = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list_NP[0], names = [0], skiprows = 24, dtype = 'int16')
tmp_bit_key = int(file_list_NP[0][index_k:index_m-34], 16) % 2


#ВЫЧИСЛИМ СТАТИСТИКУ ПИРСОНА
X = np.mean(tmp_trace)[0]

n = len(tmp_trace)
sigma = math.sqrt(np.std(tmp_trace)[0])

alpha = 0.01
u = stats.norm.ppf(1 - alpha)#квантиль N(0,1)
beta = stats.norm.cdf((u - math.sqrt(n) * (mu1 - mu0)/sigma))

S = (mu0 + (sigma/math.sqrt(n)) * u)

if X >= S:
    print("Гипотеза о том,что bit=0 отвергается, tmp_bit_key=%s"%tmp_bit_key)
else:
    print("Гипотеза о том,что bit=0 принимается, tmp_bit_key=%s"%tmp_bit_key)


#Посчитаем количество необходимого материала

#Нейман-Пирсон
N_pearson = int((sigma * (stats.norm.ppf(alpha) + stats.norm.ppf(beta)) / (mu1 - mu0))**2) + 1

#критерий Вальда

N_vald = - 2 * sigma**2 / (mu1 - mu0)**2 * ((1 - alpha) * math.log(beta/(1 - alpha)) + alpha * math.log((1 - beta) / alpha))

N_vald / N_pearson


M = - 2  / (stats.norm.ppf(alpha) + stats.norm.ppf(beta))**2 * ((1 - alpha) * math.log(beta/(1 - alpha)) + alpha * math.log((1 - beta) / alpha))




#Вычислим ошибку определения 1-го бита ключа на 1000 ключах
file_list_NP = os.listdir('D:\Trace\DPA_public_base')
np.random.shuffle(file_list_NP)

TP = 0
FP = 0
FN = 0
TN = 0
alpha = 0.01
u = stats.norm.ppf(1 - alpha)#квантиль N(0,1)  
n = 3253
bit_prediction = 0
bit_expected = []
bit_pred = []
for i in range(2000):
    
    index_k = file_list_NP[i].find('k=') + 2
    index_m = file_list_NP[i].find('_m=') + 3
    tmp_trace = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list_NP[i], names = [0], skiprows = 24, dtype = 'int16')
    bit_key = int(file_list_NP[i][index_k:index_m-34], 16) % 2
    bit_expected.append(bit_key)
    
    #ВЫЧИСЛИМ СТАТИСТИКУ ПИРСОНА
    X = np.mean(tmp_trace)[0]      
    sigma = math.sqrt(np.std(tmp_trace)[0])     
    S = (mu0 + (sigma/math.sqrt(n)) * u)
    if X >= S:
        # print("Гипотеза о том,что bit=0 отвергается, tmp_bit_key=%s"%bit_key)
        bit_prediction = 1
        bit_pred.append(bit_prediction)
        if bit_prediction == bit_key:
            TN = TN + 1
        else:
            FN = FN + 1  
    else:
        # print("Гипотеза о том,что bit=0 принимается, tmp_bit_key=%s"%bit_key)
        bit_prediction = 0
        bit_pred.append(bit_prediction)
        if bit_prediction == bit_key:
            TP = TP + 1
        else:
            FP = FP + 1  
            
accuracy = (TP + TN)/(TP + TN + FP + FN)
print(accuracy)
from sklearn import metrics
class_report = metrics.classification_report(bit_expected, bit_pred)
confusion_matr = metrics.confusion_matrix(bit_expected, bit_pred)
accuracy = metrics.accuracy_score(bit_expected, bit_pred)
print(class_report)
print('confusion_matrix \n', confusion_matr)
print('\n accuracy =', accuracy)






















#------------------------------------------------------------------------------
#теперь попробуем различить байт ключа с помощью критерия Неймана-Пирсона

import os
import math
import scipy.stats
import pandas as pd
import numpy as np
import scipy 
from sklearn import preprocessing
import scipy.stats as stats
def to_bin(x):
    res = bin(x)[2:]
    while len(res) < 8:
        res = '0' + res
    return res
    
N = 200
bit_mu0 = []
bit_mu1 = []
for j in range(8):
    print(j)
    #while 1:
    keys = []
    dfX = pd.DataFrame()
    dfY = pd.DataFrame
    file_list_NP = os.listdir('D:\Trace\DPA_public_base')
    np.random.shuffle(file_list_NP)
    for i in range(N):
        index_k = file_list_NP[i].find('k=') + 2
        index_m = file_list_NP[i].find('_m=') + 3
        keys.append(to_bin(int(file_list_NP[i][index_k:index_m-33], 16)))
        df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list_NP[i], names = [i], skiprows = 24, dtype = 'int16')
        dfX = pd.concat([dfX, df_temp], axis=1)
    
    dfX = dfX.transpose()
    dfY = pd.DataFrame(keys)
    bit0 = pd.DataFrame()
    bit1 = pd.DataFrame()
    for i in range(N):
        if dfY.iloc[i][0][j] == '0':
            bit0 =  pd.concat([bit0, dfX.iloc[i]], axis=1)
        else:
            bit1 =  pd.concat([bit1, dfX.iloc[i]], axis=1)
        
    bit0 = bit0.transpose()
    bit1 = bit1.transpose()
        
    mu_mass0 = []
    mu_mass1 = []
    for trace in range(len(bit0)):
        mu_mass0.append(bit0.iloc[trace].mean())
    for trace in range(len(bit1)):
        mu_mass1.append(bit1.iloc[trace].mean())
    mu0 = np.mean(mu_mass0)
    mu1 = np.mean(mu_mass1)
    print("mu0 = %s\nmu1 = %s \nmu1 > mu0:%s\n\n"%(mu0, mu1, mu1 > mu0))
    bit_mu0.append(mu0)
    bit_mu1.append(mu1)
        # if mu1 > mu0:
        #     bit_mu0.append(mu0)
        #     bit_mu1.append(mu1)
        #     break
    



TP = 0
FP = 0
FN = 0
TN = 0
alpha = 0.05
u = stats.norm.ppf(1 - alpha)#квантиль N(0,1)  
n = 3253
bit_prediction = 0
byte_key_expected = []
byte_key_prediction = []
for i in range(1000):    
    index_k = file_list_NP[i].find('k=') + 2
    index_m = file_list_NP[i].find('_m=') + 3
    tmp_trace = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list_NP[i], names = [0], skiprows = 24, dtype = 'int16')
    byte_key = to_bin(int(file_list_NP[i][index_k:index_m-33], 16))
    byte_key_expected.append(hex(int(byte_key, 2)))
    byte_prediction = ''
    
    X = np.mean(tmp_trace)[0]      
    sigma = math.sqrt(np.std(tmp_trace)[0])   
    for i in range(8): 
        #ВЫЧИСЛИМ СТАТИСТИКУ ПИРСОНА
        #Это условие для того, чтобы подогнать под условие статистики mu1 > mu0 и взять за нул. гипотезу либо bit=0?, либо bit=1
        if bit_mu1[i] > bit_mu0[i]:
            S = (int(bit_mu0[i]) + (sigma/math.sqrt(n)) * u)
            if X >= S:
                # print("Гипотеза о том,что bit=0 отвергается, tmp_bit_key=%s"%bit_key)
                bit_prediction = 1
                byte_prediction = byte_prediction + '1'
                if bit_prediction == byte_key[i]:
                    TN = TN + 1
                else:
                    FN = FN + 1  
            else:
                # print("Гипотеза о том,что bit=0 принимается, tmp_bit_key=%s"%bit_key)
                bit_prediction = 0
                byte_prediction = byte_prediction + '0'
                if bit_prediction == byte_key[i]:
                    TP = TP + 1
                else:
                    FP = FP + 1  
        else:
            S = (int(bit_mu1[i]) + (sigma/math.sqrt(n)) * u)
            if X >= S:
                # print("Гипотеза о том,что bit=0 отвергается, tmp_bit_key=%s"%bit_key)
                bit_prediction = 0
                byte_prediction = byte_prediction + '0'
                if bit_prediction == byte_key[i]:
                    TN = TN + 1
                else:
                    FN = FN + 1  
            else:
                # print("Гипотеза о том,что bit=0 принимается, tmp_bit_key=%s"%bit_key)
                bit_prediction = 1
                byte_prediction = byte_prediction + '1'
                if bit_prediction == byte_key[i]:
                    TP = TP + 1
                else:
                    FP = FP + 1  
    byte_key_prediction.append(hex(int(byte_prediction, 2)))
           
    
    
# accuracy = (TP + TN)/(TP + TN + FP + FN)
# print(accuracy)
from sklearn import metrics
class_report = metrics.classification_report(byte_key_expected, byte_key_prediction)
confusion_matr = metrics.confusion_matrix(byte_key_expected, byte_key_prediction)
accuracy = metrics.accuracy_score(byte_key_expected, byte_key_prediction)
print(class_report)
print('confusion_matrix \n', confusion_matr)
print('\n accuracy =', accuracy)

print(set(byte_key_expected))
print(set(byte_key_prediction))








#______________________________________________________________________________
#______________________________________________________________________________
#ПОПРОБУЕМ ОБУЧИТЬ ПЕРСЕПТРОН НА КЛАССИФИКАЦИЮ


import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import scipy
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from tqdm import tqdm

#file_list = os.listdir('D:\Trace\DPA_public_base')
file_list = os.listdir('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23')
np.random.shuffle(file_list)

keys = []
index_k = file_list[0].find('k=') + 2
index_m = file_list[0].find('_m=') + 3
#dfX = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
dfX = pd.read_csv('D:\\Trace\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
keys.append('0x' + file_list[0][index_k:index_m-33])
for i in tqdm(range(2000)):
    index_k = file_list[i].find('k=') + 2
    index_m = file_list[i].find('_m=') + 3
    keys.append('0x' + file_list[i][index_k:index_m-33])
   # df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    df_temp = pd.read_csv('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    dfX = pd.concat([dfX, df_temp], axis=1)
dfX = dfX.transpose()
dfY = pd.DataFrame(keys)

# Строим таблицу с зависимостями между признаками
import seaborn as sns
sns.pairplot(dfX[[0,1,2,3,4]])


#Теперь посмотрим на математические значения зависимостей
dfX[[0,1,2,3,4,5,6]].corr()
import seaborn as sns
corr = dfX.transpose().iloc[:50].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True, annot=True, linewidths=.5)
    ax.fig.set_size_inches(15,15)



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# standardize the data attributes
from sklearn import preprocessing
dfX_scale = preprocessing.scale(dfX)

from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX, dfY,test_size = 0.2, random_state = 17)

from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX_scale, dfY,test_size = 0.2, random_state = 17)


#model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3253, 2), random_state=1)
model = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(3253,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=17,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
model.fit(X_train, np.ravel(y_train))
expected = np.ravel(y_valid)
predicted = model.predict(X_valid)
# summarize the fit of the model
class_report = metrics.classification_report(expected, predicted)
confusion_matr = metrics.confusion_matrix(expected, predicted)
accuracy = metrics.accuracy_score(expected, predicted)     
print(class_report)
print(confusion_matr)
print('\n accuracy =', accuracy)

file_res_fit_svm = open('D:\Trace\ResultFitMLPClassifier.txt', 'a')
file_res_fit_svm.write('\n\n WITH OneVSRest on 5000 trace\n' + str(model)) 
file_res_fit_svm.write( '\n' + str(model.activation) + '   ' + str(model.hidden_layer_sizes))
file_res_fit_svm.write( '\n' + str(class_report))
file_res_fit_svm.close()


# --------------------------------------------------------------------------------
# MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
#        beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(3253,), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#        verbose=False, warm_start=False)

#  accuracy = 0.761194029851
# -------------------------------------------------------------------------------


layer = [(3253,), (3253,2)]
from sklearn.model_selection import GridSearchCV
parameters = [{'activation':['identity', 'logistic', 'tanh', 'relu'],
               'solver' : ['lbfgs', 'sgd', 'adam'],
               'hidden_layer_sizes' : (3253,),
               'learning_rate' : ['constant', 'invscaling', 'adaptive']
               }]
model = MLPClassifier(verbose=True)
gs = GridSearchCV(estimator = model, param_grid = parameters, scoring='accuracy', cv = 2, n_jobs = -1)
gs.fit(X_train, y_train[0])
print(gs)
# make predictions
expected, predicted = y_valid, gs.predict(X_valid)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



# standardize the data attributes
from sklearn import preprocessing
dfX1 = preprocessing.scale(dfX)

from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX1, dfY, test_size = 0.2, random_state = 17)

model = MLPClassifier(verbose=True)
gs1 = GridSearchCV(estimator = model, param_grid = parameters, scoring='accuracy', cv = 2, n_jobs = 3)
gs1.fit(X_train, y_train[0])
print(gs1)
# make predictions
expected, predicted = y_valid, gs.predict(X_valid)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))






import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import scipy
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from tqdm import tqdm



dfX.to_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfX_test_diplom_1000.csv', index=False)
dfY.to_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfY_test_diplom_1000.csv', index=False)

dfX = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfX_test_diplom_20000.csv')
dfY = pd.read_csv('D:\\Trace\\test_and_valid_set\\diplom\\dfY_test_diplom_20000.csv')

dfX = dfX.drop([x for x in range(10000, 20001)], axis=0)
dfY = dfY.drop([x for x in range(10000, 20001)], axis=0)




# standardize the data attributes

dfX_scale = pd.DataFrame(preprocessing.scale(dfX))


X_train, X_valid, y_train, y_valid = train_test_split(dfX, dfY,test_size = 0.1, random_state = 17)

from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX_scale, dfY,test_size = 0.1, random_state = 17)

act = ['relu']
HLS = [(2000,2), (3253,2)]
#model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3253, 2), random_state=1)
from tqdm import tqdm
tqdm.pandas()
for a in act:
    for h in HLS:
        model = MLPClassifier(activation=a, hidden_layer_sizes=h, max_iter=100, random_state = 17, verbose=1) 
        model.fit(X_train, np.ravel(y_train))
        expected = np.ravel(y_valid)
        predicted = model.predict(X_valid)
        # summarize the fit of the model
        class_report = metrics.classification_report(expected, predicted)
        confusion_matr = metrics.confusion_matrix(expected, predicted)
        accuracy = metrics.accuracy_score(expected, predicted)     
        print(str(a) + '    ' + str(h) + '    withpreprocessing')
    
        print(class_report)
        print('\n accuracy =', accuracy)
        
        file_res_fit_svm = open('D:\Trace\ResultFitMLPClassifier_with_preprocessing_10000.txt', 'a')        
        file_res_fit_svm.write( '\n\hline\n' + str(model.activation) + ' & ' 
                               + str(model.hidden_layer_sizes) + ' & да &' 
                               + class_report[class_report.find('total') + 12:class_report.find('total') + 16] + 
                               ' & ' + class_report[class_report.find('total') + 22:class_report.find('total') + 26] +
                               ' & ' + class_report[class_report.find('total') + 32:class_report.find('total') + 36] + '\\\\')
        file_res_fit_svm.close()



from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX, dfY,test_size = 0.1, random_state = 17)

for a in act:
    for h in HLS:
        model = MLPClassifier(activation=a, hidden_layer_sizes=h, max_iter=100, random_state = 17, verbose=1) 
        model.fit(X_train, np.ravel(y_train))
        expected = np.ravel(y_valid)
        predicted = model.predict(X_valid)
        # summarize the fit of the model
        class_report = metrics.classification_report(expected, predicted)
        confusion_matr = metrics.confusion_matrix(expected, predicted)
        accuracy = metrics.accuracy_score(expected, predicted)   
        print(str(a) + '    ' + str(h) + '    without preprocessing')
        print(class_report)
        print(confusion_matr)
        print('\n accuracy =', accuracy)
        
        file_res_fit_svm = open('D:\Trace\ResultFitMLPClassifier_without_preprocessing_10000.txt', 'a')        
        file_res_fit_svm.write( '\n\hline\n' + str(model.activation) + ' & ' 
                               + str(model.hidden_layer_sizes) + ' & нет &' 
                               + class_report[class_report.find('total') + 12:class_report.find('total') + 16] + 
                               ' & ' + class_report[class_report.find('total') + 22:class_report.find('total') + 26] +
                               ' & ' + class_report[class_report.find('total') + 32:class_report.find('total') + 36] + '\\\\')
        file_res_fit_svm.close()





act = ['identity', 'logistic', 'tanh', 'relu']
HLS = [(100,), (256,), (1000,), (2000,), (3253,) , (100,2), (256,2), (1000,2), (2000,2), (3253,2)]
from sklearn.model_selection import GridSearchCV
#model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3253, 2), random_state=1)
from tqdm import tqdm
tqdm.pandas()
for a in act:
    for h in HLS:
        model = MLPClassifier(activation=a, hidden_layer_sizes=h, max_iter=100, random_state = 17, verbose=1) 
        gs = GridSearchCV(estimator = model, scoring='f1', cv = 1, n_jobs = -1)
        gs.fit(X_train, y_train[0])
        print(gs)
        # make predictions
        expected, predicted = y_valid, gs.predict(X_valid)
        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))

        model.fit(X_train, np.ravel(y_train))
        expected = np.ravel(y_valid)
        predicted = model.predict(X_valid)
        # summarize the fit of the model
        class_report = metrics.classification_report(expected, predicted)
        confusion_matr = metrics.confusion_matrix(expected, predicted)
        accuracy = metrics.accuracy_score(expected, predicted)     
        print(str(a) + '    ' + str(h) + '    withpreprocessing')
    
        print(class_report)
        print('\n accuracy =', accuracy)
        
        file_res_fit_svm = open('D:\Trace\ResultFitMLPClassifier_with_preprocessing_20000_1.txt', 'a')        
        file_res_fit_svm.write( '\n\hline\n' + str(model.activation) + ' & ' 
                               + str(model.hidden_layer_sizes) + ' & да &' 
                               + class_report[class_report.find('total') + 12:class_report.find('total') + 16] + 
                               ' & ' + class_report[class_report.find('total') + 22:class_report.find('total') + 26] +
                               ' & ' + class_report[class_report.find('total') + 32:class_report.find('total') + 36] + '\\\\')
        file_res_fit_svm.close()











import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn import preprocessing
dfX_scale = pd.DataFrame(preprocessing.scale(dfX))

param_grid = [
    {
        'activation': ['identity'],#, 'logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(100,), (256,)]#,, (1000,) (2000,), (3253,) , (100,2), (256,2), (1000,2), (2000,2), (3253,2)]
    }
]

mlp = MLPClassifier(max_iter=100, random_state = 17, verbose=1)

gs = GridSearchCV(estimator=mlp,
                  param_grid=param_grid,
                  scoring = {'precision': 'precision_samples', 'recall':'recall_samples', 'f1': make_scorer(f1_score)},
                  n_jobs=-1,
                  verbose=2,
                  refit='f1',
                  return_train_score=True)

gs.fit(dfX_scale,  np.ravel(dfY))


#______________________________________________________________________________
#______________________________________________________________________________
#СВЕРТОЧНЫЕ НЕЙРОННЫЕ СЕТИ


import os
import pandas as pd
import numpy as np
from sklearn import metrics
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import scipy
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from tqdm import tqdm

#file_list = os.listdir('D:\Trace\DPA_public_base')
file_list = os.listdir('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23')
np.random.seed(17)
np.random.shuffle(file_list)

keys = []
index_k = file_list[0].find('k=') + 2
index_m = file_list[0].find('_m=') + 3
#dfX = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
dfX = pd.read_csv('D:\\Trace\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
#keys.append('0x' + file_list[0][index_k:index_m-33])
keys.append(int('0x' + file_list[0][index_k:index_m-33], 16))
for i in tqdm(range(2000)):
    index_k = file_list[i].find('k=') + 2
    index_m = file_list[i].find('_m=') + 3
    #keys.append('0x' + file_list[i][index_k:index_m-33])
    #df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    keys.append(int('0x' + file_list[i][index_k:index_m-33], 16))
    df_temp = pd.read_csv('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    dfX = pd.concat([dfX, df_temp], axis=1)
dfX = dfX.transpose()
dfY = pd.DataFrame(keys)





# Нормализуем данные
# X_train = X_train.astype('float32')
# X_valid = X_test.astype('float32')
# X_train /= 255
# X_valid /= 255
from sklearn.model_selection import train_test_split
# standardize the data attributes
from sklearn import preprocessing
dfX_scale = pd.DataFrame(preprocessing.scale(dfX))
from sklearn import metrics
X_train, X_valid, y_train, y_valid = train_test_split(dfX, dfY,test_size = 0.2, random_state = 17)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(dfX_scale, dfY,test_size = 0.2, random_state = 17)

N = int(np.sqrt(3253))















import os
import pandas as pd
import numpy as np
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tqdm import tqdm
import os

import tensorflow as tf
from keras import backend as K

num_cores = 4


config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : 1})
session = tf.Session(config=config)
K.set_session(session)

def replace(x):
    repl_arr = np.unique(x)
    return x.replace(repl_arr, range(len(repl_arr)))
    


# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


#file_list = os.listdir('D:\Trace\DPA_public_base')
file_list = os.listdir('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23')
np.random.seed(17)
np.random.shuffle(file_list)

keys = []
index_k = file_list[0].find('k=') + 2
index_m = file_list[0].find('_m=') + 3
#dfX = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
dfX = pd.read_csv('D:\\Trace\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[0], names = [0], skiprows = 24, dtype = 'int16')
#keys.append('0x' + file_list[0][index_k:index_m-33])
keys.append(int('0x' + file_list[0][index_k:index_m-33], 16))
for i in tqdm(range(200)):
    index_k = file_list[i].find('k=') + 2
    index_m = file_list[i].find('_m=') + 3
    #keys.append('0x' + file_list[i][index_k:index_m-33])
    #df_temp = pd.read_csv('D:\\Trace\\DPA_public_base\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    keys.append(int('0x' + file_list[i][index_k:index_m-33], 16))
    df_temp = pd.read_csv('D:\\Trace\\public\\unpacked\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\'+file_list[i], names = [i], skiprows = 24, dtype = 'int16')
    dfX = pd.concat([dfX, df_temp], axis=1)
dfX = dfX.transpose()
dfY = pd.DataFrame(keys)

dfX.drop([3249, 3250, 3251, 3252], axis=1,inplace=True)

batch_size = 100
num_classes = len(np.unique(dfY))
epochs = 10
data_augmentation = True
num_predictions = 20

from sklearn import preprocessing
dfX_scale = pd.DataFrame(preprocessing.scale(dfX))
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(dfX_scale, dfY,test_size = 0.2, random_state = 17)
# process the data to fit in a keras CNN properly
# input data needs to be (N, C, X, Y) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size
X = np.array(X_train).reshape((16000, 1, 57, 57))
X_test = np.array(X_valid).reshape((4001, 1, 57, 57))

# Convert class vectors to binary class matrices.
y = keras.utils.to_categorical(replace(y_train), num_classes)
y_test = keras.utils.to_categorical(replace(y_valid), num_classes)

model = Sequential()
model.add(Conv2D(1, (53, 53), padding='same', data_format='channels_first',
                 input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(2, (2, 2), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Dropout(0.3))#лучайное исключение 30% нейронов в слое, чтобы уменьшить переобучение.

model.add(Conv2D(4, (2, 2), padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(Conv2D(4, (2, 2), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
model.add(Dropout(0.25))

model.add(Flatten())# Flatten - слой, который преобразует данные двумерной матрицы в вектор
model.add(Dense(512))#полносвязный слой из 512 нейронов
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.SGD(nesterov = True)



# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


model.fit(X, y,
          batch_size=100,
          epochs=100,
          validation_data=(X_test, y_test),
          shuffle=True)
 


    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




























#____________________________________________________________________________
#                 РАСЧТ МАТРИЦЫ КОВАРИАЦИИ
#____________________________________________________________________________
z = np.vstack((dfX.iloc[0], dfX.iloc[0])).astype(float)
c = np.cov(z.T)
np.set_printoptions(precision=3, suppress=True)
print(c)
