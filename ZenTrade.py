########################################################
#Citations
#Run function and mode dispatcher format taken from 15-112 website
#Ideas/code bits courtesy of https://sklearn.org/stable/modules/generated/sklearn.svm.SVC.html
#Math procedures from https://towardsdatascience.com/understanding-support -vector-machine-
# part-2-kernel-trick-mercers-theorem-e1e6848c6c4d
########################################################
#Machine Learning Code
########################################################
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import numpy as np
import cutCSV
#training vectors (subset of full data)
X = np.array([[40763710.13107367, 126.52258886336628],\
 [1044969.1605663702, -263.135668893372],\
       [29.908637228669004, 0], [0.7413149352321979, 0],\
        [2.830938433438445, 0], [6.377318609955799, 0]])

#class assignments
y = np.array([1,1,1,-1,-1,-1])
#plt.plot(X,y)
#plt.show()
from sklearn.svm import SVC
clf = SVC(gamma = 'auto')
clf.fit(X, y)
svc = SVC(C = 1.0, kernel = "rbf", gamma = "auto")
svc.fit(X, y) 

#dot product of two vectors xi, xj
def dotProduct(xi, xj):
    return np.dot(xi, xj)

#subtract two vectors xi, xj
def subtractVectors(xi, xj):
    if (len(xi) != len(xj)):
        return 0
    else:
        newVec = []
        for i in range(len(xi)):
            newVec.append(xi[i] - xj[i])
    return newVec

#Let K(xi, xj) = α * e**(-γ||xi-xj||**2) --> Kernel Trick
def decisionFunction(dual, radial, intercept, newVector):
    supportVectors = svc.support_vectors_.tolist()
    dualVector = dual.tolist()
    decision = 0
    tuner = 0.030 #tuning variable
    gamma = 0.1 
    for i in range(len(supportVectors)):
        decision += tuner * dualVector[0][i]\
         * np.exp(-0.001 * gamma *\
          dotProduct(subtractVectors(supportVectors[i],newVector),\
           subtractVectors(supportVectors[i],newVector)))
    decision += -1 * intercept[0]
    decision += gamma
    if decision >= 0:
        return 1
    else:
        return -1

########################################
#Technical Analysis Installation
#Quandl Documentation:
#https://blog.quandl.com/getting-started-with-the-quandl-api
#Technical Analysis calculation module/documentation from:
# https://github.com/bukosabino/ta
########################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import quandl
from ta import *
import datetime
import cutCSV
quandl.ApiConfig.api_key = "5hP1qvLqKeLFqAHnA9yG"
extract = None

#storage of ta file as a data struct
class Struct(object): pass
taStorage = Struct()
taStorage.current = ""

#returns standard deviation of an indicator in a given time range
def volatilityScore(df, column, pastDays):
    colSum = 0
    for item in df[column][len(df[column]) - pastDays : len(df[column])]:
        colSum += item
    mean = colSum / len(df[column])
    diffSum = 0
    for item in df[column][len(df[column]) - pastDays : len(df[column])]:
        diffSum += (mean-item)**2
    return (diffSum / (len(df[column]) - 1))**(0.5)

#find average percentage gains over given time period
def findAverageGain(df, column, pastDays):
    gainz = 0
    try:
        for i in range(len(df[column][len(df[column])-pastDays : len(df[column])]) - 1):
            if (df[column][len(df[column])-pastDays : len(df[column])][i+1])\
             != 0 and (df[column][len(df[column])-pastDays : len(df[column])][i]) != 0:
                if df[column][len(df[column])-pastDays : len(df[column])][i+1]\
                 /  df[column][len(df[column])-pastDays : len(df[column])][i]  > 0:
                    gainz += df[column][len(df[column])-pastDays : len(df[column])][i+1]\
                     /  df[column][len(df[column])-pastDays : len(df[column])][i]
        return gainz/(len(df[column][len(df[column])-pastDays: len(df[column])]))
    except:
        return 0

#find average percentage losses over given time period
def findAverageLosses(df, column, pastDays):
    losses = 0
    try:
        for i in range(len(df[column][len(df[column])-pastDays : len(df[column])]) - 1):
            if (df[column][len(df[column])-pastDays : len(df[column])][i+1]) != 0 and\
             (df[column][len(df[column])-pastDays : len(df[column])][i]) != 0:
                if df[column][len(df[column])-pastDays : len(df[column])][i+1] /\
                  df[column][len(df[column])-pastDays : len(df[column])][i] < 0:
                    losses += df[column][len(df[column])-pastDays : len(df[column])][i+1]\
                     /  df[column][len(df[column])-pastDays : len(df[column])][i]
        return losses/(len(df[column][len(df[column])-pastDays: len(df[column])]))
    except:
        return 0

#compute RSI index for indicator data over given time period
def strengthIndex(df, column, pastDays):
    avgGain = findAverageGain(df, column, pastDays)
    avgLoss = findAverageLosses(df, column, pastDays)
    if avgLoss == 0:
        return 0
    if (1 + (avgGain / avgLoss)) == 0:
        return 0
    else:
        return 100 - (100 / (1 + (avgGain / avgLoss)))

#use Technical Analysis package to add all features
def assignTAFeatures(ticker, days, taStorage):
    # Load datas
    cutCSV.getCSV(ticker, days)
    df = pd.read_csv('Stock_Data/%s-%d.csv'%(ticker, days), sep=',')
    if "volatility_bbl" not in df.columns:
        try:
            df = add_all_ta_features(df, "open", "High", "Low", "Close", "Volume", fillna = True)
        except:
            ValError()
    df = df.iloc[::1]
    df.to_csv("Stock_Data/%s-%d.csv"%(ticker, days))
    length = len(df["Adj. Close"])
    taStorage.tuples = getAllPlotPoints(df, days)
    taStorage.file = df

#use Technical Analysis package to add all features
def assignCustomTAFeatures(tickerList, days, taStorage):
    # Load data
    try:
        cutCSV.getCSV(tickerList[0], 30)
        cutCSV.getCSV(tickerList[1], 30)
        cutCSV.getCSV(tickerList[2], 30)
        cutCSV.getCSV(tickerList[3], 30)
    except:
        ValError()
    
    df1 = pd.read_csv('Stock_Data/%s-%d.csv'%(tickerList[0], days), sep=',')
    if "volume_adi" not in df1.columns:
        df1 = add_all_ta_features(df1, "open", "High", "Low", "Close",\
         "Volume", fillna=True)
        df1.to_csv("Stock_Data/%s-%d.csv"%(tickerList[0], days))
    
    df2 = pd.read_csv('Stock_Data/%s-%d.csv'%(tickerList[1], days), sep=',')
    if "volume_adi" not in df2.columns:
        df2 = add_all_ta_features(df2, "open", "High", "Low", "Close",\
         "Volume", fillna=True)
        df2.to_csv("Stock_Data/%s-%d.csv"%(tickerList[1], days))
    
    df3 = pd.read_csv('Stock_Data/%s-%d.csv'%(tickerList[2], days), sep=',')
    if "volume_adi" not in df3.columns:
        df3 = add_all_ta_features(df3, "open", "High", "Low", "Close",\
         "Volume", fillna=True)
        df3.to_csv("Stock_Data/%s-%d.csv"%(tickerList[2], days))
    
    df4 = pd.read_csv('Stock_Data/%s-%d.csv'%(tickerList[3], days), sep=',')
    if "volume_adi" not in df4.columns:
        df4 = add_all_ta_features(df4, "open", "High", "Low", "Close",\
         "Volume", fillna=True)
        df3.to_csv("Stock_Data/%s-%d.csv"%(tickerList[3], days))  

    finaldf = df1.add(df2)
    finaldf = finaldf.add(df3)
    finaldf = finaldf.add(df4)
    finaldf = finaldf.iloc[::-1]
    finaldf.to_csv("Stock_Data/%s,%s,%s,%s-%d.csv"%(tickerList[0],\
     tickerList[1], tickerList[2], tickerList[3], days))
    length = len(finaldf["Adj. Close"])
    taStorage.tuples = getAllPlotPoints(finaldf, days)
    taStorage.file = finaldf
        
#convert acronyms to indicator module format
def getInverseReadable(indicator):
    readable = ""
    if indicator == "Accumulation/ \n Distribution Index":
        readable = "volume_adi"
    elif indicator == "On Balance Volume":
        readable = "volume_obv"
    elif indicator == "On Balance Volume Mean":
        readable = "volume_obvm"
    elif indicator == "Chaikin Money Flow":
        readable = "volume_cmf"
    elif indicator == "Force Index":
        readable = "volume_fi"
    elif indicator == "Ease Of Movement":
        readable = "volume_em"
    elif indicator == "Volume-Price Trend":
        readable = "volume_vpt"
    elif indicator == "Negative Volume Index":
        readable = "volume_nvi"
    elif indicator == "Average True Range":
        readable = "volatility_atr"
    elif indicator == "Bollinger High Band":
        readable = "volatility_bbh"
    elif indicator == "Bollinger Low Band":
        readable = "volatility_bbl"
    elif indicator == "Bollinger Bands Mean":
        readable = "volatility_bbm"
    elif indicator == "Bollinger High Bands Indicator":
        readable =  "volatility_bbhi"#returns 1 and 0
    elif indicator == "Bollinger Low Bands Indicator":
        readable = "volatility_bbli"
    elif indicator == "Keltner Channel Average":
        readable = "volatility_kcc"
    elif indicator =="Keltner Channel High Band":
        readable =  "volatility_kch"
    elif indicator == "Keltner Channel Low Band":
        readable = "volatility_kcl"
    elif indicator == "Keltner Channel \n High Band Indicator":
        readable = "volatility_kchi"
    elif indicator == "Keltner Channel \n Low Band Indicator":
        readable = "volatility_kcli"
    elif indicator == "Donchian Channel High":
        readable = "volatility_dch"
    elif indicator == "Donchain Channel Low":
        readable = "volatility_dcl"
    elif indicator == "Donchain Channel \n High Band Indicator":
        readable = "volatility_dchi"
    elif indicator == "Donchain Channel \n Low Band Indicator":
        readable = "volatility_dcli"
    elif indicator == "Moving Average \n Convergence Divergence":
        readable = "trend_macd"
    elif indicator == "Moving Average \n Conv-Div Signal":
        readable = "trend_macd_signal"
    elif indicator == "Moving Average \n Conv-Div Difference":
        readable =  "trend_macd_diff"
    elif indicator == "Exponential Moving \n Average Fast":
        readable = "trend_ema_fast"
    elif indicator == "Exponential Moving \n Average Slow":
        readable = "trend_ema_slow"
    elif indicator == "Average Directional \n Movement Index":
        readable = "trend_adx"
    elif indicator ==  "Average Directional \n Movement Index Positive":
        readable ="trend_adx_pos"
    elif indicator == "Average Directional \n Movement Index Negative":
        readable = "trend_adx_neg"
    elif indicator == "Vortex Indicator Positive":
        readable = "trend_vortex_ind_pos"
    elif indicator == "Vortex Indicator Negative":
        readable = "trend_vortex_ind_neg"
    elif indicator == "Vortex Indicator Difference":
        readable = "trend_vortex_ind_diff"
    elif indicator == "Trix":
        readable = "trend_trix"
    elif indicator == "Mass Index":
        readable = "trend_mass_index"
    elif indicator ==  "Commodity Channel Index":
        readable ="trend_cci"
    elif indicator == "Detrended Price Oscillator":
        readable = "trend_dpo"
    elif indicator == "KST Oscillator":
        readable = "trend_kst"
    elif indicator == "KST Oscillator Signal":
        readable = "trend_kst_sig"
    elif indicator == "KST Oscillator Difference":
        readable = "trend_kst_diff"
    elif indicator == "Ichimoku Kinkō Hyō A":
        readable = "trend_ichimoku_a"
    elif indicator == "Ichimoku Kinkō Hyō B":
        readable = "trend_ichimoku_b"
    elif indicator == "Ichimoku Kinkō Hyō A Visual":
        readable = "trend_visual_ichimoku_a"
    elif indicator == "Ichimoku Kinkō Hyō B Visual":
        readable = "trend_visual_ichimoku_b"
    elif indicator == "Aroon Indicator Up":
        readable = "trend_aroon_up"
    elif indicator == "Aroon Indicator Down":
        readable = "trend_aroon_down"
    elif indicator == "Aroon Index Indicator":
        readable = "trend_aroon_ind"
    elif indicator == "Relative Strength Index":
        readable = "momentum_rsi"
    elif indicator == "Money Flow Index":
        readable = "momentum_mfi"
    elif indicator == "True Strength Index":
        readable = "momentum_tsi"
    elif indicator == "Ultimate Oscillator":
        readable = "momentum_uo"
    elif indicator == "Stochastic Oscillator":
        readable = "momentum_stoch"
    elif indicator == "Stochastic Oscillator Signal":
        readable = "momentum_stoch_signal"
    elif indicator == "Williams %R":
        readable = "momentum_wr"
    elif indicator == "Awesome Oscillator":
        readable = "momentum_ao"
    elif indicator == "Daily Return":
        readable = "others_dr"
    elif indicator == "Daily Log Return":
        readable = "others_dlr"
    elif indicator == "Cumulative Return":
        readable = "others_cr"
    return readable

#convert indicator acronyms to understandable format
def getReadable(indicator):
    readable = ""
    if indicator == "volume_adi":
        readable = "Accumulation/ \n Distribution Index"
    elif indicator == "volume_obv":
        readable = "On Balance Volume"
    elif indicator == "volume_obvm":
        readable = "On Balance Volume Mean"
    elif indicator == "volume_cmf":
        readable = "Chaikin Money Flow"
    elif indicator == "volume_fi":
        readable = "Force Index"
    elif indicator == "volume_em":
        readable = "Ease Of Movement"
    elif indicator == "volume_vpt":
        readable = "Volume-Price Trend"
    elif indicator == "volume_nvi":
        readable = "Negative Volume Index"
    elif indicator == "volatility_atr":
        readable = "Average True Range"
    elif indicator == "volatility_bbh":
        readable = "Bollinger High Band"
    elif indicator == "volatility_bbl":
        readable = "Bollinger Low Band"
    elif indicator == "volatility_bbm":
        readable = "Bollinger Bands Mean"
    elif indicator == "volatility_bbhi":
        readable = "Bollinger High Bands Indicator" #returns 1 and 0
    elif indicator == "volatility_bbli":
        readable = "Bollinger Low Bands Indicator"
    elif indicator == "volatility_kcc":
        readable = "Keltner Channel Average"
    elif indicator == "volatility_kch":
        readable = "Keltner Channel High Band"
    elif indicator == "volatility_kcl":
        readable = "Keltner Channel Low Band"
    elif indicator == "volatility_kchi":
        readable = "Keltner Channel \n High Band Indicator"
    elif indicator == "volatility_kcli":
        readable = "Keltner Channel \n Low Band Indicator"
    elif indicator == "volatility_dch":
        readable = "Donchian Channel High"
    elif indicator == "volatility_dcl":
        readable = "Donchain Channel Low"
    elif indicator == "volatility_dchi":
        readable = "Donchain Channel \n High Band Indicator"
    elif indicator == "volatility_dcli":
        readable = "Donchain Channel \n Low Band Indicator"
    elif indicator == "trend_macd":
        readable = "Moving Average \n Convergence Divergence"
    elif indicator == "trend_macd_signal":
        readable = "Moving Average \n Conv-Div Signal"
    elif indicator == "trend_macd_diff":
        readable = "Moving Average \n Conv-Div Difference"
    elif indicator == "trend_ema_fast":
        readable = "Exponential Moving \n Average Fast"
    elif indicator == "trend_ema_slow":
        readable = "Exponential Moving \n Average Slow"
    elif indicator == "trend_adx":
        readable = "Average Directional \n Movement Index"
    elif indicator == "trend_adx_pos":
        readable = "Average Directional \n Movement Index Positive"
    elif indicator == "trend_adx_neg":
        readable = "Average Directional \n Movement Index Negative"
    elif indicator == "trend_vortex_ind_pos":
        readable = "Vortex Indicator Positive"
    elif indicator == "trend_vortex_ind_neg":
        readable = "Vortex Indicator Negative"
    elif indicator == "trend_vortex_ind_diff":
        readable = "Vortex Indicator Difference"
    elif indicator == "trend_trix":
        readable = "Trix"
    elif indicator == "trend_mass_index":
        readable = "Mass Index"
    elif indicator == "trend_cci":
        readable = "Commodity Channel Index"
    elif indicator == "trend_dpo":
        readable = "Detrended Price Oscillator"
    elif indicator == "trend_kst":
        readable = "KST Oscillator"
    elif indicator == "trend_kst_sig":
        readable = "KST Oscillator Signal"
    elif indicator == "trend_kst_diff":
        readable = "KST Oscillator Difference"
    elif indicator == "trend_ichimoku_a":
        readable = "Ichimoku Kinkō Hyō A"
    elif indicator == "trend_ichimoku_b":
        readable = "Ichimoku Kinkō Hyō B"
    elif indicator == "trend_visual_ichimoku_a":
        readable = "Ichimoku Kinkō Hyō A Visual"
    elif indicator == "trend_visual_ichimoku_b":
        readable = "Ichimoku Kinkō Hyō B Visual"
    elif indicator == "trend_aroon_up":
        readable = "Aroon Indicator Up"
    elif indicator == "trend_aroon_down":
        readable = "Aroon Indicator Down"
    elif indicator == "trend_aroon_ind":
        readable = "Aroon Index Indicator"
    elif indicator == "momentum_rsi":
        readable = "Relative Strength Index"
    elif indicator == "momentum_mfi":
        readable = "Money Flow Index"
    elif indicator == "momentum_tsi":
        readable = "True Strength Index"
    elif indicator == "momentum_uo":
        readable = "Ultimate Oscillator"
    elif indicator == "momentum_stoch":
        readable = "Stochastic Oscillator"
    elif indicator == "momentum_stoch_signal":
        readable = "Stochastic Oscillator Signal"
    elif indicator == "momentum_wr":
        readable = "Williams %R"
    elif indicator == "momentum_ao":
        readable = "Awesome Oscillator"
    elif indicator == "others_dr":
        readable = "Daily Return"
    elif indicator == "others_dlr":
        readable = "Daily Log Return"
    elif indicator == "others_cr":
        readable = "Cumulative Return"
    return readable
   
#plots any indicator column against a range of integers
def plotIndicator(taStorage, ticker, days, indicator):
    assignTAFeatures(ticker, days, taStorage)
    plt.figure(figsize=(8,6))
    y1 = taStorage.file["%s"%(indicator)]
    x = [i for i in range(len(taStorage.file["Close"]))]
    y3 = taStorage.file["Adj. Close"]
    readable = getReadable(indicator)
    title = "(%s) %s: past %d days" %(ticker, readable, days)
    plt.title(title) 
    plt.plot(x,y1, label = readable)
    plt.xlabel("Past %s days" %(days))
    plt.legend()
    plt.show()

#plotting function for custom screen. Takes in list of stocks to plot
def plotCustomIndicator(taStorage, tickerList, days, indicator):
    plt.figure(figsize=(8,6))
    y1 = taStorage.file["%s"%(indicator)]
    x = [i for i in range(len(taStorage.file["Close"]))]
    y3 = taStorage.file["Adj. Close"]
    readable = getReadable(indicator)
    title = "("+ tickerList[0]+ ", " + tickerList[1]+", "+tickerList[2]+", "+\
     tickerList[3] + ") " +"%s: past %d days" %( readable, days)
    plt.title(title) 
    plt.plot(x,y1, label = readable)
    plt.xlabel("Past %s days" %(days))
    plt.legend()
    plt.show()

#store feature values as a 2d list for ML processing
def getAllPlotPoints(df, days):
    tuples = []
    for column in ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']:
        tuples.append([volatilityScore(df, column, days), strengthIndex(df, column, days)])
    return(tuples)


#get feature tuples and run ML model on new vectors
def testModel(taStorage, ticker, days, data):
    assignTAFeatures(ticker, days, taStorage)
    # testElements =[[40763710.13107367, 126.52258886336628], [37885713.464623876, 618.4197888845191],\
    #  [9469989.666643852, 354.13955856324407], [0.1971876501214658, 155.7218253721363], [80756873.87393586,\
    #   105.83443389887171], [7.968563051353424e-08, 106.9502232868236], [1044969.1605663702, -263.135668893372],\
    #    [29.908637228669004, 0], [0.7413149352321979, 0], [2.830938433438445, 0], [6.377318609955799, 0],\
    #     [3.471241046307652, 0], [0.0, 0], [0.2959013411369334, 0], [5.002341860467579, 0], [4.110603490548902, 0],\
    #      [6.120916481884704, 0], [0.0, 0], [0.0, 0], [2.916943039352749, 0], [5.683117161674838, 0], [0.24580452980260492, 0],\
    #       [0.2959013411369334, 0], [1.7386032350488025, 121.82694279360047], [1.4041466963807236, 108.22323432032155], \
    #       [0.9063355133879694, 107.80239479935324], [4.198709355853161, 0], [3.04786814553082, 0], [9.97693191852648, 0],\
    #        [8.586775814821836, 0], [8.586775814821836, 0], [0.177122829216337, 0], [0.18046167453098394, 0], [0.186879342659801, 0],\
    #         [0.30141806309738645, 102.73255950909466], [7.7521432874761045, 0], [83.88214367915286, 126.85703971030495],\
    #          [4.1355719456535915, -313.6064031474488], [34.69930234465697, 109.15466155409824], [31.075703923412995,\
    #           102.83122410165191], [20.726242501306825, 107.0235400814984], [4.177926267748812, 0], [3.302729059373675, 0],\
    #            [3.2642704252716324, 0], [2.33085815004353, 0], [25.5968187987587, 0], [32.55559662300438, 0],\
    #             [48.2665410628845, 114.80505181236549], [18.229175936270455, 0], [12.150100738052954, 0],\
    #              [16.443637855754858, 107.11549130310814], [6.351359521576162, 0], [30.940304498740208, 0],\
    #               [28.59652147366238, 0], [30.94030449874021, 0], [5.517673411914127, 100.2607998530663],\
    #                [1.6554062180789109, -173.43876223400088], [1.6337934915168952, -173.19127004643656], [3.6497927224050115, 115.72430169023998]]
    testElements = taStorage.tuples
    machinePredicts = []
    machinePredictsFix = []
    myPredicts = []
    for element in testElements:
        machinePredicts.append(clf.predict([element]))
        myPredicts.append(decisionFunction(svc.dual_coef_, 1, svc.intercept_, element))
    for elem in machinePredicts:
        machinePredictsFix.append(elem.tolist()[0])
    #print(machinePredictsFix)
    #print(myPredicts)
    correct = 0
    for i in range(len(machinePredictsFix)):
        if machinePredictsFix[i] == myPredicts[i]:
            correct += 1
    #print("accuracy", correct*100/len(machinePredictsFix))
    data.currentAccuracy = correct*100/len(machinePredictsFix)
    data.priority = myPredicts
    return myPredicts

#get feature tuples and run ML model
def testCustomModel(taStorage, tickerList, days, data):
    assignCustomTAFeatures(tickerList, 30, taStorage)
    # testElements =[[40763710.13107367, 126.52258886336628], [37885713.464623876, 618.4197888845191],\
    #  [9469989.666643852, 354.13955856324407], [0.1971876501214658, 155.7218253721363], [80756873.87393586,\
    #   105.83443389887171], [7.968563051353424e-08, 106.9502232868236], [1044969.1605663702, -263.135668893372],\
    #    [29.908637228669004, 0], [0.7413149352321979, 0], [2.830938433438445, 0], [6.377318609955799, 0],\
    #     [3.471241046307652, 0], [0.0, 0], [0.2959013411369334, 0], [5.002341860467579, 0], [4.110603490548902, 0],\
    #      [6.120916481884704, 0], [0.0, 0], [0.0, 0], [2.916943039352749, 0], [5.683117161674838, 0], [0.24580452980260492, 0],\
    #       [0.2959013411369334, 0], [1.7386032350488025, 121.82694279360047], [1.4041466963807236, 108.22323432032155], \
    #       [0.9063355133879694, 107.80239479935324], [4.198709355853161, 0], [3.04786814553082, 0], [9.97693191852648, 0],\
    #        [8.586775814821836, 0], [8.586775814821836, 0], [0.177122829216337, 0], [0.18046167453098394, 0], [0.186879342659801, 0],\
    #         [0.30141806309738645, 102.73255950909466], [7.7521432874761045, 0], [83.88214367915286, 126.85703971030495],\
    #          [4.1355719456535915, -313.6064031474488], [34.69930234465697, 109.15466155409824], [31.075703923412995,\
    #           102.83122410165191], [20.726242501306825, 107.0235400814984], [4.177926267748812, 0], [3.302729059373675, 0],\
    #            [3.2642704252716324, 0], [2.33085815004353, 0], [25.5968187987587, 0], [32.55559662300438, 0],\
    #             [48.2665410628845, 114.80505181236549], [18.229175936270455, 0], [12.150100738052954, 0],\
    #              [16.443637855754858, 107.11549130310814], [6.351359521576162, 0], [30.940304498740208, 0],\
    #               [28.59652147366238, 0], [30.94030449874021, 0], [5.517673411914127, 100.2607998530663],\
    #                [1.6554062180789109, -173.43876223400088], [1.6337934915168952, -173.19127004643656], [3.6497927224050115, 115.72430169023998]]
    testElements = taStorage.tuples
    machinePredicts = []
    machinePredictsFix = []
    myPredicts = []
    for element in testElements:
        machinePredicts.append(clf.predict([element]))
        myPredicts.append(decisionFunction(svc.dual_coef_, 1, svc.intercept_, element))
    for elem in machinePredicts:
        machinePredictsFix.append(elem.tolist()[0])
    #print(machinePredictsFix)
    #print(myPredicts)
    correct = 0
    for i in range(len(machinePredictsFix)):
        if machinePredictsFix[i] == myPredicts[i]:
            correct += 1
    #print("accuracy", correct*100/len(machinePredictsFix))
    data.currentAccuracy = correct*100/len(machinePredictsFix)
    data.priority = myPredicts
    return myPredicts

###########################################################
#UI Functions
###########################################################
from tkinter import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import quandl
import datetime
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key= "5hP1qvLqKeLFqAHnA9yG"
matplotlib.use('TkAgg')
globalStock = ""

#sector buttons with different labels on left-pane
class SectorButton(object):
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.width = 150
        self.height = 40
    def draw(self, canvas):
        margin = 20
        canvas.create_rectangle(self.x - self.width,\
         self.y - self.height, self.x + self.width, self.y + self.height,\
          outline = "white", width = 5)
        canvas.create_text(self.x, self.y, text = self.label,\
         font = "Arial 20", fill = "white")
    def isIndicator(self):
        return False
    def isClicked(self, eventx, eventy):
        global globalStock
        globalStock = ""
        if self.x-self.width < eventx < self.x+self.width and\
         self.y-self.height < eventy < self.y+self.height:
            #print("isClicked", self.label)
            return True
        else:
            return False

#creates buttons on right-pane showing indicators
class IndicatorButton(SectorButton):
    def __init__(self, x, y, label, fill):
        super().__init__(x, y, label)
        self.width = 120
        self.fill = fill
    def isIndicator(self):
        return True
    def draw(self, canvas):
        margin = 20
        canvas.create_rectangle(self.x - self.width,\
         self.y - self.height, self.x + self.width, self.y + self.height,\
          outline = self.fill, width = 5)
        canvas.create_text(self.x, self.y, text = self.label,\
         font = "Arial 13", fill = "white", anchor = "center")
    def isClicked(self, eventx, eventy):
        if self.x-self.width < eventx < self.x+self.width and\
         self.y-self.height < eventy < self.y+self.height:
            #print("isClicked", self.label)
            return True
        else:
            return False

#sends user to screen with index of indicators
class InfoButton(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = 20
        self.label = "Info"

    def isIndicator(self):
        return False

    def draw(self, canvas):
        canvas.create_oval(self.x - self.r,\
         self.y - self.r, self.x + self.r, self.y + self.r,\
          fill = "white", width = 5, outline="white")
        canvas.create_text(self.x, self.y, text = "i",\
         font = "Times 20 bold", fill = "blue")

    def isClicked(self, eventx, eventy):
        if ((self.x-eventx)**2 + (self.y-eventy)**2)**(0.5) < self.r:
            return True
        else:
            return False

#runs ml model on selected stock
class RefreshButton(object):
    def __init__(self, x, y, label, fill):
        self.x = x
        self.y = y
        self.label = label
        self.width = 80
        self.height = 20
        self.fill = fill
    def isIndicator(self):
        return False
    def draw(self, canvas):
        margin = 20
        canvas.create_rectangle(self.x - self.width,\
         self.y - self.height, self.x + self.width, self.y + self.height,\
          outline = self.fill, width = 5)
        canvas.create_text(self.x, self.y, text = self.label,\
         font = "Arial 20", fill = "white")
    def isClicked(self, eventx, eventy):
        if self.x-self.width < eventx < self.x+self.width and\
         self.y-self.height < eventy < self.y+self.height:
            #print("isClickeddd", self.label)
            return True
        else:
            return False

#click this button to update global stock
class GlobalResetButton(object):
    def __init__(self, x, y, label, fill):
        self.x = x
        self.y = y
        self.label = label
        self.width = 20
        self.height = 20
        self.fill = fill
    def isIndicator(self):
        return False
    def draw(self, canvas):
        margin = 20
        canvas.create_rectangle(self.x - self.width,\
         self.y - self.height, self.x + self.width, self.y + self.height,\
          outline = self.fill, width = 5)
        canvas.create_text(self.x, self.y, text = self.label,\
         font = "Arial 12", fill = "white")
    def isClicked(self, eventx, eventy):
        if self.x-self.width < eventx < self.x+self.width and\
         self.y-self.height < eventy < self.y+self.height:
            #print("isClickeddd", self.label)
            return True
        else:
            return False

dataDays = 10        
class timeButtons(object):
    def __init__(self, x, y, label, fill):
        self.x = x
        self.y = y
        self.label = label
        self.width = 60
        self.height = 20
        self.clickCount = 0
        self.fill = fill
        self.clicked = False
    def isIndicator(self):
        return False
    def draw(self, canvas):
        margin = 20
        if self.clicked == True: 
            fill = "red"
        elif self.clicked == False:
            fill = "white" 
        canvas.create_rectangle(self.x - self.width,\
         self.y - self.height, self.x + self.width, self.y + self.height,\
          outline = fill, width = 5)
        canvas.create_text(self.x, self.y, text = self.label,\
         font = "Arial 20", fill = "white")    
    def isClicked(self, eventx, eventy):
        self.clicked = not self.clicked
        if self.x-self.width < eventx < self.x+self.width and\
         self.y-self.height < eventy < self.y+self.height:
            #print("isClicked", self.label)
            global dataDays
            dataDays = int(self.label[0]+self.label[1])
            #print("Data.days", dataDays)
            return True
        else:
            return False

#create custom input pop-up screen
import tkinter as tk
class EntryButton(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Get", command=self.on_button)
        self.entry.pack()
        self.button.pack()
        self.contents = ""
    def on_button(self):
        self.contents = (self.entry.get())
        global globalStock
        globalStock = self.contents
        #print(self.contents)
        #print("globalstock",globalStock)    
    def __repr__(self):
        return self.contents
    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

#creates lines under selected date box
class DateLines():
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.length = 50
        self.on = False
        self.label = label
    def draw(self, canvas):
        if self.on == True:
            canvas.create_line(self.x - self.length, self.y, self.x + self.length,\
             self.y, fill = "white", width = 5)

#update global variable
def updateGlobalStock(data):
    global globalStock
    #print("global",globalStock)
    if globalStock != "":
        data.currentStock = globalStock

#get rbg string from three integers
def rgbString(red, green, blue):
    return "#%02x%02x%02x" % (red, green, blue)

#create 10 buttons for menu, each with different market sector
def drawSectorButtons(data, canvas):
    leftButtonsX = 400
    rightButtonsX = 800
    buttonsY = [175,275,375,475,575]
    data.sectorButtons.append(SectorButton(leftButtonsX,buttonsY[0],\
     "Energy"))
    data.sectorButtons.append(SectorButton(leftButtonsX,buttonsY[1],\
     "Materials"))
    data.sectorButtons.append(SectorButton(leftButtonsX,buttonsY[2],\
     "Industrials"))
    data.sectorButtons.append(SectorButton(leftButtonsX,buttonsY[3],\
     "Consumer"))
    data.sectorButtons.append(SectorButton(leftButtonsX,buttonsY[4],\
     "Health Care"))
    data.sectorButtons.append(SectorButton(rightButtonsX,buttonsY[0],\
     "Financials"))
    data.sectorButtons.append(SectorButton(rightButtonsX,buttonsY[1],\
     "Technology"))
    data.sectorButtons.append(SectorButton(rightButtonsX,buttonsY[2],\
     "Telecom"))
    data.sectorButtons.append(SectorButton(rightButtonsX,buttonsY[3],\
     "Utilities"))
    data.sectorButtons.append(SectorButton(rightButtonsX,buttonsY[4],\
     "Real Estate"))


#return [] of -1 and 1 for individual ticker
def getHighPriorityIndicators(taStorage,ticker, days, data):
    return testModel(taStorage, ticker, dataDays, data)

#return [] of -1 and 1 for custom screen; takes in a tickerlist
def getCustomHighPriorityIndicators(taStorage, tickerList, days, data):
    return testCustomModel(taStorage, tickerList, 30, data)

####################################################################
#Main TK Functions
####################################################################
#main init function
def init(data):
    data.currentScreen = "Start Menu"
    data.prevScreen = ""
    data.margin = 60
    data.days = dataDays
    data.Behr = rgbString(170, 186, 176)
    data.blue = rgbString(0,154,20)
    data.sectorButtons = []
    data.energyButtons = []
    data.indicatorButtons =[]
    data.timeButtons = []
    data.currentStock = "Select A Stock To Analyze"
    data.priority = None
    data.highestPriority = None
    data.currentAccuracy = 0.000
    data.currentPortfolio = ["AAPL","GOOGL" ,"CMG", "BA"]

#main mousePressed controller function
def mousePressed(event, data):
    if data.currentScreen == "Start Menu":
        startMenuMousePressed(event, data)
    elif data.currentScreen == "Custom Portfolio":
        customPortfolioMousePressed(event, data)
    elif data.currentScreen == "Choose Sector":
        chooseSectorMousePressed(event, data)    
    elif data.currentScreen == "Energy":
        energyMousePressed(event, data)
    elif data.currentScreen == "Info Screen":
        infoScreenMousePressed(event, data)
    elif data.currentScreen == "Materials":
        materialsScreenMousePressed(event, data)
    elif data.currentScreen == "Industrials":
        industrialsScreenMousePressed(event, data)
    elif data.currentScreen == "Consumer":
        consumerScreenMousePressed(event, data)
    elif data.currentScreen == "Health Care":
        healthCareScreenMousePressed(event, data)
    elif data.currentScreen == "Financials":
        financialsScreenMousePressed(event, data)
    elif data.currentScreen == "Technology":
        technologyScreenMousePressed(event, data)
    elif data.currentScreen == "Telecom":
        telecomScreenMousePressed(event, data)
    elif data.currentScreen == "Utilities":
        utilitiesScreenMousePressed(event, data)
    elif data.currentScreen == "Real Estate":
        realEstateScreenMousePressed(event, data)

#set up entry button   
def customStockChooser(data):
    w = EntryButton()
    #print(w.ticks)
    w.mainloop()
    updateGlobalStock(data)

#main keypressed controller
def keyPressed(event, data):
    pass

#main timerfired controller
def timerFired(data):
    #constantly update global stock variable for each timerfire
    updateGlobalStock(data)


#overall redrawall controller  
def redrawAll(canvas, data):
    #print("data.currentstock", data.currentStock)
    if data.currentScreen == "Start Menu":
        updateGlobalStock(data)
        startMenuRedrawAll(canvas, data)   
    elif data.currentScreen == "Custom Portfolio":
        updateGlobalStock(data)
        customPortfolioRedrawAll(canvas, data)
    elif data.currentScreen == "Choose Sector":
        updateGlobalStock(data)
        chooseSectorRedrawAll(canvas, data)
    elif data.currentScreen == "Energy":
        updateGlobalStock(data)
        energyScreenRedrawAll(canvas,data)        
    elif data.currentScreen == "Info Screen":
        updateGlobalStock(data)
        infoScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Materials":
        updateGlobalStock(data)
        materialsScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Industrials":
        updateGlobalStock(data)
        industrialsScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Consumer":
        updateGlobalStock(data)
        consumerScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Health Care":
        updateGlobalStock(data)
        healthCareScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Financials":
        updateGlobalStock(data)
        financialsScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Technology":
        updateGlobalStock(data)
        technologyScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Telecom":
        updateGlobalStock(data)
        telecomScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Utilities":
        updateGlobalStock(data)
        utilitiesScreenRedrawAll(canvas, data)
    elif data.currentScreen == "Real Estate":
        updateGlobalStock(data)
        realEstateScreenRedrawAll(canvas, data)

##########################################################################
#Start Menu Functions
##########################################################################

#mouse pressed function for start menu
def startMenuMousePressed(event, data):
    for button in data.startButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "Sector Analysis":
                data.currentScreen = "Choose Sector"
            elif button.label == "Custom Portfolio":
                data.currentScreen = "Custom Portfolio"
    for button in data.resetButton:
        if button.isClicked(event.x, event.y) == True:
            print("Reset")
            global globalStock, custom1, custom2, custom3, custom4
            globalStock = ""
            custom1 = "Custom 1"
            custom2 = "Custom 2"
            custom3 = "Custom 3"
            custom4 = "Custom 4"
            init(data)
#draw start streen and 2 buttons
def startMenuRedrawAll(canvas, data):
    canvas.create_rectangle(0, 0, data.width, data.height,\
     fill = data.Behr)
    canvas.create_text(data.width//2, data.height//4, text = "ZenTrade ", fill = "white", font = "Arial 75 bold")
    canvas.create_text(data.width//2, data.height//2, text = "*Disclaimer: This application does not provide direct financial advice, nor does it serve as a brokerage.", fill = "white", font = "Arial 10")
    data.startButtons = []
    data.startButtons.append(SectorButton(data.width//2-200, data.height - 200,"Sector Analysis"))
    data.startButtons.append(SectorButton(data.width//2+200, data.height - 200,"Custom Portfolio"))
    data.resetButton = []
    data.resetButton.append(GlobalResetButton(data.width-30, data.height - 30, "R", "red" ))
    for button in data.startButtons:
        button.draw(canvas)
    #for button in data.resetButton:
     #   button.draw(canvas)

##########################################################################
#Custom Portfolio Functions
##########################################################################
class Error(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.button = tk.Button(self, text="4 Stocks Required!!!", command=self.on_button)
        self.button.pack()
        self.contents = ""

    def on_button(self):
        pass
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

class ValError(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.button = tk.Button(self, text="Value Error: Try Later", command=self.on_button)
        self.button.pack()
        self.contents = ""

    def on_button(self):
        pass
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

def customPortfolioMousePressed(event, data):
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Start Menu"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "Custom 1":
                customEntry1()
            elif button.label == "Custom 2":
                customEntry2()
            elif button.label == "Custom 3":
                customEntry3()
            elif button.label == "Custom 4":
                customEntry4()
            elif button.label == "REFRESH":

                if "Custom 1" in data.currentPortfolio or "Custom 2" in data.currentPortfolio or "Custom 3" in data.currentPortfolio or "Custom 4" in data.currentPortfolio:
                    Error()
                else:
                    try:
                        data.indicatorButtons = []
                        tickerList = data.currentPortfolio
                        getCustomHighPriorityIndicators(taStorage, tickerList, 30, data)
                        data.highestPriority = []
                        for i in range(len(data.priority)):
                            if data.priority[i] == 1:
                                data.highestPriority.append(i)
                        drawHighPriorityIndicators(data.highestPriority, data)
                    #error handling for occasional missing key error
                    except KeyError:
                        data.indicatorButtons = []
                        tickerList = data.currentPortfolio
                        getCustomHighPriorityIndicators(taStorage, tickerList, 30, data)
                        data.highestPriority = []
                        for i in range(len(data.priority)):
                            if data.priority[i] == 1:
                                data.highestPriority.append(i)
                        drawHighPriorityIndicators(data.highestPriority, data)
                    except ValueError:
                        ValError()
            elif button.label == "Info":
                #print("clicked Info")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotCustomIndicator(taStorage, data.currentPortfolio, 30, getInverseReadable(indicator.label))       

custom1 = "Custom 1"
custom2 = "Custom 2"
custom3 = "Custom 3"
custom4 = "Custom 4"

#entry box for 1st stock
class customEntry1(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Custom 1", command=self.on_button)
        self.entry.pack()
        self.button.pack()
        self.contents = ""

    def on_button(self):
        self.contents = (self.entry.get())
        global custom1
        custom1 = self.contents
        #print(self.contents)
        #print(globalStock)
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

#entry box for 2nd stock
class customEntry2(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Custom 2", command=self.on_button)
        self.entry.pack()
        self.button.pack()
        self.contents = ""

    def on_button(self):
        self.contents = (self.entry.get())
        global custom2
        custom2 = self.contents
        #print(self.contents)
        #print(globalStock)
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

#entry box for 3rd stock
class customEntry3(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Custom 3", command=self.on_button)
        self.entry.pack()
        self.button.pack()
        self.contents = ""

    def on_button(self):
        self.contents = (self.entry.get())
        global custom3
        custom3 = self.contents
        #print(self.contents)
        #print(globalStock)
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

#entry box for 4th stock
class customEntry4(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Custom 4", command=self.on_button)
        self.entry.pack()
        self.button.pack()
        self.contents = ""

    def on_button(self):
        self.contents = (self.entry.get())
        global custom4
        custom4 = self.contents
        #print(self.contents)
        #print(globalStock)
    
    def __repr__(self):
        return self.contents

    def returnStock(self):
        data.currentStock = self.entry.get()
        return self.contents

def customPortfolioRedrawAll(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height - 25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Custom Portfolio Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Enter 4 Ticker Symbols",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []    
    data.energyButtons.append(SectorButton(160,175,"Custom 1"))
    data.energyButtons.append(SectorButton(160,300,"Custom 2"))
    data.energyButtons.append(SectorButton(160,425,"Custom 3"))
    data.energyButtons.append(SectorButton(160,550,"Custom 4"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    data.currentPortfolio[0] = custom1
    data.currentPortfolio[1] = custom2
    data.currentPortfolio[2] = custom3
    data.currentPortfolio[3] = custom4
    canvas.create_text(600,200,text = data.currentPortfolio[0], fill = "white", font = "Arial 30 bold")
    canvas.create_text(600,300,text = data.currentPortfolio[1], fill = "white", font = "Arial 30 bold")
    canvas.create_text(600,400,text = data.currentPortfolio[2], fill = "white", font = "Arial 30 bold")
    canvas.create_text(600,500,text = data.currentPortfolio[3], fill = "white", font = "Arial 30 bold")

    data.timeButtons = []
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    data.timeLines = []
    data.timeLines.append(DateLines(400,635,"10"))
    data.timeLines.append(DateLines(600,635,"30"))
    data.timeLines.append(DateLines(800,635,"90"))
    canvas.create_text(600,650,text = "30 Day Custom Portfolio Analysis", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

##########################################################################
#Choose Sector Functions
##########################################################################
#choose sector mouse function
def chooseSectorMousePressed(event, data):
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Start Menu"
    for button in data.sectorButtons:
            if button.isClicked(event.x, event.y) == True:
                if button.label == "Energy":
                    data.currentScreen = "Energy"
                elif button.label == "Materials":
                    data.currentScreen = "Materials"
                elif button.label == "Industrials":
                    data.currentScreen = "Industrials"
                elif button.label == "Consumer":
                    data.currentScreen = "Consumer"
                elif button.label == "Health Care":
                    data.currentScreen = "Health Care"
                elif button.label == "Financials":
                    data.currentScreen = "Financials"
                elif button.label == "Technology":
                    data.currentScreen = "Technology"
                elif button.label == "Telecom":
                    data.currentScreen = "Telecom"
                elif button.label == "Utilities":
                    data.currentScreen = "Utilities"
                elif button.label == "Real Estate":
                    data.currentScreen = "Real Estate"
                break

#draw 8 buttons with major sectors
def chooseSectorRedrawAll(canvas, data):
    canvas.create_rectangle(0,0,data.width, data.height,\
     fill = data.Behr)
    backButtonSize = 50
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin,\
     anchor ="center", text = "StockGenie: Choose A Sector To Analyze",\
      font = "Arial 30 bold underline", fill = "white" )
    drawSectorButtons(data, canvas)
    for button in data.sectorButtons:
        button.draw(canvas)

##########################################################################
#Energy Screen Functions
##########################################################################
def energyMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    
    data.clickTracker = [0,0,0]
    for i in range(len(data.timeButtons)):
        if data.timeButtons[i].isClicked(event.x, event.y) == True:
            data.days = data.timeButtons[i].label
            data.clickTracker[i] += 1
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), data.blue))
    
def drawEnergyScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Energy Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"PEGI"))
    data.energyButtons.append(SectorButton(160,250,"NEE"))
    data.energyButtons.append(SectorButton(160,350,"XOM"))
    data.energyButtons.append(SectorButton(160,450,"LNG"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    data.timeLines = []
    data.timeLines.append(DateLines(400,635,"10"))
    data.timeLines.append(DateLines(600,635,"30"))
    data.timeLines.append(DateLines(800,635,"90"))
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

def energyScreenRedrawAll(canvas, data):
    updateGlobalStock(data)
    drawEnergyScreen(canvas, data)

##################################################################
#Info Screen Functions
##################################################################
def infoScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "%s"%(data.prevScreen)

def infoScreenRedrawAll(canvas, data):
    backButtonSize = 50
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height - 25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,25, text="Index of Indicators", font = "Arial 24 underline", fill = "white")
    canvas.create_text(25,75, text = "Accumulation/Distribution Index (ADI) = -1 when the close is the low of the day, +1 when close is high of the day " , anchor = "w", fill = "white")
    canvas.create_text(25,90, text = "Aroon Indicator (AI) = Identifies when trends are likely to change direction" , anchor = "w", fill = "white")
    canvas.create_text(25,105, text = "Average Directional Movement Index (ADX) = Smoothed average to detect direction and strength of trend" , anchor = "w", fill = "white")
    canvas.create_text(25,120, text = "Average True Range (ATR) = Provides indication of degree of price volatility" , anchor = "w", fill = "white")
    canvas.create_text(25,135, text = "Awesome Oscillator = Simple Moving Avg(5 Periods) - Simple Moving Average(34 Periods)" , anchor = "w", fill = "white")
    canvas.create_text(25,150, text = "Bollinger Bands = High band at moving average + deviation. Low band at moving average - deviation" , anchor = "w", fill = "white")
    canvas.create_text(25,165, text = "Bollinger Bands Indicators = Returns +1 if close higher/lower than bollinger high/low band, else 0." , anchor = "w", fill = "white")
    canvas.create_text(25,180, text = "Chaikin Money Flow (CMF) = Amount of money flow volume over a specific period" , anchor = "w", fill = "white")
    canvas.create_text(25,195, text = "Commodity Channel Index (CCI) = Difference between a security's price change and its average price change." , anchor = "w", fill = "white")
    canvas.create_text(25,210, text = "Cumulative Return (CR) = Total return over period of time" , anchor = "w", fill = "white")
    canvas.create_text(25,225, text = "Daily Log Return (DLR) = Logarithmig returns of price" , anchor = "w", fill = "white")
    canvas.create_text(25,240, text = "Daily Return (DR) = Daily return of asset as a percentage" , anchor = "w", fill = "white")
    canvas.create_text(25,255, text = "Detrended Price Oscillator (DPO) = Removes price trend to identify cycles" , anchor = "w", fill = "white")
    canvas.create_text(25,270, text = "Donchian Channel (DC) = upper band marks highest price over time period" , anchor = "w", fill = "white")
    canvas.create_text(25,285, text = "Donchian Channel Indicator = Returns +1 if close higher/lower than Donchian high/low channel, else 0." , anchor = "w", fill = "white")
    canvas.create_text(25,300, text = "Ease of Movement (EoM) = Relates asset's price change to volume; assess strength of trend" , anchor = "w", fill = "white")
    canvas.create_text(25,315, text = "Exponential Moving Average = EMA Indicator" , anchor = "w", fill = "white") 
    canvas.create_text(25,330, text = "Force Index (FI) = Illustrates how strong buying/selling pressure is. High = bullish, Low = bearish." , anchor = "w", fill = "white")
    canvas.create_text(25,345, text = "Ichimoku Kinko Hyo = Identifies the trend and looks for potential signals within that trend" , anchor = "w", fill = "white")
    canvas.create_text(25,360, text = "Keltner Channel (KC) = Simple moving average line of typical price" , anchor = "w", fill = "white")
    canvas.create_text(25,375, text = "Keltner Channel Indicator = Returns +1 if close higher/lower than Keltner high/low band, else 0." , anchor = "w", fill = "white")
    canvas.create_text(25,390, text = "KST Oscillator (KST) = Measures long-term market cycles" , anchor = "w", fill = "white")
    canvas.create_text(25,405, text = "Mass Index (MI) = Identifies trend reversals and bulges" , anchor = "w", fill = "white")
    canvas.create_text(25,420, text = "Money Flow Index (MFI) = Positive when price rises(buying pressure), Negative when price declines(selling pressure)" , anchor = "w", fill = "white")
    canvas.create_text(25,435, text = "Moving Average Convergence Divergence (MACD) = Trend-following momentum indicator; shows relationship between two moving averages" , anchor = "w", fill = "white")
    canvas.create_text(25,450, text = "Negative Volume Index (NVI) = Cululative Indicator that predicts change in volume" , anchor = "w", fill = "white")
    canvas.create_text(25,465, text = "On-Balance Volume = Cumulative total volume" , anchor = "w", fill = "white")
    canvas.create_text(25,480, text = "Relative Strength Index (RSI) = Measures speed and change of price movements" , anchor = "w", fill = "white")
    canvas.create_text(25,495, text = "Stochastic Oscillator = Represents closing price in terms of high and low ranges" , anchor = "w", fill = "white")
    canvas.create_text(25,510, text = "Stochastic Oscillator Signal = 3 Day Moving Average of Stochastic Oscillator" , anchor = "w", fill = "white")
    canvas.create_text(25,525, text = "Trix (TRIX) = Shows percentage change of exponentially smoothed moving avg." , anchor = "w", fill = "white")
    canvas.create_text(25,540, text = "True Strength Index = Shows trend direction / overbought or oversold conditions" , anchor = "w", fill = "white")
    canvas.create_text(25,555, text = "Ultimate Oscillator = Captures momentum across previous 3 weeks" , anchor = "w", fill = "white")
    canvas.create_text(25,570, text = "Volume-Price Trend: Cumulative volume +/- proportion of price trend" , anchor = "w", fill = "white")
    canvas.create_text(25,585, text = "Vortex Indicator (VI) = Negative values indicate bearish, postive indicate bullish; captures pos/neg trends" , anchor = "w", fill = "white")
    canvas.create_text(25,600, text = "Williams %R = More negative correlates to more oversold" , anchor = "w", fill = "white")
    
###########################################################
#Materials Screen Functions
###########################################################
def materialsScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
            data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked Info")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for i in range(len(data.timeButtons)):
        if data.timeButtons[i].isClicked(event.x, event.y) == True:
            data.days = data.timeButtons[i].label

def drawMaterialsScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Material Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"FCX"))
    data.energyButtons.append(SectorButton(160,250,"NEM"))
    data.energyButtons.append(SectorButton(160,350,"MOS"))
    data.energyButtons.append(SectorButton(160,450,"DWDP"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

def materialsScreenRedrawAll(canvas, data):
    drawMaterialsScreen(canvas, data)

###########################################################
#Industrials Screen Functions
###########################################################
def industrialsScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try: 
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label

def drawIndustrialsScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Industrials Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"GE"))
    data.energyButtons.append(SectorButton(160,250,"AAL"))
    data.energyButtons.append(SectorButton(160,350,"BA"))
    data.energyButtons.append(SectorButton(160,450,"UPS"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

def industrialsScreenRedrawAll(canvas, data):
    drawIndustrialsScreen(canvas, data)

###########################################################
#Consumer Screen Functions
###########################################################
def consumerScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()

            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
           # print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawConsumerScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Consumer Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"SBUX"))
    data.energyButtons.append(SectorButton(160,250,"AMZN"))
    data.energyButtons.append(SectorButton(160,350,"TGT"))
    data.energyButtons.append(SectorButton(160,450,"F"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

    
def consumerScreenRedrawAll(canvas, data):

    drawConsumerScreen(canvas, data)

###########################################################
#Health Care Screen Functions
###########################################################
def healthCareScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawHealthCareScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Health Care Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"PFE"))
    data.energyButtons.append(SectorButton(160,250,"BMY"))
    data.energyButtons.append(SectorButton(160,350,"CNC"))
    data.energyButtons.append(SectorButton(160,450,"BSX"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)
    
def healthCareScreenRedrawAll(canvas, data):
    drawHealthCareScreen(canvas, data)

###########################################################
#Financials Screen Functions
###########################################################
def financialsScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked Info")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawFinancialsScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Financials Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"BAC"))
    data.energyButtons.append(SectorButton(160,250,"WFC"))
    data.energyButtons.append(SectorButton(160,350,"JPM"))
    data.energyButtons.append(SectorButton(160,450,"GS"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)
    
def financialsScreenRedrawAll(canvas, data):
    drawFinancialsScreen(canvas, data)

###########################################################
#Technology Screen Functions
###########################################################
def technologyScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
            data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawTechnologyScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Technology Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"AAPL"))
    data.energyButtons.append(SectorButton(160,250,"MSFT"))
    data.energyButtons.append(SectorButton(160,350,"NVDA"))
    data.energyButtons.append(SectorButton(160,450,"INTC"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: " + str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

    
def technologyScreenRedrawAll(canvas, data):
    drawTechnologyScreen(canvas, data)

###########################################################
#Telecom Screen Functions
###########################################################
def telecomScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break

    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))   

    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawTelecomScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Telecom Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"T"))
    data.energyButtons.append(SectorButton(160,250,"FB"))
    data.energyButtons.append(SectorButton(160,350,"VZ"))
    data.energyButtons.append(SectorButton(160,450,"EA"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)
    
def telecomScreenRedrawAll(canvas, data):
    drawTelecomScreen(canvas, data)

###########################################################
#Utilities Screen Functions
###########################################################
def utilitiesScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
           # print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]), "green"))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]), "green"))
    
def drawUtilitiesScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Utilities Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"NRG"))
    data.energyButtons.append(SectorButton(160,250,"AES"))
    data.energyButtons.append(SectorButton(160,350,"EXC"))
    data.energyButtons.append(SectorButton(160,450,"DUK"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

def utilitiesScreenRedrawAll(canvas, data):
    drawUtilitiesScreen(canvas, data)

###########################################################
#Real Estate Screen Functions
###########################################################
def realEstateScreenMousePressed(event, data):
    #back button mouse press
    if 0 < event.x < 50 and  data.height - 50 < event.y < data.height:
        data.currentScreen = "Choose Sector"
    for button in data.energyButtons:
        if button.isClicked(event.x, event.y) == True:
            if button.label == "ADD CUSTOM":
                EntryButton()
            elif button.label == "REFRESH":
                try:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except KeyError:
                    data.indicatorButtons = []
                    ticker = data.currentStock
                    getHighPriorityIndicators(taStorage, ticker, dataDays, data)
                    data.highestPriority = []
                    for i in range(len(data.priority)):
                        if data.priority[i] == 1:
                            data.highestPriority.append(i)
                    drawHighPriorityIndicators(data.highestPriority, data)
                except ValueError:
                    ValError()
            elif button.label == "Info":
                #print("clicked INfo")
                data.prevScreen = data.currentScreen
                data.currentScreen = "Info Screen"
            else:
                data.currentStock = button.label
            break
    for indicator in data.indicatorButtons:
        if indicator.isClicked(event.x, event.y) == True:
            #print("indicLabel", indicator.label)
            plotIndicator(taStorage, data.currentStock, dataDays, getInverseReadable(indicator.label))       
    for time in data.timeButtons:
        if time.isClicked(event.x, event.y) == True:
            data.days = time.label
        
def removeExistingIndicators(data):
    for button in data.energyButtons:
        if button.isIndicator() == True:
            data.energyButtons.remove(button)

import random
def drawHighPriorityIndicators(lst ,data):
    sample = random.sample(population = lst, k=5)
    columnNames = ['volume_adi', 'volume_obv', 'volume_obvm', 'volume_cmf','volume_fi', 'volume_em', 'volume_vpt', 'volume_nvi',\
     'volatility_atr','volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi','volatility_bbli', 'volatility_kcc',\
      'volatility_kch', 'volatility_kcl', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',\
    'volatility_dcl', 'volatility_dchi', 'volatility_dcli', 'trend_macd','trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast',\
    'trend_ema_slow', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg','trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff',\
    'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',\
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a','trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b',\
     'trend_aroon_up', 'trend_aroon_down','trend_aroon_ind', 'momentum_rsi', 'momentum_mfi', 'momentum_tsi','momentum_uo', 'momentum_stoch',\
      'momentum_stoch_signal', 'momentum_wr','momentum_ao', 'others_dr', 'others_dlr', 'others_cr']
    data.indicatorButtons.append(IndicatorButton(1050,160,getReadable(columnNames[sample[0]]),data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,260,getReadable(columnNames[sample[1]]),data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,360,getReadable(columnNames[sample[2]]),data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,460,getReadable(columnNames[sample[3]]),data.blue))
    data.indicatorButtons.append(IndicatorButton(1050,560,getReadable(columnNames[sample[4]]),data.blue))
    
def drawRealEstateScreen(canvas, data):
    backButtonSize = 50
    data.sectorButtons = []
    canvas.create_rectangle(0,0,data.width, data.height, fill = data.Behr)
    #back button
    canvas.create_rectangle(0, data.height - backButtonSize, backButtonSize,\
     data.height, fill = "pink")
    canvas.create_text(25,data.height-25, text="←", font = "Arial 14")
    canvas.create_text(data.width//2,data.margin//2, anchor ="center",\
     text = "Real Estate Stocks Analysis", font = "Arial 20 bold underline",\
      fill = "white" )  
    canvas.create_line(data.width//4 + 20, 75 , data.width//4 + 20,\
     data.height, fill = "white")
    canvas.create_text(data.width//7.5 , 75, text = "Recommended Stocks",\
     font = "Arial 17 bold underline", fill = "white")
    canvas.create_text(data.width - data.width//8 , 85,\
     text = "High Priority Indicators", font = "Arial 17 bold underline",\
      fill = "white")
    canvas.create_line(data.width - data.width//4, 75 ,\
     data.width - data.width//4, data.height, fill = "white")
    data.energyButtons = []
    data.energyButtons.append(SectorButton(160,150,"IRM"))
    data.energyButtons.append(SectorButton(160,250,"WY"))
    data.energyButtons.append(SectorButton(160,350,"HST"))
    data.energyButtons.append(SectorButton(160,450,"HCP"))
    data.energyButtons.append(SectorButton(160,550,"ADD CUSTOM"))
    data.energyButtons.append(RefreshButton(1050,35,"REFRESH", "red"))
    data.energyButtons.append(InfoButton(1150,650))
    updateGlobalStock(data)
    canvas.create_text(600,350,text= data.currentStock, fill = "white", font = "Arial 30 bold")
    data.timeButtons = []
    data.timeButtons.append(timeButtons(800,600,"90 days", "white"))
    data.timeButtons.append(timeButtons(600,600,"30 days", "white"))
    data.timeButtons.append(timeButtons(400,600,"10 days", "white"))
    #accuracy text
    canvas.create_text(1120,650, text = "Current Accuracy: "+ str(data.currentAccuracy)[0:5] + "%", fill = "white", font = "Arial 13 bold", anchor = "e")
    canvas.create_text(600,550,text = "Date Range: " + str(dataDays) + " days", fill = "white", font = "Arial 14")
    for button in data.energyButtons:
        button.draw(canvas)
    for button in data.indicatorButtons:
        button.draw(canvas)
    for button in data.timeButtons:
        button.draw(canvas)

    
def realEstateScreenRedrawAll(canvas, data):
    drawRealEstateScreen(canvas, data)

#Run function taken from 15-112 website
####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    
    def redrawAllWrapper(canvas, data):
        try:
            canvas.delete(ALL)
            canvas.create_rectangle(0, 0, data.width, data.height,
                                    fill='white', width=0)
            redrawAll(canvas, data)
            canvas.update() 
        except:
            pass  

    def mousePressedWrapper(event, canvas, data):
        try:
            mousePressed(event, data)
            redrawAllWrapper(canvas, data)
        except:
            ValError()

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):

        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)

    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    canvas.winfo_toplevel().title("Stock Genie")
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    print(root.winfo_x(), root.winfo_y())
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(1200, 700)
