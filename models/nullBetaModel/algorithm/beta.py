#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd

def downloadData():
    return pd.read_csv('../../../analysisData/aedesData.csv.gz',compression='gzip')

def sumAcrossTrapType(d):
    temp = d.groupby(['year','month','state','county']).apply(lambda x: pd.Series({ 'sAlbo':x['pAlbo'].sum(), 'sAegy':x['pAegy'].sum()})).reset_index()
    temp = temp.rename(columns={'sAlbo':'pAlbo','sAegy':'pAegy'})
    return temp

def grabCurrentStateCounty(d):
    return (d.state.iloc[0],d.county.iloc[0])

def reComputeTrainingData(trainingData, monthData):
    if trainingData.shape[0]==0:
        return monthData

    state,county = monthData.state.iloc[0], monthData.county.iloc[0]
    currentStateAndCounty = grabCurrentStateCounty(trainingData)
    if (state,county) == currentStateAndCounty:
        trainingData = trainingData.append(monthData)
    else:
        trainingData = monthData
    return trainingData

def nullBetaModel(data,priorAlphaBeta = [1.,1.]):
    N = len(data)
    binarize = [ 1 if p>0 else 0 for p in data] 
    if N==0:
        return priorAlphaBeta[1]/sum(priorAlphaBeta)
    MAP = sum(binarize+priorAlphaBeta[1])/(N+sum(priorAlphaBeta))
    return MAP

def addPrediction2AllResults(results, state, county, year, month,target,probPresent):
    results['location'].append("{:s}-{:s}".format(state,county))
    results['year'].append('{:d}'.format(year))
    results['month'].append('{:02d}'.format(month))
    results['target'].append('Ae. albopictus' if target == 'pAlbo' else 'Ae. aegypti')
    results['type'].append('binary')
    results['unit'].append('present')
    results['value'].append(probPresent)
    return results
    
if __name__ == "__main__":

    results = {'location':[],'year':[],'month':[],'target':[],'type':[],'unit':[],'value':[]}
    data = downloadData()
    data = sumAcrossTrapType(data)

    for target in ['pAlbo','pAegy']:
        trainingData = pd.DataFrame()
        for (state,county,year,month), monthData in data.groupby(['state','county','year','month']):
            sys.stdout.write('\rstate={:15s},county={:20s},year={:4d},month={:2d}\r'.format(state,county,year,month))
            sys.stdout.flush()
            
            trainingData   = reComputeTrainingData(trainingData, monthData)
            trainingValues = trainingData['{:s}'.format(target)].values

            probPresent = nullBetaModel(trainingValues, priorAlphaBeta = np.array([1,1]))
            predictedMonth = month+1 if month<12 else -1
            
            results = addPrediction2AllResults(results, state, county, year, predictedMonth, target, probPresent)

    forecastData = pd.DataFrame(results)
    forecastData.to_csv('./nullBetaForecast.csv')
