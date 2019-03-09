#mcandrew

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

def downloadData():
    return pd.read_csv('../scoreForecastModels/logScoresNullModels.csv')

def sumAcrossTrapType(d):
    temp = d.groupby(['year','month','state','county']).apply(lambda x: pd.Series({ 'sAlbo':x['pAlbo'].sum(), 'sAegy':x['pAegy'].sum()})).reset_index()
    temp = temp.rename(columns={'sAlbo':'pAlbo','sAegy':'pAegy'})
    return temp

def includeForecast(fileName,rename,drop):
    d = pd.read_csv(fileName)
    d = d.rename(columns={'value':"{:s}".format(rename)})
    if drop:
        d = d.drop(columns=['unit','type'])
    return d

def mergeAllForecasts(aedesData,forecasts):
    for f in forecasts:
        aedesData = aedesData.merge(f,on=['location','year','month','target'])
    return aedesData

def createDate(d):
    d['year']  = d.year.astype(str)
    d['month'] = d.month.astype(str)
    d['YYYY-MM']  = pd.to_datetime(d['year']+'-'+d['month'], format='%Y-%m')
    return d
    
def computeLogScore(d,v):
    def logScore(x):
        x['logScore_{:s}'.format(v)] = np.log(x[v]) if x['presence'] == 1 else np.log(1-x[v])
        return x
    return d.apply( logScore ,1)

def mm2inch(x):
    return x/25.4




   
if __name__ == "__main__":

    aedesPresenceAndForecasts = downloadData()
    aedesPresenceAndForecasts['YYYY-MM'] = pd.to_datetime(aedesPresenceAndForecasts['YYYY-MM'],format='%Y-%m')
    
    sanDiegoAeg = aedesPresenceAndForecasts[(aedesPresenceAndForecasts.location=='California-San Diego') & (aedesPresenceAndForecasts.target=='Ae. aegypti')]
    sanDiegoAlb = aedesPresenceAndForecasts[(aedesPresenceAndForecasts.location=='California-San Diego') & (aedesPresenceAndForecasts.target=='Ae. albopictus')]

    mpl.rcParams['xtick.labelsize'] = 8.
    mpl.rcParams['ytick.labelsize'] = 8.
    

    gs00 = gridspec.GridSpec(2,1,hspace=0.0,top=0.95,bottom=0.55,left=0.075,right=0.45,height_ratios = [10,1])
    ax = plt.subplot(gs00[0])

    times = sanDiegoAeg['YYYY-MM']
    
    #ax.plot(times, sanDiegoAeg.presence    , 'b', label = 'Presence/Absence')
    ax.plot(times,[-1 for x in times],'b-',label = 'Presence of Aedes')
    
    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg._1stOrderMM , 'g', label ='_None_')
    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg.nullBeta    , 'r', label = '_None_')
    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg._5050       , 'k', label = '_None_')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Pred. prob. of Aegy', fontsize=10)
    ax.tick_params(direction='in',size=2)

    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0,1.25)
    ax.set_yticks(ax.get_yticks()[:-1])

    ax.legend(frameon=False,loc='upper left',fontsize=8)

    ax = plt.subplot(gs00[1])
    times = sanDiegoAeg['YYYY-MM']
    pres  = sanDiegoAeg.presence
    
    ax.vlines([t for (t,x) in zip(times,pres) if x],0,1,'b',lw=3.0)
    ax.set_xlabel('', fontsize=10)
    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0.9,1.0)
    ax.set_yticks([])

    ax.set_xlim(min(times),max(times))

    gs01 = gridspec.GridSpec(2,1,hspace=0.0,top=0.95,bottom=0.55,right=0.925,left=0.55,height_ratios = [10,1])
    ax = plt.subplot(gs01[0])
    #ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg.presence    , 'b', label = 'Presence/Absence')

    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg.logScore__1stOrderMM , 'g', label ='1st order Markov')
    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg.logScore_nullBeta    , 'r', label = 'Null Beta')
    ax.plot(sanDiegoAeg['YYYY-MM'], sanDiegoAeg.logScore__5050       , 'k', label = '50/50')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Model logscore', fontsize=10)
    ax.tick_params(direction='in',size=2)

    ax.set_xlim(min(times),max(times))
    ax.set_ylim(-4.,0)
    ax.legend(frameon=False,loc='lower right',fontsize=8)
    
    ax = plt.subplot(gs01[1])
    times = sanDiegoAeg['YYYY-MM']
    pres  = sanDiegoAeg.presence
    ax.vlines([t for (t,x) in zip(times,pres) if x],0,1,'b',lw=3.0)
    ax.set_xlabel('', fontsize=10)
    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0.9,1.0)
    ax.set_yticks([])

    gs10 = gridspec.GridSpec(2,1,hspace=0.0,top=0.45,bottom=0.05,right=0.45,left=0.075,height_ratios = [10,1])
    
    ax = plt.subplot(gs10[0])
    #ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.presence    , 'b', label = 'Presence/Absence')

    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb._1stOrderMM , 'g', label ='1st order Markov')
    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.nullBeta    , 'r', label = 'Null Beta')
    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb._5050       , 'k', label = '50/50')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Pred. prob. of Albo', fontsize=10)
    ax.tick_params(direction='in',size=2)
    sns.despine(bottom=True,top=False,right=False,left=False,ax=ax)
    ax.set_xticks([])
    ax.set_xlim(min(times),max(times))
    
    ax = plt.subplot(gs10[1])
    times = sanDiegoAlb['YYYY-MM']
    pres  = sanDiegoAlb.presence
    ax.vlines([t for (t,x) in zip(times,pres) if x],0,1,'b',lw=3.0) 
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0.9,1.0)
    ax.set_yticks([])

    gs11 = gridspec.GridSpec(2,1
                             ,height_ratios = [10,1]
                             ,hspace=0.0
                             ,top=0.45
                             ,bottom=0.05
                             ,right=0.925
                             ,left=0.55)
    ax = plt.subplot(gs11[0])
    #ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.presence    , 'b', label = 'Presence/Absence')

    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.logScore__1stOrderMM , 'g', label ='1st order Markov')
    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.logScore_nullBeta    , 'r', label = 'Null Beta')
    ax.plot(sanDiegoAlb['YYYY-MM'], sanDiegoAlb.logScore__5050       , 'k', label = '50/50')
    
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Model logscore', fontsize=10)
    ax.tick_params(direction='in',size=2)
    ax.set_ylim(-4.,0)
    ax.set_xlim(min(times),max(times))

    ax = plt.subplot(gs11[1])
    times = sanDiegoAlb['YYYY-MM']
    pres  = sanDiegoAlb.presence
    ax.vlines([t for (t,x) in zip(times,pres) if x],0,1,'b',lw=3.0) 
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0.9,1.0)
    ax.set_yticks([])

   
    fig = plt.gcf()
    fig.set_size_inches(mm2inch(183),mm2inch(183)/1.6)
    plt.subplots_adjust(hspace=0.01, right=0.95, left = 0.075,top=0.95,bottom=0.05)
    plt.savefig('./nullModelPerformance.pdf')
    plt.savefig('./nullModelPerformance.png')
    plt.savefig('./nullModelPerformance.eps')
    plt.close()
   
