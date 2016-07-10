#!/apollo/bin/env python2.7

import pandas as pd
import numpy as np
import sys

inputFile = "/tmp/train.csv"
outputFile = "/tmp/cleanTD.csv"

def RR_Zh(hour):
    hour = hour.sort('minutes_past', ascending=True)
    ref = hour['Ref']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        if np.isfinite(dbz):
            mmperhr = 2.62 * pow(10, -2) * pow(dbz, 0.687)
            sum = sum + mmperhr * hours
    return sum

def RR_Zh_MP(hour):
    hour = hour.sort('minutes_past', ascending=True)
    ref = hour['Ref']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum

def RR_Zh_Zdr(hour):
    hour = hour.sort('minutes_past', ascending=True)
    ref = hour['Ref']
    zdr = hour['Zdr']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, db, hours in zip(ref, zdr, valid_time):
        if np.isfinite(dbz) and np.isfinite(db):
            mmperhr = 7.46 * pow(10, -3) * pow(dbz, 0.945) * pow(db, -4.76)
            sum = sum + mmperhr * hours
    return sum


def RR_kdp_Zdr(hour):
    hour = hour.sort('minutes_past', ascending=True)
    kdp = hour['Kdp']
    zdr = hour['Zdr']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for k, db, hours in zip(kdp, zdr, valid_time):
        if np.isfinite(k) and np.isfinite(db):
            mmperhr = np.sign(k) * 136 * pow(abs(k), 0.968) * pow(db, -2.86)
            sum = sum + mmperhr * hours
    return sum

def RR_kdp_Zdr(hour):
    hour = hour.sort('minutes_past', ascending=True)
    kdp = hour['Kdp']
    zdr = hour['Zdr']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for k, db, hours in zip(kdp, zdr, valid_time):
        if np.isfinite(k) and np.isfinite(db):
            mmperhr = np.sign(k) * 136 * pow(abs(k), 0.968) * pow(db, -2.86)
            sum = sum + mmperhr * hours
    return sum

def RR_kdp(hour):
    hour = hour.sort('minutes_past', ascending=True)
    kdp = hour['Kdp']
    minutes_past = hour['minutes_past']
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for k, hours in zip(kdp, valid_time):
        if np.isfinite(k):
            mmperhr = np.sign(k) * 54.3 * pow(abs(k), 0.806)
            sum = sum + mmperhr * hours
    return sum

def main():
    allData = pd.read_csv(inputFile)

    # Removal of Outliers
    if sys.argv[1]=="train":
        allData = allData[allData['Expected'] < 70]
        allData = allData[(allData['Ref'] > 5) & (allData['Ref'] < 53)]
    print 'Outlier Removal Complete'
    # Grouping data
    allData.set_index('Id')
    allDataGrouped = allData.groupby("Id")
    finalTD = pd.DataFrame()
    finalTD['num_of_scans'] = np.array(allDataGrouped.Ref.count())
    print 'Number of Scans Completed'
    finalTD['mean_dist'] = np.array(allDataGrouped.radardist_km.mean())
    finalTD['mean_ref'] = np.array(allDataGrouped.Ref.mean())
    finalTD['mean_refcomp'] = np.array(allDataGrouped.RefComposite.mean())
    finalTD['mean_RhoHV'] = np.array(allDataGrouped.RhoHV.mean())
    finalTD['Rainfall_MP'] = np.array(allDataGrouped.apply(RR_Zh_MP))
    finalTD['Rainfall_Zh'] = np.array(allDataGrouped.apply(RR_Zh))
    # Rainfall_Zh_Zdr,Rainfall_kdp_Zdr
    finalTD['Rainfall_Zh_Zdr'] = np.array(allDataGrouped.apply(RR_Zh_Zdr))
    finalTD['Rainfall_kdp_Zdr'] = np.array(allDataGrouped.apply(RR_kdp_Zdr))
    finalTD['Rainfall_kdp'] = np.array(allDataGrouped.apply(RR_kdp))
    finalTD['exp_rainfall'] = np.array(allDataGrouped.Expected.mean())

    cleanFinalTD = finalTD

    cleanFinalTD.to_csv(outputFile, header=True,sep=",",index=True)


if __name__=='__main__':
    main()
