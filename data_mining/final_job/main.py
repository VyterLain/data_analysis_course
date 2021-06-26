import json
import os
import pandas as pd
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.transformer import *
from adtk.detector import *


def detector_print(anomalies):
    print("our detector:")
    for p in range(0, len(anomalies)):
        if anomalies['value'][p]:
            print(anomalies.index[p])


def save_result(raw, out, name):
    new_one = pd.concat([raw, out], axis=1)
    new_one.to_csv("./results/" + name + '.csv')


dir_path = './data'
file_names = os.listdir(dir_path)
csv_paths = []
csv_name = []
label_path = ''
for file_name in file_names:
    path_tmp = dir_path + '/' + file_name
    if path_tmp.endswith('.csv'):
        csv_paths.append(path_tmp)
        csv_name.append(file_name)
    elif path_tmp.endswith('.json'):
        label_path = path_tmp
    print(path_tmp)

# read csv
res = []
all_df = []
for csv_path in csv_paths:
    df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    all_df.append(df)
    break  # for test

# read label
label_f = open(label_path)
label = json.load(label_f)

index = 0
print()
print("current csv " + csv_name[index])
cur_label = label[csv_name[index]]
for res in cur_label:
    print(res)
cur = all_df[index]
data = validate_series(cur)
# origin data
# plot(data)
# data rolling
# data_rolling = RollingAggregate(agg='quantile', agg_params={"q":[0.25,0.75]},window=5).transform(data)
# plot(data_rolling)
# double rolling
# data_double_rolling = DoubleRollingAggregate(agg='median', window=5, diff='diff').transform(data)
# plot(data_double_rolling)
# season decomposition
# however, could not find significant seasonality for ec2_cpu_utilization_24ae8d
# data_season = ClassicSeasonalDecomposition(trend=True).fit_transform(data)
# plot(data_season)
# detector
# threshold_m = ThresholdAD(high=1.5, low=0)
# threshold_anomalies = threshold_m.detect(data)
# detector_print(threshold_anomalies)
# save_result(data, threshold_anomalies, csv_name[index][:-5] + '_threshold')
# plot(data, threshold_anomalies,anomaly_tag='marker')
# quantile_m = QuantileAD(high=0.999, low=0.001)
# quantile_anomalies = quantile_m.fit_detect(data)
# detector_print(quantile_anomalies)
# save_result(data, quantile_anomalies, csv_name[index][:-5] + '_quantile')
# plot(data, quantile_anomalies,anomaly_tag='marker')
# esd_m = GeneralizedESDTestAD(alpha=0.0001)
# esd_anomalies = esd_m.fit_detect(data)
# detector_print(esd_anomalies)
# save_result(data, esd_anomalies, csv_name[index][:-5] + '_esd')
# plot(data, esd_anomalies,anomaly_tag='marker')
# persist_m = PersistAD(window=3, c=3, side='positive')
# persist_anomalies = persist_m.fit_detect(data)
# detector_print(persist_anomalies)
# save_result(data, persist_anomalies, csv_name[index][:-5] + '_persist')
# plot(data, persist_anomalies)
# level_shift_m = LevelShiftAD(window=5)
# level_shift_anomalies = level_shift_m.fit_detect(data)
# detector_print(level_shift_anomalies)
# save_result(data, level_shift_anomalies, csv_name[index][:-5] + '_level_shift')
# plot(data, level_shift_anomalies)
# season_m = SeasonalAD()
# season_anomalies = season_m.fit_detect(data)
# detector_print(season_anomalies)
# save_result(data, season_anomalies, csv_name[index][:-5] + '_season')
# plot(data, season_anomalies, anomaly_tag='marker')
