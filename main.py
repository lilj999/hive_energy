import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import convert_tsf_to_dataframe
from pred_time import predict_series


def align_two(consumption_data, consumption_start_timestamp, solar_data, solar_start_timestamp):
    # 假设时间戳是字符串类型，例如 "2023-10-01T00:00:00"
    # 将时间戳转换为 pandas 的 datetime 格式，以便进行对比
    consumption_start_time = pd.to_datetime(consumption_start_timestamp)
    solar_start_time = pd.to_datetime(solar_start_timestamp)

    # 选择较晚的时间作为对齐的起点
    start_time = max(consumption_start_time, solar_start_time)

    # 计算每个数据点对应的时间戳列表
    # 假设每个数据点间隔 15 分钟（即 15 * 60 秒）
    time_intervals = pd.date_range(start=start_time, periods=len(consumption_data), freq="15T")

    # 对比时间戳，找出从对齐起点后开始的索引
    consumption_start_idx = np.where(time_intervals >= consumption_start_time)[0][0]
    solar_start_idx = np.where(time_intervals >= solar_start_time)[0][0]

    # 截取对齐后的数据
    aligned_consumption_data = consumption_data[consumption_start_idx:]
    aligned_solar_data = solar_data[solar_start_idx:]

    # 更新时间戳列表
    aligned_time_stamps = time_intervals[max(consumption_start_idx, solar_start_idx):]

    # 输出aligned_time_stamps为列表格式
    aligned_time_stamps_list = aligned_time_stamps.tolist()
    #print("Aligned Time Stamps List:", aligned_time_stamps_list)
    return aligned_consumption_data, aligned_solar_data, aligned_time_stamps

def calculate_savings(predicted_solar, predicted_consumption, electricity_rate=0.12, solar_savings_fraction=0.10):
    """
    Calculate the monthly savings based on predicted solar generation and electricity consumption.
    :param predicted_solar: Predicted solar generation data (kWh).
    :param predicted_consumption: Predicted electricity consumption data (kWh).
    :param electricity_rate: The cost per kWh of electricity (in dollars).
    :param solar_savings_fraction: Fraction of solar generation that reduces electricity bills.
    :return: Monthly savings (in dollars).
    """
    original_cost=np.sum(predicted_consumption) * electricity_rate
    # Solar generation that contributes to savings
    savings = predicted_solar * solar_savings_fraction
    # Assuming consumption is higher than solar generation, the savings is limited to solar generation
    total_savings = np.sum(savings) * electricity_rate
    # 
    final_cost= original_cost -total_savings
    return total_savings, original_cost, final_cost


if __name__ == "__main__":
    # Load and preprocess data
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("data/phase_1_data.tsf")
    for i in range(len(loaded_data['series_value'])):
        loaded_data['series_value'].ffill( inplace=True)
    
    # Load only consumption data and solar data
    consumption_data = np.array(loaded_data['series_value'][2])#df['Consumption'].values
    solar_data = np.array(loaded_data['series_value'][6])#df['SolarGeneration'].values
    consumption_start_timestamp= loaded_data['start_timestamp'][2]
    solar_start_timestamp= loaded_data['start_timestamp'][6]
    
    # to predict next 30 days
    predicted_consumption = predict_series(consumption_data,  displayTitle='Consumption', save_path='predicted_consumption.png',display=True)
    predicted_solar = predict_series(solar_data, displayTitle='Solar', save_path='predicted_solar.png',display=True)

    # Calculate predicted cost and savings
    predicted_savings, predicted_original_cost, predicted_final_cost = calculate_savings(predicted_solar, predicted_consumption, electricity_rate=0.12)
    print(f"Predicted monthly original cost: ${predicted_original_cost:.2f}")
    print(f"Predicted monthly savings: ${predicted_savings:.2f}")
    print(f"Predicted monthly final cost: ${predicted_final_cost:.2f}")
