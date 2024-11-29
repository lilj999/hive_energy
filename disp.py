import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from data_loader import convert_tsf_to_dataframe
from pred_time import predict_series

# 假设的预测模型（仅作为示例）
def predict_time_series(data, forecast_steps):
    """
    假设模型会基于历史数据预测未来`forecast_steps`步的数据
    :param data: 现有的历史时序数据
    :param forecast_steps: 预测的时间步数
    :return: 预测的未来数据
    """
    # 假设线性增长（可以替换为实际的预测模型）
    forecast = data[-1] * (1 + np.linspace(0, 0.1, forecast_steps))  # 简单的增长预测
    return forecast

# 用于生成模拟的12条时序数据
def generate_mock_data(n_series=12, n_points=1000):
    data = {}
    start_time = pd.to_datetime("2023-01-01 00:00:00")  # 设置起始时间戳
    for i in range(n_series):
        # 每条数据是一个带噪声的正弦波
        time_stamps = [start_time + timedelta(minutes=15 * j) for j in range(n_points)]
        data[f"Series {i+1}"] = {
            "data": np.sin(np.linspace(0, 100, n_points)) + np.random.normal(0, 0.1, n_points),
            "time": time_stamps
        }
    return data

# 用于生成模拟的12条时序数据
def load_default(tsf_filename, sample_count):
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(tsf_filename)
    data = {}
    for i in range(len(loaded_data['series_value'])):
        series_name = loaded_data['series_name'][i]
        df = pd.DataFrame({
            'Demand': pd.to_numeric(loaded_data['series_value'][i], errors='coerce') 
        }) 
        df['Demand'].ffill( inplace=True)
        start_timestamp= pd.Timestamp(loaded_data['start_timestamp'][i])
        last_timestamp= start_timestamp + timedelta(minutes=15 * (len(df['Demand'].values)-1))
        series_data = np.array(df['Demand'].values)[-sample_count:]
        start_timestamp = last_timestamp - timedelta(minutes=15 * (sample_count-1))
        n_points = len(series_data)
        assert n_points == sample_count
        time_stamps = [start_timestamp + timedelta(minutes=15 * j) for j in range(n_points)]
        data[f"{series_name}"] = {
            "data": series_data,
            "time": time_stamps
        }
    
    return data

def get_pair(input_name):
    # 定义配对关系
    pairs = {
        0: ("Building0", "Solar0"),
        1: ("Building1", "Solar1"),
        2: ("Building3", "Solar2"),
        3: ("Building4", "Solar3"),
        4: ("Building5", "Solar4"),
        5: ("Building6", "Solar5"),
    }

    # 遍历配对字典，查找输入名字属于哪个组
    for group, (building, solar) in pairs.items():
        if input_name == building or input_name == solar:
            return pairs[group]  # 返回该组的配对

    return None  # 如果没有找到配对，返回 None

# Streamlit应用主函数
def main():
    dataset_names = ["data/phase_1_data.tsf", "data/phase_2_data.tsf","data/final_test_data.tsf"]
    dataset_filename = dataset_names[0]
    # 初始化 session_state 的属性（如果它们还没有初始化）
    if 'dataset_filename' not in st.session_state:
        st.session_state.dataset_filename = dataset_filename  # 设置默认数据集

    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = load_default(tsf_filename=dataset_filename, sample_count=5000)    
       
    # 设置标题
    st.title("Energy Data Management and Prediction")
    
    # 页面导航：选择页面
    page = st.sidebar.radio("Select a Page", ["Choose Dataset","View Original Data", "Prediction and Results"])
    
    if page == "Choose Dataset":
        st.subheader("Choose Dataset")
        # 选择数据文件
        dataset_filename = st.selectbox("Select a dataset", dataset_names, index=dataset_names.index(st.session_state.dataset_filename) if st.session_state.dataset_filename else 0)
        default_sample_count = 5000
        sample_count = st.number_input("Input sample count", value=default_sample_count, min_value=100)
        time_series_data = load_default(tsf_filename=dataset_filename,sample_count=sample_count) 
        st.session_state.time_series_data = time_series_data
        st.session_state.dataset_filename =dataset_filename
         # 显示所有时序数据的原始图
        st.subheader("All Time Series Data (Original Data)")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in time_series_data.items():
            ax.plot(data["time"], data["data"], label=f"{name}")  # 绘制所有时序数据
        ax.set_title("Original Time Series Data - All Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        #st.write(time_series_data)

    # 1. 查看时序数据页面
    if page == "View Original Data":
        st.subheader("View Original Data")
        st.markdown(f"### Dataset: {st.session_state.dataset_filename}")
        time_series_data =st.session_state.time_series_data 
        
        # 选择时序数据
        series_names = list(time_series_data.keys())[:5]
        selected_series = st.selectbox("Select a building", series_names)
        buildingname, solarname= get_pair(selected_series)
 
        series_data = time_series_data[buildingname]
        # 显示时序数据图
        st.markdown("### Engergy Consumption")
        st.line_chart(pd.DataFrame(series_data["data"], index=series_data["time"]), color='#ff0000')

        series_data = time_series_data[solarname]
        # 显示时序数据图
        st.markdown("### Solar")
        st.line_chart(pd.DataFrame(series_data["data"], index=series_data["time"]), color='#ffaa00')
       
    
    # 2. 预测范围及结果页面
    if page == "Prediction and Results":
        st.subheader("Prediction and Results")
        st.markdown(f"### Dataset: {st.session_state.dataset_filename}")
        time_series_data =st.session_state.time_series_data 
        # 选择时序数据和预测范围
        series_names = list(time_series_data.keys())[:5]
        selected_series = st.selectbox("Select a building", series_names, key="prediction_series")
        buildingname, solarname= get_pair(selected_series)

        building_data = time_series_data[buildingname]
        solar_data = time_series_data[buildingname]
        st.markdown(f"### Original: {buildingname}, {solarname}")
        st.line_chart(pd.DataFrame(building_data["data"], index=building_data["time"]), color='#ff0000')
        st.line_chart(pd.DataFrame(solar_data["data"], index=solar_data["time"]), color='#ffaa00')

        # 输入预测终点时间
        last_time_stamp = pd.Timestamp(building_data["time"][-1])
        st.write(f"Last time point: {last_time_stamp}")
        
        # 设置默认值为 last_time_stamp 后的 10 天
        default_prediction_end_time = last_time_stamp + pd.Timedelta(days=10)
        prediction_end_time = st.date_input("Select Prediction End Time", value=default_prediction_end_time)
        
        # 确保 prediction_end_time 和 last_time_stamp 是相同类型
        prediction_end_time = pd.Timestamp(prediction_end_time)

        # 计算预测步数
        time_diff = prediction_end_time - last_time_stamp  # 时间差
        forecast_steps = int(time_diff.total_seconds() // 60 // 15)  # 总秒数除以每步15分钟的秒数，得到预测步数

        st.write(f"Number of forecast steps: {forecast_steps}")
        
        # 生成预测按钮
        if st.button("Generate Prediction"):

            with st.spinner('Generating predictions (about two minutes)...'):
            # 预测未来的数据
                predicted_consumption = predict_series(building_data["data"], forecast_steps=forecast_steps,epochs=50,displayTitle='Consumption', save_path='predicted_consumption.png',display=False)
                predicted_solar = predict_series(solar_data["data"], forecast_steps=forecast_steps,epochs=50,displayTitle='Solar', save_path='predicted_solar.png',display=False)
                st.success("Prediction generated successfully!")

                st.session_state.predicted_consumption = predicted_consumption
                st.session_state.predicted_solar =predicted_solar
                # 生成未来的时间戳
                future_time_stamps = [last_time_stamp + timedelta(minutes=15 * i) for i in range(1, forecast_steps)]
                
                # 绘制原始数据和预测数据
                st.markdown(f"### Predctions: {buildingname}, {solarname}")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(building_data["time"], building_data["data"], label="Actual Consumption", color='red')
                ax.plot(future_time_stamps, predicted_consumption, label="Predicted Consumption", color='pink', linestyle='--')  # 虚线
                ax.axvline(x=building_data["time"][-1], color='red', linestyle='--', label="Prediction Start")
                ax.set_title(f"{selected_series} - Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(building_data["time"], solar_data["data"], label="Actual Solar", color='orange')
                ax.plot(future_time_stamps, predicted_solar, label="Predicted Solar", color='yellow', linestyle='--')  # 虚线
                ax.axvline(x=building_data["time"][-1], color='red', linestyle='--', label="Prediction Start")
                ax.set_title(f"{selected_series} - Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)
            
            # 计算预测误差和节省（假设模型为一个简单的电费节省模型）
            original_cost=np.sum(predicted_consumption) * 0.12
            # Solar generation that contributes to savings
            savings = predicted_solar * 0.1
            # Assuming consumption is higher than solar generation, the savings is limited to solar generation
            total_savings = np.sum(savings) * 0.12
            # 
            final_cost= original_cost -total_savings
            #return total_savings, original_cost, final_cost
            st.write(f"Predicted monthly original cost: ${original_cost:.2f}")
            st.write(f"Predicted monthly savings: ${total_savings:.2f}")
            st.write(f"Predicted monthly final cost: ${final_cost:.2f}")
    
    
    # 显示时序数据的总结统计
    if page == "Choose Dataset":
        st.markdown("### Summary Statistics of All Time Series")
        stats = {name: {"mean": np.mean(data["data"]), "std": np.std(data["data"])} for name, data in time_series_data.items()}
        stats_df = pd.DataFrame(stats).T
        st.write(stats_df)

if __name__ == "__main__":
    main()
