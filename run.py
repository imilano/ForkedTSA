import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import streamlit as st
import sys

sys.path.insert(0, 'lib/')
#sys.tracebacklimit = 0 # Hide traceback on errors

from decompose_series import decompose_series
from file_selector import file_selector
from find_acf_pacf import find_acf_pacf
from generate_code import generate_code
from grid_search_arima import grid_search_arima
from mean_abs_pct_error import mean_abs_pct_error
from plot_forecast import plot_forecasts
from predict_set import predict_set
from sidebar_menus import sidebar_menus
from test_stationary import test_stationary
from train_ts_model import train_ts_model
from transform_time_series import transform_time_series

pd.set_option('display.float_format', lambda x: '%.3f' % x) # 确保 pandas 不会对浮点数使用科学计数法

description =   '''
               时序分析演示平台.
                '''
# 描述信息
# st.image('img/banner.png')
# st.write('*An equivalent exchange: you give me data, I give you answers*')
st.write(description)

### SIDEBAR
st.sidebar.title('数据')

filename, df = file_selector()

st.markdown('## **数据预览**')
st.dataframe(df.head(10)) # 数据前10行

ds_column, y, data_frequency, test_set_size, exog_variables = sidebar_menus('feature_target', df=df)

# 外源变量名
exog_variables_names = exog_variables

# 如果没有外源变量，返回 None
exog_variables = df[exog_variables] if len(exog_variables) > 0 else None

# 选择是否对以下图表进行展示
plot_menu_title = st.sidebar.markdown('### 图表')
plot_menu_text = st.sidebar.text('选择你想看的图表')
show_absolute_plot = sidebar_menus('absolute')
show_seasonal_decompose = sidebar_menus('seasonal')
show_adfuller_test = sidebar_menus('adfuller')
show_train_prediction = sidebar_menus('train_predictions')
show_test_prediction = sidebar_menus('test_predictions')
force_transformation = sidebar_menus('force_transformations') # 平稳时序

difference_size = None
seasonal_difference_size = None

if ('Custom Difference') in force_transformation:
    # 如果用户选择了自定义转换, 那么就激活差分选项
    difference_size = st.sidebar.slider('Difference size: ', 0, 30, 1)
    seasonal_difference_size = st.sidebar.slider('Seasonal Difference size: ', 0, 30, 1)

plot_adfuller_result = False
if show_adfuller_test:
    plot_adfuller_result = True

# 把pandas数据帧转换为pandas序列
df = transform_time_series(df, ds_column, data_frequency, y)

# 展示历史数据图像
if show_absolute_plot:
    st.markdown('# 历史数据 ')
    df[y].plot(color='green')
    plt.title('绝对历史数据')
    st.pyplot()

# 展示分解图
if show_seasonal_decompose:
    st.markdown('# 季节性分解')
    decompose_series(df)  # 将季节性分解应用到时序，它会产生一个季节图、趋势图和残差图

# 检查序列平稳性
st.title('检查平稳性')

# If a function is not forced by the user, use the default pipeline
if force_transformation == None:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = test_stationary(df[y], plot_adfuller_result, data_frequency)
else:
    ts, d, D, seasonality, acf_pacf_data, transformation_function, test_stationarity_code = test_stationary(df[y], plot_adfuller_result, data_frequency, 
                                                                                                            force_transformation_technique = force_transformation, 
                                                                                                            custom_transformation_size = (difference_size, seasonal_difference_size))

st.title('ACF 和 PACF 估计')
p, q, P, Q = find_acf_pacf(acf_pacf_data, seasonality)
st.markdown('**模型的建议参数**: {}x{}{}'.format((p, d, q), (P, D, Q), (seasonality)))

st.title('Time to train!')
st.write('选择sidebar的参数然后点击 "开始展现黑魔法!" 按钮')

try:
    p, d, q, P, D, Q, s, train_model, periods_to_forecast, execute_grid_search = sidebar_menus('terms', test_set_size, seasonality, (p, d, q, P, D, Q, seasonality), df=ts)
except ValueError:
    error_message = '''
                    在尝试为 p、d、和 q寻找初始最优参数的过程中国出现了错误。
                    请检查数据集的频率是否正确. 比如, 如果数据集按天收集, 检查FREQUENCY字段是否选择了Daily.
                    '''
    raise ValueError(error_message)

# 网格搜索代价太大时展示警告信息
if execute_grid_search:
    if data_frequency in ['Hourly', 'Daily'] or p >= 5 or q >= 5:
        warning_grid_search = '''
                            使用这些设置在对数据集进行网格搜索时代价太大了，请确保你有足够的内存来进行这个操作，否则该操作很可能失败
                            '''
        st.sidebar.warning(warning_grid_search)

# 如果训练按钮被点击了
if train_model:
    exog_train = None
    exog_test = None

    # 如果 exog_variables 非空，调整endog和exog变量（Aligning endog and exog variables index）
    if type(exog_variables) == type(pd.DataFrame()):
        exog_variables.index = ts.index
        exog_train = exog_variables.iloc[:-test_set_size]
        exog_test = exog_variables.iloc[-test_set_size:]

    train_set = transformation_function(ts.iloc[:-test_set_size])
    
    test_set = transformation_function(ts.iloc[-test_set_size:])
    
    try:
        model = train_ts_model(train_set, p, d, q, P, D, Q, s, exog_variables=exog_train, quiet=False)
    except ValueError as ve:
        if ve.args[0] == 'maxlag should be < nobs':
            raise ValueError('似乎你数据不足. 尝试对  AR 和 MA 使用更小的参数(p, q, P, Q)')
        else:
            raise ve

    st.markdown('## **训练集预测**')
    st.write('模型使用本地数据进行预测')
    if transformation_function == np.log1p:
        predict_set(train_set.iloc[-24:], y, seasonality, np.expm1, model, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    else:
        predict_set(train_set.iloc[-24:], y, seasonality, transformation_function, model, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    
    st.markdown('## **测试集预测**')
    st.write('不可见数据. 不会用这部分数据来训练模型，而是用它测试训练出的模型')
    if transformation_function == np.log1p:
        predict_set(test_set, y, seasonality, np.expm1, model, exog_variables=exog_test,forecast=True, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)
    else:
        predict_set(test_set, y, seasonality, transformation_function, model, exog_variables=exog_test, forecast=True, show_train_prediction=show_train_prediction, show_test_prediction=show_test_prediction)

    # 执行网格搜索
    if execute_grid_search:
        st.markdown('# 执行网格搜索')
        st.markdown('''
                    开始为你的模型找出最优参数.这个操作会花费比较长的时间. 
                    ''')
        p, d, q, P, D, Q, s = grid_search_arima(train_set, exog_train,  range(p+2), range(q+2), range(P+2), range(Q+2), d=d, D=D, s=s)
        
    # 预测数据
    st.markdown('# 样本外数据的预测')
    
    # 创建最终模型
    with st.spinner('正在用数据集训练模型，请稍等.'):
        final_model = train_ts_model(transformation_function(ts), p, d, q, P, D, Q, s, exog_variables=exog_variables, quiet=True)
    st.success('完成!')
    
    if type(exog_variables) == type(pd.DataFrame()):
        st.write('你正在使用外源变量，无法对外源变量做出预测.，你可以尝试改变下面的代码.' )
    else:
        if transformation_function == np.log1p:
            forecasts = np.expm1(final_model.forecast(periods_to_forecast))
            confidence_interval = np.expm1(final_model.get_forecast(periods_to_forecast).conf_int())

        else:
            forecasts = final_model.forecast(periods_to_forecast)
            confidence_interval = final_model.get_forecast(periods_to_forecast).conf_int()

        confidence_interval.columns = ['ci_lower', 'ci_upper']
        plot_forecasts(forecasts, confidence_interval, data_frequency)

    st.write('# 代码生成')
    st.markdown(generate_code(filename, ds_column, y, test_stationarity_code, test_set_size, 
                              seasonality, p, d, q, P, D, Q, s, exog_variables_names, transformation_function, 
                              periods_to_forecast, data_frequency))