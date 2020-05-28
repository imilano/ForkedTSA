import streamlit as st
import sys

def sidebar_menus(menu_name, test_set_size=None, seasonality=None, terms=(0, 0, 0, 0, 0, 0, 0), df=None):
    '''
    生成 streamlit sidebar 菜单. 根据menu_name parameter的参数, 为每种情况返回不同的菜单.

    Args.
        menu_name (str): 展示在sidebar的菜单栏. 包括以下几种参数: absolute, seasonal, adfuller, train_predictions, test_predictions, feature_target, 或者 terms
        seasonality (str, optional): 要被数字代替的值. 比如: 如果数据是小时为单位的，那么就会考虑用24作为季节性的参数
        terms (七个值的 tuple): 七个整数值的tuple，包含 p, d, q, P, D, Q, and s
        df (Pandas DataFrame, optional): 包含一些时间序列数据的pandas数据帧，以便于提取列
    '''
    seasonality_dict = {'Secondly':60,
                        'Minutely':60,
                        'Hourly': 24, 
                        'Daily': 7, 
                        'Monthly': 12, 
                        'Quarterly': 4, 
                        'Yearly': 5}

    if menu_name == 'absolute':
        show_absolute_plot = st.sidebar.checkbox('Historical data', value=True)
        return show_absolute_plot
    elif menu_name == 'seasonal':
        show_seasonal_decompose = st.sidebar.checkbox('Seasonal decompose', value=True)
        return show_seasonal_decompose
    elif menu_name == 'adfuller':
        show_adfuller = st.sidebar.checkbox('Dickey-Fuller statistical test', value=True)
        return show_adfuller
    elif menu_name == 'train_predictions':
        show_train_predict_plot = st.sidebar.checkbox('Train set predictions', value=True)
        return show_train_predict_plot
    elif menu_name == 'test_predictions':
        show_test_predict_plot = st.sidebar.checkbox('Test set forecast', value=True)
        return show_test_predict_plot
    elif menu_name == 'feature_target':
        data_frequency = st.sidebar.selectbox('数据的采集频率是? ', ['选择频率', 'Secondly','Minutely','Hourly', 'Daily', 'Monthly', 'Quarterly', 'Yearly'], 0)
        
        # 如果没有为数据集选择频率, 那么就raise 一个 error
        if data_frequency == '选择频率':
            # 隐藏 traceback 以只展示错误信息
            sys.tracebacklimit = 0
            raise ValueError('请选择数据集的频率')
        
        # Show traceback error
        sys.tracebacklimit = None

        st.sidebar.markdown('### 选择列')
        ds_column = st.sidebar.selectbox('哪一个是日期列?', df.columns, 0)
        y = st.sidebar.selectbox('你想对哪一列数据进行预测?', df.columns, 1)
        exog_variables = st.sidebar.multiselect('外源变量（exogenous variables)是哪一个?', df.drop([ds_column, y], axis=1).columns)
        test_set_size = st.sidebar.slider('验证集（validation set）大小', 3, 30, seasonality_dict[data_frequency])
        return ds_column, y, data_frequency, test_set_size, exog_variables
    elif menu_name == 'force_transformations':
        st.sidebar.markdown('### 强制数据转换 (可选)')
        # transformation_techniques_list = ['Choose the best one', 'No transformation', 'First Difference', 
        #                                   'Log transformation', 'Seasonal Difference', 'Log First Difference', 
        #                                   'Log Difference + Seasonal Difference', 'Custom Difference']
        transformation_techniques_list = ['Choose the best one', 'No transformation', 'First Difference', 
                                          'Log transformation', 'Seasonal Difference', 'Log First Difference', 
                                          'Log Difference + Seasonal Difference', 'Custom Difference']
        transformation_techniques = st.sidebar.selectbox('Transformation technique', transformation_techniques_list, 0)
        return transformation_techniques
    elif menu_name == 'terms':
        st.sidebar.markdown('### 模型参数')
        st.sidebar.text('(p, d, q)x(P, D, Q)s 的参数')
        p = st.sidebar.slider('p (AR)', 0, 30, min([terms[0], 30]))
        d = st.sidebar.slider('d (I)', 0, 3, min([terms[1], 3]))
        q = st.sidebar.slider('q (MA)', 0, 30, min([terms[2], 30]))
        P = st.sidebar.slider('P (Seasonal AR)', 0, 30, min([terms[3], 30]))
        D = st.sidebar.slider('D (Amount of seasonal difference)', 0, 3, min([terms[4], 3]))
        Q = st.sidebar.slider('Q (Seasonal MA)', 0, 30, min([terms[5], 30]))
        s = st.sidebar.slider('s (Seasonal frequency)', 0, 30, min([terms[6], 30]))
        
        st.sidebar.markdown('# 预测时间')
        periods_to_forecast = st.sidebar.slider('要预测多少时间?', 1, int(len(df.iloc[:-test_set_size])/3), int(seasonality/2))
        
        grid_search = st.sidebar.checkbox('为我找到最优参数')
        train_model = st.sidebar.button('开始展示黑魔法!')

        return p, d, q, P, D, Q, s, train_model, periods_to_forecast, grid_search