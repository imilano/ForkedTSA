import os
import pandas as pd
import requests
import streamlit as st

def file_selector(folder_path='datasets/'):
    '''
    选择一个CSV数据用作模型的数据集

    Args:
        folder_path (str): 数据集，目录的绝对路径
    Return:
        OS文件路径
        df (DataFrame): 数据集的pandas数据帧
    '''

    filenames = os.listdir(folder_path)
    filenames.sort()
    default_file_index = filenames.index('monthly_air_passengers.csv') if 'monthly_air_passengers.csv' in filenames else 0
    selected_filename = st.sidebar.selectbox('选择一个文件', filenames, default_file_index)
    
    # 检查文件是否按照格式正确划分
    if str.lower(selected_filename.split('.')[-1]) in ['csv', 'txt']:
        try:
            df = pd.read_csv(os.path.join(folder_path, selected_filename))
        except pd._libs.parsers.ParserError:
            try:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), delimiter=';')
            except UnicodeDecodeError:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), delimiter=';', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), encoding='latin1')
            except pd._libs.parsers.ParserError:
                df = pd.read_csv(os.path.join(folder_path, selected_filename), encoding='latin1', delimiter=';')

    elif str.lower(selected_filename.split('.')[-1]) == 'xls' or str.lower(selected_filename.split('.')[-1]) == 'xlsx':
        try:
            df = pd.read_excel(os.path.join(folder_path, selected_filename))
        except pd._libs.parsers.ParserError:
            try:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), delimiter=';')
            except UnicodeDecodeError:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), delimiter=';', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), encoding='latin1')
            except pd._libs.parsers.ParserError:
                df = pd.read_excel(os.path.join(folder_path, selected_filename), encoding='latin1', delimiter=';')
    else:
        st.error('This file format is not supported yet')

    if len(df) < 30:
        data_points_warning = '''
                              文件中的数据量太少，可能无法预测. 
                              推荐至少有50条数据，最好能有100条以上，否则可能导致预测结果不准确.
                              '''
        st.warning(data_points_warning)
    return os.path.join(folder_path, selected_filename), df