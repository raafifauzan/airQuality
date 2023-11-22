#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title = 'Air Quality Dashboard',
    page_icon = '⛅',
    layout = 'wide'
)

folder_path = 'dataset'
all_data = []  # List untuk menyimpan semua DataFrames

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)
    all_data.append(data)

# Menggabungkan semua DataFrames menjadi satu DataFrame tunggal
data = pd.concat(all_data, ignore_index=True)

data.drop('No', axis=1, inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
date_column = data.pop('date')

# Memasukkan kolom 'date' ke posisi pertama
data.insert(0, 'date', date_column)

columns_to_drop = ['year', 'month', 'day']
data.drop(columns=columns_to_drop, inplace=True)

data['CO'] = data['CO'] / 100

# Pilih hanya kolom-kolom numerik
numerical_columns = data.select_dtypes(include=[np.number])

# Hitung Q1 dan Q3 untuk setiap kolom numerik
Q1 = numerical_columns.quantile(0.25)
Q3 = numerical_columns.quantile(0.75)
# Hitung IQR untuk setiap kolom numerik
IQR = Q3 - Q1

# Tentukan Upper Fence dan Lower Fence
upper_fence = Q3 + 1.5 * IQR
lower_fence = Q1 - 1.5 * IQR

# Fungsi untuk mendeteksi outlier
def is_outlier(row):
    for col in numerical_columns.columns:
        if row[col] < lower_fence[col] or row[col] > upper_fence[col]:
            return False
    return True

# Hapus outlier
data = data[data.apply(is_outlier, axis=1)]


# Menghitung rata-rata harian dari data per jam
daily_data = data.groupby(['date', 'station']).agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean',
}).reset_index()

# Menampilkan hasil

daily_data = daily_data.round(2)
daily_data.set_index('date')

def calculate_aqi(pm25, pm10, so2, no2, co, o3):
    # Tentukan batas-batas konsentrasi (threshold) berdasarkan regulasi negara/wilayah
    def calculate_aqi_component(C, breakpoints):
        Ii = None
        for i in range(len(breakpoints) - 1):
            if breakpoints[i][0] <= C <= breakpoints[i][1]:
                Ii = (breakpoints[i+1][1] - breakpoints[i+1][0]) / (breakpoints[i][1] - breakpoints[i][0]) * (C - breakpoints[i][0]) + breakpoints[i+1][0]
        if Ii is not None:
            return Ii
        else:
            return 0

    # Komponen AQI untuk masing-masing parameter
    I_pm25 = calculate_aqi_component(pm25, [(0, 12), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, 350.4), (350.5, 500.4)])
    I_pm10 = calculate_aqi_component(pm10, [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 504), (505, 604)])
    I_so2 = calculate_aqi_component(so2, [(0, 35), (36, 75), (76, 185), (186, 304), (305, 604)])
    I_no2 = calculate_aqi_component(no2, [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249), (1250, 2049), (2050, 4049)])
    I_co = calculate_aqi_component(co, [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 40.4), (40.5, 50.4)])
    I_o3 = calculate_aqi_component(o3, [(0, 54), (55, 70), (71, 85), (86, 105), (106, 200)])

    # Hitung AQI
    aqi = max(I_pm25, I_pm10, I_so2, I_no2, I_co, I_o3)

    return aqi

# Iterasi melalui setiap baris DataFrame
for index, row in daily_data.iterrows():
    aqi = calculate_aqi(row['PM2.5'], row['PM10'], row['SO2'], row['NO2'], row['CO'], row['O3'])
    daily_data.at[index, 'AQI'] = aqi

daily_data.round(2)

# ========================================================

import calendar

# Filter data hanya untuk tahun 2016
data_2016 = daily_data[daily_data['date'].dt.year == 2016]

# Daftar stasiun
stasiun = data_2016['station'].unique()

# Daftar bulan (konversi dari angka bulan menjadi nama bulan)
bulan = data_2016['date'].dt.month.unique()
nama_bulan = [calendar.month_name[bln] for bln in bulan]

# Inisialisasi DataFrame untuk menyimpan rata-rata AQI per bulan
rata_rata_aqi_per_bulan = pd.DataFrame()

for s in stasiun:
    data_stasiun = data_2016[data_2016['station'] == s]
    rata_rata_aqi = data_stasiun.groupby(data_stasiun['date'].dt.month)['AQI'].mean()
    rata_rata_aqi_per_bulan[s] = rata_rata_aqi

# Membuat DataFrame untuk plot
plot_data = rata_rata_aqi_per_bulan.reset_index().melt(id_vars='date', var_name='station', value_name='AQI')

# Menyesuaikan nama kolom bulan
plot_data['date'] = plot_data['date'].map({bulan: nama for bulan, nama in zip(bulan, nama_bulan)})

# Filter data hanya untuk 3 bulan terakhir tahun 2016
data_last_3_months = daily_data[(daily_data['date'] >= '2016-08-01') & (daily_data['date'] <= '2016-11-30')]

# Plotly Express untuk perbandingan rata-rata AQI per bulan
fig1 = px.bar(plot_data, x='date', y='AQI',
              title='Perbandingan Rata-rata AQI per Bulan (Tahun 2016)',
              labels={'AQI': 'Rata-rata AQI', 'date': 'Bulan', 'station': 'Stasiun'},
              barmode='group',
              color='station',
              color_discrete_sequence=px.colors.qualitative.Light24_r
              )
fig1.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=700,  # Lebar grafik
    height=400  # Tinggi grafik
)

# Plotly Express untuk perbandingan AQI harian per stasiun
fig2 = px.line(data_last_3_months, x='date', y='AQI',
               title='Perbandingan AQI Harian per Stasiun (Agustus - Desember 2016)',
               labels={'AQI': 'AQI', 'date': 'Tanggal', 'station': 'Stasiun'},
               color='station',
               color_discrete_sequence=px.colors.qualitative.Light24_r
               )
fig2.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=1400,  # Lebar grafik
    height=400  # Tinggi grafik
)


# Filter data hanya untuk tahun 2016-2017
data_2016_2017 = daily_data[(daily_data['date'] >= '2016-01-01') & (daily_data['date'] <= '2017-12-31')]

# Menghitung rata-rata AQI per stasiun
average_aqi_per_station = data_2016_2017.groupby('station')['AQI'].mean().reset_index()

# Urutkan stasiun berdasarkan rata-rata AQI
average_aqi_per_station = average_aqi_per_station.sort_values(by='AQI')

# Membuat plot dengan Plotly Express
fig3 = px.bar(
    average_aqi_per_station, 
    x='AQI', 
    y='station', 
    orientation='h',
    title='Rata-rata AQI per Stasiun (2016-2017)',
    labels={'AQI': 'Rata-rata AQI'},
    color='station',  # Mengatur warna berdasarkan stasiun
    color_discrete_sequence=px.colors.qualitative.Light24_r)  # Gunakan palet warna kualitatif


fig3.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=700,  # Lebar grafik
    height=400  # Tinggi grafik
)

import plotly.graph_objects as go
# Data AQI (disesuaikan dengan data Anda)
data_2016_2017 = daily_data[(daily_data['date'] >= '2014-01-01') & (daily_data['date'] <= '2017-12-31')]
aqi_data = data_2016_2017['AQI']

# Tentukan kategori AQI berdasarkan rentang tertentu
categories = ['Baik', 'Sedang', 'Kurang Sehat', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Beracun']
category_ranges = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, float('inf'))]

# Hitung jumlah data dalam masing-masing grup
category_counts = [sum(1 for aqi in aqi_data if aqi >= range_min and aqi <= range_max) for range_min, range_max in category_ranges]

# Hitung persentase masing-masing grup
total_data = len(aqi_data)
category_percentages = [count / total_data * 100 for count in category_counts]

# Buat diagram pie setengah lingkaran dengan Plotly
fig4 = go.Figure(data=[go.Pie(labels=categories, values=category_percentages, hole=0.4)])
fig4.update_traces(textinfo='percent+label')

fig4.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=500,  # Lebar grafik
    height=400,  # Tinggi grafik
    title='AQI by Categories',  # Menambahkan judul
)

# Filter data hanya untuk tahun 2013-2017
data_2013_2017 = daily_data[(daily_data['date'] >= '2013-01-01') & (daily_data['date'] <= '2017-12-31')]

# Extract the year from the 'date' column
data_2013_2017['year'] = data_2013_2017['date'].dt.year

# Calculate the average AQI per year
average_aqi_per_year = data_2013_2017.groupby('year')['AQI'].mean().reset_index()

fig5 = px.bar(
    average_aqi_per_year,
    x='year',
    y='AQI',
    labels={'year': 'Tahun', 'AQI': 'Rata-rata AQI'},
    title='Perbandingan Rata-rata AQI per Tahun (2013-2017)',
    barmode='group',
    color='AQI',
    color_discrete_sequence=px.colors.qualitative.Light24_r
)
fig5.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=600,  # Lebar grafik
    height=400  # Tinggi grafik
)

st.title('Dashboard Air Quality Beijing 2013 - 2017')
st.markdown('_______')

st.subheader('Overview')
placeholder = st.empty()
with placeholder.container():
  
    col1, col2 = st.columns(2)
    with col1:
     st.plotly_chart(fig4)
    with col2:
     st.plotly_chart(fig5)
     
def show_filtered_graph(data, start_date_user, end_date_user):
    filtered_data = data[(data['date'] >= start_date_user) & (data['date'] <= end_date_user)]

    fig2 = px.line(filtered_data, x='date', y='AQI',
                   title='Perbandingan AQI Harian per Stasiun',
                   labels={'AQI': 'AQI', 'date': 'Tanggal', 'station': 'Stasiun'},
                   color='station',
                   color_discrete_sequence=px.colors.qualitative.Light24_r)

    fig2.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        width=1400,
        height=400
    )

    st.plotly_chart(fig2)

def main():
  
    min_date = pd.to_datetime(daily_data['date'].min())
    max_date = pd.to_datetime(daily_data['date'].max())

    default_start_date = pd.Timestamp(2016, 9, 1)
    default_end_date = pd.Timestamp(2017, 1, 1)
    st.subheader('Daily AQI')
    start_date_user = st.date_input("Pilih Tanggal Awal", min_value=min_date, max_value=max_date, value=default_start_date)
    end_date_user = st.date_input("Pilih Tanggal Akhir", min_value=min_date, max_value=max_date, value=default_end_date)

    if start_date_user and end_date_user:
        start_date_user = pd.to_datetime(start_date_user)
        end_date_user = pd.to_datetime(end_date_user)

        if st.button("Proses Data"):
            if end_date_user is not None:
                if start_date_user is not None and (end_date_user - start_date_user).days > 140:
                    st.error("Maaf, rentang tanggal maksimal adalah 4 bulan.")
                else:
                    show_filtered_graph(daily_data, start_date_user, end_date_user)
            else:
                st.info("Mohon pilih tanggal akhir untuk memproses.")
        else:
            # Menampilkan grafik default sebelum proses
            show_filtered_graph(daily_data, default_start_date, default_end_date)

if __name__ == "__main__":
    main()
    


import calendar
st.subheader('Yearly AQI')
# Filter data hanya untuk tahun yang diinputkan pengguna
selected_year = st.number_input("Masukkan Tahun", min_value=daily_data['date'].dt.year.min(), max_value=daily_data['date'].dt.year.max(), value=2013)

data_selected_year = daily_data[daily_data['date'].dt.year == selected_year]

# Daftar stasiun
stations = data_selected_year['station'].unique()

# Daftar bulan (konversi dari angka bulan menjadi nama bulan)
months = data_selected_year['date'].dt.month.unique()
month_names = [calendar.month_name[month] for month in months]

# Inisialisasi DataFrame untuk menyimpan rata-rata AQI per bulan
average_aqi_per_month = pd.DataFrame()

for station in stations:
    station_data = data_selected_year[data_selected_year['station'] == station]
    average_aqi = station_data.groupby(station_data['date'].dt.month)['AQI'].mean()
    average_aqi_per_month[station] = average_aqi

# Membuat DataFrame untuk plot
plot_data = average_aqi_per_month.reset_index().melt(id_vars='date', var_name='station', value_name='AQI')

# Menyesuaikan nama kolom bulan
plot_data['date'] = plot_data['date'].map({month: name for month, name in zip(months, month_names)})

# Plotly Express untuk perbandingan rata-rata AQI per bulan
fig1 = px.bar(plot_data, x='date', y='AQI',
              title=f'Perbandingan Rata-rata AQI per Bulan ({selected_year})',
              labels={'AQI': 'Rata-rata AQI', 'date': 'Bulan', 'station': 'Stasiun'},
              barmode='group',
              color='station',
              color_discrete_sequence=px.colors.qualitative.Light24_r
              )
fig1.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=700,  # Lebar grafik
    height=400  # Tinggi grafik
)

# Menghitung rata-rata AQI per stasiun untuk tahun yang diinputkan
average_aqi_per_station = data_selected_year.groupby('station')['AQI'].mean().reset_index()

# Urutkan stasiun berdasarkan rata-rata AQI
average_aqi_per_station = average_aqi_per_station.sort_values(by='AQI')

# Membuat plot dengan Plotly Express
fig3 = px.bar(
    average_aqi_per_station, 
    x='AQI', 
    y='station', 
    orientation='h',
    title=f'Rata-rata AQI per Stasiun ({selected_year})',
    labels={'AQI': 'Rata-rata AQI'},
    color='station',  # Mengatur warna berdasarkan stasiun
    color_discrete_sequence=px.colors.qualitative.Light24_r)  # Gunakan palet warna kualitatif


fig3.update_layout(
    margin=dict(l=50, r=50, t=50, b=50),  # Sesuaikan nilai ini untuk mengatur margin
    width=700,  # Lebar grafik
    height=400  # Tinggi grafik
)
 
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig1)
with col4:
    st.plotly_chart(fig3) 

