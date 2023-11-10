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


# In[2]:

st.title('Analisis Kualitas Udara di Kota Beijing 2013 - 2017')
st.write('Rafif Fauzan Almahdy | Data Analyst')

folder_path = 'dataset'
all_data = []  # List untuk menyimpan semua DataFrames

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)
    all_data.append(data)

# Menggabungkan semua DataFrames menjadi satu DataFrame tunggal
data = pd.concat(all_data, ignore_index=True)

data
st.caption('dataset kotor sebelum dilakukan analisa.')
# Deskripsi dataset
st.header('Deskripsi Dataset:')
st.write('Dataset ini berisi informasi tentang kualitas udara di Kota Beijing selama periode tahun 2013 hingga 2017. Data berasal dari berbagai stasiun udara di kota ini dan mencakup beberapa kolom kunci, termasuk:')
st.write('- **Date:** Tanggal pengukuran.')
st.write('- **Station:** Nama stasiun tempat pengukuran udara.')
st.write('- **PM2.5:** Konsentrasi partikel halus dengan diameter kurang dari 2,5 mikrometer dalam mikrogram per meter kubik (µg/m³).')
st.write('- **PM10:** Konsentrasi partikel halus dengan diameter kurang dari 10 mikrometer dalam µg/m³.')
st.write('- **SO2:** Konsentrasi sulfur dioksida dalam µg/m³.')
st.write('- **NO2:** Konsentrasi nitrogen dioksida dalam µg/m³.')
st.write('- **CO:** Konsentrasi karbon monoksida dalam mg/m³.')
st.write('- **O3:** Konsentrasi ozon dalam µg/m³.')
# Tambahkan caption

st.header('Pertanyaan/Hipotesis:')
st.write('- Apakah ada tren peningkatan atau penurunan dalam kualitas udara selama periode 2013-2017?')
st.write('- Bagaimana korelasi antara parameter kualitas udara seperti PM2.5, PM10, SO2, NO2, CO, dan O3?')
st.markdown('---')
# In[3]:
data.drop('No', axis=1, inplace=True)
data.describe()
# In[4]:
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
# In[6]:
# Deskripsi Data Wrangling
st.header('Data Wrangling')

# Menampilkan data.describe()

st.write(data.describe())

st.write('Data wrangling adalah langkah awal dalam analisis data, di mana kita membersihkan, mengubah, dan mempersiapkan data '
         'sebelum memulai analisis yang lebih mendalam. Dalam tahap ini, kita melakukan beberapa tindakan seperti menghapus kolom '
         'yang tidak relevan, menggabungkan kolom, dan merapihkan data.')

# Menghapus Kolom 'No'
st.write('- Pada langkah pertama, kita menghapus kolom "No", yang berisi nomor baris atau identifikasi yang tidak relevan.')

# Menggabungkan Kolom Tanggal dan Waktu
st.write('- Selanjutnya, kita menggabungkan kolom "year", "month", "day", dan "hour" menjadi satu kolom tunggal "date". Ini membantu kita dalam menganalisis data berdasarkan tanggal dan waktu dengan lebih mudah.')
st.write('- Menghapus kolom-kolom lain yang mengandung data kosong (NaN). Dengan menghilangkan data kosong, kita memastikan bahwa data yang kita gunakan dalam analisis adalah data yang valid.')

data.info()


# In[7]:


data['date'] = pd.to_datetime(data[['year', 'month', 'day']])


# In[8]:


date_column = data.pop('date')

# Memasukkan kolom 'date' ke posisi pertama
data.insert(0, 'date', date_column)




# In[10]:


columns_to_drop = ['year', 'month', 'day']
data.drop(columns=columns_to_drop, inplace=True)


# In[11]:


data['CO'] = data['CO'] / 100

# In[12]:


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




# In[13]:

st.write('- Kemudian dilakukan penyederhanaan data dengan melakukan rata - rata harian dari data perjam')

# Menghitung rata-rata harian dari data per jam
daily_data = data.groupby(['date', 'station']).agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean',
    'TEMP': 'mean',
    'PRES': 'mean',
    'DEWP': 'mean', 
    'RAIN': 'mean', 
    'WSPM': 'mean'
}).reset_index()

# Menampilkan hasil

daily_data = daily_data.round(2)
daily_data.set_index('date')


# In[14]:


data.describe()

daily_data
# In[15]:


numerical_data = daily_data.select_dtypes(include='number')

# Menghitung korelasi antara kolom-kolom numerik
correlation_matrix = numerical_data.corr()

# Mengatur ukuran plot
plt.figure(figsize=(12, 8))

# Membuat heatmap korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='crest', linewidths=0.5)


# Streamlit app
st.header("Analisis Korelasi")

 
# Display the heatmap using Streamlit
st.pyplot(plt)

st.write("Analisis korelasi adalah langkah penting dalam memahami hubungan antara parameter kualitas udara. Hasil korelasi ini membantu kami menentukan sejauh mana satu parameter berkaitan dengan yang lainnya. Korelasi positif menunjukkan bahwa jika satu parameter meningkat, kemungkinan lainnya juga akan meningkat. Sebaliknya, korelasi negatif menunjukkan bahwa jika satu parameter meningkat, kemungkinan lainnya akan menurun. Hasil analisis ini membantu kami dalam pemahaman lebih mendalam tentang faktor-faktor apa yang berkontribusi terhadap kualitas udara di Beijing.")
daily_data = data.groupby(['date', 'station']).agg({
    'PM2.5': 'mean',
    'PM10': 'mean',
    'SO2': 'mean',
    'NO2': 'mean',
    'CO': 'mean',
    'O3': 'mean',
}).reset_index()
# In[16]:
st.header("Perhitungan Nilai AQI")
st.write("Perhitungan Indeks Kualitas Udara (AQI) adalah cara untuk menggambarkan tingkat polusi udara dengan angka yang mudah dimengerti. AQI dihitung berdasarkan konsentrasi berbagai polutan udara seperti PM2.5, PM10, SO2, NO2, CO, dan O3.")

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

daily_data

st.markdown("------")
# In[17]:

st.header('Visualisasi Data')
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


# In[18]:


daily_data.max()


# In[19]:


# Filter data hanya untuk tahun 2016
data_2016 = daily_data[daily_data['date'].dt.year == 2016]

# Daftar stasiun
stasiun = data_2016['station'].unique()

# Daftar bulan
bulan = data_2016['date'].dt.strftime('%b').unique()

# Warna untuk setiap stasiun
colors = sns.color_palette("Spectral", len(stasiun))

# Lebar setiap bar
bar_width = 0.18

# Inisialisasi DataFrame untuk menyimpan rata-rata AQI per bulan
rata_rata_aqi_per_bulan = pd.DataFrame()

for s in stasiun:
    data_stasiun = data_2016[data_2016['station'] == s]
    rata_rata_aqi = data_stasiun.groupby(data_stasiun['date'].dt.month)['AQI'].mean()
    rata_rata_aqi_per_bulan[s] = rata_rata_aqi

# Streamlit app
st.subheader('Perbandingan Rata-rata AQI per Bulan (Tahun 2016)')
st.write('Perbandingan rata-rata AQI per bulan untuk stasiun-stasiun berbeda dalam tahun 2016.')

# Membuat plot dengan Matplotlib
plt.figure(figsize=(15, 5))
sns.set(style="whitegrid", rc={"grid.linewidth": 0.5})

for i, s in enumerate(stasiun):
    x = np.arange(len(bulan)) + i * bar_width
    plt.bar(x, rata_rata_aqi_per_bulan[s], bar_width, label=s, color=colors[i])

plt.xlabel('Bulan')
plt.ylabel('Rata-rata AQI')
plt.xticks(np.arange(len(bulan)) + 0.3, bulan)
plt.legend(fontsize='large', loc='upper center', ncol=len(stasiun), bbox_to_anchor=(0.5, 1.17))

# Menampilkan plot dengan Streamlit
st.pyplot(plt)


# In[20]:

# Filter data hanya untuk 3 bulan terakhir tahun 2016
data_last_3_months = daily_data[(daily_data['date'] >= '2016-08-01') & (daily_data['date'] <= '2016-11-30')]

# Daftar stasiun
stasiun = data_last_3_months['station'].unique()

# Warna untuk setiap stasiun
colors = sns.color_palette("Spectral", len(stasiun))

# Streamlit app
st.subheader('Perbandingan AQI Harian per Stasiun (Agustus - Desember 2016)')
st.write('Perbandingan AQI harian per stasiun untuk periode Agustus - Desember 2016.')

# Membuat plot dengan Matplotlib
plt.figure(figsize=(20, 5))
sns.set(style="whitegrid", rc={"grid.linewidth": 0.5})

for i, s in enumerate(stasiun):
    data_stasiun = data_last_3_months[data_last_3_months['station'] == s]
    plt.plot(data_stasiun['date'], data_stasiun['AQI'], label=s, color=colors[i], linewidth=3)

plt.xlabel('Tanggal')
plt.ylabel('AQI')
plt.legend(fontsize='x-large')

# Mengatur interval label bulan
months = MonthLocator(interval=1)
plt.gca().xaxis.set_major_locator(months)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Menampilkan plot dengan Streamlit
st.pyplot(plt)


# In[21]:

# Filter data hanya untuk tahun 2016-2017
data_2016_2017 = daily_data[(daily_data['date'] >= '2016-01-01') & (daily_data['date'] <= '2017-12-31')]

# Menghitung rata-rata AQI per stasiun
average_aqi_per_station = data_2016_2017.groupby('station')['AQI'].mean().reset_index()

# Temukan stasiun dengan AQI tertinggi
stasiun_dengan_aqi_tertinggi = average_aqi_per_station[average_aqi_per_station['AQI'] == average_aqi_per_station['AQI'].max()]

# Temukan stasiun dengan AQI terendah
stasiun_dengan_aqi_terendah = average_aqi_per_station[average_aqi_per_station['AQI'] == average_aqi_per_station['AQI'].min()]


# In[22]:


# Filter data hanya untuk tahun 2016-2017
data_2016_2017 = daily_data[(daily_data['date'] >= '2016-01-01') & (daily_data['date'] <= '2017-12-31')]

# Menghitung rata-rata AQI per stasiun
average_aqi_per_station = data_2016_2017.groupby('station')['AQI'].mean().reset_index()

# Urutkan stasiun berdasarkan rata-rata AQI
average_aqi_per_station = average_aqi_per_station.sort_values(by='AQI')

# Streamlit app
st.subheader('Rata-rata AQI per Stasiun (2016-2017)')
st.write('Perbandingan rata-rata AQI per stasiun untuk periode tahun 2016-2017.')

# Membuat plot dengan Seaborn
plt.figure(figsize=(12, 4))
sns.set(style="whitegrid", rc={"grid.linewidth": 0.4})

# Menggunakan palet warna Seaborn yang sama
colors = sns.color_palette("Spectral")
colors = list(reversed(sns.color_palette("Spectral")))

# Membuat bar chart
ax = sns.barplot(x='AQI', y='station', data=average_aqi_per_station, palette=colors)

# Menambahkan nilai di atas setiap batang
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 7, p.get_y() + p.get_height() / 2, f'{width:.2f}', ha='center', va='center', fontsize=10)

# Menampilkan plot dengan Streamlit
st.pyplot(plt)


# In[23]:

# Data AQI (disesuaikan dengan data Anda)
data_2016_2017 = daily_data[(daily_data['date'] >= '2014-01-01') & (daily_data['date'] <= '2017-12-31')]
aqi_data = data_2016_2017['AQI']

# Tentukan kategori AQI berdasarkan rentang tertentu
categories = ['Baik', 'Sedang', 'Kurang Sehat', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Beracun']
category_ranges = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, float('inf'))]

# Hitung jumlah data dalam masing-masing grup
category_counts = [sum(1 for aqi in aqi_data if aqi >= range_min and aqi <= range_max) for range_min, range_max in category_ranges]

# Hitung persentase masing-masing grup
total_data = len(aqi_data)
category_percentages = [count / total_data * 100 for count in category_counts]

# Streamlit app
st.subheader('Persentase Kategori AQI (Setengah Lingkaran)')
st.write('Diagram setengah lingkaran menampilkan persentase kategori AQI.')

# Membuat diagram setengah lingkaran dengan Seaborn style
plt.figure(figsize=(5, 5))
plt.title('Persentase Kategori AQI (Setengah Lingkaran)')

# Buat palet warna Seaborn
colors = sns.color_palette("Spectral")
colors = list(reversed(sns.color_palette("Spectral")))

# Tambahkan label persentase yang lebih besar
plt.pie(
    category_percentages, 
    labels=[f'{cat}\n{perc:.1f}%' for cat, perc in zip(categories, category_percentages)], 
    colors=colors, 
    startangle=90, 
    counterclock=False, 
    radius=1.0
)

# Buat lingkaran kecil di atasnya (sebagai pusat)
small_circle = plt.Circle((0, 0), 0.45, color='white')
plt.gca().add_artist(small_circle)

# Menampilkan diagram setengah lingkaran dengan Streamlit
st.pyplot(plt)


# In[24]:

# Filter data hanya untuk tahun 2013-2017
data_2013_2017 = daily_data[(daily_data['date'] >= '2013-01-01') & (daily_data['date'] <= '2017-12-31')]

# Extract the year from the 'date' column
data_2013_2017['year'] = data_2013_2017['date'].dt.year

# Calculate the average AQI per year
average_aqi_per_year = data_2013_2017.groupby('year')['AQI'].mean().reset_index()

# Streamlit app
st.subheader('Perbandingan Rata-rata AQI per Tahun (2013-2017)')
st.write('Diagram batang menampilkan rata-rata AQI per tahun.')

# Set the color palette to "Spectral"
colors = sns.color_palette("Spectral")

# Membuat diagram batang
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid", rc={"grid.linewidth": 0.4})

# Membuat subplot
ax = sns.barplot(x='year', y='AQI', data=average_aqi_per_year, palette=colors, errcolor=None)

plt.xlabel('Tahun')
plt.ylabel('Rata-rata AQI')
plt.title('Perbandingan Rata-rata AQI per Tahun (2013-2017)')

# Menambahkan nilai di atas setiap bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), textcoords='offset points')

# Menambahkan jarak antara label sumbu y dan grafik
plt.subplots_adjust(left=0.1)
plt.xticks(rotation=45)
plt.yticks(range(0, 276, 25))
# Menampilkan diagram batang dengan Streamlit
st.pyplot(plt)


# In[25]:


daily_data['year'] = daily_data['date'].dt.year
daily_data['month'] = daily_data['date'].dt.month

# Menghitung rata-rata AQI per bulan
monthly_aqi = daily_data.groupby(['year', 'month'])['AQI'].mean().reset_index()

# Menggabungkan kolom "year" dan "month" menjadi kolom "date"
monthly_aqi['date'] = monthly_aqi['year'].astype(str) + '-' + monthly_aqi['month'].astype(str)

# Mengubah kolom "date" menjadi format tanggal
monthly_aqi['date'] = pd.to_datetime(monthly_aqi['date'], format='%Y-%m')

# Sort data berdasarkan tanggal
monthly_aqi = monthly_aqi.sort_values('date')

# Set the color palette to "Spectral"
colors = sns.color_palette("Spectral")
st.markdown("-----")
# Judul
st.header("Kesimpulan")

# Paragraf 1
st.write("Air Quality Index (AQI) merupakan indikator yang terbentuk dari variabel kualitas udara seperti PM2.5, PM10, SO2, NO2, dan O3. Hubungan yang erat dan korelasi yang cukup tinggi antara variabel-variabel ini memberikan dasar bagi perhitungan AQI sebagai metode untuk mengukur kualitas udara secara komprehensif. Penggunaan AQI membantu dalam pemahaman lebih baik tentang sejauh mana parameter-parameter ini berkaitan dan bagaimana pengaruh mereka terhadap kualitas udara secara keseluruhan.")

# Paragraf 2
st.write("Namun, hasil analisis menunjukkan bahwa selama periode 2013-2017, kualitas udara di Beijing menunjukkan tren penurunan secara umum. Meskipun ada fluktuasi tahunan dan tingkat polusi tertinggi pada tahun 2014, nilai rata-rata AQI tetap di atas 180, yang merupakan indikator 'tidak sehat'. Ini menunjukkan bahwa masalah polusi udara masih menjadi isu serius di kota ini dan perlu adanya upaya yang lebih besar untuk meningkatkan kualitas udara di masa depan.")






