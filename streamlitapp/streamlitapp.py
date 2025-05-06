import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore')

# Заголовок приложения
st.title('Анализ и прогнозирование цен на драгоценные металлы')

# Загрузка данных
@st.cache_data
def load_data():
    # Чтение данных из CSV файла
    data = pd.read_csv(r'streamlitapp/metals.csv')
    # Преобразование столбца Date в datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # Установка Date в качестве индекса
    data.set_index('Date', inplace=True)
    # Сортировка по дате (на случай, если данные не упорядочены)
    data.sort_index(inplace=True)
    return data

try:
    df = load_data()
    
    # Проверка данных
    if df.empty:
        st.error("Данные не загружены. Проверьте файл metals.csv")
        st.stop()
    
    # Выбор металла
    st.sidebar.header('Настройки')
    metal = st.sidebar.selectbox('Выберите металл', df.columns)
    days_to_predict = st.sidebar.slider('Количество дней для прогноза', 1, 30, 7)
    
    # Отображение исторических данных
    st.subheader(f'Исторические данные цены на {metal}')
    st.line_chart(df[metal])
    
    # Прогнозирование с помощью ARIMA
    st.subheader(f'Прогноз цены на {metal} на {days_to_predict} дней')
    
    try:
        # Подготовка данных для модели
        series = df[metal].dropna()
        
        if len(series) < 10:
            st.error("Недостаточно данных для прогнозирования")
            st.stop()
        
        # Обучение модели ARIMA
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        
        # Прогнозирование
        forecast = model_fit.forecast(steps=days_to_predict)
        
        # Создание дат для прогноза
        last_date = series.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series.index, series, label='Исторические данные')
        ax.plot(forecast_dates, forecast, label='Прогноз', color='red', linestyle='--')
        ax.legend()
        ax.set_title(f'Прогноз цены на {metal}')
        ax.grid(True)
        st.pyplot(fig)
        
        # Отображение прогноза в таблице
        forecast_df = pd.DataFrame({
            'Дата': forecast_dates,
            'Прогнозируемая цена': forecast.round(2)
        })
        st.table(forecast_df)
        
        # Отображение последних исторических данных для сравнения
        st.subheader('Последние исторические данные')
        st.table(df.tail(5)[[metal]])
        
    except Exception as e:
        st.error(f'Ошибка при прогнозировании: {str(e)}')
        st.stop()
    
except Exception as e:
    st.error(f'Ошибка при загрузке данных: {str(e)}')
    st.stop()

# Дополнительная информация
st.sidebar.markdown('''
### Инструкция:
1. Выберите металл из списка
2. Укажите количество дней для прогноза
3. Наблюдайте график и прогноз

Автор:
Дмитрий Сауков
''')
