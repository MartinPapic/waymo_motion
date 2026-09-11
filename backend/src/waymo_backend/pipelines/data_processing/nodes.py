import pandas as pd
import numpy as np

def clean_and_process_data(df: pd.DataFrame) -> dict:
    """
    Realiza la limpieza básica de datos (Silver layer):
    - Elimina duplicados
    - Trata valores nulos (speed_x/speed_y, si el dataset trae capa de movimiento)
    - Elimina outliers (Regla IQR) en velocidad si está disponible, y siempre en
      box_length (dimensión física garantizada en la tabla LiDAR estática)
    
    Retorna un diccionario con dos DataFrames:
    - uno completo para analítica (Parquet)
    - uno reducido/optimizado para el frontend (JSON)
    """
    # 1. Manejo de duplicados
    df_clean = df.drop_duplicates().copy()
    
    # 2. Manejo de nulos (Imputación con la Mediana)
    # Rellenar con 0 alteraría la distribución y sesgaría los cuartiles.
    # Usar la mediana es más robusto para los algoritmos estadísticos de ML.
    if 'speed_x' in df_clean.columns:
        mediana_x = df_clean['speed_x'].median()
        df_clean['speed_x'] = df_clean['speed_x'].fillna(mediana_x)
    if 'speed_y' in df_clean.columns:
        mediana_y = df_clean['speed_y'].median()
        df_clean['speed_y'] = df_clean['speed_y'].fillna(mediana_y)
        
    # Calcular velocidad absoluta si no existe
    if 'speed_x' in df_clean.columns and 'speed_y' in df_clean.columns:
        df_clean['velocity_abs'] = np.sqrt(df_clean['speed_x']**2 + df_clean['speed_y']**2)
    
    # 3. Detección de Outliers (Filtro IQR)
    # 3a. Sobre velocity_abs, SOLO si el dataset trae capa de movimiento (speed_x/speed_y).
    if 'velocity_abs' in df_clean.columns:
        Q1 = df_clean['velocity_abs'].quantile(0.25)
        Q3 = df_clean['velocity_abs'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Filtrar outliers
        df_clean = df_clean[(df_clean['velocity_abs'] >= limite_inferior) & (df_clean['velocity_abs'] <= limite_superior)]

    # 3b. Sobre box_length, SIEMPRE disponible en la tabla estática de cajas LiDAR.
    # Cubre el caso (real, en este proyecto) en que el dataset descargado no trae velocidades:
    # sin esto, el pipeline nunca detectaría ninguna anomalía.
    if 'box_length' in df_clean.columns:
        Q1 = df_clean['box_length'].quantile(0.25)
        Q3 = df_clean['box_length'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean['box_length'] >= limite_inferior) & (df_clean['box_length'] <= limite_superior)]

    # 4. Preparar versión ligera para el Frontend (JSON)
    # Filtraremos solo a Vehículos y Peatones, y tomaremos columnas esenciales
    columnas_frontend = ['object_type', 'box_center_x', 'box_center_y', 'box_center_z', 'box_length', 'box_width', 'box_height', 'speed_x', 'speed_y', 'velocity_abs']
    columnas_presentes = [col for col in columnas_frontend if col in df_clean.columns]
    
    df_frontend = df_clean[columnas_presentes].copy()
    
    # Opcional: reducir la muestra para no colapsar el navegador (ej. 1000 registros aleatorios)
    if len(df_frontend) > 1000:
        df_frontend = df_frontend.sample(1000, random_state=42)
        
    return {
        "clean_data": df_clean,
        "frontend_data": df_frontend
    }
