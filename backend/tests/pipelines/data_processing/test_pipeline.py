"""
Tests unitarios para el nodo de limpieza de datos (waymo_backend.pipelines.data_processing.nodes).

Cubren los casos reales del proyecto:
- Deduplicacion de registros.
- Imputacion por mediana cuando SI hay columnas de velocidad (speed_x/speed_y).
- Comportamiento correcto cuando NO hay columnas de velocidad (caso real de este
  dataset: la tabla LiDAR estatica de Waymo no trae speed_x/speed_y).
- Deteccion de anomalias (IQR) sobre box_length, garantizada incluso sin velocidad.
- Limite de muestreo a 1000 registros para el dataset del frontend.
"""
import numpy as np
import pandas as pd
import pytest

from waymo_backend.pipelines.data_processing.nodes import clean_and_process_data


def _dataset_base(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "object_type": rng.choice(["vehicle", "pedestrian", "cyclist", "sign"], n),
        "box_center_x": rng.uniform(-50, 50, n),
        "box_center_y": rng.uniform(-50, 50, n),
        "box_center_z": rng.uniform(0, 3, n),
        "box_length": rng.uniform(0.5, 5, n),
        "box_width": rng.uniform(0.5, 3, n),
        "box_height": rng.uniform(0.5, 2, n),
    })


def test_elimina_duplicados():
    df = _dataset_base(n=50)
    df_con_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)

    resultado = clean_and_process_data(df_con_dup)

    assert len(resultado["clean_data"]) <= len(df)


def test_sin_columnas_de_velocidad_no_crea_velocity_abs():
    df = _dataset_base(n=100)
    assert "speed_x" not in df.columns and "speed_y" not in df.columns

    resultado = clean_and_process_data(df)

    assert "velocity_abs" not in resultado["clean_data"].columns
    assert "velocity_abs" not in resultado["frontend_data"].columns


def test_imputa_nulos_de_velocidad_con_mediana_cuando_existen():
    df = _dataset_base(n=100)
    df["speed_x"] = np.random.default_rng(1).normal(5, 2, 100)
    df["speed_y"] = np.random.default_rng(2).normal(0, 1, 100)
    df.loc[0:4, "speed_x"] = np.nan
    df.loc[0:4, "speed_y"] = np.nan

    resultado = clean_and_process_data(df)
    df_clean = resultado["clean_data"]

    assert df_clean["speed_x"].isna().sum() == 0
    assert df_clean["speed_y"].isna().sum() == 0
    assert "velocity_abs" in df_clean.columns


def test_detecta_outliers_de_box_length_sin_velocidad():
    df = _dataset_base(n=300, seed=3)
    outliers = _dataset_base(n=10, seed=99).assign(box_length=lambda d: d["box_length"] + 100)
    df_con_outliers = pd.concat([df, outliers], ignore_index=True)

    resultado = clean_and_process_data(df_con_outliers)
    df_clean = resultado["clean_data"]

    assert df_clean["box_length"].max() < 50
    assert len(df_clean) < len(df_con_outliers)


def test_frontend_data_limita_a_1000_registros():
    df = _dataset_base(n=5000, seed=4)
    resultado = clean_and_process_data(df)

    assert len(resultado["frontend_data"]) == 1000


def test_frontend_data_no_incluye_columnas_ausentes():
    df = _dataset_base(n=50, seed=5)
    resultado = clean_and_process_data(df)

    columnas_prohibidas = {"speed_x", "speed_y", "velocity_abs"}
    assert columnas_prohibidas.isdisjoint(resultado["frontend_data"].columns)
