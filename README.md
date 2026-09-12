# Waymo Motion - Machine Learning EP1

**Equipo:** Martín Papic y Matías Retamal

## Descripción del Proyecto
Este proyecto aplica la metodología **CRISP-DM** para la ingesta y análisis exploratorio del *Waymo Open Dataset (Motion / Perception)*. Incluye la orquestación de datos mediante **Kedro** (Arquitectura Medallón), análisis estadístico con detección de sesgos éticos, y un **Dashboard Interactivo en React** para visualizaciones en 3D.

> 📄 **El informe técnico del proyecto está en [INFORME_TECNICO.md](INFORME_TECNICO.md)**: problema de negocio, objetivos, KPIs, fuentes de datos, EDA, metodología CRISP-DM y auditoría ética.
>
> Este README es la guía de instalación y ejecución.

### Entregables

| Documento | Contenido |
|---|---|
| [INFORME_TECNICO.md](INFORME_TECNICO.md) | Informe técnico completo del proyecto |
| [backend/notebooks/EP1_Memoria_Waymo_Motion.ipynb](backend/notebooks/EP1_Memoria_Waymo_Motion.ipynb) | Memoria ejecutable: reproduce cada etapa del informe |
| [backend/src/waymo_backend/](backend/src/waymo_backend/) | Pipeline de datos en Kedro |
| [frontend/](frontend/) | Dashboard interactivo de presentación |

---

## Requisitos Previos (Prerequisites)
1. **Python 3.10 o superior**
2. **Node.js y npm** (Para ejecutar el frontend interactivo)
3. **Google Cloud SDK** (Obligatorio para interactuar con los buckets del dataset de Waymo)

---

## 1. Instalación y Autenticación en Google Cloud
El dataset de Waymo reside de forma nativa en Google Cloud Storage. Necesitas credenciales de tu cuenta de Google.

1. Descarga e instala [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) para tu sistema operativo.
2. Abre tu terminal y ejecuta el siguiente comando para loguearte:
   ```bash
   gcloud auth application-default login
   ```
3. Acepta los permisos en la ventana del navegador que se abrirá automáticamente. Esto le dará permiso a los scripts de Python para descargar los datos.

---

## 2. Configuración del Backend (Python y Kedro)
El ecosistema backend se encarga del proceso ETL: Ingesta efímera, eliminación de duplicados, imputación robusta de nulos (Mediana) y filtro de anomalías con la regla del Rango Intercuartílico (IQR).

1. **Crear y activar un entorno virtual (Recomendado):**
   ```bash
   python -m venv env
   # En Windows:
   .\env\Scripts\activate
   # En Mac/Linux:
   source env/bin/activate
   ```
2. **Instalar dependencias completas:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   *(El archivo `requirements.txt` incluye explícitamente `kedro`, `pandas`, `pyarrow`, `seaborn` y las librerías de GCP).*

3. **Ejecución rápida, sin credenciales (recomendado para revisar el proyecto):**
   El repositorio incluye una muestra de 12 segmentos y 103.430 detecciones (4,3 MB), suficiente para correr todo el pipeline al clonar:
   ```bash
   cd backend
   kedro run --env muestra
   ```

4. **Descargar el dataset completo (opcional, 2,4 millones de detecciones):**
   Requiere el login de Google Cloud del paso 1, con una cuenta que haya aceptado la licencia de Waymo.
   ```bash
   cd backend
   python scripts/descargar_waymo.py --lote 200
   ```
   *El script deja `detecciones_reales.parquet` en `backend/datos/waymo_real/`. Cópialo a `backend/data/01_raw/` y luego ejecuta el pipeline:*
   ```bash
   kedro run
   ```
   *Este comando limpia los millones de datos crudos y bifurca el resultado en `clean_waymo_data.parquet` (Capa Intermediate para Machine Learning) y exporta `waymo_frontend.json` (Capa Primary).*

---

## 3. Ejecución del Cuaderno de Jupyter (Memoria CRISP-DM)
Para revisar el Análisis Exploratorio de Datos (EDA) que exige la pauta (Distribución de clases, Matriz de Correlación térmica, y la Auditoría Ética):
```bash
cd backend
jupyter notebook notebooks/EP1_Memoria_Waymo_Motion.ipynb
```

---

## 4. Ejecución del Frontend (Dashboard en React)
Levanta la interfaz gráfica para visualizar las métricas clave del EDA generadas por el pipeline.

1. **Asegurar inyección de datos:**
   Asegúrate de que el archivo `waymo_frontend.json` (generado por `kedro run`) esté copiado dentro de la carpeta `frontend/src/`.
2. **Instalar dependencias web:**
   ```bash
   cd frontend
   npm install
   ```
3. **Iniciar el servidor de desarrollo:**
   ```bash
   npm run dev
   ```
4. Abre tu navegador web en `http://localhost:5173`. Podrás interactuar con la Presentación tipo Carousel usando las flechas de tu teclado, analizando el Sesgo en el gráfico de Barras, BoxPlots de tamaño, el Gráfico de Dispersión 2D (Clústers separables), y la impresionante Nube de Puntos Espacial 3D desde la perspectiva del Ego-Vehicle.

---

## Tecnologías y Arquitectura
- **Ingeniería de Datos:** Kedro, PyArrow (Parquet), Pandas, Google Cloud Storage.
- **Data Science:** Scikit-Learn, Seaborn, Matplotlib, Imbalanced-Learn (SMOTE).
- **Experiencia de Usuario (UX):** React, Vite, Recharts, Plotly.js.
