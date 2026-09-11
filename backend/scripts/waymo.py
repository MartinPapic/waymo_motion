"""Descarga de componentes del Waymo Open Dataset, en Colab o en local.

Existe porque **las credenciales funcionan distinto en cada entorno**:

- En **Google Colab**, ``auth.authenticate_user()`` autentica a Python pero
  **no** al CLI: ``gsutil`` responde *"You are attempting to access protected
  data with no configured credentials"*. Hay que usar el cliente Python
  ``google.cloud.storage``.
- En **local**, ``gcloud auth login`` habilita ``gsutil``. Si corriste
  ``gcloud auth application-default login`` (ADC), el cliente Python
  ``google.cloud.storage`` autentica aunque ``gcloud auth list`` esté vacío
  **o** liste una cuenta con el token caducado. ADC gana: no se pide
  ``gcloud auth login`` de nuevo.

El curso usa Perception **v2** (parquet). ``CATALOGO_BUCKETS`` documenta también
Motion, End-to-End camera y Perception v1.4.3: se listan y se baja un objeto
chico, nunca el bucket entero.

Poner esto en un módulo y no en una celda del notebook tiene una razón: así se
puede probar. ``tests/test_waymo_descarga.py`` verifica la lógica de selección
de entorno sin necesidad de credenciales.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "waymo_open_dataset_v_2_0_1"
SPLIT = "training"
COMPONENTES_LIVIANOS = ("lidar_box", "stats")
# Tablas parquet de un segmento (KB–cientos de KB). El alumno las abre en pandas;
# no entran al Random Forest. JPEG / nubes / tfrecord no están aquí.
COMPONENTES_MANIPULABLES = ("camera_box", "vehicle_pose", "camera_calibration")
TAMANO_MAXIMO_MB = 50.0
TAMANO_MAXIMO_CLASE_MB = 250.0
LOTE_CLASE = 8
PAGINA_DESCARGA = "https://waymo.com/open/download/"
FAQ_WAYMO = "https://waymo.com/open/faq/"

# Lo que hace Waymo en sus propios tutoriales: 2 frames de muestra, no el bucket.
# No copiamos esos binarios (licencia). El alumno abre el Colab oficial.
TUTORIALES_OFICIALES: dict[str, dict[str, str]] = {
    "percepcion_dos_frames": {
        "colab": (
            "https://colab.research.google.com/github/waymo-research/"
            "waymo-open-dataset/blob/master/tutorial/tutorial.ipynb"
        ),
        "que": (
            "FAQ de Waymo: el tutorial usa frames de muestra, no el dataset. "
            "Ahí sí se ven JPEG y cajas. Son 2 fotogramas, no 1,5 GB."
        ),
    },
    "percepcion_v2": {
        "colab": (
            "https://colab.research.google.com/github/waymo-research/"
            "waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb"
        ),
        "que": "Parquet modular (el mismo formato que el lote de clase).",
    },
    "motion": {
        "colab": (
            "https://colab.research.google.com/github/waymo-research/"
            "waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb"
        ),
        "que": "Hay un ejemplo en el tutorial. Un shard real de Motion sigue siendo ~1 GB.",
    },
    "e2e": {
        "colab": (
            "https://colab.research.google.com/github/waymo-research/"
            "waymo-open-dataset/blob/master/tutorial/"
            "tutorial_vision_based_e2e_driving.ipynb"
        ),
        "que": "Pide un tfrecord E2E (~1,6 GB). En clase usamos el JSON de 479 clusters.",
    },
}

# FAQ oficial (2026-09-08): "the tutorial currently uses some sample frames —
# it does not access the actual dataset files." No copiamos esos binarios.
MUESTRAS_OFICIALES_GITHUB = (
    "https://github.com/waymo-research/waymo-open-dataset/tree/master/tutorial"
)
# Carpeta tutorial/ del repo oficial (leída 2026-09-09). No copiar los .ipynb.
TUTORIALES_REPO_OFICIAL: tuple[str, ...] = (
    "tutorial.ipynb",
    "tutorial_v2.ipynb",
    "tutorial_local.ipynb",
    "tutorial_motion.ipynb",
    "tutorial_vision_based_e2e_driving.ipynb",
    "tutorial_camera_only.ipynb",
    "tutorial_keypoints.ipynb",
    "tutorial_maps.ipynb",
    "tutorial_2d_pvps.ipynb",
    "tutorial_3d_semseg.ipynb",
    "tutorial_object_asset.ipynb",
    "tutorial_occupancy_flow.ipynb",
    "tutorial_sim_agents.ipynb",
    "tutorial_scenario_gen.ipynb",
    "tutorial_womd_camera.ipynb",
    "tutorial_womd_lidar.ipynb",
)
TUTORIALES_EN_CLASE: frozenset[str] = frozenset(
    {
        "tutorial.ipynb",
        "tutorial_v2.ipynb",
        "tutorial_motion.ipynb",
        "tutorial_vision_based_e2e_driving.ipynb",
    }
)
VIEWER_PARQUET_V2 = {
    "url": "https://egolens.org",
    "repo": "https://github.com/egolens/egolens",
    "que": (
        "EgoLens (curso OMSCS CS 7638; antes waymo-perception-studio): "
        "arrastras parquet v2 en el navegador. Para ver JPEG hace falta "
        "camera_image (~320 MB). En clase dibujamos un frame con camera_box "
        "sobre el lienzo de calibración."
    ),
}

# Qué hay en un segmento de Perception v2 y qué baja el curso.
# Pesos: un segmento GCS (10017090168044687777_…), 2026-09-08. camera_box es tabla 2D.
COMPONENTES_V2: dict[str, dict[str, str]] = {
    "lidar_box": {
        "uso": "curso",
        "peso": "0,25–0,95 MB",
        "que": "cajas 3D de cada detección LiDAR (tabla del RF)",
    },
    "stats": {
        "uso": "curso",
        "peso": "0,02 MB",
        "que": "clima, hora y ciudad del segmento",
    },
    "camera_box": {
        "uso": "opcional",
        "peso": "0,08–0,28 MB",
        "que": "cajas 2D sobre la foto: sigue siendo tabla, no es la imagen",
    },
    "camera_calibration": {
        "uso": "opcional",
        "peso": "0,01 MB",
        "que": "intrínsecos de cámara (tabla); se abre, no entra al RF",
    },
    "vehicle_pose": {
        "uso": "opcional",
        "peso": "0,04 MB",
        "que": "pose del vehículo (tabla); trayectoria x/y, no entra al RF",
    },
    "camera_image": {
        "uso": "no",
        "peso": "~320 MB por segmento",
        "que": "JPEG de las cámaras: no se baja en clase",
    },
    "lidar": {
        "uso": "no",
        "peso": "~165 MB por segmento",
        "que": "nube de puntos: no se baja en clase",
    },
    "lidar_camera_projection": {
        "uso": "no",
        "peso": "~71 MB",
        "que": "proyección LiDAR→cámara: pasa el tope de 50 MB/archivo",
    },
}

# Los cuatro productos de https://waymo.com/open/download/ en GCS.
# El curso usa Perception v2 (parquet, componentes livianos). Los otros tres
# son tfrecord: se listan y, si cabe, se baja un objeto chico. Nunca el bucket.
CATALOGO_BUCKETS: dict[str, dict[str, str]] = {
    "percepcion_v2": {
        "bucket": "waymo_open_dataset_v_2_0_1",
        "formato": "parquet",
        "prefijo_muestra": "training/lidar_box/",
        "consola": "https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_2_0_1",
        "pagina": PAGINA_DESCARGA,
        "para_que": (
            "Perception v2.0.1 (modular, sin mapas): el hilo del curso. "
            "Un segmento de lidar_box (0,25–0,95 MB) + stats (0,02 MB) alcanza."
        ),
        "en_clase": "tabla",
        "tamano_medido": "lidar_box 0,25–0,95 MB · stats 0,02 MB (GCS 2026-09-08)",
        "sustituto": "Esta SÍ es la tabla del curso. No hace falta otra.",
    },
    "percepcion_v1": {
        "bucket": "waymo_open_dataset_v_1_4_3",
        "formato": "tfrecord",
        "prefijo_muestra": "individual_files/training/",
        "consola": "https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_3",
        "pagina": PAGINA_DESCARGA,
        "para_que": (
            "Perception v1.4.3 (con mapas): un Frame protobuf por registro, "
            "imágenes y LiDAR pegados. Cada tfrecord mide 894–1.062 MB; el curso usa v2."
        ),
        "en_clase": "listar",
        "tamano_medido": "tfrecord 894–1.062 MB por segmento (GCS 2026-09-08)",
        "sustituto": (
            "No se abre el video ni el Frame protobuf. Perception v2 ya trae las "
            "cajas 3D en parquet (~1 MB)."
        ),
    },
    "motion": {
        "bucket": "waymo_open_dataset_motion_v_1_3_1",
        "formato": "tfrecord",
        "prefijo_muestra": "uncompressed/tf_example/training/",
        "consola": "https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_1",
        "pagina": PAGINA_DESCARGA,
        "para_que": (
            "Motion v1.3.1: trayectorias a 9 s y mapa. Un shard tf_example mide "
            "1,17–1,32 GB (1000 shards); scenario 434–480 MB. En clase solo se listan."
        ),
        "en_clase": "listar",
        "tamano_medido": "tf_example 1,17–1,32 GB · scenario 434–480 MB (GCS 2026-09-08)",
        "sustituto": (
            "No hay trayectorias a 9 s en este curso. Lo más parecido que cabe: "
            "vehicle_pose (x/y del auto, ~40 KB) y speed_mps en la tabla v2."
        ),
    },
    "e2e_camara": {
        "bucket": "waymo_open_dataset_end_to_end_camera_v_1_0_0",
        "formato": "tfrecord",
        "prefijo_muestra": "val_sequence_name_to_scenario_cluster.json",
        "consola": (
            "https://console.cloud.google.com/storage/browser/"
            "waymo_open_dataset_end_to_end_camera_v_1_0_0"
        ),
        "pagina": PAGINA_DESCARGA,
        "para_que": (
            "E2E cámara v1.0.0: los tfrecord miden 1,59–1,68 GB. El JSON de metadatos "
            "(0,03 MB, 479 clusters) sí cabe; no hay carpeta val_sequence/."
        ),
        "en_clase": "json",
        "tamano_medido": "JSON 0,03 MB · tfrecord 1,59–1,68 GB (GCS 2026-09-08)",
        "sustituto": (
            "El video no se baja (el más chico mide 1,56 GB). En su lugar: JSON de "
            "479 clusters + camera_box (qué hay en el cuadro, sin JPEG)."
        ),
    },
}

_MENSAJE_403 = (
    "Google Cloud respondió 403: la cuenta con la que está abierto este entorno no "
    "tiene acceso al Waymo Open Dataset.\n\n"
    "  Causa habitual: Colab está abierto con una cuenta de Google distinta de la que "
    "aceptó los términos en https://waymo.com/open/download/\n\n"
    "  Solución: cambia de cuenta en Colab (avatar arriba a la derecha) y usa la misma "
    "con la que te registraste en Waymo, o acepta los términos con esta cuenta.\n\n"
    "  Mensaje original: {error}"
)


def en_colab() -> bool:
    """True si el código se está ejecutando dentro de Google Colab."""
    return "google.colab" in sys.modules


def hay_adc() -> bool:
    """True si hay Application Default Credentials (el login que abre el navegador de ADC)."""
    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env and Path(env).expanduser().exists():
        return True
    return (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    ).exists()


def hay_cuenta_gcloud() -> bool:
    """True si `gcloud auth list` muestra alguna cuenta (no ADC)."""
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return False
    resultado = subprocess.run(
        [gcloud, "auth", "list", "--format=value(account)"],
        capture_output=True,
        text=True,
    )
    return bool(resultado.stdout.strip())


def usar_cliente_python() -> bool:
    """Colab, o local con ADC (aunque `gcloud auth list` muestre una cuenta)."""
    if en_colab():
        return True
    return hay_adc()


def exigir_credenciales_gcs() -> str:
    """``python`` o ``gsutil``, o error que dice qué login falta."""
    if usar_cliente_python():
        return "python"
    if hay_cuenta_gcloud() and shutil.which("gsutil"):
        return "gsutil"
    raise RuntimeError(
        "No hay credenciales para Google Cloud Storage.\n"
        "  Opción A:  gcloud auth login\n"
        "  Opción B:  gcloud auth application-default login\n"
        "  Usa la MISMA cuenta con la que aceptaste los términos en\n"
        f"  {PAGINA_DESCARGA}"
    )


def _importar_gcs():
    """Cliente Python de GCS. El extra `waymo` no está en el entorno base."""
    try:
        from google.api_core import exceptions
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "Falta el extra `waymo` (google-cloud-storage).\n"
            "  uv sync --extra waymo\n"
            "Eso usa las Application Default Credentials "
            "(gcloud auth application-default login)."
        ) from error
    return storage, exceptions


def ruta_gcs(componente: str, segmento: str) -> str:
    """Ruta del objeto dentro del bucket, sin el prefijo gs://."""
    return f"{SPLIT}/{componente}/{segmento}.parquet"


def _descargar_con_cliente_python(componente: str, segmento: str, destino: Path) -> None:
    """Vía de Colab: usa las credenciales de ``auth.authenticate_user()``.

    El proyecto que se indica solo se usa para facturación; para **leer** un
    objeto sirve cualquier nombre, y las cuentas personales no tienen proyecto
    por defecto.

    Raises:
        RuntimeError: si el bucket responde 403, con el nombre de la cuenta que
            se está usando. Es el error más frecuente y el más confuso: Colab
            suele estar abierto con una cuenta personal, mientras que los
            términos de Waymo se aceptaron con otra.
    """
    storage, exceptions = _importar_gcs()

    bucket = storage.Client(project="mly1101").bucket(BUCKET)
    try:
        bucket.blob(ruta_gcs(componente, segmento)).download_to_filename(str(destino))
    except exceptions.Forbidden as error:
        destino.unlink(missing_ok=True)
        raise RuntimeError(_MENSAJE_403.format(error=error)) from error


def _descargar_con_gsutil(componente: str, segmento: str, destino: Path) -> None:
    """Vía local: usa las credenciales de ``gcloud auth login``."""
    gsutil = shutil.which("gsutil")
    if not gsutil:
        raise RuntimeError(
            "No se encontró gsutil. Instálalo con: brew install --cask google-cloud-sdk"
        )
    resultado = subprocess.run(
        [gsutil, "cp", f"gs://{BUCKET}/{ruta_gcs(componente, segmento)}", str(destino)],
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"gsutil falló: {resultado.stderr.strip()[:300]}")


def descargar(componente: str, segmento: str, carpeta: Path, forzar: bool = False) -> Path:
    """Descarga un componente de un segmento y devuelve la ruta local.

    Elige automáticamente la vía que funciona en el entorno actual. Si el
    archivo ya existe, no lo vuelve a bajar salvo que ``forzar`` sea True.

    Args:
        componente: ``"lidar_box"`` o ``"stats"``.
        segmento: nombre del segmento, sin extensión.
        carpeta: directorio de destino; se crea si no existe.
        forzar: vuelve a descargar aunque el archivo ya esté.

    Raises:
        RuntimeError: si la descarga falla, con el mensaje del proveedor.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{componente}.parquet"

    if destino.exists() and not forzar:
        return destino

    if usar_cliente_python():
        _descargar_con_cliente_python(componente, segmento, destino)
    else:
        _descargar_con_gsutil(componente, segmento, destino)
    return destino


def descargar_segmento(segmento: str, carpeta: Path) -> dict[str, Path]:
    """Descarga los dos componentes livianos de un segmento.

    Returns:
        ``{"lidar_box": ruta, "stats": ruta}``
    """
    rutas = {}
    for componente in COMPONENTES_LIVIANOS:
        ruta = descargar(componente, segmento, carpeta)
        tamano = ruta.stat().st_size / 1024**2
        print(f"{componente}: {tamano:.2f} MB  ({ruta})")
        rutas[componente] = ruta
    return rutas


def descargar_camera_box(segmento: str, carpeta: Path) -> Path:
    """Cajas 2D de un segmento (~50–360 KB). Es tabla, no JPEG."""
    return descargar("camera_box", segmento, carpeta)


def descargar_tablas_chicas(segmento: str, carpeta: Path) -> dict[str, Path]:
    """Baja las tablas manipulables de un segmento (cajas 2D, pose, calibración).

    No baja JPEG, nubes ni tfrecord. Cada archivo cabe en Colab. Si ya está en
    disco, no vuelve a GCS.
    """
    rutas: dict[str, Path] = {}
    for componente in COMPONENTES_MANIPULABLES:
        rutas[componente] = descargar(componente, segmento, carpeta)
    return rutas


def completar_tablas_chicas(
    muestra: Path, limite: int | None = LOTE_CLASE
) -> list[str]:
    """Baja cajas 2D, pose y calibración en segmentos **ya** completos.

    No toca ``lidar_box``. Por defecto los primeros 8 (lote de clase).
    ``limite=None`` recorre todos. Lista vacía si aún no hay ``muestra/``.
    """
    if not muestra.is_dir():
        return []
    carpetas = segmentos_completos(muestra)
    if limite is not None:
        carpetas = carpetas[:limite]
    nombres: list[str] = []
    for carpeta in carpetas:
        descargar_tablas_chicas(carpeta.name, carpeta)
        nombres.append(carpeta.name)
    return nombres


def producto(clave: str) -> dict[str, str]:
    """Devuelve una entrada de ``CATALOGO_BUCKETS`` o lista las claves válidas."""
    try:
        return CATALOGO_BUCKETS[clave]
    except KeyError as error:
        disponibles = ", ".join(CATALOGO_BUCKETS)
        raise KeyError(
            f"No hay un producto '{clave}'. Claves: {disponibles}."
        ) from error


def que_hacer_con_el_producto(clave: str) -> str:
    """Una frase para clase: ¿se baja, se lista, o se sustituye?"""
    info = producto(clave)
    return (
        f"{clave}: en_clase={info['en_clase']} · {info['tamano_medido']}. "
        f"{info['sustituto']}"
    )


def listar_objetos(bucket: str, prefijo: str = "", limite: int = 8) -> list[dict]:
    """Lista objetos de un bucket (nombre, tamaño, URI gs://), sin descargarlos.

    En Colab usa el cliente Python; en local, ``gsutil ls -l``. El límite evita
    paginar terabytes: se mira un prefijo, se elige un archivo, y recién ahí se
    copia.
    """
    if usar_cliente_python():
        return _listar_con_cliente_python(bucket, prefijo, limite)
    return _listar_con_gsutil(bucket, prefijo, limite)


def descargar_objeto(
    bucket: str,
    blob: str,
    carpeta: Path,
    forzar: bool = False,
    tamano_maximo_mb: float | None = TAMANO_MAXIMO_MB,
) -> Path:
    """Copia un objeto cualquiera a ``carpeta / nombre_del_archivo``.

    Args:
        bucket: nombre del bucket, sin ``gs://``.
        blob: ruta del objeto dentro del bucket.
        carpeta: directorio de destino; se crea si no existe.
        forzar: vuelve a descargar aunque el archivo ya esté.
        tamano_maximo_mb: tope de seguridad. ``None`` lo desactiva. Un tfrecord
            de Perception v1 o Motion puede ser cientos de MB o varios GB.

    Raises:
        RuntimeError: si el archivo supera el tope o si GCS responde 403.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / Path(blob).name
    if destino.exists() and not forzar:
        return destino

    if tamano_maximo_mb is not None:
        bytes_ = _tamano_blob(bucket, blob)
        tope = tamano_maximo_mb * 1024 * 1024
        if bytes_ > tope:
            mb = bytes_ / 1024**2
            raise RuntimeError(
                f"gs://{bucket}/{blob} pesa {mb:.0f} MB, por encima del tope de "
                f"{tamano_maximo_mb:.0f} MB. Un tfrecord de Perception v1 o Motion "
                "puede ser cientos de MB o varios GB: no lo bajes en Colab por "
                "accidente. Si lo necesitas de verdad, llama con "
                "tamano_maximo_mb=None."
            )

    if usar_cliente_python():
        _copiar_blob_python(bucket, blob, destino)
    else:
        _copiar_blob_gsutil(bucket, blob, destino)
    return destino


def elegir_muestra(
    objetos: list[dict],
    tamano_maximo_mb: float | None = TAMANO_MAXIMO_CLASE_MB,
) -> dict:
    """Elige el objeto más chico que cabe bajo el tope.

    Así se trabaja con datos reales en clase: un fragmento, no el bucket.
    """
    if not objetos:
        raise RuntimeError(
            "El listado está vacío. 403 suele ser cuenta distinta a la del "
            "registro en Waymo; lista vacía, un prefijo que no existe."
        )
    tope = None if tamano_maximo_mb is None else tamano_maximo_mb * 1024 * 1024
    candidatos = [
        o for o in objetos if tope is None or o["bytes"] <= tope
    ]
    if not candidatos:
        mas_chico = min(objetos, key=lambda o: o["bytes"])
        mb = mas_chico["bytes"] / 1024**2
        raise RuntimeError(
            f"Nada del listado cabe bajo {tamano_maximo_mb:.0f} MB. "
            f"El más chico es {mas_chico['nombre']} ({mb:.0f} MB). "
            "En local, si tienes disco, llama con tamano_maximo_mb=None."
        )
    return min(candidatos, key=lambda o: o["bytes"])


def descargar_muestra(
    clave: str,
    carpeta: Path,
    limite: int = 12,
    tamano_maximo_mb: float | None = TAMANO_MAXIMO_CLASE_MB,
) -> Path:
    """Lista el prefijo del producto y copia el fragmento real más chico que cabe.

    Args:
        clave: una clave de ``CATALOGO_BUCKETS`` (``percepcion_v2``, ``motion``, …).
        carpeta: destino local (``datos/waymo_real/``).
        limite: cuántos objetos pedir a GCS al listar.
        tamano_maximo_mb: tope de clase. ``None`` desactiva el tope.
    """
    info = producto(clave)
    objetos = listar_objetos(info["bucket"], info["prefijo_muestra"], limite=limite)
    elegido = elegir_muestra(objetos, tamano_maximo_mb=tamano_maximo_mb)
    return descargar_objeto(
        info["bucket"],
        elegido["nombre"],
        carpeta,
        tamano_maximo_mb=tamano_maximo_mb,
    )


def resumir_fragmento(ruta: Path) -> dict:
    """Abre un archivo ya descargado y dice qué hay: formato, tamaño, columnas.

    No baja nada. Sirve para validar el tratamiento después de conectar a GCS.
    """
    if not ruta.exists():
        raise FileNotFoundError(ruta)

    nombre = ruta.name.lower()
    if ruta.suffix == ".parquet" or nombre.endswith(".parquet"):
        import pandas as pd

        tabla = pd.read_parquet(ruta)
        numericas = [
            c for c in tabla.columns if pd.api.types.is_numeric_dtype(tabla[c])
        ]
        return {
            "formato": "parquet",
            "filas": int(len(tabla)),
            "columnas": int(tabla.shape[1]),
            "numericas": int(len(numericas)),
            "nombres": list(tabla.columns),
        }

    if ruta.suffix == ".json" or nombre.endswith(".json"):
        with ruta.open(encoding="utf-8") as fh:
            contenido = json.load(fh)
        if isinstance(contenido, dict):
            return {
                "formato": "json",
                "claves": list(contenido.keys()),
                "elementos": len(contenido),
            }
        if isinstance(contenido, list):
            return {"formato": "json", "claves": [], "elementos": len(contenido)}
        return {"formato": "json", "claves": [], "elementos": 1}

    if "tfrecord" in nombre:
        return {"formato": "tfrecord", "bytes": int(ruta.stat().st_size)}

    return {"formato": ruta.suffix.lstrip(".") or "binario", "bytes": int(ruta.stat().st_size)}


def _parsear_listado_gsutil(stdout: str, bucket: str) -> list[dict]:
    """Convierte ``gsutil ls -l`` en una lista de objetos (sin directorios)."""
    objetos: list[dict] = []
    prefijo_gs = f"gs://{bucket}/"
    for linea in stdout.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("TOTAL"):
            continue
        partes = linea.split()
        if len(partes) < 2:
            continue
        try:
            tamano = int(partes[0])
        except ValueError:
            continue
        gs = partes[-1]
        if not gs.startswith(prefijo_gs) or gs.endswith("/"):
            continue
        objetos.append(
            {"nombre": gs[len(prefijo_gs) :], "bytes": tamano, "gs": gs}
        )
    return objetos


def _error_gsutil(stderr: str) -> RuntimeError:
    """Traduce los fallos de gsutil a un mensaje que el alumno puede ejecutar."""
    texto = (stderr or "").strip()
    bajo = texto.lower()
    if "reauthentication" in bajo or "cannot prompt" in bajo:
        return RuntimeError(
            "gcloud pide volver a iniciar sesión (el token caducó).\n"
            "  Opción A:  gcloud auth login\n"
            "  Opción B:  gcloud auth application-default login\n"
            "             y  uv sync --extra waymo\n"
            "  Usa la MISMA cuenta con la que aceptaste los términos en "
            "https://waymo.com/open/download/"
        )
    if "accessdenied" in bajo or "403" in texto:
        return RuntimeError(_MENSAJE_403.format(error=texto[:300]))
    return RuntimeError(f"gsutil falló: {texto[:300]}")


def _listar_con_gsutil(bucket: str, prefijo: str, limite: int) -> list[dict]:
    gsutil = shutil.which("gsutil")
    if not gsutil:
        raise RuntimeError(
            "No se encontró gsutil. Instálalo con: brew install --cask google-cloud-sdk"
        )
    uri = f"gs://{bucket}/{prefijo}" if prefijo else f"gs://{bucket}"
    resultado = subprocess.run(
        [gsutil, "ls", "-l", uri], capture_output=True, text=True
    )
    if resultado.returncode != 0:
        raise _error_gsutil(resultado.stderr)
    return _parsear_listado_gsutil(resultado.stdout, bucket)[:limite]


def _listar_con_cliente_python(bucket: str, prefijo: str, limite: int) -> list[dict]:
    storage, exceptions = _importar_gcs()

    try:
        blobs = storage.Client(project="mly1101").list_blobs(
            bucket, prefix=prefijo or None, max_results=limite
        )
        objetos = []
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            objetos.append(
                {
                    "nombre": blob.name,
                    "bytes": int(blob.size or 0),
                    "gs": f"gs://{bucket}/{blob.name}",
                }
            )
        return objetos[:limite]
    except exceptions.Forbidden as error:
        raise RuntimeError(_MENSAJE_403.format(error=error)) from error


def _tamano_blob(bucket: str, blob: str) -> int:
    """Tamaño en bytes de un objeto, sin descargarlo."""
    if usar_cliente_python():
        storage, _ = _importar_gcs()

        remoto = storage.Client(project="mly1101").bucket(bucket).get_blob(blob)
        if remoto is None:
            raise RuntimeError(f"No existe gs://{bucket}/{blob}")
        return int(remoto.size or 0)

    for objeto in _listar_con_gsutil(bucket, blob, 1):
        if objeto["nombre"] == blob:
            return int(objeto["bytes"])
    raise RuntimeError(f"No existe gs://{bucket}/{blob}")


def _copiar_blob_python(bucket: str, blob: str, destino: Path) -> None:
    storage, exceptions = _importar_gcs()

    try:
        storage.Client(project="mly1101").bucket(bucket).blob(blob).download_to_filename(
            str(destino)
        )
    except exceptions.Forbidden as error:
        destino.unlink(missing_ok=True)
        raise RuntimeError(_MENSAJE_403.format(error=error)) from error


def _copiar_blob_gsutil(bucket: str, blob: str, destino: Path) -> None:
    gsutil = shutil.which("gsutil")
    if not gsutil:
        raise RuntimeError(
            "No se encontró gsutil. Instálalo con: brew install --cask google-cloud-sdk"
        )
    resultado = subprocess.run(
        [gsutil, "cp", f"gs://{bucket}/{blob}", str(destino)],
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise _error_gsutil(resultado.stderr)


# ===========================================================================
# Traducción del esquema real de Waymo v2 al esquema de la asignatura
# ===========================================================================
#
# Vivía dentro del notebook 00 como texto de celda. Aquí es una función pura,
# así que la usan por igual el notebook, el pipeline de Kedro y los tests.
# El esquema fue verificado contra un Parquet real el 2026-08-26.

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LB = "[LiDARBoxComponent]"
_ST = "[StatsComponent]"

EQUIVALENCIAS_CAJAS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    "key.laser_object_id": "id_interno",
    f"{_LB}.type": "object_type",
    f"{_LB}.box.center.x": "box_center_x",
    f"{_LB}.box.center.y": "box_center_y",
    f"{_LB}.box.center.z": "box_center_z",
    f"{_LB}.box.size.x": "box_length",
    f"{_LB}.box.size.y": "box_width",
    f"{_LB}.box.size.z": "box_height",
    f"{_LB}.speed.x": "speed_x",
    f"{_LB}.speed.y": "speed_y",
    f"{_LB}.num_lidar_points_in_box": "num_lidar_points",
    f"{_LB}.difficulty_level.detection": "detection_difficulty",
}

EQUIVALENCIAS_STATS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    f"{_ST}.time_of_day": "time_of_day",
    f"{_ST}.weather": "weather",
    f"{_ST}.location": "location",
}

# El entero que Waymo usa para el tipo de objeto.
TIPOS_DE_OBJETO = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}

# Cámaras de Perception v2 (enum CameraName). 1 = frente; no son archivos JPEG.
NOMBRES_DE_CAMARA = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

_CB = "[CameraBoxComponent]"
_CC = "[CameraCalibrationComponent]"
_VP = "[VehiclePoseComponent]"

EQUIVALENCIAS_CAMERA_BOX = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    "key.camera_name": "camera_name",
    "key.camera_object_id": "id_interno",
    f"{_CB}.type": "object_type",
    f"{_CB}.box.center.x": "box_center_x_px",
    f"{_CB}.box.center.y": "box_center_y_px",
    f"{_CB}.box.size.x": "box_width_px",
    f"{_CB}.box.size.y": "box_height_px",
    f"{_CB}.difficulty_level.detection": "detection_difficulty",
}

EQUIVALENCIAS_CALIBRACION = {
    "key.segment_context_name": "segment_id",
    "key.camera_name": "camera_name",
    f"{_CC}.intrinsic.f_u": "f_u",
    f"{_CC}.intrinsic.f_v": "f_v",
    f"{_CC}.intrinsic.c_u": "c_u",
    f"{_CC}.intrinsic.c_v": "c_v",
    f"{_CC}.width": "ancho_px",
    f"{_CC}.height": "alto_px",
}


def _renombrar(datos: pd.DataFrame, equivalencias: dict) -> pd.DataFrame:
    """Se queda con las columnas conocidas y les pone el nombre de la clase."""
    presentes = {k: v for k, v in equivalencias.items() if k in datos.columns}
    return datos[list(presentes)].rename(columns=presentes)


def traducir_esquema(cajas: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Convierte los componentes reales de Waymo v2 al esquema de la asignatura.

    Tres traducciones no son un simple cambio de nombre, y son justamente las que
    conviene mirar en clase:

    1. **La velocidad es un vector.** Waymo entrega ``speed.x`` y ``speed.y`` por
       separado; la rapidez es su módulo, ``√(x² + y²)``. Quedarse solo con
       ``speed.x`` es un error silencioso: da valores plausibles y equivocados.

    2. **El tipo de objeto es un entero**, no una cadena. Hay que traducirlo con
       ``TIPOS_DE_OBJETO``, y existe el valor ``0`` (*unknown*).

    3. **El ``NaN`` de la dificultad NO es un dato faltante.** Waymo solo rellena
       ``difficulty_level.detection`` cuando la detección es difícil (valor ``2``);
       si está vacío significa ``LEVEL_1``. En el segmento verificado son 15.356
       ``NaN`` de 18.633 filas: tratarlos como faltantes borraría el 82 % de los
       datos y dejaría un dataset con una sola clase.

    Args:
        cajas: contenido de ``lidar_box.parquet``.
        stats: contenido de ``stats.parquet``.

    Returns:
        DataFrame de detecciones del curso (una fila por caja LiDAR), más
        ``location``.
    """
    tabla = _renombrar(cajas, EQUIVALENCIAS_CAJAS).merge(
        _renombrar(stats, EQUIVALENCIAS_STATS),
        on=["segment_id", "timestamp_micros"],
        how="left",
    )

    # 1. La rapidez es el módulo del vector velocidad.
    if {"speed_x", "speed_y"} <= set(tabla.columns):
        tabla["speed_mps"] = np.sqrt(tabla["speed_x"] ** 2 + tabla["speed_y"] ** 2)
        tabla = tabla.drop(columns=["speed_x", "speed_y"])

    # 2. El tipo de objeto llega como entero.
    if "object_type" in tabla.columns:
        tabla["object_type"] = (
            tabla["object_type"].map(TIPOS_DE_OBJETO).fillna("desconocido")
        )

    # 3. El NaN de la dificultad significa LEVEL_1, no "falta el dato".
    if "detection_difficulty" in tabla.columns:
        tabla["detection_difficulty"] = np.where(
            tabla["detection_difficulty"] == 2, "LEVEL_2", "LEVEL_1"
        )

    return tabla.reset_index(drop=True)


def _traducir_tipo_y_dificultad(tabla: pd.DataFrame) -> pd.DataFrame:
    """Mismos enteros y el mismo NaN = LEVEL_1 que en ``traducir_esquema``."""
    if "object_type" in tabla.columns:
        tabla["object_type"] = (
            tabla["object_type"].map(TIPOS_DE_OBJETO).fillna("desconocido")
        )
    if "detection_difficulty" in tabla.columns:
        tabla["detection_difficulty"] = np.where(
            tabla["detection_difficulty"] == 2, "LEVEL_2", "LEVEL_1"
        )
    return tabla


def traducir_camera_box(cajas: pd.DataFrame) -> pd.DataFrame:
    """Pasa ``camera_box`` a nombres de clase. Las cajas quedan en **píxeles**.

    No cruce esta tabla con el LiDAR fila a fila: metros ≠ píxeles. Sirve para
    contar tipos, ver cámaras y el NaN de dificultad (el mismo reverso que en
    ``lidar_box``).
    """
    tabla = _renombrar(cajas, EQUIVALENCIAS_CAMERA_BOX).copy()
    tabla = _traducir_tipo_y_dificultad(tabla)
    if "camera_name" in tabla.columns:
        tabla["camara"] = tabla["camera_name"].map(NOMBRES_DE_CAMARA).fillna(
            "desconocida"
        )
    return tabla.reset_index(drop=True)


def traducir_calibracion_camara(calibracion: pd.DataFrame) -> pd.DataFrame:
    """Intrínsecos y tamaño de cada cámara (una fila por cámara del segmento)."""
    tabla = _renombrar(calibracion, EQUIVALENCIAS_CALIBRACION).copy()
    if "camera_name" in tabla.columns:
        tabla["camara"] = tabla["camera_name"].map(NOMBRES_DE_CAMARA).fillna(
            "desconocida"
        )
    return tabla.reset_index(drop=True)


def traducir_pose_vehiculo(pose: pd.DataFrame) -> pd.DataFrame:
    """Una fila por frame: segmento, tiempo y traslación ``(x, y, z)`` en mundo.

    La matriz 4×4 se deja fuera: en clase basta la trayectoria. Vacío si no hay
    columna de transform.
    """
    claves = {
        "key.segment_context_name": "segment_id",
        "key.frame_timestamp_micros": "timestamp_micros",
    }
    tabla = _renombrar(pose, claves).copy()
    col_tf = f"{_VP}.world_from_vehicle.transform"
    if col_tf not in pose.columns:
        return tabla.reset_index(drop=True)

    xs, ys, zs = [], [], []
    for valor in pose[col_tf]:
        arr = np.asarray(valor, dtype=float).reshape(-1)
        if arr.size >= 12:
            xs.append(float(arr[3]))
            ys.append(float(arr[7]))
            zs.append(float(arr[11]))
        else:
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
    tabla["pos_x"] = xs
    tabla["pos_y"] = ys
    tabla["pos_z"] = zs
    return tabla.reset_index(drop=True)


def ficha_tabla(tabla: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: dtype, nulos y valores distintos. Para explorar."""
    if tabla.empty:
        return pd.DataFrame(columns=["columna", "dtype", "nulos_pct", "n_unicos"])
    filas = []
    for columna in tabla.columns:
        serie = tabla[columna]
        filas.append(
            {
                "columna": columna,
                "dtype": str(serie.dtype),
                "nulos_pct": round(float(serie.isna().mean() * 100), 2),
                "n_unicos": int(serie.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(filas)


def comparar_conteos_por_tipo(
    lidar: pd.DataFrame, camara: pd.DataFrame
) -> pd.DataFrame:
    """Conteos de ``object_type`` en LiDAR (metros) vs cámara (píxeles).

    No une filas. ``sign`` suele aparecer en LiDAR y no en ``camera_box``: es el
    dato, no un bug de merge.
    """
    def _conteo(origen: pd.DataFrame) -> pd.Series:
        if origen.empty or "object_type" not in origen.columns:
            return pd.Series(dtype="int64")
        return origen["object_type"].value_counts()

    juntos = pd.DataFrame(
        {"lidar_n": _conteo(lidar), "camara_n": _conteo(camara)}
    ).fillna(0)
    juntos["lidar_n"] = juntos["lidar_n"].astype(int)
    juntos["camara_n"] = juntos["camara_n"].astype(int)
    juntos.index.name = "object_type"
    return juntos.reset_index()


def recorte_de_un_frame(
    camara: pd.DataFrame,
    *,
    segmento: str | None = None,
    timestamp_micros: int | None = None,
    nombre_camara: str = "FRONT",
) -> pd.DataFrame:
    """Cajas 2D de **un** instante y **una** cámara. Es el “fotograma” sin JPEG.

    Waymo, en su Colab oficial, muestra 2 frames de muestra embebidos. Aquí el
    alumno usa las tablas que ya bajó: mismo gesto (ver el cuadro), sin el video
    de 1,6 GB.
    """
    if camara.empty:
        return camara.copy()
    trabajo = camara
    if "camara" in trabajo.columns:
        trabajo = trabajo.loc[trabajo["camara"] == nombre_camara]
    if trabajo.empty:
        return trabajo.copy()
    if segmento is None:
        segmento = str(trabajo["segment_id"].iloc[0])
    trabajo = trabajo.loc[trabajo["segment_id"] == segmento]
    if trabajo.empty:
        return trabajo.copy()
    if timestamp_micros is None:
        timestamp_micros = int(trabajo["timestamp_micros"].iloc[0])
    return trabajo.loc[trabajo["timestamp_micros"] == timestamp_micros].reset_index(
        drop=True
    )


def tamano_del_lienzo(
    calibracion: pd.DataFrame, nombre_camara: str = "FRONT"
) -> tuple[int, int]:
    """Ancho × alto en píxeles de esa cámara. 1920×1280 si no hay calibración (v2)."""
    if calibracion.empty or "camara" not in calibracion.columns:
        return 1920, 1280
    fila = calibracion.loc[calibracion["camara"] == nombre_camara]
    if fila.empty:
        return 1920, 1280
    return int(fila["ancho_px"].iloc[0]), int(fila["alto_px"].iloc[0])


def rectangulos_del_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Esquina superior-izquierda + tamaño, listos para ``matplotlib.patches.Rectangle``.

    Waymo da el centro y el tamaño en píxeles. Sin JPEG igual se puede ver el
    cuadro: es el mismo gesto que el Colab oficial de 2 frames.
    """
    vacio = pd.DataFrame(columns=["object_type", "x0", "y0", "ancho", "alto"])
    necesarias = {
        "box_center_x_px",
        "box_center_y_px",
        "box_width_px",
        "box_height_px",
    }
    if frame.empty or not necesarias <= set(frame.columns):
        return vacio
    x = frame["box_center_x_px"].astype(float)
    y = frame["box_center_y_px"].astype(float)
    ancho = frame["box_width_px"].astype(float)
    alto = frame["box_height_px"].astype(float)
    tipos = (
        frame["object_type"]
        if "object_type" in frame.columns
        else pd.Series(["desconocido"] * len(frame), index=frame.index)
    )
    return pd.DataFrame(
        {
            "object_type": tipos.to_numpy(),
            "x0": (x - ancho / 2).to_numpy(),
            "y0": (y - alto / 2).to_numpy(),
            "ancho": ancho.to_numpy(),
            "alto": alto.to_numpy(),
        }
    ).reset_index(drop=True)


def informe_analitica(tabla: pd.DataFrame) -> dict:
    """Resumen corto para no perderse en el volumen: cuántas filas, qué hay, qué sigue."""
    tipos = (
        tabla["object_type"].value_counts().to_dict()
        if "object_type" in tabla.columns
        else {}
    )
    sin_puntos = 0
    if "num_lidar_points" in tabla.columns:
        sin_puntos = int((tabla["num_lidar_points"] == 0).sum())
    segmentos = (
        int(tabla["segment_id"].nunique()) if "segment_id" in tabla.columns else 1
    )
    return {
        "filas": int(len(tabla)),
        "segmentos": segmentos,
        "tipos": tipos,
        "cajas_sin_puntos": sin_puntos,
        "siguiente": (
            "Esta tabla alimenta las Act. 1.1–3.3, el proyecto (notebook 10) y "
            "el pipeline Kedro (`kedro run`). Parte train/test por segment_id, "
            "no al azar."
        ),
    }


def texto_informe(informe: dict) -> str:
    """Versión imprimible del informe de analítica."""
    lineas = [
        f"{informe['filas']} detecciones  ·  {informe['segmentos']} segmentos",
        "tipos: " + ", ".join(f"{k}={v}" for k, v in informe["tipos"].items()),
        f"cajas con 0 puntos LiDAR: {informe['cajas_sin_puntos']}",
        informe["siguiente"],
    ]
    return "\n".join(lineas)


def segmentos_completos(muestra: Path) -> list[Path]:
    """Carpetas de ``muestra/`` que tienen ``lidar_box`` y ``stats``."""
    if not muestra.is_dir():
        return []
    completos: list[Path] = []
    for carpeta in sorted(p for p in muestra.iterdir() if p.is_dir()):
        if (carpeta / "lidar_box.parquet").exists() and (
            carpeta / "stats.parquet"
        ).exists():
            completos.append(carpeta)
    return completos


def ensamblar_muestra(muestra: Path) -> pd.DataFrame:
    """Traduce todos los segmentos completos de ``muestra/`` a una sola tabla.

    Es la misma traducción que usa el pipeline Kedro: varios segmentos, no uno.
    Con un solo segmento no se puede partir train/test sin fuga.
    """
    carpetas = segmentos_completos(muestra)
    if not carpetas:
        raise RuntimeError(
            "No hay segmentos completos (lidar_box + stats) en "
            f"{muestra}. Bájalos con: python herramientas/descargar_waymo.py --lote 8"
        )
    piezas = [
        traducir_esquema(
            pd.read_parquet(carpeta / "lidar_box.parquet"),
            pd.read_parquet(carpeta / "stats.parquet"),
        )
        for carpeta in carpetas
    ]
    return pd.concat(piezas, ignore_index=True)


def inventario_muestra(muestra: Path) -> pd.DataFrame:
    """Qué componentes hay en disco, por segmento. Para ver que no bajamos imágenes."""
    filas: list[dict] = []
    if not muestra.is_dir():
        return pd.DataFrame(columns=["segmento", "componente", "mb", "completo"])
    nombres_completos = {carpeta.name for carpeta in segmentos_completos(muestra)}
    for carpeta in sorted(p for p in muestra.iterdir() if p.is_dir()):
        for parquet in sorted(carpeta.glob("*.parquet")):
            filas.append(
                {
                    "segmento": carpeta.name,
                    "componente": parquet.stem,
                    "mb": round(parquet.stat().st_size / 1024**2, 3),
                    "completo": carpeta.name in nombres_completos,
                }
            )
    return pd.DataFrame(filas)


def inventario_fuentes(carpeta: Path) -> pd.DataFrame:
    """Qué productos de Waymo hay en disco y cuáles entran al modelo.

    El clasificador del curso solo usa Perception v2 (``lidar_box`` + ``stats``).
    El resto se lista para que el pipeline las *vea* sin mezclarlas: ``camera_box``
    es tabla 2D, E2E/Motion/v1 son otro problema y otro tamaño.
    """
    muestra = inventario_muestra(carpeta / "muestra")

    def _de_componente(nombre: str) -> tuple[int, float]:
        if muestra.empty:
            return 0, 0.0
        sub = muestra.loc[muestra["componente"] == nombre]
        return int(sub["segmento"].nunique()), float(sub["mb"].sum())

    n_v2 = len(segmentos_completos(carpeta / "muestra"))
    mb_v2 = 0.0
    if not muestra.empty:
        mb_v2 = float(
            muestra.loc[
                muestra["componente"].isin(("lidar_box", "stats"))
                & muestra["completo"],
                "mb",
            ].sum()
        )
    n_box, mb_box = _de_componente("camera_box")
    n_img, mb_img = _de_componente("camera_image")

    jsons_e2e = list(carpeta.glob("val_sequence*.json")) + list(
        carpeta.glob("test_sequence*.json")
    )
    tf_v1 = [
        p
        for p in carpeta.rglob("*.tfrecord*")
        if "segment-" in p.name and "camera_labels" in p.name
    ]
    tf_motion = [
        p
        for p in carpeta.rglob("*.tfrecord*")
        if "tfexample" in p.name.lower() or "motion" in str(p).lower()
    ]
    tf_e2e = [
        p
        for p in carpeta.glob("*.tfrecord*")
        if p.name.startswith("test_") or p.name.startswith("val_")
    ]

    def _mb(rutas: list[Path]) -> float:
        return round(sum(p.stat().st_size for p in rutas) / 1024**2, 3)

    filas = [
        {
            "fuente": "percepcion_v2",
            "archivos": n_v2,
            "mb": round(mb_v2, 3),
            "entra_al_modelo": True,
            "nota": "lidar_box + stats: el hilo del curso (Act. 2.2 por segmento)",
        },
        {
            "fuente": "camera_box",
            "archivos": n_box,
            "mb": round(mb_box, 3),
            "entra_al_modelo": False,
            "nota": "cajas 2D (tabla). El pipeline las ve; no van al Random Forest",
        },
        {
            "fuente": "camera_image",
            "archivos": n_img,
            "mb": round(mb_img, 3),
            "entra_al_modelo": False,
            "nota": "~330 MB por segmento: no se baja en clase",
        },
        {
            "fuente": "e2e_camara",
            "archivos": len(jsons_e2e) + len(tf_e2e),
            "mb": round(_mb(jsons_e2e) + _mb(tf_e2e), 3),
            "entra_al_modelo": False,
            "nota": "JSON de metadatos (~36 KB) sí; tfrecord ~1,6 GB no",
        },
        {
            "fuente": "percepcion_v1",
            "archivos": len(tf_v1),
            "mb": round(_mb(tf_v1), 3),
            "entra_al_modelo": False,
            "nota": "tfrecord con mapas (~1 GB). No entra al grafo del curso",
        },
        {
            "fuente": "motion",
            "archivos": len(tf_motion),
            "mb": round(_mb(tf_motion), 3),
            "entra_al_modelo": False,
            "nota": "trayectorias (~1 GB por shard). No entra al grafo del curso",
        },
    ]
    return pd.DataFrame(filas)


def ensamblar_componente(muestra: Path, componente: str) -> pd.DataFrame:
    """Concatena ``muestra/*/<componente>.parquet``. Vacío si no hay archivos."""
    if not muestra.is_dir():
        return pd.DataFrame()
    piezas = [
        pd.read_parquet(ruta)
        for ruta in sorted(muestra.glob(f"*/{componente}.parquet"))
    ]
    if not piezas:
        return pd.DataFrame()
    return pd.concat(piezas, ignore_index=True)


def ensamblar_camera_box(muestra: Path) -> pd.DataFrame:
    """Concatena los ``camera_box.parquet`` de ``muestra/``. Vacío si no hay."""
    return ensamblar_componente(muestra, "camera_box")


def leer_metadatos_e2e(carpeta: Path) -> pd.DataFrame:
    """Lee el JSON liviano de E2E si está. Vacío si no hay archivo."""
    candidatos = sorted(carpeta.glob("val_sequence*.json")) + sorted(
        carpeta.glob("test_sequence*.json")
    )
    if not candidatos:
        return pd.DataFrame(columns=["secuencia", "cluster"])
    with candidatos[0].open(encoding="utf-8") as fh:
        contenido = json.load(fh)
    if not isinstance(contenido, dict):
        return pd.DataFrame(columns=["secuencia", "cluster"])

    def _etiqueta(valor) -> str:
        if isinstance(valor, dict):
            if "scenario_cluster" in valor:
                return str(valor["scenario_cluster"])
            if valor:
                return str(next(iter(valor.values())))
            return ""
        return str(valor)

    return pd.DataFrame(
        {
            "secuencia": list(contenido.keys()),
            "cluster": [_etiqueta(v) for v in contenido.values()],
        }
    )


def _contar_segmentos_parquet(ruta: Path) -> int:
    tabla = pd.read_parquet(ruta)
    if "segment_id" not in tabla.columns:
        return 0
    return int(tabla["segment_id"].nunique())


def partir_por_grupo(
    tabla: pd.DataFrame,
    columna_grupo: str = "segment_id",
    test_size: float = 0.25,
    semilla: int = 42,
) -> pd.DataFrame:
    """Marca ``entrenamiento`` / ``prueba`` sin partir un segmento a la mitad.

    Un segmento son ~20 s de la misma calle, el mismo clima y los mismos objetos
    frame a frame. Si una detección cae en train y la del fotograma siguiente en
    test, la métrica mide memoria, no generalización.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if columna_grupo not in tabla.columns:
        raise ValueError(
            f"Falta la columna '{columna_grupo}' para partir por grupo."
        )
    n_grupos = tabla[columna_grupo].nunique()
    if n_grupos < 2:
        raise ValueError(
            f"Solo hay {n_grupos} valor de '{columna_grupo}'. Hacen falta al "
            "menos 2 segmentos: uno entero va a entrenamiento o a prueba, nunca "
            "a los dos. Baja más con "
            "`python herramientas/descargar_waymo.py --lote 8` "
            "(o `--muestra 40` para el pipeline Kedro)."
        )
    marcada = tabla.reset_index(drop=True).copy()
    separador = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=semilla
    )
    _, indices_prueba = next(
        separador.split(marcada, groups=marcada[columna_grupo])
    )
    marcada["particion"] = "entrenamiento"
    marcada.loc[marcada.index[indices_prueba], "particion"] = "prueba"
    return marcada


def preparar_lote(
    carpeta: Path,
    n: int = LOTE_CLASE,
    segmentos: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, Path]:
    """Baja N segmentos livianos, los traduce y deja UNA tabla.

    Solo ``lidar_box`` + ``stats`` (~1 MB por segmento). No toca imágenes ni
    nubes de puntos. El resultado es ``detecciones_reales.parquet``.
    """
    if segmentos is None:
        objetos = listar_objetos(BUCKET, f"{SPLIT}/lidar_box/", limite=n)
        segmentos = [Path(objeto["nombre"]).stem for objeto in objetos][:n]
    if not segmentos:
        raise RuntimeError("No hay segmentos para armar el lote.")

    piezas: list[pd.DataFrame] = []
    for segmento in segmentos:
        rutas = descargar_segmento(segmento, carpeta / "muestra" / segmento)
        piezas.append(
            traducir_esquema(
                pd.read_parquet(rutas["lidar_box"]),
                pd.read_parquet(rutas["stats"]),
            )
        )
    tabla = pd.concat(piezas, ignore_index=True)
    salida = carpeta / "detecciones_reales.parquet"
    tabla.to_parquet(salida, index=False)
    return tabla, informe_analitica(tabla), salida


def cargar_o_preparar(
    carpeta: Path,
    n: int = LOTE_CLASE,
) -> tuple[pd.DataFrame, dict, Path]:
    """Una tabla para el curso, preferiendo varios segmentos si ya están en disco.

    Orden:

    1. ``muestra/`` con ≥2 segmentos completos — es lo que usa Kedro; si el
       parquet suelto tiene menos segmentos, lo reescribe.
    2. ``detecciones_reales.parquet`` ya armado.
    3. Un par ``lidar_box`` + ``stats`` suelto (un segmento: no alcanza para
       partir sin fuga).
    4. Baja ``n`` segmentos livianos (pide GCS).
    """
    salida = carpeta / "detecciones_reales.parquet"
    completos = segmentos_completos(carpeta / "muestra")
    if len(completos) >= 2:
        n_parquet = _contar_segmentos_parquet(salida) if salida.exists() else 0
        if n_parquet < len(completos):
            tabla = ensamblar_muestra(carpeta / "muestra")
            tabla.to_parquet(salida, index=False)
            return tabla, informe_analitica(tabla), salida
        tabla = pd.read_parquet(salida)
        return tabla, informe_analitica(tabla), salida

    if salida.exists():
        tabla = pd.read_parquet(salida)
        return tabla, informe_analitica(tabla), salida

    suelto = carpeta / "lidar_box.parquet"
    stats = carpeta / "stats.parquet"
    if suelto.exists() and stats.exists():
        tabla = traducir_esquema(pd.read_parquet(suelto), pd.read_parquet(stats))
        tabla.to_parquet(salida, index=False)
        return tabla, informe_analitica(tabla), salida

    return preparar_lote(carpeta, n=n)


def encontrar_raiz(desde: Path | None = None) -> Path:
    """Raíz del clone aunque el kernel arranque en ``notebooks/`` o en el repo.

    Cursor y VS Code suelen dejar el cwd en la raíz; Jupyter clásico, en
    ``notebooks/``. Caminar hacia los padres evita el ``FileNotFoundError``
    de quien ya bajó el parquet y el notebook lo busca un nivel más arriba.
    """
    inicio = Path(desde or Path.cwd()).resolve()
    for carpeta in (inicio, *inicio.parents):
        if (carpeta / "src" / "waymo.py").exists():
            return carpeta
    raise FileNotFoundError(
        "No encuentro la raíz del repo (falta src/waymo.py). "
        "Arranca el kernel en el clone de mly1101-machine-learning o en notebooks/."
    )


MENSAJE_SIN_DATOS_REALES = (
    "Faltan los datos reales de Waymo. No van en el repositorio "
    "(licencia de uso no comercial, sin redistribución). "
    "Acepta los términos en https://waymo.com/open/terms/ y corre:\n"
    "  uv run python herramientas/descargar_waymo.py --muestra 40\n"
    "Si ya los bajaste y esto igual falla, el kernel no está en el clone "
    "(en Cursor debe arrancar en la raíz del repo, no un nivel más arriba)."
)


def ruta_detecciones_reales(raiz: Path) -> Path:
    """Única tabla de trabajo del curso: Perception v2 ensamblada en local."""
    return Path(raiz) / "datos" / "waymo_real" / "detecciones_reales.parquet"


def exigir_detecciones_reales(raiz: Path) -> Path:
    """Ruta del parquet real, o error que dice cómo bajarlo. Sin fallback."""
    ruta = ruta_detecciones_reales(raiz)
    if ruta.exists():
        return ruta
    try:
        alternativa = ruta_detecciones_reales(encontrar_raiz())
    except FileNotFoundError:
        alternativa = None
    if alternativa is not None and alternativa.exists():
        return alternativa
    raise FileNotFoundError(MENSAJE_SIN_DATOS_REALES)


def leer_tabla(ruta: Path | str) -> pd.DataFrame:
    """Lee CSV o Parquet. Una función para no preguntar el formato en cada notebook."""
    texto = str(ruta)
    if texto.startswith(("http://", "https://")):
        return pd.read_csv(texto)
    archivo = Path(ruta)
    if archivo.suffix == ".parquet":
        return pd.read_parquet(archivo)
    return pd.read_csv(archivo)


def cargar_tabla_curso(raiz: Path) -> tuple[pd.DataFrame, str, Path]:
    """Tabla de trabajo del curso: solo Perception v2 real.

    Returns:
        ``(tabla, origen, ruta)`` con ``origen`` siempre ``"real"``.
    """
    ruta = exigir_detecciones_reales(raiz)
    return pd.read_parquet(ruta), "real", ruta
