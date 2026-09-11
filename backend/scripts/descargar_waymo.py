"""Descarga un segmento del Waymo Open Dataset v2 para verificar el mapeo.

Baja solo los dos componentes livianos que necesita el análisis de la EA1
(``lidar_box`` y ``stats``) de un único segmento de conducción. **No** baja
imágenes ni nubes de puntos, que son los componentes pesados.

Requisitos previos (una sola vez):

    brew install --cask google-cloud-sdk
    gcloud auth login          # o: gcloud auth application-default login
                               # misma cuenta que aceptó los términos en
                               # https://waymo.com/open/download/

Uso:

    python herramientas/descargar_waymo.py                 # primer segmento disponible
    python herramientas/descargar_waymo.py --lote 8        # recomendado: tabla de clase
    python herramientas/descargar_waymo.py --tablas-chicas # camera_box + pose + calibración
    python herramientas/descargar_waymo.py --muestra 40    # mismos livianos, para Kedro
    python herramientas/descargar_waymo.py --censo-stats   # stats de los 798 segmentos

Los archivos quedan en ``datos/waymo_real/``, que está en .gitignore: la
licencia de Waymo es de uso no comercial y no permite redistribuirlos.

``--tablas-chicas`` no baja JPEG. El fotograma (cajas en el lienzo) se dibuja
en el notebook 14; Kedro traduce ``camera_box`` y no lo mete al RF.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "datos" / "waymo_real"
BUCKET = "gs://waymo_open_dataset_v_2_0_1/training"
COMPONENTES = ["lidar_box", "stats"]
sys.path.insert(0, str(RAIZ / "src"))


def _ejecutar(comando: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(comando, capture_output=True, text=True)


def _comprobar_requisitos() -> str | None:
    """Comprueba login de GCS. Devuelve gsutil si hay cuenta gcloud; si no, ADC."""
    import waymo

    try:
        via = waymo.exigir_credenciales_gcs()
    except RuntimeError as error:
        sys.exit(str(error))
    if via == "python":
        print("Credenciales: Application Default Credentials (cliente Python).")
    else:
        cuentas = _ejecutar(["gcloud", "auth", "list", "--format=value(account)"])
        print(f"Cuenta activa: {cuentas.stdout.strip().splitlines()[0]}")
    try:
        waymo.listar_objetos(waymo.BUCKET, "training/lidar_box/", limite=1)
    except RuntimeError as error:
        sys.exit(str(error))
    return shutil.which("gsutil")


def listar_segmentos(cantidad: int = 5) -> list[str]:
    """Lista los primeros segmentos disponibles del componente lidar_box."""
    import waymo

    objetos = waymo.listar_objetos(waymo.BUCKET, "training/lidar_box/", limite=cantidad)
    nombres = []
    for objeto in objetos:
        stem = Path(objeto["nombre"]).stem
        if stem:
            nombres.append(stem)
    if not nombres:
        sys.exit(
            "Acceso denegado o listado vacío del bucket de Waymo.\n"
            "  Suele significar que la cuenta autenticada no es la misma con la que\n"
            "  aceptaste los términos en https://waymo.com/open/download/"
        )
    return nombres[:cantidad]


def descargar(segmento: str) -> None:
    """Copia lidar_box y stats del segmento indicado a datos/waymo_real/."""
    import waymo

    DESTINO.mkdir(parents=True, exist_ok=True)
    rutas = waymo.descargar_segmento(segmento, DESTINO)
    for componente, destino in rutas.items():
        print(f"  {destino.relative_to(RAIZ)}  ({destino.stat().st_size / 1024**2:.1f} MB)")
    (DESTINO / "SEGMENTO.txt").write_text(segmento + "\n", encoding="utf-8")


def descargar_muestra(segmentos: list[str], solo_stats: bool = False) -> Path:
    """Descarga varios segmentos para el análisis de sesgo de muestreo.

    Cada segmento queda en ``datos/waymo_real/muestra/{nombre}/``. Se usa desde
    ``herramientas/analizar_sesgo_waymo.py``: un solo segmento no sirve para
    estudiar el sesgo, porque el clima y la hora son propiedades del segmento
    completo y no varían dentro de él.

    Args:
        solo_stats: baja únicamente el componente ``stats`` (~23 KB por
            segmento en vez de ~1 MB). Suficiente para caracterizar clima,
            hora y ubicación sobre muchos segmentos a bajo costo.
    """
    import waymo

    muestra = DESTINO / "muestra"
    for numero, segmento in enumerate(segmentos, start=1):
        carpeta = muestra / segmento
        carpeta.mkdir(parents=True, exist_ok=True)
        requeridos = ["stats"] if solo_stats else COMPONENTES
        pendientes = [c for c in requeridos if not (carpeta / f"{c}.parquet").exists()]
        if not pendientes:
            print(f"[{numero}/{len(segmentos)}] {segmento[:28]}… ya estaba")
            continue
        for componente in pendientes:
            try:
                waymo.descargar(componente, segmento, carpeta)
            except RuntimeError as error:
                print(f"   ⚠️ falló {componente} de {segmento}: {str(error)[:90]}")
        peso = sum(f.stat().st_size for f in carpeta.glob("*.parquet")) / 1024**2
        print(f"[{numero}/{len(segmentos)}] {segmento[:28]}…  {peso:.2f} MB")
    return muestra


def descargar_censo_stats(gsutil: str) -> Path:
    """Descarga el componente ``stats`` de **todos** los segmentos de training.

    Son ~798 segmentos de ~23 KB, unos 18 MB en total. Con ``gsutil -m`` y un
    comodín se copian en paralelo en una sola llamada, así que tarda minutos y
    no horas.

    A diferencia de ``--muestra N``, esto no es un muestreo: es el censo del
    split de entrenamiento. Elimina la pregunta de si la muestra era
    representativa, que es la objeción obvia a cualquier cifra de sesgo.
    """
    destino = DESTINO / "censo_stats"
    destino.mkdir(parents=True, exist_ok=True)
    print("Copiando stats de todos los segmentos de training (~18 MB, en paralelo)…")
    resultado = _ejecutar([gsutil, "-m", "cp", f"{BUCKET}/stats/*.parquet", str(destino)])
    archivos = list(destino.glob("*.parquet"))
    if not archivos:
        sys.exit(f"No se descargó nada:\n{resultado.stderr.strip()[:400]}")
    peso = sum(f.stat().st_size for f in archivos) / 1024**2
    print(f"{len(archivos)} segmentos · {peso:.1f} MB en {destino.relative_to(RAIZ)}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lote",
        type=int,
        metavar="N",
        help="baja N segmentos livianos, los traduce y deja detecciones_reales.parquet (recomendado en clase: 8)",
    )
    parser.add_argument("--segmento", help="nombre del segmento (sin .parquet)")
    parser.add_argument("--listar", action="store_true", help="solo listar segmentos disponibles")
    parser.add_argument(
        "--muestra",
        type=int,
        metavar="N",
        help="descarga N segmentos a datos/waymo_real/muestra/ para el análisis de sesgo",
    )
    parser.add_argument(
        "--solo-stats",
        action="store_true",
        help="con --muestra, baja solo el componente stats (~23 KB por segmento)",
    )
    parser.add_argument(
        "--censo-stats",
        action="store_true",
        help="baja el stats de TODOS los segmentos de training (~798, ~18 MB)",
    )
    parser.add_argument(
        "--tablas-chicas",
        action="store_true",
        help="en segmentos ya bajados, completa camera_box + pose + calibración (KB, no JPEG)",
    )
    args = parser.parse_args()

    if args.tablas_chicas:
        import waymo

        muestra = DESTINO / "muestra"
        existentes = waymo.segmentos_completos(muestra)
        if not existentes:
            sys.exit(
                "No hay segmentos completos en datos/waymo_real/muestra/.\n"
                "  Primero:  python herramientas/descargar_waymo.py --lote 8"
            )
        limite = args.lote if args.lote else waymo.LOTE_CLASE
        print(
            f"\nTablas chicas en hasta {limite} segmentos ya bajados "
            f"(hay {len(existentes)} completos)."
        )
        _comprobar_requisitos()
        hechos = waymo.completar_tablas_chicas(muestra, limite=limite)
        print(f"Listo: {len(hechos)} segmentos con {list(waymo.COMPONENTES_MANIPULABLES)}.")
        print("Siguiente: notebooks/14_opcional_waymo_buckets.ipynb sección 5.")
        return

    if args.lote:
        import waymo

        print(f"\nLote de clase: hasta {args.lote} segmentos livianos (~1 MB cada uno).")
        print("Si ya hay varios en muestra/, los ensambla (no pide GCS).")
        ya_hay = (
            len(waymo.segmentos_completos(DESTINO / "muestra")) >= 2
            or (DESTINO / "detecciones_reales.parquet").exists()
            or (
                (DESTINO / "lidar_box.parquet").exists()
                and (DESTINO / "stats.parquet").exists()
            )
        )
        if not ya_hay:
            _comprobar_requisitos()
        tabla, informe, ruta = waymo.cargar_o_preparar(DESTINO, n=args.lote)
        print(waymo.texto_informe(informe))
        print(f"\nTabla única: {ruta.relative_to(RAIZ)}")
        print("Siguiente: notebooks/14_opcional_waymo_buckets.ipynb")
        return

    if args.muestra:
        import waymo

        existentes = waymo.segmentos_completos(DESTINO / "muestra")
        if len(existentes) >= args.muestra:
            print(
                f"\nYa hay {len(existentes)} segmentos completos en muestra/. "
                "No baja de nuevo."
            )
            tabla, informe, ruta = waymo.cargar_o_preparar(DESTINO, n=args.muestra)
            print(waymo.texto_informe(informe))
            print(f"\nTabla única: {ruta.relative_to(RAIZ)}")
            print("Pipeline: cd kedro_mly1101 && uv run kedro run --pipeline waymo_real")
            return
        _comprobar_requisitos()
        cantidad = args.muestra
        disponibles = listar_segmentos(cantidad=cantidad)
        peso = "~23 KB" if args.solo_stats else "~1 MB"
        print(f"\nDescargando {len(disponibles)} segmentos ({peso} cada uno)…")
        muestra = descargar_muestra(disponibles, solo_stats=args.solo_stats)
        print(f"Datos en: {muestra.relative_to(RAIZ)}")
        if not args.solo_stats:
            tabla, informe, ruta = waymo.cargar_o_preparar(DESTINO, n=args.muestra)
            print(waymo.texto_informe(informe))
            print(f"\nTabla única: {ruta.relative_to(RAIZ)}")
        print("\nListo. Pipeline: cd kedro_mly1101 && uv run kedro run --pipeline waymo_real")
        return

    _comprobar_requisitos()

    if args.censo_stats:
        gsutil = shutil.which("gsutil")
        if not gsutil:
            sys.exit(
                "El censo de stats usa gsutil -m. Instálalo o baja con --muestra 40."
            )
        descargar_censo_stats(gsutil)
        print("\nListo. Ahora puedes ejecutar:  python herramientas/analizar_sesgo_waymo.py")
        return

    cantidad = args.muestra if args.muestra else 5
    disponibles = listar_segmentos(cantidad=cantidad)

    if args.listar:
        print(f"\nSegmentos disponibles (primeros {len(disponibles)}):")
        for nombre in disponibles:
            print("  ", nombre)
        return

    segmento = args.segmento or disponibles[0]
    print(f"\nSegmento: {segmento}")
    descargar(segmento)
    print("\nListo. Ahora puedes ejecutar:  pytest tests/test_mapeo_waymo.py -v")


if __name__ == "__main__":
    main()
