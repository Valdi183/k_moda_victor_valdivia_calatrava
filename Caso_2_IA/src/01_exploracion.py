"""
K-Moda MMM — Paso 01: Exploración inicial de datos
=====================================================

Requisitos:
    pip install pandas tabulate colorama
"""

import os
import pandas as pd
from pathlib import Path

# ── Configuración de rutas ────────────────────────────────────────────────────
RAW_PATH = Path("data/raw")          # Carpeta con los CSVs originales
REPORT_PATH = Path("data/processed") # Carpeta de salida

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# ── Colores para consola (sin dependencias externas si no están instaladas) ───
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    OK    = Fore.GREEN  + "✔" + Style.RESET_ALL
    WARN  = Fore.YELLOW + "⚠" + Style.RESET_ALL
    ERR   = Fore.RED    + "✘" + Style.RESET_ALL
except ImportError:
    OK, WARN, ERR = "OK", "WARN", "ERR"

# ── Separadores a probar (detecta automáticamente) ────────────────────────────
SEP_CANDIDATES = [",", ";", "\t", "|"]

def detect_sep(filepath: Path) -> str:
    """Lee las primeras líneas del archivo y detecta el separador más probable."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(2048)
    counts = {sep: sample.count(sep) for sep in SEP_CANDIDATES}
    return max(counts, key=counts.get)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Carga un CSV detectando separador y encoding automáticamente."""
    sep = detect_sep(filepath)
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
            return df
        except Exception:
            continue
    raise ValueError(f"No se pudo leer: {filepath}")


# ── Función principal de exploración ─────────────────────────────────────────
def explorar_tabla(filepath: Path) -> dict:
    """
    Analiza un CSV y devuelve un diccionario con toda la información relevante.
    """
    nombre = filepath.stem  # nombre sin extensión
    print(f"\n{'='*60}")
    print(f"  {nombre}")
    print(f"{'='*60}")

    df = load_csv(filepath)

    # Dimensiones
    filas, cols = df.shape
    print(f"  {OK} Shape: {filas:,} filas × {cols} columnas")

    # Columnas y tipos
    print(f"\n  Columnas y tipos:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_nulos = df[col].isnull().sum()
        pct_nulos = (n_nulos / filas * 100) if filas > 0 else 0
        estado = OK if n_nulos == 0 else (WARN if pct_nulos < 10 else ERR)
        print(f"    {estado}  {col:<35} {dtype:<12} nulos: {n_nulos:>6} ({pct_nulos:.1f}%)")

    # Duplicados exactos
    n_dup = df.duplicated().sum()
    estado_dup = OK if n_dup == 0 else WARN
    print(f"\n  {estado_dup} Filas duplicadas exactas: {n_dup:,}")

    # Muestra de las primeras filas
    print(f"\n  Primeras 3 filas:")
    print(df.head(3).to_string(index=False))

    # Estadísticas para columnas numéricas
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  Estadísticas numéricas:")
        print(df[num_cols].describe().round(2).to_string())

    # Detección de posibles columnas de fecha
    posibles_fechas = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["fecha", "date", "semana", "week", "periodo"])
    ]
    if posibles_fechas:
        print(f"\n  Columnas de fecha detectadas: {posibles_fechas}")
        for col in posibles_fechas:
            muestra = df[col].dropna().astype(str).head(5).tolist()
            print(f"    {col}: {muestra}")

    # Valores únicos en columnas categóricas (baja cardinalidad)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 20:
            valores = df[col].unique().tolist()
            print(f"\n  Valores únicos en '{col}' ({n_unique}): {valores}")

    return {
        "tabla": nombre,
        "filas": filas,
        "columnas": cols,
        "nulos_total": int(df.isnull().sum().sum()),
        "duplicados": int(n_dup),
        "col_nombres": list(df.columns),
    }


# ── Ejecución ─────────────────────────────────────────────────────────────────
def main():
    archivos = sorted(RAW_PATH.glob("*.csv"))

    if not archivos:
        print(f"\n{ERR}  No se encontraron archivos .csv en: {RAW_PATH.resolve()}")
        print("  Comprueba que la carpeta existe y contiene los archivos del caso.")
        return

    print(f"\n{'#'*60}")
    print(f"  K-Moda MMM — Exploración de {len(archivos)} tabla(s)")
    print(f"{'#'*60}")

    resumen = []
    for csv_file in archivos:
        info = explorar_tabla(csv_file)
        resumen.append(info)

    # ── Resumen final en tabla ────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*60}")
    print(f"  {'Tabla':<35} {'Filas':>8} {'Cols':>5} {'Nulos':>7} {'Dups':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*5} {'-'*7} {'-'*6}")
    for r in resumen:
        estado = OK if r["nulos_total"] == 0 and r["duplicados"] == 0 else WARN
        print(f"  {estado} {r['tabla']:<33} {r['filas']:>8,} {r['columnas']:>5} "
              f"{r['nulos_total']:>7,} {r['duplicados']:>6,}")

    # ── Guardar resumen en CSV ────────────────────────────────────────────────
    df_resumen = pd.DataFrame(resumen)
    out = REPORT_PATH / "00_resumen_exploracion.csv"
    df_resumen.to_csv(out, index=False)
    print(f"\n  {OK} Resumen guardado en: {out}")

    # ── Próximos pasos ────────────────────────────────────────────────────────
    print(f"""
  PRÓXIMOS PASOS
  ──────────────
  1. Revisa las tablas con WARN o ERR arriba
  2. Ejecuta: python src/02_limpieza.py
  3. Prioriza PEDIDOS y VENTAS_LINEAS (son la variable dependiente Yt del MMM)
""")


if __name__ == "__main__":
    main()