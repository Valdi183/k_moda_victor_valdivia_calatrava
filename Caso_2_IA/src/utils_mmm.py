"""
utils_mmm.py — Funciones compartidas para el proyecto K-Moda MMM
=================================================================
Importar en cualquier notebook con:
    import sys; sys.path.append('../src')
    from utils_mmm import adstock_geometrico, hill_saturation, normalizar_mmm
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Rutas estándar del proyecto ───────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
RAW   = ROOT / 'data' / 'raw'
PROC  = ROOT / 'data' / 'processed'
MODS  = ROOT / 'models'
FIGS  = ROOT / 'reports' / 'figures'


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONES DE MEDIOS
# ─────────────────────────────────────────────────────────────────────────────

def adstock_geometrico(serie: pd.Series, decay: float) -> pd.Series:
    """
    Adstock geométrico: A_t = x_t + decay * A_{t-1}

    Modela la memoria acumulada de la publicidad: el efecto de invertir
    en la semana t persiste parcialmente en semanas posteriores.

    Args:
        serie: Serie temporal de inversión (ordenada cronológicamente).
        decay: Tasa de retención λ ∈ [0, 1). 0 = sin memoria, 0.9 = memoria larga.

    Returns:
        Serie transformada con el mismo índice que la entrada.
    """
    if not (0 <= decay < 1):
        raise ValueError(f"decay debe estar en [0, 1). Recibido: {decay}")

    resultado = np.empty(len(serie))
    resultado[0] = serie.iloc[0]
    for i in range(1, len(serie)):
        resultado[i] = serie.iloc[i] + decay * resultado[i - 1]
    return pd.Series(resultado, index=serie.index, name=serie.name)


def adstock_geometrico_df(df: pd.DataFrame, columnas: list[str],
                           decays: dict[str, float]) -> pd.DataFrame:
    """
    Aplica adstock_geometrico a múltiples columnas de un DataFrame.

    Args:
        df:       DataFrame ordenado cronológicamente.
        columnas: Lista de columnas a transformar.
        decays:   Dict {columna: decay}. Si una columna no está en decays
                  se usa decay=0 (sin transformación).

    Returns:
        DataFrame con columnas adicionales sufijadas con '_adstock'.
    """
    df = df.copy()
    for col in columnas:
        d = decays.get(col, 0.0)
        df[f'{col}_adstock'] = adstock_geometrico(df[col].fillna(0), d)
    return df


def hill_saturation(x: np.ndarray | pd.Series,
                    K: float, S: float) -> np.ndarray | pd.Series:
    """
    Función Hill para modelar rendimientos decrecientes (saturación):
        f(x) = x^S / (K^S + x^S)

    Args:
        x: Valores de inversión (no negativos).
        K: Punto de media saturación (inflexión de la curva).
        S: Exponente de forma (S>1 → curva en S; S<1 → cóncava desde el origen).

    Returns:
        Valores transformados en [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return x**S / (K**S + x**S)


def hill_saturation_df(df: pd.DataFrame, columnas: list[str],
                        params: dict[str, dict]) -> pd.DataFrame:
    """
    Aplica hill_saturation a múltiples columnas.

    Args:
        df:       DataFrame de entrada.
        columnas: Lista de columnas a transformar.
        params:   Dict {columna: {'K': float, 'S': float}}.

    Returns:
        DataFrame con columnas adicionales sufijadas con '_sat'.
    """
    df = df.copy()
    for col in columnas:
        p = params.get(col, {'K': df[col].median(), 'S': 1.0})
        df[f'{col}_sat'] = hill_saturation(df[col].fillna(0), p['K'], p['S'])
    return df


def pipeline_media(df: pd.DataFrame, columnas: list[str],
                   decays: dict[str, float],
                   hill_params: dict[str, dict]) -> pd.DataFrame:
    """
    Pipeline completo de transformación de medios:
        inversión raw → adstock → saturación Hill

    Aplica adstock y luego Hill sobre el resultado del adstock.
    El output es la variable de medios lista para entrar al modelo.

    Returns:
        DataFrame con columnas '_transformed' (adstock + Hill aplicados).
    """
    df = adstock_geometrico_df(df, columnas, decays)
    adstock_cols = [f'{c}_adstock' for c in columnas]

    # Construir params de Hill para las columnas de adstock
    hill_params_ads = {
        f'{c}_adstock': hill_params.get(c, {'K': df[f'{c}_adstock'].median(), 'S': 1.0})
        for c in columnas
    }
    df = hill_saturation_df(df, adstock_cols, hill_params_ads)

    # Renombrar resultado final para claridad
    for c in columnas:
        df[f'{c}_transformed'] = df[f'{c}_adstock_sat']

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN Y PREPROCESADO
# ─────────────────────────────────────────────────────────────────────────────

def normalizar_mmm(df: pd.DataFrame, columnas: list[str],
                   metodo: str = 'minmax') -> tuple[pd.DataFrame, dict]:
    """
    Normaliza columnas para el modelo MMM.

    Args:
        df:       DataFrame de entrada.
        columnas: Columnas a normalizar.
        metodo:   'minmax' (escala [0,1]) | 'zscore' (media 0, std 1).

    Returns:
        (df_normalizado, params_dict) — params_dict permite invertir la transformación.
    """
    df = df.copy()
    params = {}
    for col in columnas:
        if metodo == 'minmax':
            mn, mx = df[col].min(), df[col].max()
            rng = mx - mn if mx != mn else 1.0
            df[f'{col}_norm'] = (df[col] - mn) / rng
            params[col] = {'min': mn, 'max': mx, 'metodo': 'minmax'}
        elif metodo == 'zscore':
            mu, sigma = df[col].mean(), df[col].std()
            sigma = sigma if sigma > 0 else 1.0
            df[f'{col}_norm'] = (df[col] - mu) / sigma
            params[col] = {'mean': mu, 'std': sigma, 'metodo': 'zscore'}
        else:
            raise ValueError(f"metodo no reconocido: {metodo}")
    return df, params


def invertir_normalizacion(serie: pd.Series, params: dict) -> pd.Series:
    """Deshace la normalización a escala original."""
    if params['metodo'] == 'minmax':
        rng = params['max'] - params['min']
        return serie * rng + params['min']
    elif params['metodo'] == 'zscore':
        return serie * params['std'] + params['mean']


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE EVALUACIÓN MMM
# ─────────────────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (evita división por cero)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rsquared_adj(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """R² ajustado por número de parámetros."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    n = len(y_true)
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)


def decomposicion_contribucion(modelo, X: pd.DataFrame,
                                y_mean: float) -> pd.DataFrame:
    """
    Calcula la contribución absoluta de cada variable al nivel de ventas.

    Para un modelo lineal: contribución_i = coef_i * X_i_mean
    Devuelve DataFrame con columna y contribución (€) y porcentaje.
    """
    contribs = {}
    for col, coef in zip(X.columns, modelo.coef_):
        contribs[col] = coef * X[col].mean()

    df = pd.DataFrame.from_dict(contribs, orient='index', columns=['contribucion_eur'])
    df['pct'] = df['contribucion_eur'] / y_mean * 100
    return df.sort_values('contribucion_eur', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE CARGA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_master(ruta: Path = None) -> pd.DataFrame:
    """Carga el dataset maestro con tipos correctos."""
    ruta = ruta or PROC / '01_master_semanal.csv'
    df = pd.read_csv(ruta, parse_dates=['semana_inicio'])
    df['ciudad'] = df['ciudad'].astype('category')
    df = df.sort_values(['ciudad', 'semana_inicio']).reset_index(drop=True)
    return df


def cargar_features(ruta: Path = None) -> pd.DataFrame:
    """Carga el dataset de features listo para modelado."""
    ruta = ruta or PROC / '02_features_mmm.csv'
    df = pd.read_csv(ruta, parse_dates=['semana_inicio'])
    df['ciudad'] = df['ciudad'].astype('category')
    df = df.sort_values(['ciudad', 'semana_inicio']).reset_index(drop=True)
    return df
