# K-Moda — Marketing Mix Model (MMM)

> **¿Cuánto vale realmente cada euro invertido en publicidad?**
> Este proyecto construye un modelo estadístico que responde exactamente esa pregunta para K-Moda, una cadena de moda con presencia en 10 ciudades de España.

---

## El problema de negocio

K-Moda invierte anualmente más de **15 millones de euros** en 8 canales de publicidad: Paid Search, Social Paid, Display, Video Online, Email/CRM, Exterior, Radio Local y Prensa.

Hasta ahora, la empresa no disponía de ninguna herramienta que le permitiese responder preguntas como:

- ¿Cuántas ventas genera cada canal de forma aislada?
- ¿Estamos invirtiendo demasiado en canales poco eficientes?
- Si el CFO nos recorta el presupuesto un 20%, ¿dónde deberíamos recortar para perder el mínimo de ventas?
- ¿Cuál sería el reparto óptimo de un presupuesto de 12M€?

---

## ¿Qué es un Marketing Mix Model?

Un **Marketing Mix Model (MMM)** es un modelo de regresión que descompone las ventas en sus causas: ¿cuánto se explica por la publicidad, por la estacionalidad, por los eventos especiales, por el contexto económico...?

La dificultad está en que la publicidad no funciona de forma instantánea ni lineal:
- Una valla publicitaria vista hoy puede generar una venta dentro de tres semanas (**efecto memoria**)
- Doblar la inversión en un canal no dobla las ventas (**rendimientos decrecientes**)

El MMM modela ambos efectos matemáticamente para obtener estimaciones fiables y defendibles ante el equipo directivo.

---

## Datos utilizados

| Fuente | Descripción |
|---|---|
| `ventas_lineas.csv` | Ventas semanales por ciudad y línea de producto |
| `inversion_medios_semanal.csv` | Inversión por canal y ciudad (2020–2024) |
| `trafico_tienda_web_diario.csv` | Sesiones web y visitas a tienda |
| `calendario_ciudad.csv` | Festivos, rebajas, Black Friday, Navidad, Semana Santa |
| `clientes.csv` / `pedidos.csv` | Datos transaccionales para variables de control |

**Estructura del panel de datos:** 10 ciudades × 261 semanas = **2.610 observaciones**  
**Periodo:** enero 2020 — diciembre 2024  
**Ciudades:** Barcelona, Madrid, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Murcia, Palma, A Coruña

---

## Pipeline del modelo

El proyecto sigue un pipeline en 4 etapas:

### 1. Exploración (NB01)
Análisis exploratorio de las series temporales de ventas e inversión por ciudad y canal. Identificación de estacionalidad, outliers y correlaciones preliminares.

### 2. Feature Engineering (NB02)
Transformación de las inversiones brutas en variables explicativas del modelo mediante dos pasos:

**Adstock geométrico** — modela el efecto memoria de la publicidad:

```
A_t = inversión_t + λ × A_{t-1}
```

Cada canal tiene su propio decay (λ) calibrado por correlación detrended con ventas dentro de rangos económicamente razonables por tipo de canal:

| Canal | Decay (λ) | Interpretación |
|---|---|---|
| Exterior | 0.75 | Memoria larga (vallas duran semanas) |
| Radio / Display / Video | 0.65 | Memoria media |
| Paid Search / Social | 0.50–0.60 | Memoria media-corta |
| Email / CRM | 0.35 | Efecto casi inmediato |

**Saturación Hill** — modela rendimientos decrecientes:

```
f(x) = x^S / (K^S + x^S)
```

Donde K es el punto de media saturación y S controla la forma de la curva. Evita que el modelo asuma que doblar la inversión dobla las ventas.

### 3. Modelado (NB03)
Arquitectura de **dos etapas** para garantizar coeficientes con sentido económico:

- **Etapa 1 — Ridge** sobre variables de control (estacionalidad, festivos, clima, turismo, dummies de ciudad): sin restricción de signo, captura el contexto externo
- **Etapa 2 — ElasticNet con coeficientes positivos** sobre variables de medios: fuerza que todos los canales tengan efecto ≥ 0 en ventas, evitando los coeficientes negativos que aparecen en modelos sin restricción cuando los canales están correlacionados entre sí

La validación se realiza con **TimeSeriesSplit** (5 folds cronológicos, nunca aleatorios) para respetar la estructura temporal de los datos.

### 4. Simulador estratégico (NB04)
Con el modelo entrenado se construye un `BudgetSimulator` que permite tres operaciones:

1. **Simular escenarios**: dado cualquier reparto de presupuesto por canal, predice las ventas resultantes
2. **ROI Marginal**: calcula cuántos euros de ventas adicionales genera cada euro extra invertido en cada canal
3. **Optimización**: encuentra el reparto que maximiza las ventas predichas dado un presupuesto total, con restricciones que mantienen la solución dentro del rango histórico observado (evita extrapolación fuera de la zona calibrada del modelo)

---

## Resultados principales

### Contribución de cada canal a las ventas

| Canal | Peso% | Posición |
|---|---|---|
| Paid Search | 25.8% | 1º |
| Exterior | 22.7% | 2º |
| Video Online | 19.4% | 3º |
| Radio Local | 14.4% | 4º |
| Display | 9.4% | 5º |
| Prensa | 6.4% | 6º |
| Social Paid | 1.8% | 7º |
| Email / CRM | 0.0% | 8º |

### Conclusión estratégica

Con un presupuesto de **12M€** redistribuido según el modelo óptimo se obtienen prácticamente las mismas ventas que con los **15.2M€** históricos. El ROI total pasa de **10.7x a 13.8x**.

La palanca no es gastar más — es **redistribuir mejor**.

---

## Dashboard interactivo

El proyecto incluye un dashboard desplegado en Hugging Face Spaces que permite explorar los resultados de forma interactiva:

**[huggingface.co/spaces/Valdi121/MMM_project](https://huggingface.co/spaces/Valdi121/MMM_project)**

El dashboard tiene 4 secciones:
- **Resumen Ejecutivo** — KPIs y mix de inversión actual
- **Contribución por Canal** — Peso% y ROI marginal por canal
- **Simulador de Presupuesto** — ajusta sliders por canal y ve el impacto en ventas
- **Optimización** — calcula el reparto óptimo para cualquier presupuesto objetivo

---

## Estructura del repositorio

```
k_moda_victor_valdivia_calatrava/
│
├── files/                          # Notebooks del proyecto
│   ├── 01_EDA_MMM.ipynb            # Exploración de datos
│   ├── 02_feature_engineering.ipynb # Adstock + Hill saturation
│   ├── 03_modelo_mmm_base.ipynb    # Entrenamiento del modelo
│   └── 04_simulador_estrategico.ipynb # Simulador y optimización
│
├── data/
│   ├── raw/                        # Datos originales sin modificar
│   └── processed/                  # CSVs generados por los notebooks
│       ├── 01_master_semanal.csv   # Dataset consolidado (2.610 filas)
│       ├── 02_features_mmm.csv     # Features para el modelo (89 columnas)
│       ├── 02_hiperparametros_medios.csv # Decays y parámetros Hill calibrados
│       ├── 03_peso_por_canal.csv   # Contribución % de cada canal
│       ├── 04_simulador_escenarios.csv  # Resultados de escenarios
│       ├── 05_marginal_roi.csv     # ROI marginal por canal
│       ├── 06_budget_redistribution.csv # Redistribución óptima
│       ├── 07_financial_projections.csv # Proyecciones financieras
│       └── 08_sensitivity_matrix.csv    # Análisis de sensibilidad
│
├── models/
│   └── mmm_elasticnet_baseline.pkl # Modelo entrenado serializado
│
├── src/
│   └── utils_mmm.py                # Funciones compartidas (adstock, Hill, métricas)
│
├── reports/figures/                # Gráficos generados por los notebooks
│
├── hf_deploy/                      # Código del dashboard (Hugging Face)
│   ├── app.py                      # App Streamlit
│   ├── requirements.txt
│   └── data/                       # Copia de los datos para el dashboard
│
└── requirements.txt                # Dependencias del proyecto
```

---

## Cómo ejecutar el proyecto

### Requisitos
- Python 3.10+
- Instalar dependencias: `pip install -r requirements.txt`

### Orden de ejecución de los notebooks

Los notebooks deben ejecutarse en orden con **Restart Kernel and Run All Cells**:

```
01_EDA_MMM.ipynb               → exploración, no genera artefactos
02_feature_engineering.ipynb   → genera 02_features_mmm.csv
03_modelo_mmm_base.ipynb       → genera mmm_elasticnet_baseline.pkl + 03_peso_por_canal.csv
04_simulador_estrategico.ipynb → genera 04_simulador_escenarios.csv a 08_sensitivity_matrix.csv
```

### Lanzar el dashboard en local
```bash
cd hf_deploy
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack tecnológico

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.12 |
| Modelado | scikit-learn (Ridge, ElasticNet), scipy (SLSQP) |
| Datos | pandas, numpy |
| Visualización | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Despliegue | Hugging Face Spaces |
| Entorno | Jupyter Notebooks, VS Code |

---

## Autor

**Víctor Valdivia Calatrava**
