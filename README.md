# K-Moda — Marketing Mix Model (MMM)

> Proyecto de modelado estadístico para la estimación de la contribución de medios publicitarios a las ventas de una cadena de moda española, con implementación de un simulador de optimización presupuestaria.

---

## Descripción del proyecto

El presente proyecto desarrolla un **Marketing Mix Model (MMM)** aplicado a K-Moda, una cadena de moda con presencia en 10 ciudades de España. El objetivo es cuantificar el efecto de cada canal publicitario sobre las ventas semanales, teniendo en cuenta que dicho efecto no es ni inmediato ni lineal.

Se trabaja con datos históricos de 5 años (2020–2024) a granularidad semanal y nivel de ciudad, lo que constituye un panel de **2.610 observaciones**. A partir de ellos se estima un modelo de regresión con regularización que permite:

- Atribuir a cada canal su contribución al volumen de ventas
- Calcular el ROI marginal de cada euro invertido en publicidad
- Simular escenarios alternativos de distribución presupuestaria
- Obtener la asignación óptima de un presupuesto dado

---

## Motivación

El principal reto de un MMM es que la publicidad presenta dos propiedades que dificultan su modelado directo:

1. **Efecto memoria (adstock):** el impacto de una inversión publicitaria no se agota en la semana en que se realiza, sino que decae gradualmente en semanas posteriores
2. **Rendimientos decrecientes (saturación):** la relación entre inversión y ventas no es lineal; a partir de cierto nivel, cada euro adicional genera un retorno marginal menor

Sin modelar estas dos propiedades, los coeficientes de regresión resultan sesgados e inútiles para la toma de decisiones. El presente proyecto propone un pipeline de transformación que aborda ambos efectos antes de la estimación del modelo.

Adicionalmente, un problema clásico en MMM es la aparición de coeficientes negativos en canales de medios cuando estos están correlacionados entre sí. Se propone una arquitectura de dos etapas que resuelve este problema imponiendo restricciones de signo únicamente sobre los canales de medios.

---

## Datos

| Fichero | Descripción |
|---|---|
| `ventas_lineas.csv` | Ventas a nivel de línea de pedido (ciudad, semana, producto) |
| `inversion_medios_semanal.csv` | Inversión semanal por canal y ciudad (2020–2024) |
| `trafico_tienda_web_diario.csv` | Sesiones web y visitas a tienda física |
| `calendario_ciudad.csv` | Variables de calendario: festivos, rebajas, Black Friday, Navidad |
| `clientes.csv` / `pedidos.csv` | Datos transaccionales para variables de control |

**Estructura del panel:** 10 ciudades × 261 semanas = 2.610 filas  
**Periodo:** enero 2020 — diciembre 2024  
**Ciudades:** Barcelona, Madrid, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Murcia, Palma, A Coruña  
**Canales de medios:** Paid Search, Social Paid, Display, Video Online, Email/CRM, Exterior, Radio Local, Prensa

---

## Metodología

El proyecto se estructura en cuatro notebooks que deben ejecutarse en orden:

### NB01 — Análisis exploratorio (EDA)

Análisis univariado y multivariado de las series temporales de ventas e inversión. Se estudia la estacionalidad, los efectos de calendario, la distribución de la variable objetivo y las correlaciones preliminares con y sin lag entre inversión y ventas.

### NB02 — Feature Engineering

Transformación de la inversión bruta en variables explicativas para el modelo mediante el siguiente pipeline:

**Paso 1: Adstock geométrico**

$$A_t = x_t + \lambda \cdot A_{t-1}, \quad \lambda \in [0, 1)$$

donde $\lambda$ es la tasa de retención calibrada por canal mediante un grid search con correlación detrended sobre el conjunto de entrenamiento. Se imponen restricciones por tipo de canal para evitar que el algoritmo capture la tendencia de ventas en lugar de la memoria publicitaria:

| Canal | Rango de búsqueda $\lambda$ |
|---|---|
| Exterior | [0.30, 0.75] |
| Radio / Display / Video | [0.20, 0.65] |
| Paid Search / Social | [0.00, 0.60] |
| Email / CRM | [0.00, 0.35] |

**Paso 2: Saturación Hill**

$$f(A_t) = \frac{A_t^S}{K^S + A_t^S}$$

donde $K$ es el punto de media saturación (estimado como la mediana del adstock por canal) y $S = 2$ para todos los canales, lo que produce una curva en S con inflexión en $K$.

Además se construyen variables de control: tendencia logarítmica, componentes de Fourier para estacionalidad anual, flags de eventos de calendario y dummies de ciudad.

### NB03 — Modelo base

Se propone una arquitectura de **dos etapas** que separa el tratamiento de variables de control y de medios:

**Etapa 1 — Ridge sobre controles:**
$$\hat{y}^{(1)} = \mathbf{x}_{ctrl}^T \boldsymbol{\beta}_{ctrl}$$

Sin restricción de signo, ya que las variables de control (temperatura, festivos, etc.) pueden tener efectos negativos perfectamente válidos.

**Etapa 2 — ElasticNet con coeficientes positivos sobre medios:**
$$\hat{y}^{(2)} = \mathbf{x}_{media}^T \boldsymbol{\beta}_{media}, \quad \beta_{media,j} \geq 0 \; \forall j$$

La restricción de positividad garantiza que ningún canal presente un efecto negativo en ventas, lo que es una condición necesaria para que los resultados sean interpretables y económicamente coherentes.

La predicción final es $\hat{y} = \hat{y}^{(1)} + \hat{y}^{(2)}$.

La selección de hiperparámetros se realiza mediante **TimeSeriesSplit con 5 folds** sobre el conjunto de entrenamiento (nunca validación cruzada aleatoria, que rompería la estructura temporal).

### NB04 — Simulador estratégico

Con el modelo entrenado se implementa la clase `BudgetSimulator` que opera de la siguiente forma:

1. Dado un presupuesto por canal $\{b_1, \ldots, b_8\}$, escala el adstock histórico proporcionalmente: $A_t^{(nuevo)} = A_t^{(hist)} \cdot \frac{b_j}{b_j^{(actual)}}$, limitado al máximo histórico observado para no extrapolar fuera de la zona calibrada de la curva Hill.
2. Re-aplica la transformación Hill al adstock escalado.
3. Predice las ventas con el modelo de dos etapas.

La optimización presupuestaria se formula como:

$$\max_{\mathbf{b}} \; \hat{y}(\mathbf{b}) \quad \text{s.t.} \quad \sum_j b_j = B, \quad b_j \in [lo_j, hi_j]$$

donde los bounds $[lo_j, hi_j]$ son asimétricos según el ROI marginal de cada canal, resolviéndose mediante SLSQP.

---

## Resultados

### Contribución de medios a las ventas (Peso%)

$$\text{Peso\%}_j = \frac{\hat{\beta}_j \cdot \bar{A}_j}{\sum_k \hat{\beta}_k \cdot \bar{A}_k} \times 100$$

| Canal | Peso% |
|---|---|
| Paid Search | 25.8% |
| Exterior | 22.7% |
| Video Online | 19.4% |
| Radio Local | 14.4% |
| Display | 9.4% |
| Prensa | 6.4% |
| Social Paid | 1.8% |
| Email / CRM | 0.0% |

### Análisis de escenarios

| Escenario | Presupuesto | Ventas predichas | ROI |
|---|---|---|---|
| Histórico real | 15.2 M€ | baseline | 10.7x |
| Reducción −20% | 12.1 M€ | −0.1% | 12.1x |
| Óptimo (12M€) | 12.0 M€ | −0.2% | 13.8x |

El resultado más relevante es que con un presupuesto un **21% inferior** al histórico, el modelo predice una caída de ventas inferior al **0.2%**, lo que implica una mejora sustancial del ROI. Esto sugiere que el mix histórico presenta ineficiencias significativas en la distribución entre canales.

---

## Dashboard interactivo

Se ha desarrollado un dashboard en Streamlit desplegado en Hugging Face Spaces que permite explorar los resultados de forma interactiva:

**[huggingface.co/spaces/Valdi121/MMM_project](https://huggingface.co/spaces/Valdi121/MMM_project)**

El dashboard incluye cuatro secciones: resumen ejecutivo, contribución por canal con ROI marginal, simulador de presupuesto con sliders por canal, y optimización con comparativa de redistribución.

---

## Estructura del repositorio

```
k_moda_victor_valdivia_calatrava/
│
├── files/                               # Notebooks del proyecto
│   ├── 01_EDA_MMM.ipynb                 # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb     # Pipeline adstock + Hill
│   ├── 03_modelo_mmm_base.ipynb         # Entrenamiento y validación del modelo
│   └── 04_simulador_estrategico.ipynb   # Simulador y optimización presupuestaria
│
├── data/
│   ├── raw/                             # Datos originales sin modificar
│   └── processed/                       # Artefactos generados por los notebooks
│       ├── 01_master_semanal.csv        # Panel consolidado (2.610 filas × 34 cols)
│       ├── 02_features_mmm.csv          # Features para el modelo (89 columnas)
│       ├── 02_hiperparametros_medios.csv # Decays λ y parámetros Hill por canal
│       ├── 03_peso_por_canal.csv        # Contribución % de cada canal
│       ├── 04_simulador_escenarios.csv  # Resultados de escenarios simulados
│       ├── 05_marginal_roi.csv          # ROI marginal por canal
│       ├── 06_budget_redistribution.csv # Redistribución óptima de presupuesto
│       ├── 07_financial_projections.csv # Proyecciones financieras por escenario
│       └── 08_sensitivity_matrix.csv   # Análisis de sensibilidad presupuestaria
│
├── models/
│   └── mmm_elasticnet_baseline.pkl     # Modelo serializado (joblib)
│
├── src/
│   └── utils_mmm.py                    # Funciones compartidas: adstock, Hill, métricas
│
├── reports/figures/                    # Gráficos generados por los notebooks
│
├── hf_deploy/                          # Código del dashboard (Hugging Face Spaces)
│   ├── app.py
│   ├── requirements.txt
│   └── data/
│
└── requirements.txt                    # Dependencias del proyecto
```

---

## Reproducción de resultados

### Requisitos

- Python 3.10+
- `pip install -r requirements.txt`

### Orden de ejecución

Los notebooks deben ejecutarse secuencialmente con **Restart Kernel and Run All Cells**:

```
01_EDA_MMM.ipynb              →  exploración, sin artefactos de salida
02_feature_engineering.ipynb  →  genera data/processed/02_features_mmm.csv
03_modelo_mmm_base.ipynb      →  genera models/mmm_elasticnet_baseline.pkl
04_simulador_estrategico.ipynb → genera data/processed/04_*.csv a 08_*.csv
```

### Dashboard en local

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
| Modelado | scikit-learn (Ridge, ElasticNet positivo), scipy (SLSQP) |
| Preprocesado | pandas, numpy |
| Visualización | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Despliegue | Hugging Face Spaces |
| Entorno de desarrollo | Jupyter Notebooks, VS Code |

---

## Autor

**Víctor Valdivia Calatrava**
