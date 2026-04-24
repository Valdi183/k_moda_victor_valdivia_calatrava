"""Actualiza las celdas markdown de los notebooks para tono academico de proyecto."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# ── NB01 ──────────────────────────────────────────────────────────────────────
with open('files/01_EDA_MMM.ipynb', encoding='utf-8') as f:
    nb1 = json.load(f)

nb1['cells'][0]['source'] = (
    "# K-Moda — Análisis Exploratorio de Datos (EDA) para Marketing Mix Modeling\n\n"
    "## Contexto del proyecto\n\n"
    "El presente notebook corresponde a la primera fase del proyecto de Marketing Mix Modeling (MMM) "
    "aplicado a K-Moda, una cadena de moda con presencia en 10 ciudades españolas "
    "(Madrid, Barcelona, Valencia, Sevilla, Bilbao, Zaragoza, Málaga, Murcia, Palma, A Coruña) "
    "con canales de venta físico, online y click & collect.\n\n"
    "El objetivo del EDA es comprender la estructura de los datos antes de cualquier modelado: "
    "distribución de la variable objetivo, comportamiento temporal, relación preliminar entre "
    "inversión en medios y ventas, y validación de la calidad de los datos disponibles.\n\n"
    "Se trabaja con **8 canales de medios**: Paid Search, Social Paid, Display, Video Online, "
    "Email CRM, Exterior, Radio Local y Prensa.\n\n"
    "> **Nota:** Este notebook no genera artefactos de salida. Su función es exploratoria "
    "y sirve de base para las decisiones de diseño del pipeline de feature engineering (NB02).\n"
).splitlines(keepends=True)

# Conclusiones NB01
for i, cell in enumerate(nb1['cells']):
    if cell.get('cell_type') == 'markdown' and '## 11. Conclusiones' in ''.join(cell['source']):
        nb1['cells'][i]['source'] = (
            "---\n## 11. Conclusiones del EDA\n\n"
            "### Hallazgos principales\n\n"
            "#### Variable objetivo\n"
            "- Las ventas semanales presentan **distribución asimétrica positiva** con coeficiente "
            "de variación superior al 30%. Se evalúa la conveniencia de aplicar log-transformación en fases posteriores.\n"
            "- Existe **estacionalidad anual clara**: picos en Navidad (semanas 50-52), Rebajas de "
            "invierno (semanas 3-6), Rebajas de verano (semanas 27-30) y Black Friday (semana 47).\n"
            "- Se observa una **tendencia creciente** en el periodo analizado, lo que motivará la "
            "inclusión de una variable de tendencia en el modelo.\n\n"
            "#### Inversión en medios\n"
            "- La distribución de inversión entre canales es heterogénea: Paid Search y Exterior "
            "concentran la mayor parte del presupuesto.\n"
            "- Las series de inversión presentan picos puntuales asociados a campañas estacionales.\n"
            "- La correlación contemporánea entre inversión y ventas es positiva pero moderada, "
            "lo que anticipa la necesidad del pipeline adstock + Hill para extraer la señal correctamente.\n\n"
            "#### Decisiones de diseño para NB02\n"
            "- Se aplicará adstock geométrico con decay calibrado por canal mediante grid search detrended.\n"
            "- Se aplicará saturación Hill con $S=2$ y $K$ estimado como mediana del adstock por canal.\n"
            "- Se incluirán variables de tendencia logarítmica, Fourier y flags de calendario como controles.\n"
        ).splitlines(keepends=True)
        break

with open('files/01_EDA_MMM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb1, f, ensure_ascii=False, indent=1)
print('NB01 OK')

# ── NB02 ──────────────────────────────────────────────────────────────────────
with open('files/02_feature_engineering.ipynb', encoding='utf-8') as f:
    nb2 = json.load(f)

nb2['cells'][0]['source'] = (
    "# K-Moda MMM — Feature Engineering\n\n"
    "**Input:** `data/processed/01_master_semanal.csv` — 2.610 filas x 34 columnas "
    "(10 ciudades x 261 semanas)  \n"
    "**Output:** `data/processed/02_features_mmm.csv` — dataset listo para modelado\n\n"
    "---\n\n## Objetivo\n\n"
    "El objetivo de este notebook es transformar las inversiones brutas en medios en "
    "**variables explicativas adecuadas para el MMM**, aplicando el pipeline de transformacion "
    "estandar en dos etapas: adstock geometrico y saturacion Hill.\n\n"
    "| Bloque | Transformacion | Justificacion |\n"
    "|---|---|---|\n"
    "| Inversion en medios | Adstock geometrico + Saturacion Hill | Modela memoria y rendimientos decrecientes |\n"
    "| Variable temporal | Tendencia logaritmica + Fourier | Captura tendencia y estacionalidad |\n"
    "| Eventos de calendario | Flags binarios | Aisla efectos puntuales |\n"
    "| Geografia | Dummies de ciudad | Controla heterogeneidad entre mercados |\n\n"
    "> **Decision de diseno:** se trabaja con la variable objetivo en niveles (`ventas_eur`), no en "
    "logaritmo. La log-transformacion se calcula y evalua pero no se emplea como target principal, "
    "ya que el modelo de dos etapas con restriccion de positividad resulta mas interpretable en escala original.\n"
).splitlines(keepends=True)

with open('files/02_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, ensure_ascii=False, indent=1)
print('NB02 OK')

# ── NB03 ──────────────────────────────────────────────────────────────────────
with open('files/03_modelo_mmm_base.ipynb', encoding='utf-8') as f:
    nb3 = json.load(f)

nb3['cells'][0]['source'] = (
    "# K-Moda MMM — Modelo Baseline\n\n"
    "**Input:** `data/processed/02_features_mmm.csv` — features preparadas  \n"
    "**Output:** modelo entrenado en `models/mmm_elasticnet_baseline.pkl`, metricas de validacion, "
    "contribuciones por canal\n\n"
    "---\n\n## Objetivo\n\n"
    "El objetivo de este notebook es estimar un modelo de regresion con regularizacion que sirva "
    "como **baseline del MMM**. Se propone una arquitectura de dos etapas que resuelve el problema "
    "de coeficientes negativos en canales de medios, habitual en modelos OLS o Ridge sin restricciones "
    "cuando los canales estan correlacionados entre si.\n\n"
    "La hipotesis de partida es que las ventas se pueden descomponer como:\n\n"
    "$$y_t = f_{ctrl}(\\mathbf{x}_{ctrl,t}) + f_{media}(\\mathbf{x}_{media,t}) + \\varepsilon_t$$\n\n"
    "donde $f_{ctrl}$ captura el efecto de variables de control (estacionalidad, festivos, clima) "
    "y $f_{media}$ captura el efecto de los medios publicitarios con coeficientes restringidos a ser no negativos.\n\n"
    "> **Nota sobre la validacion:** se utiliza **TimeSeriesSplit** (5 folds cronologicos) para la "
    "seleccion de hiperparametros, lo que garantiza que en cada fold el conjunto de validacion es "
    "siempre posterior al de entrenamiento, evitando filtracion de informacion futura.\n"
).splitlines(keepends=True)

with open('files/03_modelo_mmm_base.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb3, f, ensure_ascii=False, indent=1)
print('NB03 OK')

# ── NB04 ──────────────────────────────────────────────────────────────────────
with open('files/04_simulador_estrategico.ipynb', encoding='utf-8') as f:
    nb4 = json.load(f)

nb4['cells'][0]['source'] = (
    "# K-Moda MMM — Simulador Estrategico de Presupuesto\n\n"
    "**Input:** Modelo ElasticNet entrenado (`models/mmm_elasticnet_baseline.pkl`), dataset de features  \n"
    "**Output:** Escenarios simulados, redistribucion optima, proyecciones financieras, "
    "matriz de sensibilidad\n\n"
    "---\n\n## Objetivo\n\n"
    "El presente notebook implementa un **simulador iterativo de presupuesto** que, a partir "
    "del modelo de dos etapas estimado en NB03, permite:\n\n"
    "1. Evaluar el impacto en ventas de diferentes asignaciones presupuestarias entre los 8 canales\n"
    "2. Calcular el ROI marginal de cada canal: cuantas ventas adicionales genera cada euro extra\n"
    "3. Obtener la distribucion optima que maximiza las ventas predichas dado un presupuesto objetivo\n"
    "4. Analizar la sensibilidad del resultado ante variaciones del presupuesto total\n\n"
    "> **Limitacion metodologica:** el simulador escala el adstock historico proporcionalmente y "
    "lo limita al maximo historico observado para no extrapolar fuera de la zona calibrada de la "
    "curva Hill. El modelo no captura efectos estructurales de largo plazo ni cambios en la "
    "efectividad de los canales fuera del periodo de entrenamiento.\n"
).splitlines(keepends=True)

# Conclusiones NB04
for i, cell in enumerate(nb4['cells']):
    if cell.get('cell_type') == 'markdown' and 'Conclusiones' in ''.join(cell['source']):
        nb4['cells'][i]['source'] = (
            "## Conclusiones del simulador\n\n"
            "### Resultados obtenidos\n\n"
            "1. **Escenario de reduccion (-10%)**  \n"
            "   La perdida de ventas estimada es inferior al 0.5%, lo que sugiere que existe "
            "margen para reducir el presupuesto total sin un impacto proporcional en ventas, "
            "siempre que la redistribucion entre canales se realice de forma eficiente.\n\n"
            "2. **Escenario de aumento (+10%)**  \n"
            "   El incremento de ventas predicho es modesto, lo que indica que varios canales "
            "se encuentran en la zona de saturacion de la curva Hill: el retorno marginal "
            "de invertir mas en el mix actual es decreciente.\n\n"
            "3. **Escenario optimo (12M euros)**  \n"
            "   La optimizacion encuentra un mix que, con un presupuesto un 21% inferior al "
            "historico (15.2M euros), produce una caida de ventas inferior al 0.2%. Esto "
            "evidencia que el mix historico presenta ineficiencias en la distribucion entre canales.\n\n"
            "4. **ROI marginal por canal**  \n"
            "   Los canales con mayor ROI marginal son Paid Search y Video Online. "
            "Email/CRM presenta coeficiente nulo, lo que indica que su contribucion no es "
            "estadisticamente distinguible en el periodo analizado.\n\n"
            "### Limitaciones\n\n"
            "- El modelo asume estacionariedad en los efectos de adstock.\n"
            "- Los parametros Hill se calibran sobre datos historicos y pueden desactualizarse "
            "si el mercado cambia estructuralmente.\n"
            "- La optimizacion esta restringida al rango historico observado; los resultados "
            "fuera de ese rango no son extrapolables con este modelo.\n"
        ).splitlines(keepends=True)
        break

with open('files/04_simulador_estrategico.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb4, f, ensure_ascii=False, indent=1)
print('NB04 OK')

print('\nTodos los notebooks actualizados.')
