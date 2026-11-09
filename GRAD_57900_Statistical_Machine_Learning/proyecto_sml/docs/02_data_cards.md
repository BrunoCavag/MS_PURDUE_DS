# Fase 2 — Fichas de datos (data cards)

Todos los datasets son públicos. Los archivos crudos están en `data/raw/` con checksums MD5 (`data/raw/CHECKSUMS.md5`) y se pueden volver a descargar con `python scripts/download_data.py`. Se obtuvieron de espejos públicos en GitHub de los archivos originales. **En el reporte se debe citar la fuente oficial** (columna "Fuente oficial"), no el espejo.

## Resumen

| Dataset | n | d (num / cat) | Default | Ratio neg:pos | % celdas faltantes | Duplicados | Régimen TFM |
|---|---:|---|---:|---:|---:|---:|---|
| Australian | 690 | 14 (6 / 8) | 55.5%* | 0.8 | 0.0 | 0 | ≤ límite TabPFNv2 |
| German | 1,000 | 20 (7 / 13) | 30.0% | 2.3 | 0.0 | 0 | ≤ límite TabPFNv2 |
| HMEQ | 5,960 | 12 (10 / 2) | 19.9% | 4.0 | 7.4 | 0 | ≤ límite TabPFNv2 |
| Taiwan | 30,000 | 23 (20 / 3) | 22.1% | 3.5 | 0.0 | 35 | ≤ límite TabPFN-2.5 |
| GMSC | 150,000 | 10 (10 / 0) | 6.7% | 14.0 | 2.2 | 609 | > 50k: submuestrear el contexto |
| Home Credit (opcional) | ~307,000 | ~120 | ~8% | — | — | — | requiere Kaggle |

\* En Australian la clase positiva es "solicitud rechazada" y no un default observado (ver la ficha).

Los duplicados de Taiwan y GMSC son filas idénticas en todas las variables y el target. Son plausibles en datos reales (clientes con perfil idéntico), así que **no se eliminan**. Se reportan como característica del dato.

**Observación para el reporte:** los cinco datasets base tienen entre 10 y 23 variables. Los datasets públicos de crédito son de baja dimensionalidad, y por eso el eje de dimensionalidad se estudia añadiendo variables no informativas de forma controlada (y, si da el tiempo, con Home Credit).

---

## 1. Default of Credit Card Clients (Taiwan)

- **Fuente oficial:** UCI Machine Learning Repository, id 350. **Cita:** Yeh & Lien (2009), *Expert Systems with Applications* 36(2).
- **Espejo usado:** `FederatedAI/FATE/examples/data/UCI_Credit_Card.csv`. Es la versión CSV idéntica a la de Kaggle. También se incluye el `.xls` original.
- **Unidad:** clientes de tarjeta de crédito en Taiwán, de abril a septiembre de 2005.
- **Target:** `default.payment.next.month` (1 = incumple el pago del mes siguiente).
- **Variables:**
  - Límite de crédito, sexo, educación, estado civil y edad.
  - `PAY_0`…`PAY_6`: estado de pago de los últimos 6 meses, en códigos ordinales de −2 a 8. Se tratan como numéricos.
  - `BILL_AMT1–6`: montos facturados.
  - `PAY_AMT1–6`: montos pagados.
- **Problemas conocidos:** `EDUCATION` tiene códigos no documentados (0, 5 y 6) y `MARRIAGE` tiene el código 0. Se dejan como categoría propia para no imponer supuestos.
- **Licencia:** CC BY 4.0 según UCI (verificar en la página del dataset).

## 2. Statlog German Credit

- **Fuente oficial:** UCI, id 144. **Cita:** Hofmann (1994).
- **Espejo usado:** `JLZml/Credit-Scoring-Data-Sets` (`german.data` y `german.doc`).
- **Unidad:** préstamos de consumo en Alemania.
- **Target:** clase 2 = mal pagador (se recodifica a 1); clase 1 = buen pagador (0).
- **Variables:** 7 numéricas (duración, monto, tasa de cuota, antigüedad de residencia, edad, número de créditos y número de dependientes) y 13 categóricas con códigos `Axx` (los nombres están en `german.doc`).
- **Nota útil:** el dataset original trae una **matriz de costos 5:1**: clasificar como bueno a un mal pagador cuesta 5 veces más que el error contrario. Sirve para un análisis opcional sensible al costo.
- **Advertencia ética:** `personal_status_sex` y `foreign_worker` son atributos protegidos. Para el benchmark se mantienen porque son parte del dataset estándar, pero vale una línea en la sección de limitaciones.
- **Licencia:** CC BY 4.0 según UCI (verificar).

## 3. Statlog Australian Credit Approval

- **Fuente oficial:** UCI, id 143 (Quinlan).
- **Espejo usado:** `JLZml/Credit-Scoring-Data-Sets`.
- **Unidad:** solicitudes de tarjeta de crédito.
- **Target:** `A15` (1 = aprobada, 0 = rechazada). **Se modela "bad" = rechazada**, que es la clase mayoritaria (55.5%).
- **Variables:** 6 numéricas y 8 categóricas, **anonimizadas** (A1…A14).
- **Limitaciones:**
  - El target es una decisión de aprobación, no un default observado. Hay que explicitarlo en el reporte. Se incluye porque es un benchmark clásico de credit scoring y es el dataset más pequeño, ideal para el régimen de n bajo.
  - El donante ya imputó los faltantes con la moda o la media, así que no hay NaN.
  - Es el mismo dato subyacente que el dataset japonés `crx`, por lo que **crx no se usa** para no duplicar.

## 4. HMEQ (Home Equity)

- **Fuente oficial:** datos complementarios de Baesens, Roesch & Scheule (2016), *Credit Risk Analytics* (Wiley). El origen es SAS.
- **Espejo usado:** `JLZml/Credit-Scoring-Data-Sets`.
- **Unidad:** préstamos sobre el valor de la vivienda en EE.UU.
- **Target:** `BAD` (1 = incumplió o tiene morosidad grave).
- **Variables:** 10 numéricas (monto del préstamo, hipoteca adeudada, valor del inmueble, años en el empleo, reportes negativos, morosidades, antigüedad de la línea más vieja, consultas recientes, número de líneas y ratio deuda/ingreso) y 2 categóricas (`REASON`, `JOB`).
- **Faltantes reales en 11 de 12 columnas.** La más afectada es `DEBTINC` (~21%). Es el mejor dataset para ver cómo cada familia maneja NaN de forma nativa.

## 5. Give Me Some Credit (GMSC)

- **Fuente oficial:** competencia de Kaggle "Give Me Some Credit" (2011), archivo `cs-training.csv`. Solo se usa el train, porque el test de Kaggle no tiene etiquetas.
- **Espejo usado:** `JLZml/Credit-Scoring-Data-Sets`. Se incluye el diccionario de datos original (`.xls`).
- **Unidad:** prestatarios de consumo en EE.UU.
- **Target:** `SeriousDlqin2yrs` (90 días o más de atraso en los próximos 2 años).
- **Variables:** 10 numéricas: utilización de líneas revolventes, edad, contadores de atraso de 30–59, 60–89 y 90+ días, ratio de deuda, ingreso mensual, líneas abiertas, préstamos inmobiliarios y dependientes.
- **Problemas conocidos:**
  - Faltantes en `MonthlyIncome` (~20%) y `NumberOfDependents` (~2.6%).
  - Los tres contadores de atraso contienen **códigos centinela 96 y 98**. Para las familias logística y red neuronal se convierten en NaN más una bandera (`SentinelFlagger`). Para árboles y modelos fundacionales se dejan crudos, porque pueden aislarlos con un corte.
- **Licencia:** reglas de la competencia de Kaggle. Para la entrega conviene **no redistribuir** el CSV: basta con el enlace y el script de descarga. Si el profesor exige los archivos, incluirlos citando la fuente.

## 6. Home Credit Default Risk (opcional)

- **Fuente oficial:** competencia de Kaggle (2018). Solo `application_train.csv`, sin las tablas relacionales.
- **Requisitos:**
  1. `pip install kaggle`.
  2. Token en `~/.kaggle/kaggle.json`.
  3. Aceptar las reglas de la competencia.
  4. Ejecutar `python scripts/download_data.py --kaggle`.
- **Por qué vale la pena:** es el único dataset con dimensionalidad real alta (~120 variables), lo que permite contrastar el eje m sintético con un caso natural.

---

## Protocolo de partición (Experimento A: benchmark en condiciones naturales)

- **Validación cruzada:** K-fold estratificado repetido, 5 folds × 3 repeticiones = 15 folds de test por dataset. La semilla es `20261217`.
- **Índices guardados:** están en `data/splits/<dataset>.json.gz`. Todos los modelos usan exactamente las mismas particiones, por lo que las comparaciones son pareadas, como exigen Friedman y Wilcoxon.
- **Tuning:** dentro de cada fold de entrenamiento se separa un 20% estratificado de validación para HPO y early stopping. El fold de test nunca se toca.
- **Preprocesamiento:** se ajusta **solo con el fold de entrenamiento**.

| Familia | Modelos | Tratamiento |
|---|---|---|
| `linear` | Regresión logística | Mediana + indicador de faltante, estandarización, one-hot (categorías con <1% agrupadas), banderas de centinela |
| `tree` | XGBoost, LightGBM, CatBoost | Numéricas crudas (NaN nativo), categóricas con codificación ordinal |
| `nn` | MLP | Igual que `linear`, pero con transformación cuantil a normal |
| `tfm` | TabPFN, TabICL | Valores crudos con NaN y categóricas ordinales. Los autores desaconsejan escalar o hacer one-hot |

## Diseño experimental (Experimento B: condiciones controladas)

Implementado en `src/sml/perturbations.py`. La tabla completa está en `data/processed/experiment_design.csv` (450 condiciones factibles = 90 combinaciones × 5 réplicas).

| Eje | Niveles | Datasets |
|---|---|---|
| **n** (tamaño de train) | 250, 500, 1k, 2.5k, 5k, 10k, 20k | Todos (hasta donde alcance cada dataset) |
| **π** (tasa de default en train y test) | natural, 10%, 5%, 2%, 1% | HMEQ, Taiwan, GMSC. En German y Australian no es factible con suficientes positivos |
| **m** (variables no informativas, × d) | 0, 1, 3 | Todos |
| **n × π** (interacción) | grilla completa | Taiwan, GMSC |

- **Punto base:** n = 2,500 (o lo que permita el dataset), π natural y m = 0. Cada eje se varía con los otros dos fijos en el punto base.
- **Muestreo:** el test se extrae primero, con tamaño fijo (5,000 o n/3) y **la misma π que el train**, para simular un portafolio de bajo default completo. Luego se muestrea el train del resto.
- **Variables no informativas:** son copias de columnas reales con las filas permutadas. Conservan la distribución marginal exacta (incluidos los tipos categóricos y los NaN) pero no tienen relación con el target.
- **Factibilidad:** una condición se descarta si quedan menos de 10 positivos en el train.
