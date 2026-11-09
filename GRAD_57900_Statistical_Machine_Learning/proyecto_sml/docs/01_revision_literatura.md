# Fase 1 — Revisión de literatura

**Proyecto:** ¿Los modelos fundacionales tabulares (TFMs) igualan o superan a los GBDT en predicción de default de crédito, y bajo qué condiciones (tamaño de muestra, desbalance de clases, número de variables)?
**Entrega:** 17 de diciembre de 2026 · **Estado de este documento:** 4 de octubre de 2026

Este documento tiene tres partes:
1. **Hallazgo crítico y posicionamiento.** Antes de leer lo demás.
2. **Bibliografía anotada en español.** Qué dice cada paper y para qué te sirve.
3. **Borrador de "Related Work" en inglés.** Listo para pegar en la plantilla y ajustar.

Las claves entre corchetes (por ejemplo `[baesens2026foundation]`) corresponden a `references.bib`.

---

## 1. Hallazgo crítico: ya existe un benchmark casi idéntico

En mayo de 2026, Baesens, Lessmann y otros diez autores publicaron en arXiv *Foundation Models for Credit Risk Prediction: A Game Changer?* `[baesens2026foundation]`. Es el grupo que firmó los benchmarks de referencia en credit scoring (Baesens 2003, Lessmann 2015, Gunnarsson 2021). El paper:

- Compara 29 métodos sobre 14 datasets de PD, entre ellos 5 TFMs: TabPFN v1, TabPFNv2, TabPFN-Real, MITRA y TabICL.
- Usa Optuna con hasta 20 trials por fold, 5-fold CV, test de Friedman y Wilcoxon con corrección de Holm.
- Incluye **Taiwan, GMSC, German, HMEQ y Home Credit**, que son 4 de nuestros 5 datasets base más el opcional.
- Concluye que TabICL tiene el mejor AUC promedio (0.752, frente a 0.749 de CatBoost). Las diferencias entre los mejores métodos son pequeñas.
- Concluye que la ventaja de los TFMs crece cuanto más pequeño es el dataset. Lo muestran con dos análisis: (a) la correlación de Spearman entre el ranking de cada método y n, y (b) curvas de aprendizaje submuestreando de 500 a 15,000 filas.

**Consecuencia:** si el proyecto solo replica "TFM vs GBDT en varios datasets + curva de aprendizaje", reproduce un resultado ya publicado. Eso está bien como punto de partida (y debe citarse), pero no sirve como contribución.

### Lo que ese paper deja abierto, y que se vuelve nuestra contribución

| Brecha en Baesens et al. (2026) | Cómo la cubrimos |
|---|---|
| **Desbalance:** lo mencionan como motivación (low-default portfolios), pero **no lo manipulan**. Sus datasets van de 6.7% a 40% de default, y el efecto del desbalance queda confundido con todo lo demás que cambia entre datasets. | **Eje π controlado:** dentro de un mismo dataset bajamos la tasa de default de forma artificial (natural → 10% → 5% → 2% → 1%), con n fijo. Además cruzamos n × π en Taiwan y GMSC. |
| **Dimensionalidad:** no la manipulan. Además, casi todos los datasets públicos de crédito tienen pocas variables (los nuestros, entre 10 y 23). | **Eje m controlado:** añadimos variables no informativas (copias permutadas de columnas reales, como en Grinsztajn et al. 2022) en proporción 0×, 1× y 3× d. |
| **Calibración:** en su propia Tabla 4, TabICL gana en AUC pero tiene **peor Brier** (0.162) que CatBoost (0.133) y TabPFNv2 (0.132). No lo discuten. En crédito la PD calibrada es lo que se usa en provisiones (IFRS 9), capital (IRB) y pricing. | Reportamos **discriminación y calibración por separado** (AUC/Gini/KS frente a Brier, log-loss y ECE) y analizamos cómo cambia la calibración de cada familia cuando baja π. |
| **Versiones:** evalúan TabICL v1 y TabPFNv2. Desde entonces salieron TabPFN-2.5, TabPFN-3 y TabICLv2. | Incluimos al menos una generación reciente (TabPFN-3 o 2.5, y TabICL o TabICLv2), con TabPFNv2 como ancla reproducible porque sus pesos son Apache 2.0. |
| **Diseño entre datasets:** su análisis del efecto de n es correlacional entre datasets, salvo la curva de aprendizaje. | **Diseño dentro de cada dataset** (one-factor-at-a-time + grilla n × π), con 5 réplicas por condición: los efectos se atribuyen al factor manipulado. |

**Pregunta refinada, para la propuesta y la introducción:**
> Given that recent large-scale evidence shows TFMs are competitive with tuned GBDTs for PD modeling on average [baesens2026foundation], *under which controlled data conditions*—training-set size, default rate, and number of uninformative features—does that advantage hold, and does it extend from discrimination to **calibration**?

Esta pregunta también responde directamente a la parte de la rúbrica que pide explicar *"what did you expect, what did you find instead"*. Hay hipótesis claras y falsables:

- **H1 (n):** los TFMs superan a los GBDT para n chico y la brecha se cierra o invierte para n grande. Esperamos replicar a Baesens et al.
- **H2 (π):** con default rate bajo, los TFMs mantienen mejor AUC que los GBDT porque no necesitan tuning con pocos positivos, pero su calibración se degrada más. Esto es abierto: es la apuesta del proyecto.
- **H3 (m):** los GBDT son robustos a variables no informativas (Grinsztajn et al. 2022). Para los TFMs el resultado es incierto: la atención sobre columnas puede diluirse.

> **Acción sugerida:** en tu próximo correo al profesor, menciona en una línea que encontraste este paper y que tu proyecto se enfoca en los ejes de desbalance, dimensionalidad y calibración que ese trabajo deja abiertos. Eso muestra dominio del estado del arte.

---

## 2. Bibliografía anotada

**Estado de verificación:** ✅ = verificado hoy contra arXiv, el editor o el repositorio oficial. ⚠️ = escrito de memoria; confirma DOI y páginas antes de entregar (también marcado con `note = {VERIFY…}` en el .bib). 📄 = PDF descargable con `scripts/download_papers.py`. 🔒 = paywall, conseguir vía Purdue Libraries.

### A. Modelos fundacionales tabulares (núcleo del proyecto)

**[baesens2026foundation]** ✅ 📄 — *Foundation Models for Credit Risk Prediction: A Game Changer?* (arXiv 2605.18147).
Es el trabajo más cercano; ver la sección 1. Úsalo en tres lugares del reporte: la introducción (motivación), la metodología (adoptamos su protocolo estadístico, Friedman + Wilcoxon-Holm) y la discusión (comparar nuestros resultados en n con los suyos). Un detalle útil para nuestra discusión: reportan que TabPFNv2 en LGD empeora por encima de unas 8,000 observaciones y no lo explican.

**[hollmann2023tabpfn]** ✅ 📄 — TabPFN v1 (ICLR 2023).
Presenta el concepto de *Prior-Data Fitted Network*: un transformer preentrenado con millones de datasets sintéticos que predice en un solo forward pass, sin entrenamiento. Estaba limitado a unas 1,000 filas y 100 variables, solo numéricas. Para nosotros, es la base conceptual del método.

**[muller2022transformers]** ✅ 📄 — *Transformers Can Do Bayesian Inference* (ICLR 2022).
Es el fundamento teórico: un PFN aproxima la **distribución predictiva posterior** p(y | x, D) bajo el prior con el que fue preentrenado. Es la conexión estadística que le va a gustar al profesor: el aprendizaje in-context como inferencia bayesiana amortizada. Úsalo en la sección de metodología para explicar *por qué* los TFMs deberían ganar con n pequeño: el prior aporta regularización.

**[hollmann2025accurate]** ✅ 📄 — TabPFNv2 (*Nature*, 2025).
Escala a unas 10,000 filas y 500 variables, y soporta variables categóricas y valores faltantes. Muestra que en datasets pequeños supera a CatBoost y XGBoost tuneados y a AutoGluon con 4 horas de presupuesto. Dato útil: incluyen un ejemplo de SHAP sobre el propio dataset de Taiwán. Sus pesos tienen licencia Apache 2.0 con atribución, así que es **el TFM más reproducible para el proyecto**.

**[grinsztajn2025tabpfn25]** ✅ 📄 — TabPFN-2.5 (arXiv 2511.08667).
Escala a 50,000 filas y 2,000 variables (20 veces más celdas que v2). Reporta que en un forward pass supera a XGBoost y CatBoost tuneados. Sus pesos tienen licencia no comercial, lo cual está bien para uso académico.

**[grinsztajn2026tabpfn3]** ✅ 📄 — TabPFN-3 (arXiv 2605.13986).
Es el default actual del paquete `tabpfn`. Según el README oficial, soporta del orden de 1,000,000 × 200, 100,000 × 2,000 o 1,000 × 20,000 (filas × variables) en GPU. En CPU, el paquete limita por defecto a 5,000 filas. Sus pesos son no comerciales y requieren login con token en Prior Labs. **Ojo práctico:** sin GPU, TabPFN es lento; usa Colab o un cluster de Purdue (por ejemplo Gilbreth o Anvil, si tienes acceso).

**[qu2025tabicl]** ✅ 📄 — TabICL (ICML 2025).
Usa una arquitectura en dos etapas (embedding por columnas, luego interacción por filas y por último ICL a nivel de dataset) que escala a 500k muestras con recursos modestos. Es el **mejor TFM en el benchmark de crédito** de Baesens et al. Su código y pesos son abiertos.

**[qu2026tabiclv2]** ✅(título) ⚠️(autores) 📄 — TabICLv2 (arXiv 2602.11139).
Es la versión mejorada y abierta. Usarla o no depende del tiempo de cómputo disponible.

**[klein2025grounding]** ✅ 📄 — *Foundation models for tabular data within systemic contexts need grounding*.
Es una advertencia crítica: los resultados de laboratorio de los PFNs pueden sobrestimar lo fácil que es usarlos en la práctica. Sirve para la sección de limitaciones y para dar balance.

### B. Árboles frente a deep learning en datos tabulares

**[grinsztajn2022tree]** ✅ 📄 — *Why do tree-based models still outperform deep learning on typical tabular data?* (NeurIPS 2022).
Identifica tres razones: las redes neuronales tienden a funciones demasiado suaves, **son poco robustas a variables no informativas** y no son invariantes a rotaciones. Su manipulación de añadir variables no informativas es justo nuestro eje m. Es la referencia metodológica directa para H3.

**[shwartzziv2022tabular]** ✅ 📄 — *Tabular data: Deep learning is not all you need* (Information Fusion, 2022).
Muestra que XGBoost supera a varias redes neuronales tabulares en sus propios datasets. Es la referencia estándar de la tesis de que los GBDT son el estado del arte.

**[gorishniy2021revisiting]** ✅ 📄 — FT-Transformer y MLP/ResNet bien tuneados (NeurIPS 2021).
Establece que un MLP bien configurado es un baseline fuerte. Justifica nuestra elección de MLP como baseline neuronal (y FT-Transformer como opcional).

**[mcelfresh2023neural]** ⚠️ 📄 — *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* (NeurIPS 2023).
Tiene el mismo espíritu de "¿bajo qué condiciones?" que nuestra pregunta. Analiza qué meta-características del dataset predicen cuándo gana cada familia. Útil para la discusión.

**[ye2024closer]** ✅ 📄 — *A Closer Look at Deep Learning on Tabular Data*.
Es un benchmark amplio (base de TALENT, el toolbox que usó Baesens et al.).

**[erickson2025tabarena]** ✅ 📄 — TabArena (2025).
Es un benchmark vivo de 51 datasets. Sus hallazgos: los GBDT siguen siendo fuertes, el deep learning alcanza a los GBDT con ensembles y mayor presupuesto, y los TFMs destacan en datasets pequeños. Además, enfatiza que **el protocolo de validación y el presupuesto de tuning cambian los rankings**. Esto justifica que fijemos el mismo presupuesto de HPO para todos.

### C. Credit scoring

**[lessmann2015benchmarking]** ✅(metadatos) 🔒 — El benchmark de referencia en credit scoring (de memoria: 41 clasificadores y 8 datasets; verificar).
Establece el protocolo (múltiples métricas, test estadísticos entre datasets). De memoria: los ensembles (heterogéneos y random forest) superan a la regresión logística. Confirma las cifras exactas al leerlo.

**[gunnarsson2021deep]** ✅(metadatos) 🔒 — *Deep learning for credit scoring: Do or don't?*
Concluye que, para datos tabulares de crédito, los modelos tipo GBM (XGBoost) superan al deep learning. El deep learning aporta con datos no estructurados. Es el *status quo* que los TFMs desafían.

**[brown2012experimental]** ✅(metadatos) 🔒 — Clasificadores con datasets de crédito desbalanceados.
Hacen precisamente lo que proponemos en el eje π: **submuestrean la clase "mala"** para crear desbalances crecientes. Según recuerdo, concluyen que el random forest y el gradient boosting aguantan bien el desbalance extremo; confírmalo al leerlo. Es el antecedente metodológico directo de H2.

**[jiang2023benchmarking]** ✅(metadatos) 🔒 — Benchmark de técnicas para desbalance en credit scoring.
Sirve para justificar *no* usar SMOTE ni técnicas de remuestreo en el benchmark principal (fuera de alcance) y mencionarlas como trabajo futuro.

**[baesens2016credit]** ⚠️ — Libro *Credit Risk Analytics*, de donde viene el dataset HMEQ.

### D. Evaluación y estadística

**[demsar2006statistical]** ✅(metadatos) 📄 — Es el estándar para comparar clasificadores en múltiples datasets: test de Friedman, post-hoc de Nemenyi y diagrama de diferencia crítica.

**[garcia2008extension]** ⚠️ 📄 — Comparaciones por pares con corrección de Holm.

**[niculescu2005predicting]** ⚠️ 🔒 — *Predicting good probabilities with supervised learning*.
Muestra que los métodos de margen máximo, como el boosting (en su versión con pérdida exponencial) y las SVM, alejan las probabilidades de 0 y 1 (una distorsión con forma sigmoidal), y que el escalado de Platt o la regresión isotónica lo corrigen. Los GBDT modernos entrenados con log-loss suelen estar mejor calibrados, así que hay que medirlo en lugar de asumirlo. Es la referencia clásica para la parte de calibración y motiva un experimento adicional opcional: ¿la ventaja de calibración de los TFMs desaparece si se calibran los GBDT?

**[akiba2019optuna]** ✅(metadatos) — Optuna, para el HPO.

### E. Implementaciones de GBDT

**[chen2016xgboost]** 📄, **[ke2017lightgbm]** 📄, **[prokhorenkova2018catboost]** 📄 — Citar al describir los modelos.

### F. Datasets

**[yeh2009comparisons]** (Taiwán), **[hofmann1994german]** ⚠️, **[quinlan1987australian]** ⚠️, **[kaggle2011gmsc]**, **[baesens2016credit]** (HMEQ) y **[kaggle2018homecredit]** (opcional). Detalle en `02_data_cards.md`.

---

## 3. Borrador — *Related Work* (inglés, ~650 palabras)

> Ajusta el tono a tu voz. Las afirmaciones están respaldadas por las fuentes citadas, pero vuelve a leer las secciones relevantes de cada paper antes de entregar.

**2. Related Work**

*Credit scoring benchmarks.* Predicting the probability of default (PD) is a binary classification problem on tabular data, and the credit-scoring literature has a long tradition of large comparative studies. Lessmann et al. [lessmann2015benchmarking] benchmarked a large set of classifiers on several retail credit datasets and found that ensemble methods consistently outperform the industry-standard logistic regression. Gunnarsson et al. [gunnarsson2021deep] later showed that deep neural networks do not improve on gradient-boosted trees for tabular credit data, positioning XGBoost-type models as the de facto standard. Class imbalance, which is pervasive in credit portfolios, has been studied by artificially under-sampling defaulters to create increasingly imbalanced versions of the same dataset [brown2012experimental]; a design we adopt.

*Trees versus deep learning on tabular data.* Outside credit risk, several large benchmarks confirm that gradient-boosted decision trees (GBDTs) such as XGBoost [chen2016xgboost], LightGBM [ke2017lightgbm] and CatBoost [prokhorenkova2018catboost] remain hard to beat on medium-sized tabular data [shwartzziv2022tabular, grinsztajn2022tree]. Grinsztajn et al. [grinsztajn2022tree] attribute this partly to the sensitivity of neural networks to uninformative features, which they demonstrate by appending such features to real datasets. Well-tuned MLPs and FT-Transformers are nonetheless strong baselines [gorishniy2021revisiting], and which family wins depends on dataset characteristics [mcelfresh2023neural] and on the validation and tuning protocol [erickson2025tabarena].

*Tabular foundation models.* Prior-data fitted networks (PFNs) are transformers pre-trained on millions of synthetic datasets so that, given a labeled training set as context, they approximate the Bayesian posterior predictive distribution p(y | x, D) in a single forward pass [muller2022transformers]. TabPFN [hollmann2023tabpfn] introduced this idea for small classification problems, and TabPFNv2 [hollmann2025accurate] extended it to mixed-type data with up to 10,000 samples, outperforming tuned GBDTs on small datasets without any hyperparameter optimization. Subsequent work has focused on scale: TabICL [qu2025tabicl] uses a column-then-row architecture to handle hundreds of thousands of samples, and TabPFN-2.5 and TabPFN-3 [grinsztajn2025tabpfn25, grinsztajn2026tabpfn3] raise the supported context size by one to two orders of magnitude. Critics caution that laboratory benchmarks may overstate how easily such models transfer to applied settings [klein2025grounding].

*Foundation models for credit risk.* The closest work to ours is Baesens et al. [baesens2026foundation], who benchmark five tabular foundation models against 24 alternatives on 14 PD datasets and find that TabICL attains the best average AUC, with foundation models relatively stronger on smaller datasets. Their analysis of dataset size relies on rank correlations across heterogeneous datasets and on learning curves, while class imbalance and dimensionality are not manipulated. Moreover, their results suggest a dissociation between discrimination and calibration: the foundation model with the highest AUC has a noticeably worse Brier score than CatBoost. Since calibrated PDs, not just rankings, drive provisioning and capital calculations, this gap matters in practice.

*Our contribution.* We complement this evidence with a controlled, within-dataset design. On five public credit datasets we vary (i) the training-set size, (ii) the default rate, down to low-default regimes of 1–2%, and (iii) the number of uninformative features. For each condition we compare tabular foundation models, three GBDT implementations, an MLP, and a logistic-regression scorecard, evaluating discrimination (AUC, Gini, KS, PR-AUC) and calibration (Brier score, log-loss, expected calibration error) separately. Statistical comparisons follow Demšar [demsar2006statistical].

---

## 4. Pendientes de esta fase

- [ ] Correr `python scripts/download_papers.py` en tu máquina; desde mi entorno arXiv estaba bloqueado.
- [ ] Bajar los 🔒 vía Purdue Libraries: Lessmann 2015, Gunnarsson 2021, Brown & Mues 2012, Jiang 2023, Niculescu-Mizil 2005.
- [ ] Leer completo Baesens et al. 2026 (25 páginas). Es la lectura prioritaria.
- [ ] Verificar las entradas ⚠️ del .bib.
- [ ] Mandar la línea de posicionamiento al profesor (opcional, pero recomendado).
