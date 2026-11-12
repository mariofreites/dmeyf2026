# Experimento colaborativo: ensambles

Hipótesis general: ensamblar varios LightGBM mejora la ganancia del baseline de la cátedra en 201910. Se prueban los tres ensambles aprobados:

1. **Semillero:** promediar 1, 5, 10, … n semillas de un mismo modelo.
2. **Undersampling de CONTINUA:** de 10 a 20 modelos, cada uno con el 100 % de los positivos y una muestra distinta de CONTINUA (1:5, 1:10, 1:20).
3. **Undersampling de variables:** de 10 a 20 modelos (o más), cada uno con un subconjunto distinto de columnas (25 %, 50 %, 75 %).

Esta carpeta tiene, por ahora, el **baseline** y el **código común** que van a usar los tres ensambles. Los ensambles todavía no están implementados.

```
experimento_ensambles/
  comun.py              datos, entrenamiento sobre subconjuntos de filas o columnas, métricas, guardado
  01_baseline/
    baseline.py         el baseline de la cátedra: bayesiana, semillas, predicciones y evaluación
```

## El baseline

Sigue la consigna de la cátedra:

| | |
|---|---|
| Dataset | `competencia_02_crudo.csv.gz`, solo los meses de 2019 |
| Clase | calculada con 2019 solamente: 201911 solo puede tener BAJA+1 y 201912 no tiene clase |
| Variables | las 152 originales + lag1, delta1, lag2 y delta2 de cada una (760 en total) |
| Entrenamiento | 201903, 201904, 201905 y 201906 completos, sin sampling. Positivos: BAJA+1 y BAJA+2 |
| Validación | 201908: AUC de la bayesiana, early stopping y elección del corte |
| Test | 201910: solo se mide la ganancia, no se elige nada con este mes |
| Bayesiana | `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction` y `bagging_fraction`, con `bagging_freq` = 1 |

### Decisiones propias (la consigna no las fija)

- **Dataset.** `competencia_01` no tiene 2019. `competencia_02` va de 201901 a 202108, así que tiene 201901 y 201902 para los lags de marzo y 201911 y 201912 para la clase de 201910. Para respetar la consigna, la clase se calcula cortando el dataset en 201912. El script lo baja solo de `https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_02_crudo.csv.gz` si no lo encuentra.
- **Lags.** El lag k es la k-ésima foto anterior del mismo cliente, como en las notebooks de la materia (`lag() over (partition by numero_de_cliente order by foto_mes)`). El delta k es valor − lag k. `numero_de_cliente` y `foto_mes` no entran al modelo.
- **LightGBM común**, no zLightGBM: es lo que pide la consigna. `max_bin` = 31, como en las notebooks de la cátedra, y el resto de los parámetros en sus valores por defecto (`FIXED_PARAMS` en `comun.py`).
- **Rangos de la bayesiana** (Optuna, TPE multivariado, 50 corridas, las primeras 15 al azar):

  | Hiperparámetro | Desde | Hasta | Escala |
  |---|---|---|---|
  | `num_leaves` | 8 | 2048 | logarítmica |
  | `learning_rate` | 0,01 | 0,2 | logarítmica |
  | `min_data_in_leaf` | 10 | 10.000 | logarítmica |
  | `feature_fraction` | 0,1 | 1,0 | lineal |
  | `bagging_fraction` | 0,2 | 1,0 | lineal |

  Las escalas logarítmicas reparten las pruebas por orden de magnitud (8, 64 y 512 hojas reciben el mismo peso). El piso de 0,01 en el learning rate es para que cada corrida no tarde horas.
- **Early stopping** en 201908 con AUC y una paciencia de 50 + 5 / learning_rate rondas (150 con 0,05 y 550 con 0,01), con un tope de 5000 iteraciones. Cada corrida guarda su mejor iteración.
- **El corte se elige en 201908.** Es la cantidad de envíos que maximiza la ganancia suavizada (media móvil de 501 posiciones, como en la primera competencia). Después se aplica esa misma cantidad en 201910. La ganancia es +1,0725 millones por BAJA+2 contactado y −0,0275 millones por cualquier otro, como en la primera competencia.
- **Semillas.** La semilla mueve lo único aleatorio de este LightGBM: el bagging de filas y la `feature_fraction`. Los 5 mejores de la bayesiana se reentrenan con 10 semillas cada uno, con la iteración que eligió el early stopping (sin volver a mirar 201908). La semilla de la bayesiana es la primera de la lista, así que ese modelo es el mismo de la bayesiana (verificado: dan predicciones idénticas).

### Qué guarda para los ensambles

Todo queda en `~/buckets/b1/exp/baseline/` en la máquina de la cátedra (o en `outputs/` en cualquier otra):

```
log.txt, config.json
bayesiana/corridas.csv                 las 50 corridas ordenadas por AUC de 201908
bayesiana/mejores_hiperparametros.json los 5 valores, la mejor iteración y los fijos
bayesiana/importancia_boNNN.csv        importancia (gain) de cada variable por corrida
predicciones/<mes>/_ids.parquet        numero_de_cliente, foto_mes, clase_ternaria
predicciones/<mes>/boNNN_sSEMILLA.npy  probabilidad de cada modelo, en el orden de _ids.parquet
resultados/modelos.csv                 AUC por mes, corte de 201908 y ganancia de 201910 de cada modelo
resultados/resumen.json                el baseline y la dispersión entre semillas
```

Se guardan las predicciones de **201907, 201908, 201909 y 201910** de **todas** las corridas de la bayesiana y de todas las semillas. 201907 y 201909 no los usa el baseline: quedan como meses que ningún modelo vio, por si algún ensamble necesita ajustar algo (pesos, cantidad de modelos) sin tocar 201908 ni 201910.

### Cómo se compara cada ensamble

Con la misma regla que el baseline (`evaluar_modelo` en `comun.py`): el corte se elige en 201908 sobre el promedio de probabilidades del ensamble y la ganancia se mide en 201910. Cada ensamble se compara contra:

- el **baseline** (un solo modelo, `resumen.json`), mirando también la dispersión entre semillas del mejor: una mejora menor que ese desvío no se distingue del azar;
- un **semillero del mismo tamaño**, para separar la mejora del subconjunto de la mejora de promediar n modelos.

## Cómo correrlo

En la máquina de la cátedra (con el bucket en `~/buckets/b1`):

```bash
git clone https://github.com/mariofreites/dmeyf2026.git ~/dmeyf2026
cd ~/dmeyf2026/experimento_ensambles/01_baseline

# 1) prueba corta: 5 % de los clientes, 4 corridas, unos minutos. Escribe en exp/baseline/prueba y no apaga la máquina
MODO_PRUEBA=1 python baseline.py

# 2) corrida real, en segundo plano
nohup python baseline.py > baseline_run.txt 2>&1 &
tail -f ~/buckets/b1/exp/baseline/log.txt
```

- Necesita `lightgbm`, `optuna`, `duckdb` y `pyarrow`. Si falta alguno, lo instala solo.
- **Es resumible.** Si Google borra la máquina spot, alcanza con lanzar el mismo comando en otra. Retoma la bayesiana desde el estudio guardado en el bucket y saltea las semillas que ya están.
- `touch ~/buckets/b1/exp/baseline/CORTAR_BAYESIANA` termina la corrida en curso y sigue con las semillas con lo mejor que haya.
- `touch ~/buckets/b1/exp/baseline/PARAR` no empieza más semillas y pasa a la evaluación.
- Al terminar (o si falla) borra la máquina virtual, como en la primera competencia. `APAGAR_AL_TERMINAR=0` lo evita.
- `python baseline.py --solo-evaluar` rehace la evaluación con las predicciones ya guardadas.
- Fuera de la máquina de la cátedra se pueden elegir las carpetas con `DMEYF_DATA_DIR`, `DMEYF_OUT_DIR` y `DMEYF_LOCAL_DIR`, y los hilos con `NUM_THREADS`.

## Cómo lo van a reutilizar los ensambles

```python
import sys
sys.path.insert(0, "/home/<usuario>/dmeyf2026/experimento_ensambles")
from comun import Datos, Experimento, armar_variables, hiperparametros_baseline, cargar_predicciones, evaluar_modelo

exp = Experimento("ens3_variables")                   # carpeta exp/ens3_variables con su log y predicciones
datos = Datos(armar_variables(exp.log), exp.log)      # reutiliza el parquet de variables si ya existe
hp = hiperparametros_baseline()                       # los de la bayesiana del baseline
valores = {**hp["valores"], "feature_fraction": 1.0}

columnas = datos.columnas_al_azar(0.25, seed=1)       # 25 % de las variables originales, con sus lags y deltas
filas = datos.filas_undersampling(10, seed=1)         # todos los positivos y 10 CONTINUA por positivo
booster = datos.entrenar(valores, seed=1, num_rounds=5000, columnas=columnas, early_stopping=True)
datos.predecir_y_guardar(booster, exp.pred_dir, "m001", booster.best_iteration)
```

`entrenar` acepta `filas`, `columnas`, las dos o ninguna. Con `early_stopping=True` cada modelo elige sus iteraciones con el AUC de 201908.
