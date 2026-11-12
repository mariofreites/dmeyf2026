"""Código común del baseline y de los experimentos de ensambles (DMEyF 2026, segunda parte).

Lo usan baseline.py y, más adelante, los scripts de cada ensamble:

    import sys; sys.path.insert(0, "<carpeta experimento_ensambles>")
    from comun import Experimento, Datos, cargar_predicciones, evaluar_modelo, ...

- Experimento: carpetas de salida (bucket en la máquina de la cátedra), log y apagado de la VM.
- armar_variables(): clase ternaria con 2019 + lag1, delta1, lag2, delta2 de todas las variables (DuckDB).
- Datos: matrices en memoria y entrenamiento de LightGBM sobre cualquier subconjunto de filas
  y/o columnas del entrenamiento, con o sin early stopping en 201908.
- Predicciones: una probabilidad por modelo y por mes en <exp>/predicciones/<mes>/<modelo>.npy,
  con las filas en el orden de <exp>/predicciones/<mes>/_ids.parquet.
- Métricas: AUC, curva de ganancia, corte elegido en 201908 y ganancia de 201910 con ese corte.
"""
from pathlib import Path
from urllib.request import urlretrieve
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

# LightGBM avisa esto cada vez que el Dataset de validación hereda el binning del de entrenamiento: es lo buscado.
warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")

# ---------------------------------------------------------------------------------------------
# Configuración común a todos los experimentos (cambiarla rompe la comparación con el baseline)
# ---------------------------------------------------------------------------------------------
MODO_PRUEBA = os.environ.get("MODO_PRUEBA", "0") == "1"
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 4))

DATA_URL = "https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_02_crudo.csv.gz"
DATA_FILE = "competencia_02_crudo.csv.gz"

# Solo 2019. La clase se calcula con estos meses: 201911 solo puede tener BAJA+1 y 201912 no tiene clase.
MESES_DATOS = (201901, 201912)
TRAIN_MONTHS = [201903, 201904, 201905, 201906]
VALID_MONTH = 201908  # AUC de la bayesiana, early stopping y elección del corte
TEST_MONTH = 201910  # solo para comparar: nunca elige nada
# Meses con clase completa que se predicen y se guardan. 201907 y 201909 no los usa el baseline:
# quedan "limpios" (ni entrenamiento ni validación) por si un ensamble necesita ajustar algo.
PRED_MONTHS = [201907, 201908, 201909, 201910]
LAG_ORDERS = [1, 2]
ID_COLUMNS = ["numero_de_cliente", "foto_mes"]  # no son variables del modelo
POSITIVOS = ["BAJA+1", "BAJA+2"]

# LightGBM: parámetros fijos. Los 5 de la bayesiana del baseline se agregan encima.
MAX_BIN = 31  # como en las notebooks de la cátedra
FIXED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "first_metric_only": True,
    "boosting": "gbdt",
    "bagging_freq": 1,
    "max_depth": -1,
    "min_gain_to_split": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "max_bin": MAX_BIN,
    "feature_pre_filter": False,  # permite cambiar min_data_in_leaf sin rearmar el Dataset
    "force_col_wise": True,
    "verbose": -1,
}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}
DATASET_PARAMS = {"max_bin": MAX_BIN, "feature_pre_filter": False, "verbose": -1, "data_random_seed": 1}

# Semillas (la primera es la de la bayesiana del baseline)
SEEDS = [101737, 303011, 602909, 707717, 906007, 568318, 641463, 591763, 616818, 193376,
         409993, 512821, 786431, 870929, 997991, 254747, 352813, 455353, 666173, 733133]
assert len(set(SEEDS)) == len(SEEDS)

# Ganancia (en millones): +1.0725 por BAJA+2 contactado, -0.0275 por cualquier otro
GAIN_BAJA2 = 1.0725
GAIN_OTHER = -0.0275
GAIN_WINDOW = 51 if MODO_PRUEBA else 501  # suavizado de la curva de ganancia para elegir el corte

PRUEBA_FRACCION_CLIENTES = 0.05

# Carpetas. En la máquina de la cátedra: resultados en el bucket y temporales en el disco local.
BUCKET = Path.home() / "buckets" / "b1"
EN_GCP = BUCKET.is_dir() and "DMEYF_OUT_DIR" not in os.environ
_HERE = Path(__file__).resolve().parent
if EN_GCP:
    EXP_ROOT = BUCKET / "exp"
    LOCAL_ROOT = Path.home() / "local"
    DATA_DIR = Path(os.environ.get("DMEYF_DATA_DIR", Path.home() / "datasets"))
else:
    EXP_ROOT = Path(os.environ.get("DMEYF_OUT_DIR", _HERE / "outputs"))
    LOCAL_ROOT = Path(os.environ.get("DMEYF_LOCAL_DIR", EXP_ROOT / "_local"))
    DATA_DIR = Path(os.environ.get("DMEYF_DATA_DIR", _HERE / "data"))
FEATURES_DIR = LOCAL_ROOT / ("variables_prueba" if MODO_PRUEBA else "variables")  # compartido entre experimentos


# ---------------------------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------------------------
class Experimento:
    """Carpetas y log de un experimento: <exp>/log.txt, <exp>/predicciones/<mes>/, etc."""

    def __init__(self, exp_id):
        self.exp_id = exp_id
        suffix = "prueba" if MODO_PRUEBA else ""
        self.out_dir = EXP_ROOT / exp_id / suffix
        self.local_dir = LOCAL_ROOT / exp_id / suffix
        self.pred_dir = self.out_dir / "predicciones"
        for folder in (self.out_dir, self.local_dir, self.pred_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.txt"
        self.stop_file = self.out_dir / "PARAR"

    def log(self, *parts):
        """Imprime y deja la línea en log.txt (en el bucket), por si se cierra la terminal."""
        text = " ".join(str(part) for part in parts)
        print(text, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + text + "\n")

    def apagar_vm(self, motivo):
        """Borra esta máquina virtual con el script de la cátedra (igual que en la primera competencia)."""
        script = Path.home() / "machina" / "direct" / "apagar_vm.sh"
        if not EN_GCP or not script.exists():
            self.log(f"[apagado] {motivo}: no estoy en una máquina virtual de la cátedra, no apago nada.")
            return
        self.log(f"[apagado] {motivo}: borrando la máquina virtual. Los resultados quedaron en {self.out_dir}")
        respaldo = f"""bash {script}
case "$(hostname)" in desktop*|instance-instalacion) exit 0;; esac
MD=http://metadata.google.internal/computeMetadata/v1; H="Metadata-Flavor: Google"
ZONA=$(curl -s -H "$H" $MD/instance/zone)
gcloud --quiet compute instances delete "$(curl -s -H "$H" $MD/instance/name)" --zone="${{ZONA##*/}}" --project="$(curl -s -H "$H" $MD/project/project-id)"
"""
        with open(self.out_dir / "apagado.txt", "a") as salida:
            subprocess.Popen(["bash", "-c", respaldo], stdout=salida, stderr=subprocess.STDOUT, start_new_session=True)


def save_atomic(path, write):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        write(handle)
    os.replace(tmp, path)


def save_json(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2, default=float)
    save_atomic(path, lambda handle: handle.write(text.encode("utf-8")))


def save_npy(path, array):
    save_atomic(path, lambda handle: np.save(handle, array))


def config_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def release_memory():
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ensure_packages():
    """lightgbm común (no zLightGBM), optuna, duckdb y pyarrow."""
    missing = []
    for module in ("lightgbm", "optuna", "duckdb", "pyarrow"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


# ---------------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------------
def auc_score(positive, scores):
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    ranks = pd.Series(np.asarray(scores, dtype=np.float64)).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def gain_curve(baja2, scores, window=None):
    """Ganancia acumulada al enviar los primeros k (k = 1..n) ordenados por probabilidad, y su versión suavizada."""
    window = window or GAIN_WINDOW
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    cumulative = np.cumsum(np.where(np.asarray(baja2, dtype=bool)[order], GAIN_BAJA2, GAIN_OTHER))
    smooth = pd.Series(cumulative).rolling(window=window, center=True, min_periods=window).mean().to_numpy()
    return cumulative, smooth


def best_cut(baja2, scores):
    """Cantidad de envíos que maximiza la ganancia suavizada. Se usa solo en 201908."""
    _, smooth = gain_curve(baja2, scores)
    return int(np.nanargmax(smooth)) + 1


def gain_at(baja2, scores, cut):
    """Ganancia real (sin suavizar) enviando los `cut` de mayor probabilidad."""
    cumulative, _ = gain_curve(baja2, scores)
    return float(cumulative[min(cut, len(cumulative)) - 1])


def evaluar_modelo(valid_ids, valid_scores, test_ids, test_scores):
    """La regla de comparación de todos los experimentos: corte elegido en 201908, ganancia medida en 201910.

    Sirve igual para un modelo solo o para un ensamble (scores = probabilidad o promedio de probabilidades).
    """
    vb2 = (valid_ids["clase_ternaria"] == "BAJA+2").to_numpy()
    tb2 = (test_ids["clase_ternaria"] == "BAJA+2").to_numpy()
    cut = best_cut(vb2, valid_scores)
    return {
        "auc_valid": auc_score(valid_ids["clase_ternaria"].isin(POSITIVOS).to_numpy(), valid_scores),
        "auc_test": auc_score(test_ids["clase_ternaria"].isin(POSITIVOS).to_numpy(), test_scores),
        "corte": cut,
        "ganancia_valid": gain_at(vb2, valid_scores, cut),
        "ganancia_test": gain_at(tb2, test_scores, cut),
    }


_pos = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
assert np.isclose(auc_score(_pos, np.array([0.9, 0.8, 0.8, 0.5, 0.4, 0.4, 0.2, 0.1])), 12 / 15)
assert np.isclose(gain_at(np.array([1, 0, 0], dtype=bool), np.array([0.9, 0.1, 0.5]), 2), GAIN_BAJA2 + GAIN_OTHER)
del _pos


# ---------------------------------------------------------------------------------------------
# Datos: clase ternaria + lags y deltas (DuckDB). Se arma una vez y lo comparten todos los experimentos.
# ---------------------------------------------------------------------------------------------
def dataset_path(log=print):
    path = DATA_DIR / DATA_FILE
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log("Bajando el dataset a", path)
        urlretrieve(DATA_URL, path.with_name(path.name + ".tmp"))
        os.replace(path.with_name(path.name + ".tmp"), path)
    return path


def armar_variables(log=print):
    """Arma (o reutiliza) un parquet con 201903-201910: ids, clase y todas las variables en float32."""
    import duckdb

    raw = dataset_path(log)
    config = {"archivo": raw.name, "tamano": raw.stat().st_size, "meses": MESES_DATOS, "lags": LAG_ORDERS,
              "guardar": sorted(set(TRAIN_MONTHS) | set(PRED_MONTHS)), "prueba": MODO_PRUEBA, "version": 1}
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / f"dataset_fe_{config_hash(config)}.parquet"
    if out.exists():
        log("Variables ya armadas:", out)
        return out

    start = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET threads = {NUM_THREADS}")
    con.execute(f"SET temp_directory = '{(FEATURES_DIR / 'duckdb_tmp').as_posix()}'")
    muestra = (f" AND (numero_de_cliente * 2654435761) % 1000 < {int(PRUEBA_FRACCION_CLIENTES * 1000)}"
               if MODO_PRUEBA else "")
    con.execute(f"""
        CREATE TABLE crudo AS
        SELECT * FROM read_csv('{raw.as_posix()}', sample_size = -1)
        WHERE foto_mes BETWEEN {MESES_DATOS[0]} AND {MESES_DATOS[1]}{muestra}
    """)
    columns = [row[0] for row in con.execute("DESCRIBE crudo").fetchall()]
    base = [c for c in columns if c not in ID_COLUMNS]

    # Clase ternaria, igual que la notebook de la cátedra (z494), pero solo con los meses de 2019.
    con.execute("""
        CREATE TABLE clase AS
        WITH p AS (
            SELECT numero_de_cliente, foto_mes, (foto_mes // 100) * 12 + foto_mes % 100 AS periodo0
            FROM crudo
        ), l AS (
            SELECT *,
                   lead(periodo0, 1) OVER w AS periodo1,
                   lead(periodo0, 2) OVER w AS periodo2,
                   max(periodo0) OVER () AS ultimo
            FROM p
            WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY periodo0)
        )
        SELECT numero_de_cliente, foto_mes,
            CASE
                WHEN periodo0 < ultimo - 1 AND periodo1 = periodo0 + 1
                     AND (periodo2 IS NULL OR periodo2 > periodo0 + 2) THEN 'BAJA+2'
                WHEN periodo0 < ultimo AND (periodo1 IS NULL OR periodo1 > periodo0 + 1) THEN 'BAJA+1'
                WHEN periodo0 < ultimo - 1 THEN 'CONTINUA'
            END AS clase_ternaria
        FROM l
    """)

    # lag k = valor de la k-ésima foto anterior del cliente (como en la materia); delta k = valor - lag k.
    exprs = [f'CAST("{c}" AS FLOAT) AS "{c}"' for c in base]
    for k in LAG_ORDERS:
        for c in base:
            exprs.append(f'CAST(lag("{c}", {k}) OVER w AS FLOAT) AS "{c}_lag{k}"')
            exprs.append(f'CAST("{c}" - lag("{c}", {k}) OVER w AS FLOAT) AS "{c}_delta{k}"')
    keep_months = sorted(set(TRAIN_MONTHS) | set(PRED_MONTHS))
    tmp = out.with_name(out.name + ".tmp")
    con.execute(f"""
        COPY (
            SELECT * FROM (
                SELECT c.numero_de_cliente, c.foto_mes, k.clase_ternaria, {", ".join(exprs)}
                FROM crudo c JOIN clase k USING (numero_de_cliente, foto_mes)
                WINDOW w AS (PARTITION BY c.numero_de_cliente ORDER BY c.foto_mes)
            )
            WHERE foto_mes IN ({", ".join(map(str, keep_months))})
            ORDER BY foto_mes, numero_de_cliente
        ) TO '{tmp.as_posix()}' (FORMAT parquet)
    """)
    os.replace(tmp, out)
    resumen = con.execute(f"""
        SELECT foto_mes, clase_ternaria, count(*) FROM read_parquet('{out.as_posix()}')
        GROUP BY ALL ORDER BY ALL
    """).fetchall()
    con.close()
    log(f"Variables armadas en {(time.perf_counter() - start) / 60:.1f} min: {len(base)} originales + "
        f"{len(base) * 2 * len(LAG_ORDERS)} lags/deltas -> {out}")
    for row in resumen:
        log("   ", *row)
    return out


def _load_months(path, months):
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("foto_mes", "in", list(months))])
    features = [c for c in table.column_names if c not in ID_COLUMNS + ["clase_ternaria"]]
    ids = table.select(ID_COLUMNS + ["clase_ternaria"]).to_pandas()
    X = np.empty((table.num_rows, len(features)), dtype=np.float32)
    for j, name in enumerate(features):
        X[:, j] = table.column(name).to_numpy(zero_copy_only=False)
    del table
    release_memory()
    return ids, X, features


# ---------------------------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------------------------
def lgb_params(values, seed):
    """FIXED_PARAMS + hiperparámetros (los de la bayesiana o los que se pasen) + semilla.

    La semilla mueve lo único aleatorio de este LightGBM: el bagging de filas y el feature_fraction.
    """
    params = {**FIXED_PARAMS, **values, "num_threads": NUM_THREADS,
              "bagging_seed": int(seed), "feature_fraction_seed": int(seed)}
    for key in INT_PARAMS & params.keys():
        params[key] = int(params[key])
    return params


class Datos:
    """Matrices en memoria: entrenamiento (201903-201906) y cada mes de PRED_MONTHS.

    - `entrenar(...)` entrena con todas las filas y columnas o con un subconjunto de cada una.
      `filas` son índices sobre las filas de entrenamiento (ver `y_train` y `train_ids`) y `columnas`
      índices sobre `features`.
    - `predecir_y_guardar(...)` guarda la probabilidad de cada mes de PRED_MONTHS en la carpeta del experimento.
    """

    def __init__(self, path, log=print):
        import lightgbm as lgb

        self.lgb, self.log = lgb, log
        ids, X, self.features = _load_months(path, sorted(set(TRAIN_MONTHS) | set(PRED_MONTHS)))
        months = ids["foto_mes"].to_numpy()
        train_rows = np.isin(months, TRAIN_MONTHS)
        if ids.loc[train_rows | (months == VALID_MONTH), "clase_ternaria"].isna().any():
            raise RuntimeError("Hay filas sin clase en entrenamiento o validación")
        self.train_ids = ids.loc[train_rows].reset_index(drop=True)
        self.y_train = self.train_ids["clase_ternaria"].isin(POSITIVOS).to_numpy().astype(np.float32)
        self.X_train = np.ascontiguousarray(X[train_rows])

        self.month_ids, self.month_X = {}, {}
        for month in PRED_MONTHS:
            rows = months == month
            self.month_ids[month] = ids.loc[rows].reset_index(drop=True)
            self.month_X[month] = np.ascontiguousarray(X[rows])
        del X, ids
        release_memory()
        self.y_valid = self.month_ids[VALID_MONTH]["clase_ternaria"].isin(POSITIVOS).to_numpy().astype(np.float32)

        # Dataset completo armado una sola vez (el binning es lo más caro): lo reutiliza la bayesiana.
        self._dtrain = lgb.Dataset(self.X_train, label=self.y_train, feature_name=self.features,
                                   params={**DATASET_PARAMS, "num_threads": NUM_THREADS},
                                   free_raw_data=False).construct()
        self._dvalid = lgb.Dataset(self.month_X[VALID_MONTH], label=self.y_valid, reference=self._dtrain,
                                   free_raw_data=False).construct()
        log(f"Entrenamiento {TRAIN_MONTHS}: {len(self.y_train):,} filas, {int(self.y_train.sum()):,} positivos | "
            f"{len(self.features)} variables")
        for month in PRED_MONTHS:
            counts = self.month_ids[month]["clase_ternaria"].value_counts(dropna=False).to_dict()
            log(f"  {month}: {len(self.month_ids[month]):,} filas {counts}")

    def _datasets(self, filas, columnas):
        lgb = self.lgb
        if columnas is None:
            dtrain = self._dtrain if filas is None else self._dtrain.subset(np.sort(np.asarray(filas)))
            return dtrain, self._dvalid
        columnas = np.asarray(columnas)
        X = self.X_train[:, columnas] if filas is None else self.X_train[np.ix_(np.asarray(filas), columnas)]
        y = self.y_train if filas is None else self.y_train[np.asarray(filas)]
        dtrain = lgb.Dataset(X, label=y, feature_name=[self.features[j] for j in columnas],
                             params={**DATASET_PARAMS, "num_threads": NUM_THREADS}, free_raw_data=True)
        dvalid = lgb.Dataset(self.month_X[VALID_MONTH][:, columnas], label=self.y_valid, reference=dtrain)
        return dtrain, dvalid

    def entrenar(self, values, seed, num_rounds, filas=None, columnas=None, early_stopping=False):
        """Entrena un LightGBM. Con early_stopping, num_rounds es el tope y se corta mirando el AUC de 201908.

        La paciencia crece cuando el learning rate es chico: 50 + 5 / learning_rate rondas.
        """
        lgb = self.lgb
        params = lgb_params(values, seed)
        dtrain, dvalid = self._datasets(filas, columnas)
        if early_stopping:
            patience = int(50 + 5 / params["learning_rate"])
            booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=[dvalid],
                                valid_names=[str(VALID_MONTH)],
                                callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
        else:
            booster = lgb.train(params, dtrain, num_boost_round=num_rounds)
        booster.columnas_ = None if columnas is None else np.asarray(columnas)
        return booster

    def predecir(self, booster, month, num_iteration=None):
        X = self.month_X[month]
        columnas = getattr(booster, "columnas_", None)
        if columnas is not None:
            X = X[:, columnas]
        return booster.predict(X, num_iteration=num_iteration).astype(np.float32)

    def guardar_ids(self, pred_dir):
        for month in PRED_MONTHS:
            folder = Path(pred_dir) / str(month)
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "_ids.parquet").exists():
                self.month_ids[month].to_parquet(folder / "_ids.parquet", index=False)

    def predecir_y_guardar(self, booster, pred_dir, nombre, num_iteration=None):
        """Guarda <pred_dir>/<mes>/<nombre>.npy para cada mes de PRED_MONTHS. Devuelve el dict mes -> scores."""
        self.guardar_ids(pred_dir)
        out = {}
        for month in PRED_MONTHS:
            out[month] = self.predecir(booster, month, num_iteration)
            save_npy(Path(pred_dir) / str(month) / f"{nombre}.npy", out[month])
        return out


    # --- Ayudas para los ensambles (no las usa el baseline) ---
    def grupos_de_variables(self):
        """Variable original -> índices de sus columnas (ella, lag1, delta1, lag2, delta2).

        Sirve para sortear subconjuntos de variables originales que lleven juntos sus lags y deltas.
        """
        suffixes = tuple(f"_{kind}{k}" for k in LAG_ORDERS for kind in ("lag", "delta"))
        index = {name: j for j, name in enumerate(self.features)}
        groups = {}
        for name in self.features:
            if not name.endswith(suffixes):
                groups[name] = [index[name]] + [index[name + s] for s in suffixes if name + s in index]
        return groups

    def columnas_al_azar(self, fraccion, seed):
        """Índices de columnas de una fracción de las variables originales (con sus lags y deltas), ordenados."""
        groups = self.grupos_de_variables()
        names = sorted(groups)
        chosen = np.random.default_rng(seed).choice(len(names), max(1, round(fraccion * len(names))), replace=False)
        return np.sort(np.concatenate([groups[names[i]] for i in chosen]))

    def filas_undersampling(self, continua_por_positivo, seed):
        """Índices de entrenamiento: todos los positivos y `continua_por_positivo` CONTINUA por cada uno, al azar."""
        positive = np.flatnonzero(self.y_train == 1)
        negative = np.flatnonzero(self.y_train == 0)
        n_negative = min(len(negative), int(continua_por_positivo * len(positive)))
        sample = np.random.default_rng(seed).choice(negative, n_negative, replace=False)
        return np.sort(np.concatenate([positive, sample]))


def predicciones_existen(pred_dir, nombre):
    return all((Path(pred_dir) / str(month) / f"{nombre}.npy").exists() for month in PRED_MONTHS)


def cargar_predicciones(pred_dir, month, patron="*"):
    """DataFrame con numero_de_cliente, foto_mes, clase_ternaria y una columna por modelo guardado."""
    folder = Path(pred_dir) / str(month)
    table = pd.read_parquet(folder / "_ids.parquet")
    for path in sorted(folder.glob(f"{patron}.npy")):
        table[path.stem] = np.load(path)
    return table


def hiperparametros_baseline():
    """Los mejores hiperparámetros de la bayesiana del baseline (y sus iteraciones con early stopping)."""
    folder = EXP_ROOT / "baseline" / ("prueba" if MODO_PRUEBA else "") / "bayesiana"
    return json.loads((folder / "mejores_hiperparametros.json").read_text(encoding="utf-8"))
