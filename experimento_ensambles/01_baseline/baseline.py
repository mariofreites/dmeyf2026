#!/usr/bin/env python3
"""Baseline de la cátedra para los experimentos colaborativos (DMEyF 2026, segunda parte).

Qué hace, en orden (el detalle y el porqué de cada decisión están en ../README.md):

1. Datos: competencia_02 solo con 2019. Clase ternaria calculada con 2019 (201911 queda con BAJA+1
   únicamente y 201912 sin clase) y lag1, delta1, lag2 y delta2 de todas las variables.
   Positivos: BAJA+1 y BAJA+2. Sin sampling.
2. Bayesiana (Optuna, TPE): num_leaves, learning_rate, min_data_in_leaf, feature_fraction y
   bagging_fraction, con bagging_freq = 1. Entrena 201903 a 201906, mide el AUC de 201908 con
   early stopping y ordena por ese AUC.
3. Semillas: los TOP_K mejores de la bayesiana se reentrenan con N_SEMILLAS semillas, con las
   iteraciones que encontró el early stopping (sin volver a mirar 201908).
4. Predicciones: cada modelo (todas las corridas de la bayesiana y todas las semillas) guarda su
   probabilidad para 201907, 201908, 201909 y 201910, para los ensambles.
5. Evaluación: el corte (cantidad de envíos) se elige en 201908 y se aplica en 201910, donde se mide
   la ganancia. 201910 no se usa para elegir nada.

Uso:
    python baseline.py                 # corrida completa (resumible: si se corta, se relanza igual)
    MODO_PRUEBA=1 python baseline.py   # 5 % de los clientes, pocas corridas, unos minutos
    python baseline.py --solo-evaluar  # rehace la evaluación con las predicciones ya guardadas

Seguimiento: tail -f ~/buckets/b1/exp/baseline/log.txt
Cortar la bayesiana antes: touch ~/buckets/b1/exp/baseline/CORTAR_BAYESIANA
No empezar más semillas:   touch ~/buckets/b1/exp/baseline/PARAR
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun import (  # noqa: E402
    EN_GCP, FIXED_PARAMS, INT_PARAMS, MODO_PRUEBA, NUM_THREADS, PRED_MONTHS, SEEDS, TEST_MONTH, TRAIN_MONTHS,
    VALID_MONTH, GAIN_BAJA2, GAIN_OTHER, GAIN_WINDOW, LAG_ORDERS, Datos, Experimento, armar_variables,
    auc_score, cargar_predicciones, config_hash, ensure_packages, evaluar_modelo, predicciones_existen,
    release_memory, save_json,
)

EXP_ID = "baseline"

# Bayesiana: (desde, hasta, escala logarítmica)
SEARCH_SPACE = {
    "num_leaves": (8, 2048, True),
    "learning_rate": (0.01, 0.2, True),
    "min_data_in_leaf": (10, 10_000, True),
    "feature_fraction": (0.1, 1.0, False),
    "bagging_fraction": (0.2, 1.0, False),
}
N_TRIALS = 50  # la cátedra sugiere 50 a 100 con 5 hiperparámetros
N_STARTUP_TRIALS = 15  # al azar antes de que el TPE empiece a guiar
MAX_ROUNDS = 5000  # tope de iteraciones; el early stopping corta antes
BO_SEED = SEEDS[0]
SAMPLER_SEED = 20261112

# Semillas para los mejores de la bayesiana (la primera es BO_SEED: ese modelo ya está entrenado)
TOP_K = 5
N_SEMILLAS = 10

if MODO_PRUEBA:
    N_TRIALS, N_STARTUP_TRIALS = 4, 2
    MAX_ROUNDS = 300
    TOP_K, N_SEMILLAS = 2, 3

EXP = Experimento(EXP_ID)
log = EXP.log
BO_DIR = EXP.out_dir / "bayesiana"
RES_DIR = EXP.out_dir / "resultados"
STOP_BO_FILE = EXP.out_dir / "CORTAR_BAYESIANA"
APAGAR_AL_TERMINAR = os.environ.get("APAGAR_AL_TERMINAR", "1" if (EN_GCP and not MODO_PRUEBA) else "0") == "1"


def model_name(trial_number, seed):
    return f"bo{trial_number:03d}_s{seed}"


# ---------------------------------------------------------------------------------------------
# Bayesiana
# ---------------------------------------------------------------------------------------------
def run_bayesian(data):
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    bo_config = json.loads(json.dumps({
        "search_space": SEARCH_SPACE, "fijos": FIXED_PARAMS, "train": TRAIN_MONTHS, "valid": VALID_MONTH,
        "seed": BO_SEED, "max_rounds": MAX_ROUNDS, "n_startup": N_STARTUP_TRIALS,
        "variables": config_hash(data.features), "filas": len(data.y_train), "prueba": MODO_PRUEBA}))
    local_db, bucket_db = EXP.local_dir / "estudio.db", BO_DIR / "estudio.db"
    if not local_db.exists() and bucket_db.exists():
        shutil.copy(bucket_db, local_db)
        log("Estudio recuperado de", bucket_db)
    study = optuna.create_study(study_name=f"{EXP_ID}_bayesiana", storage=f"sqlite:///{local_db.as_posix()}",
                                direction="maximize", load_if_exists=True)
    if not study.trials:
        study.set_user_attr("config", bo_config)
    elif study.user_attrs.get("config") != bo_config:
        raise RuntimeError(f"{local_db} es de otra configuración. Movelo (y el del bucket) para empezar de cero.")
    study.sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS,
                                               seed=SAMPLER_SEED + len(study.trials), multivariate=True)

    def completed():
        return [t for t in study.trials if t.state == TrialState.COMPLETE]

    def objective(trial):
        values = {}
        for key, (low, high, log_scale) in SEARCH_SPACE.items():
            suggest = trial.suggest_int if key in INT_PARAMS else trial.suggest_float
            values[key] = suggest(key, low, high, log=log_scale)
        start = time.perf_counter()
        booster = data.entrenar(values, BO_SEED, MAX_ROUNDS, early_stopping=True)
        best_iter = int(booster.best_iteration) or booster.current_iteration()
        auc_valid = float(booster.best_score[str(VALID_MONTH)]["auc"])
        # Se guardan las 4 predicciones de cada corrida, pero acá solo se mira 201908.
        scores = data.predecir_y_guardar(booster, EXP.pred_dir, model_name(trial.number, BO_SEED), best_iter)
        importance = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        del booster
        release_memory()
        assert abs(auc_score(data.y_valid > 0, scores[VALID_MONTH]) - auc_valid) < 1e-3
        minutes = (time.perf_counter() - start) / 60
        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("minutos", round(minutes, 2))
        importance.sort_values(ascending=False).to_csv(BO_DIR / f"importancia_bo{trial.number:03d}.csv",
                                                       header=["gain"])
        log(f"  corrida {len(completed()) + 1:3d}/{N_TRIALS} | "
            + " ".join(f"{k} {v:.4g}" for k, v in values.items())
            + f" -> AUC {VALID_MONTH} {auc_valid:.5f} | {best_iter} iter | {minutes:.1f} min")
        return auc_valid

    def checkpoint(study_, trial_):
        rows = [{"corrida": t.number, "auc_valid": t.value, **t.params, **t.user_attrs} for t in completed()]
        table = pd.DataFrame(rows).sort_values("auc_valid", ascending=False) if rows else pd.DataFrame()
        try:
            table.to_csv(BO_DIR / "corridas.csv", index=False)
            shutil.copy(local_db, bucket_db.with_name(bucket_db.name + ".tmp"))
            os.replace(bucket_db.with_name(bucket_db.name + ".tmp"), bucket_db)
        except Exception as error:  # un error pasajero del bucket no frena la corrida
            log(f"[aviso] no pude copiar el estudio: {error!r}")
        if STOP_BO_FILE.exists() and completed():
            log(f"Encontré {STOP_BO_FILE.name}: corto la bayesiana con {len(completed())} corridas.")
            study_.stop()

    remaining = N_TRIALS - len(completed())
    if remaining > 0 and not (STOP_BO_FILE.exists() and completed()):
        log(f"Bayesiana: faltan {remaining} de {N_TRIALS} corridas")
        study.optimize(objective, n_trials=remaining, callbacks=[checkpoint], gc_after_trial=True)
    else:
        log(f"Bayesiana: ya estaban las {len(completed())} corridas")
    checkpoint(study, None)

    trials = sorted(completed(), key=lambda t: t.value, reverse=True)
    best = trials[0]
    save_json(BO_DIR / "mejores_hiperparametros.json", {
        "corrida": best.number, "auc_valid": best.value, "valores": best.params,
        "best_iteration": best.user_attrs["best_iteration"], "semilla": BO_SEED, "fijos": FIXED_PARAMS})
    log(f"Mejor corrida: {best.number} | AUC {VALID_MONTH} {best.value:.5f} | {best.params} | "
        f"{best.user_attrs['best_iteration']} iter")
    return trials


# ---------------------------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------------------------
def run_seeds(data, trials):
    """Reentrena los TOP_K con N_SEMILLAS semillas y las iteraciones del early stopping de la bayesiana.

    Orden: todas las semillas del primero, después las del segundo, etc. Con PARAR se corta prolijo.
    """
    pending = [(t, s) for t in trials[:TOP_K] for s in SEEDS[:N_SEMILLAS]
               if not predicciones_existen(EXP.pred_dir, model_name(t.number, s))]
    log(f"Semillas: {min(TOP_K, len(trials))} configuraciones x {N_SEMILLAS} semillas, faltan {len(pending)} modelos")
    for i, (trial, seed) in enumerate(pending, start=1):
        if EXP.stop_file.exists():
            log(f"Encontré {EXP.stop_file.name}: no sigo con las semillas.")
            break
        start = time.perf_counter()
        best_iter = int(trial.user_attrs["best_iteration"])
        booster = data.entrenar(trial.params, seed, best_iter)
        data.predecir_y_guardar(booster, EXP.pred_dir, model_name(trial.number, seed))
        del booster
        release_memory()
        log(f"  semilla {i}/{len(pending)}: corrida {trial.number} semilla {seed} | {best_iter} iter | "
            f"{(time.perf_counter() - start) / 60:.1f} min")


# ---------------------------------------------------------------------------------------------
# Evaluación (solo lee predicciones guardadas)
# ---------------------------------------------------------------------------------------------
def evaluate():
    """Para cada modelo: AUC por mes, corte elegido en 201908 y ganancia en 201910 con ese corte."""
    bo = pd.read_csv(BO_DIR / "corridas.csv")
    ranking = {int(row.corrida): i + 1 for i, row in enumerate(bo.itertuples())}
    preds = {month: cargar_predicciones(EXP.pred_dir, month, "bo*_s*") for month in PRED_MONTHS}
    models = [c for c in preds[VALID_MONTH].columns if c.startswith("bo")]

    rows = []
    for name in models:
        trial, seed = int(name[2:5]), int(name.split("_s")[1])
        metrics = evaluar_modelo(preds[VALID_MONTH], preds[VALID_MONTH][name],
                                 preds[TEST_MONTH], preds[TEST_MONTH][name])
        row = {"modelo": name, "corrida": trial, "ranking_bo": ranking.get(trial), "semilla": seed,
               f"corte_{VALID_MONTH}": metrics["corte"],
               f"ganancia_{VALID_MONTH}": metrics["ganancia_valid"],
               f"ganancia_{TEST_MONTH}": metrics["ganancia_test"]}
        for month in PRED_MONTHS:
            positive = preds[month]["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).to_numpy()
            row[f"auc_{month}"] = auc_score(positive, preds[month][name])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["ranking_bo", "semilla"]).reset_index(drop=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RES_DIR / "modelos.csv", index=False)

    gtest = f"ganancia_{TEST_MONTH}"
    best_trial = int(bo.iloc[0]["corrida"])
    baseline = table[(table["corrida"] == best_trial) & (table["semilla"] == BO_SEED)].iloc[0]
    seeds_best = table[table["corrida"] == best_trial][gtest]
    per_config = (table[table["ranking_bo"] <= TOP_K].groupby("corrida")
                  .agg(ranking_bo=("ranking_bo", "first"), semillas=("semilla", "count"),
                       ganancia_media=(gtest, "mean"), ganancia_desvio=(gtest, "std"))
                  .sort_values("ranking_bo").reset_index())
    summary = {
        "baseline": {
            "modelo": baseline["modelo"],
            "hiperparametros": json.loads((BO_DIR / "mejores_hiperparametros.json").read_text(encoding="utf-8")),
            f"auc_{VALID_MONTH}": float(baseline[f"auc_{VALID_MONTH}"]),
            f"auc_{TEST_MONTH}": float(baseline[f"auc_{TEST_MONTH}"]),
            f"corte_elegido_en_{VALID_MONTH}": int(baseline[f"corte_{VALID_MONTH}"]),
            gtest: float(baseline[gtest]),
        },
        "semillas_del_mejor": {
            "n": int(len(seeds_best)), "media": float(seeds_best.mean()),
            "desvio": float(seeds_best.std(ddof=1)) if len(seeds_best) > 1 else None,
            "min": float(seeds_best.min()), "max": float(seeds_best.max()),
        },
        "top_k_por_configuracion": json.loads(per_config.to_json(orient="records")),
    }
    save_json(RES_DIR / "resumen.json", summary)

    log("=" * 70)
    log(f"BASELINE ({baseline['modelo']}): AUC {VALID_MONTH} {baseline[f'auc_{VALID_MONTH}']:.5f} | "
        f"AUC {TEST_MONTH} {baseline[f'auc_{TEST_MONTH}']:.5f} | corte (elegido en {VALID_MONTH}) "
        f"{int(baseline[f'corte_{VALID_MONTH}']):,} | ganancia {TEST_MONTH} {baseline[gtest]:.1f}")
    if len(seeds_best) > 1:
        log(f"Semillas del mejor ({len(seeds_best)}): ganancia {TEST_MONTH} media {seeds_best.mean():.1f} | "
            f"desvío {seeds_best.std(ddof=1):.1f} | min {seeds_best.min():.1f} | max {seeds_best.max():.1f}")
    log("Resultados en", RES_DIR)
    return table, summary


# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-evaluar", action="store_true", help="solo rehace la evaluación")
    args = parser.parse_args()
    for folder in (BO_DIR, RES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    if args.solo_evaluar:
        evaluate()
        return

    ensure_packages()
    import lightgbm
    import optuna

    if EXP.stop_file.exists():  # un PARAR viejo frenaría las semillas apenas empiezan
        EXP.stop_file.unlink()
    log("=" * 70)
    log(f"{EXP_ID} | prueba: {MODO_PRUEBA} | Google Cloud: {EN_GCP} | hilos: {NUM_THREADS} | "
        f"lightgbm {lightgbm.__version__} | optuna {optuna.__version__}")
    log("Resultados:", EXP.out_dir, "| temporales:", EXP.local_dir)
    save_json(EXP.out_dir / "config.json", {
        "train": TRAIN_MONTHS, "valid": VALID_MONTH, "test": TEST_MONTH, "pred_months": PRED_MONTHS,
        "lags": LAG_ORDERS, "fijos": FIXED_PARAMS, "search_space": SEARCH_SPACE, "n_trials": N_TRIALS,
        "n_startup": N_STARTUP_TRIALS, "max_rounds": MAX_ROUNDS, "bo_seed": BO_SEED, "top_k": TOP_K,
        "seeds": SEEDS[:N_SEMILLAS], "gain": [GAIN_BAJA2, GAIN_OTHER], "gain_window": GAIN_WINDOW,
        "prueba": MODO_PRUEBA})

    try:
        data = Datos(armar_variables(log), log)
        trials = run_bayesian(data)
        run_seeds(data, trials)
        evaluate()
    except BaseException as error:
        log(f"[error] {type(error).__name__}: {error}")
        if APAGAR_AL_TERMINAR and not isinstance(error, KeyboardInterrupt):
            EXP.apagar_vm("falló la corrida")
        raise
    if APAGAR_AL_TERMINAR:
        EXP.apagar_vm("terminó")


if __name__ == "__main__":
    main()
