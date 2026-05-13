"""Hulpfuncties voor het groepsnotebook: feature engineering, evaluatie en modellering."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

# evaluation.py staat in dezelfde directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import compute_prediction_utility

# Constanten

VITAL_SIGNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

LAB_VALUES = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets',
]

DEMOGRAPHICS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

EXCLUDE_COLS = ['SepsisLabel', 'Patient_ID']


# Data laden

def load_data(train_path, test_path):
    """Laad ruwe CSV-bestanden en verwijder indexkolom.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    if 'Unnamed: 0' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in df_test.columns:
        df_test  = df_test.drop(columns=['Unnamed: 0'])
    return df_train, df_test


# Feature engineering

def engineer_features(df):
    """Uitgebreide feature engineering.

    - Rolling vensters: 3h, 6h, 12h
    - Differentiaal: 6h (naast 1h en 3h)
    - Vitale interacties: HR*Resp, HR/MAP, Temp*HR, Resp/O2Sat,
      shock_index_trend_6h
    - Time-since-last-measurement
    - Missende waarde variabele + totaal aantal missende vitale/lab-waarden
    - ICULOS-afgeleiden: icos_squared, log_iculos
    - Cumulatieve instabiliteitsmaat: cumulatieve som van absolute veranderingen
    - Na alle feature-berekeningen ook de feature-kolommen forward-fillen
      per patiënt (klinisch verdedigbaar: arts weet ook nog wat het laatste
      lab was)

    Parameters
    ----------
    df : pd.DataFrame
        Ruwe trainings- of testdata

    Returns
    -------
    pd.DataFrame
        Originele kolommen + alle nieuwe features, forward-filled per patiënt.
    """
    df = df.copy()
    df = df.sort_values(['Patient_ID', 'ICULOS']).reset_index(drop=True)

    clinical_vars = VITAL_SIGNS + LAB_VALUES
    df_filled = df.groupby('Patient_ID')[clinical_vars].ffill()

    new_features = {}
    key_vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']
    key_labs   = ['WBC', 'Lactate', 'Creatinine', 'BUN', 'Platelets',
                  'Bilirubin_total', 'pH', 'FiO2']
    key_vars   = key_vitals + key_labs

    # 1. Rolling statistieken (3h, 6h, 12h)
    for window in [3, 6, 12]:
        for col in key_vars:
            grp = df_filled.groupby(df['Patient_ID'])[col]
            roll = grp.rolling(window, min_periods=1)
            new_features[f'{col}_mean_{window}h'] = roll.mean().reset_index(level=0, drop=True)
            new_features[f'{col}_std_{window}h']  = roll.std().reset_index(level=0, drop=True)
            new_features[f'{col}_min_{window}h']  = roll.min().reset_index(level=0, drop=True)
            new_features[f'{col}_max_{window}h']  = roll.max().reset_index(level=0, drop=True)

    # 2. Differentialen (1h, 3h, 6h)
    for col in key_vars:
        grp = df_filled.groupby(df['Patient_ID'])[col]
        new_features[f'{col}_diff_1h'] = grp.diff(1)
        new_features[f'{col}_diff_3h'] = grp.diff(3)
        new_features[f'{col}_diff_6h'] = grp.diff(6)

    # 3. Klinische indices
    sbp  = df_filled['SBP'].replace(0, float('nan'))
    dbp  = df_filled['DBP'].replace(0, float('nan'))
    hr   = df_filled['HR']
    resp = df_filled['Resp']
    o2   = df_filled['O2Sat'].replace(0, float('nan'))
    map_ = df_filled['MAP'].replace(0, float('nan'))
    temp = df_filled['Temp']

    new_features['shock_index']          = hr / sbp
    new_features['pulse_pressure']       = df_filled['SBP'] - df_filled['DBP']
    new_features['map_dbp_ratio']        = map_ / dbp
    new_features['bun_creatinine_ratio'] = df_filled['BUN'] / df_filled['Creatinine'].replace(0, float('nan'))

    # 4. Nieuwe vitale interacties
    new_features['hr_resp_product']      = hr * resp
    new_features['hr_map_ratio']         = hr / map_
    new_features['temp_hr']              = temp * hr
    new_features['resp_o2sat']           = resp / o2
    shock_idx_ser = hr / sbp
    # diff(5) = huidige waarde minus 5 uur geleden = equivalent aan rolling(6).last - rolling(6).first
    new_features['shock_index_trend_6h'] = shock_idx_ser.groupby(df['Patient_ID']).diff(5)

    # 5. Time-since-last-measurement
    for col in key_labs:
        last_meas = df['ICULOS'].where(~df[col].isna())
        last_meas = last_meas.groupby(df['Patient_ID']).ffill()
        new_features[f'{col}_time_since'] = (df['ICULOS'] - last_meas).clip(lower=0)

    # 6. Missende waarde indicatoren
    for col in key_vitals:
        new_features[f'{col}_missing'] = df[col].isnull().astype(int)
    new_features['n_missing_vitals'] = df[VITAL_SIGNS].isnull().sum(axis=1)
    new_features['n_missing_labs']   = df[LAB_VALUES].isnull().sum(axis=1)

    # 7. ICULOS-afgeleiden
    new_features['iculos_squared'] = df['ICULOS'] ** 2
    new_features['log_iculos']     = np.log1p(df['ICULOS'])

    # 8. Cumulatieve instabiliteitsmaat
    for col in ['HR', 'MAP', 'Resp', 'Temp']:
        abs_diff = df_filled.groupby(df['Patient_ID'])[col].diff(1).abs()
        new_features[f'{col}_cumulative_change'] = abs_diff.groupby(df['Patient_ID']).cumsum()

    result = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)

    # 9. Forward-fill alle feature-kolommen per patiënt
    feat_cols = [c for c in result.columns if c not in EXCLUDE_COLS]
    result[feat_cols] = result.groupby('Patient_ID')[feat_cols].ffill()

    return result


# Officiële utility parameters (PhysioNet Challenge 2019)

EVAL_PARAMS = dict(dt_early=-12, dt_optimal=-6, dt_late=3,
                   max_u_tp=1, min_u_fn=-2, u_fp=-0.2, u_tn=0)


# Evaluatiefuncties

def utility_score(df, pred_col='SepsisLabel_pred', label_col='SepsisLabel',
                  patient_col='Patient_ID'):
    """Genormaliseerde utility score (officieel challenge format).

    Normalisatie: (U_observed - U_inaction) / (U_best - U_inaction)
    """
    observed, best, inaction = [], [], []
    for _, pat in df.groupby(patient_col):
        labels = pat[label_col].values
        preds  = pat[pred_col].values
        n      = len(labels)

        opt_preds = np.zeros(n)
        if np.any(labels):
            t_sep = int(np.argmax(labels)) - EVAL_PARAMS['dt_optimal']
            s = max(0, t_sep + EVAL_PARAMS['dt_early'])
            e = min(t_sep + EVAL_PARAMS['dt_late'] + 1, n)
            opt_preds[s:e] = 1

        observed.append(compute_prediction_utility(labels, preds,        **EVAL_PARAMS))
        best.append(    compute_prediction_utility(labels, opt_preds,    **EVAL_PARAMS))
        inaction.append(compute_prediction_utility(labels, np.zeros(n),  **EVAL_PARAMS))

    denom = sum(best) - sum(inaction)
    return (sum(observed) - sum(inaction)) / denom if denom != 0 else 0.0


def compute_utility_labels(df, patient_col='Patient_ID', label_col='SepsisLabel'):
    """Per-tijdstap trainingsdoel: marginale utility van alarm slaan vs. stil zijn.

    Positief = alarm is op dit moment winstgevend.
    Negatief = alarm kost meer dan het oplevert (te vroeg, te laat, of geen sepsis).
    """
    p  = EVAL_PARAMS
    m1 = p['max_u_tp'] / (p['dt_optimal'] - p['dt_early'])
    b1 = -m1 * p['dt_early']
    m2 = -float(p['max_u_tp']) / (p['dt_late'] - p['dt_optimal'])
    b2 = -m2 * p['dt_late']
    m3 = float(p['min_u_fn']) / (p['dt_late'] - p['dt_optimal'])
    b3 = -m3 * p['dt_optimal']

    out = pd.Series(np.nan, index=df.index)
    for pid, grp in df.groupby(patient_col):
        labels = grp[label_col].values
        n      = len(labels)
        septic = bool(np.any(labels))
        t_sep  = int(np.argmax(labels)) - p['dt_optimal'] if septic else n + 99

        diff = np.zeros(n)
        for t in range(n):
            if t <= t_sep + p['dt_late']:
                if septic:
                    s1 = max(m1*(t-t_sep)+b1, p['u_fp']) if t <= t_sep+p['dt_optimal'] \
                         else m2*(t-t_sep)+b2
                    s0 = 0 if t <= t_sep+p['dt_optimal'] else m3*(t-t_sep)+b3
                else:
                    s1, s0 = p['u_fp'], p['u_tn']
            else:
                s1, s0 = 0.0, 0.0
            diff[t] = s1 - s0

        out[grp.index] = diff
    return out


# Train/val split

def split_train_val(df_fe, feature_cols, test_size=0.2, random_state=42):
    """Patient-level train/val split zonder data leakage.

    Returns
    -------
    X_tr, y_tr, X_val, y_val, df_tr, df_val
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, val_idx = next(gss.split(
        df_fe, df_fe['SepsisLabel'], groups=df_fe['Patient_ID']
    ))
    tr_pats  = df_fe.iloc[tr_idx]['Patient_ID'].unique()
    val_pats = df_fe.iloc[val_idx]['Patient_ID'].unique()
    assert not (set(tr_pats) & set(val_pats)), 'Data leakage!'

    df_tr  = df_fe[df_fe['Patient_ID'].isin(tr_pats)]
    df_val = df_fe[df_fe['Patient_ID'].isin(val_pats)]
    return (
        df_tr[feature_cols],  df_tr['SepsisLabel'],
        df_val[feature_cols], df_val['SepsisLabel'],
        df_tr, df_val,
    )


# Modellen trainen

def train_binary(X_tr, y_tr, X_val, y_val, scale_pos_weight=8, seed=42):
    """LightGBM binary classifier met class weighting.

    Returns
    -------
    model, val_proba
    """
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 8,
        'min_child_samples': 80, 'scale_pos_weight': scale_pos_weight,
        'subsample': 0.7, 'subsample_freq': 1, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_gain_to_split': 0.01,
        'verbose': -1, 'n_jobs': -1, 'seed': seed,
    }
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model  = lgb.train(
        params, ds_tr, num_boost_round=5000,
        valid_sets=[ds_tr, ds_val], valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
    )
    return model, model.predict(X_val, num_iteration=model.best_iteration)


def train_utility_reg(X_tr, y_util_tr, X_val, y_util_val, sw_ratio=10.0, seed=42):
    """LGBMRegressor met sample weighting op positieve utility-rijen.

    Returns
    -------
    model, val_scores
    """
    sw_tr  = np.where(np.asarray(y_util_tr)  > 0, sw_ratio, 1.0)
    sw_val = np.where(np.asarray(y_util_val) > 0, sw_ratio, 1.0)
    model  = LGBMRegressor(
        n_estimators=5000, learning_rate=0.03, num_leaves=63, max_depth=8,
        min_child_samples=80, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
        n_jobs=-1, random_state=seed, verbose=-1,
    )
    model.fit(
        X_tr, y_util_tr, sample_weight=sw_tr,
        eval_set=[(X_val, y_util_val)], eval_sample_weight=[sw_val],
        eval_metric='l2',
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
    )
    return model, model.predict(X_val)


# Threshold optimalisatie & blend

def find_best_threshold(scores, df_val_meta, score_fn, n_steps=80):
    """Zoek de drempelwaarde die de utility op de validatieset maximaliseert.

    Returns
    -------
    threshold : float
    utility   : float
    """
    mn, mx = float(scores.min()), float(scores.max())
    step   = (mx - mn) / (n_steps + 1)
    best_u, best_thr = -np.inf, (mn + mx) / 2

    for thr in np.arange(mn + step, mx, step):
        df_eval = df_val_meta[['Patient_ID', 'SepsisLabel']].copy()
        df_eval['SepsisLabel_pred'] = (scores >= thr).astype(int)
        u = score_fn(df_eval)
        if u > best_u:
            best_u, best_thr = u, float(thr)
    return best_thr, best_u


def run_optuna_blend(preds_dict, df_val_meta, score_fn, n_trials=100, seed=42):
    """Blend-optimalisatie via Optuna TPE.

    Parameters
    ----------
    preds_dict : dict[str, np.ndarray]
        Naam → val-scores per model. Worden genormaliseerd naar [0,1].
    df_val_meta : pd.DataFrame
        Heeft Patient_ID en SepsisLabel kolommen, positie-aligned met de score-arrays.
    score_fn : callable
        Neemt df met Patient_ID/SepsisLabel/SepsisLabel_pred, geeft utility float terug.

    Returns
    -------
    optuna.Study
    """
    assert optuna is not None, "pip install optuna"
    names  = list(preds_dict.keys())
    y_true = df_val_meta['SepsisLabel'].values

    normed = {}
    for n, arr in preds_dict.items():
        mn, mx = arr.min(), arr.max()
        normed[n] = (arr - mn) / (mx - mn + 1e-9)

    def objective(trial):
        raw_w   = np.array([trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in names])
        w       = raw_w / (raw_w.sum() + 1e-9)
        blended = sum(w[i] * normed[names[i]] for i in range(len(names)))
        thr     = trial.suggest_float('threshold', 0.10, 0.90)

        preds = (blended >= thr).astype(int)
        df_eval = df_val_meta[['Patient_ID', 'SepsisLabel']].copy()
        df_eval['SepsisLabel_pred'] = preds
        u  = score_fn(df_eval)

        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        trial.set_user_attr('tp', tp)
        trial.set_user_attr('fp', fp)
        trial.set_user_attr('fp_tp', round(fp / tp, 2) if tp > 0 else float('inf'))
        return u

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def apply_blend(study, preds_dict):
    """Pas de beste blend-configuratie uit een Optuna study toe.

    Returns
    -------
    blended_scores : np.ndarray
    threshold      : float
    binary_preds   : np.ndarray
    """
    names  = list(preds_dict.keys())
    normed = {}
    for n, arr in preds_dict.items():
        mn, mx = arr.min(), arr.max()
        normed[n] = (arr - mn) / (mx - mn + 1e-9)

    p     = study.best_params
    raw_w = np.array([p[f'w_{n}'] for n in names])
    w     = raw_w / (raw_w.sum() + 1e-9)

    blended = sum(w[i] * normed[names[i]] for i in range(len(names)))
    thr     = p['threshold']
    return blended, thr, (blended >= thr).astype(int)


# Gevectoriseerde utility & patient-smoothing

def build_utility_fast(df_val, y_val):
    """Pre-compute per-rij delta_u en denominator voor vectorized utility.

    Parameters
    ----------
    df_val : pd.DataFrame  — bevat kolom Patient_ID
    y_val  : array-like    — binaire labels, positie-aligned met df_val

    Returns
    -------
    delta_u_vec       : np.ndarray  shape (n,)
    denom             : float
    patient_boundaries: list of (start, end) tuples
    """
    p = EVAL_PARAMS
    m1 = p['max_u_tp'] / (p['dt_optimal'] - p['dt_early'])
    b1 = -m1 * p['dt_early']
    m2 = -float(p['max_u_tp']) / (p['dt_late'] - p['dt_optimal'])
    b2 = -m2 * p['dt_late']
    m3 = float(p['min_u_fn']) / (p['dt_late'] - p['dt_optimal'])
    b3 = -m3 * p['dt_optimal']

    pids   = df_val['Patient_ID'].values
    labels = np.asarray(y_val)
    n_val  = len(labels)

    delta_u   = np.zeros(n_val, dtype=np.float64)
    U_inaction = 0.0
    U_best     = 0.0
    boundaries = []

    changes = np.where(np.diff(pids) != 0)[0] + 1
    starts  = np.concatenate([[0], changes])
    ends    = np.concatenate([changes, [n_val]])

    for s, e in zip(starts, ends):
        boundaries.append((s, e))
        lbl = labels[s:e]
        n   = e - s
        septic = bool(np.any(lbl))

        if not septic:
            delta_u[s:e] = p['u_fp'] - p['u_tn']
            U_inaction += n * p['u_tn']
            U_best     += n * p['u_tn']
        else:
            t_sep   = int(np.argmax(lbl)) - p['dt_optimal']
            u_alarm = np.zeros(n)
            u_silent = np.zeros(n)
            best_p  = np.zeros(n)
            for t in range(n):
                if t <= t_sep + p['dt_late']:
                    if t <= t_sep + p['dt_optimal']:
                        u_alarm[t]  = max(m1 * (t - t_sep) + b1, p['u_fp'])
                        u_silent[t] = 0.0
                    else:
                        u_alarm[t]  = m2 * (t - t_sep) + b2
                        u_silent[t] = m3 * (t - t_sep) + b3
            delta_u[s:e] = u_alarm - u_silent
            U_inaction  += np.sum(u_silent)
            bs = max(0, int(t_sep + p['dt_early']))
            be = min(int(t_sep + p['dt_late']) + 1, n)
            best_p[bs:be] = 1
            U_best += np.sum(best_p * u_alarm + (1 - best_p) * u_silent)

    denom = U_best - U_inaction
    return delta_u, denom, boundaries


def make_patient_smoother(patient_boundaries):
    """Geeft een smoothing-functie terug die runs van 1 pakt als min_consecutive=1.

    Parameters
    ----------
    patient_boundaries : list of (start, end) tuples  — uit build_utility_fast

    Returns
    -------
    smoother(preds, min_consecutive) -> np.ndarray
    """
    def smoother(preds, min_consecutive):
        if min_consecutive <= 1:
            return preds
        out = np.zeros_like(preds)
        for s, e in patient_boundaries:
            p = preds[s:e]
            o = np.zeros(len(p), dtype=preds.dtype)
            count = 0
            for i in range(len(p)):
                if p[i] == 1:
                    count += 1
                    if count >= min_consecutive:
                        o[i] = 1
                else:
                    count = 0
            out[s:e] = o
        return out
    return smoother


# Per-model Optuna hyperparameter tuning

def run_optuna_binary_fe(X_tr, y_tr, X_val, utility_fast_fn,
                         smoother_fn=None, n_trials=60, seed=42, y_val=None, timeout=None):
    """Optuna-studie voor LightGBM binaire classifier op FE-data.

    Parameters
    ----------
    X_tr, y_tr       : treindata
    X_val            : validatiedata (features)
    utility_fast_fn  : callable(preds_binary) -> float  — vectorized utility
    smoother_fn      : callable(preds, min_consecutive) -> preds  of None
    n_trials         : int
    seed             : int

    Returns
    -------
    optuna.Study
    """
    assert optuna is not None, "pip install optuna"
    _y_tr_arr  = np.asarray(y_tr)
    _val_labels = np.asarray(y_val) if y_val is not None else np.zeros(len(X_val))

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc',
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 50.0),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 15, 128),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'min_child_samples':trial.suggest_int('min_child_samples', 20, 300),
            'subsample':        trial.suggest_float('subsample', 0.3, 0.95),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.95),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.001, 200.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.001, 200.0, log=True),
            'verbose': -1, 'n_jobs': -1, 'seed': seed,
        }
        thr = trial.suggest_float('threshold', 0.10, 0.92)
        mc  = trial.suggest_int('min_consecutive', 1, 2)

        _dtr  = lgb.Dataset(X_tr,  label=_y_tr_arr)
        _dval = lgb.Dataset(X_val, label=_val_labels, reference=_dtr)
        mdl   = lgb.train(params, _dtr, num_boost_round=2000, valid_sets=[_dval],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        proba = mdl.predict(X_val)
        preds = (proba >= thr).astype(int)
        if smoother_fn is not None and mc > 1:
            preds = smoother_fn(preds, mc)

        trial.set_user_attr('best_iter', mdl.best_iteration)
        return utility_fast_fn(preds)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    print(f"binary_fe  ·  utility {study.best_value:.4f}  ·  trial #{study.best_trial.number}")
    return study


def run_optuna_utility_reg(X_tr, y_util_tr, X_val, utility_fast_fn,
                           smoother_fn=None, n_trials=60, seed=42, y_val=None, timeout=None):
    """Optuna-studie voor LightGBM utility-regressie op FE-data.

    Returns
    -------
    optuna.Study
    """
    assert optuna is not None, "pip install optuna"
    _y_tr = np.asarray(y_util_tr)
    # Gebruik echte utility-val-labels voor l2-monitoring (zelfde stopconditie als baseline).
    _val_labels = np.asarray(y_val) if y_val is not None else np.zeros(len(X_val))

    def objective(trial):
        sw_ratio = trial.suggest_float('sw_ratio', 2.0, 100.0, log=True)
        thr      = trial.suggest_float('threshold', 0.0, 1.6)
        mc       = trial.suggest_int('min_consecutive', 1, 2)
        params = {
            'objective': 'regression', 'metric': 'l2',
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 15, 128),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'min_child_samples':trial.suggest_int('min_child_samples', 20, 300),
            'subsample':        trial.suggest_float('subsample', 0.3, 0.95),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.95),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.001, 200.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.001, 200.0, log=True),
            'verbose': -1, 'n_jobs': -1, 'seed': seed,
        }
        sw     = np.where(_y_tr > 0, sw_ratio, 1.0)
        _dtr   = lgb.Dataset(X_tr,  label=_y_tr, weight=sw)
        _dval  = lgb.Dataset(X_val, label=_val_labels, reference=_dtr)
        mdl    = lgb.train(params, _dtr, num_boost_round=2000, valid_sets=[_dval],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        scores = mdl.predict(X_val)
        preds  = (scores >= thr).astype(int)
        if smoother_fn is not None and mc > 1:
            preds = smoother_fn(preds, mc)

        trial.set_user_attr('best_iter', mdl.best_iteration)
        return utility_fast_fn(preds)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    print(f"utility_reg  ·  utility {study.best_value:.4f}  ·  trial #{study.best_trial.number}")
    return study


def run_optuna_xgb(X_tr, y_tr, X_val, utility_fast_fn,
                   smoother_fn=None, n_trials=60, seed=42, y_val=None, timeout=None):
    """Optuna-studie voor XGBoost binaire classifier op ruwe data.

    Returns
    -------
    optuna.Study
    """
    assert optuna is not None, "pip install optuna"
    _y_tr       = np.asarray(y_tr)
    _y_val_es   = np.asarray(y_val) if y_val is not None else None

    def objective(trial):
        thr = trial.suggest_float('threshold', 0.10, 0.92)
        mc  = trial.suggest_int('min_consecutive', 1, 2)
        mdl = XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            tree_method='hist', n_jobs=-1, random_state=seed,
            n_estimators=1000, early_stopping_rounds=50,
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            max_depth         = trial.suggest_int('max_depth', 3, 10),
            min_child_weight  = trial.suggest_float('min_child_weight', 1.0, 12.0),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
            gamma             = trial.suggest_float('gamma', 0.0, 5.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 0.2, 20.0, log=True),
            scale_pos_weight  = trial.suggest_float('scale_pos_weight', 2.0, 50.0),
        )
        _es = [(X_val, _y_val_es)] if _y_val_es is not None else None
        mdl.fit(X_tr, _y_tr, eval_set=_es, verbose=False)
        proba = mdl.predict_proba(X_val)[:, 1]
        preds = (proba >= thr).astype(int)
        if smoother_fn is not None and mc > 1:
            preds = smoother_fn(preds, mc)

        trial.set_user_attr('best_iter', mdl.best_iteration)
        return utility_fast_fn(preds)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    print(f"xgb_raw  ·  utility {study.best_value:.4f}  ·  trial #{study.best_trial.number}")
    return study


# Hertraining met beste Optuna-parameters

def retrain_best_lgb_binary(study, X_tr, y_tr, X_val, seed=42):
    """Hertraining LightGBM binary met best_params uit studie.

    Returns
    -------
    model, val_proba, threshold, min_consecutive
    """
    p   = study.best_params.copy()
    thr = p.pop('threshold')
    mc  = p.pop('min_consecutive', 1)
    n_est = study.best_trial.user_attrs.get('best_iter', 500)
    p.update({'objective': 'binary', 'metric': 'auc',
              'subsample_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': seed})
    _dtr = lgb.Dataset(X_tr, label=np.asarray(y_tr))
    mdl  = lgb.train(p, _dtr, num_boost_round=n_est, callbacks=[lgb.log_evaluation(0)])
    return mdl, mdl.predict(X_val), thr, mc


def retrain_best_lgb_reg(study, X_tr, y_util_tr, X_val, seed=42):
    """Hertraining LightGBM utility-reg met best_params uit studie.

    Returns
    -------
    model, val_scores, threshold, min_consecutive
    """
    p        = study.best_params.copy()
    thr      = p.pop('threshold')
    mc       = p.pop('min_consecutive', 1)
    sw_ratio = p.pop('sw_ratio')
    n_est    = study.best_trial.user_attrs.get('best_iter', 500)
    p.update({'objective': 'regression', 'metric': 'l2',
              'subsample_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': seed})
    _y  = np.asarray(y_util_tr)
    sw  = np.where(_y > 0, sw_ratio, 1.0)
    _dtr = lgb.Dataset(X_tr, label=_y, weight=sw)
    mdl  = lgb.train(p, _dtr, num_boost_round=n_est, callbacks=[lgb.log_evaluation(0)])
    return mdl, mdl.predict(X_val), thr, mc


def retrain_best_xgb(study, X_tr, y_tr, X_val, seed=42):
    """Hertraining XGBoost met best_params uit studie.

    Returns
    -------
    model, val_proba, threshold, min_consecutive
    """
    p     = study.best_params.copy()
    thr   = p.pop('threshold')
    mc    = p.pop('min_consecutive', 1)
    n_est = study.best_trial.user_attrs.get('best_iter', 300)
    mdl   = XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', n_jobs=-1, random_state=seed,
        n_estimators=n_est + 1,
        **p,
    )
    mdl.fit(X_tr, np.asarray(y_tr), verbose=False)
    return mdl, mdl.predict_proba(X_val)[:, 1], thr, mc

