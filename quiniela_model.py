"""
quiniela_model.py  ·  Pronóstico 1X2 para La Quiniela (Dixon-Coles + XGBoost calibrado)

INSTALACIÓN
    pip install pandas numpy scipy scikit-learn xgboost requests

EJECUCIÓN
    python quiniela_model.py

ESTRUCTURA
    Módulo 1  Ingesta de datos      -> descarga/limpia CSV de football-data.co.uk (SP1)
    Módulo 2  Dixon-Coles           -> ataque, defensa, ventaja de campo, rho (scipy.optimize)
    Módulo 3  Capa XGBoost          -> combina Dixon-Coles + fatiga/bajas/forma y calibra
    Módulo 4  Evaluación y main     -> compara contra Dixon-Coles solo y contra el mercado

CONVENCIÓN: clase 0 = "1" (gana local), 1 = "X" (empate), 2 = "2" (gana visitante).
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# ======================================================================================
# MÓDULO 1 · INGESTA DE DATOS
# ======================================================================================
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
CACHE_DIR = Path("data_cache")

# Tripletas de cuotas (local, empate, visitante) por orden de preferencia.
# Pinnacle de cierre es la mejor referencia de mercado; si falta, se usa la siguiente.
ODDS_COLUMNS = [
    ("PSCH", "PSCD", "PSCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
]


def season_code(start_year: int) -> str:
    """2026 -> '2627' (formato de football-data.co.uk)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: pd.Timestamp | None = None) -> int:
    """La temporada empieza en verano: nov-2026 -> 2026; mar-2027 -> 2026."""
    today = today or pd.Timestamp.today()
    return today.year if today.month >= 7 else today.year - 1


def download_season(start_year: int, league: str = "SP1", force: bool = False) -> pd.DataFrame:
    """Descarga (y cachea en disco) el CSV de una temporada."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{league}_{season_code(start_year)}.csv"
    if force or not path.exists():
        url = BASE_URL.format(code=season_code(start_year), league=league)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def clean_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas: Date, Home, Away, HG, AG, Result (0/1/2) y probs. del mercado."""
    df = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    out = pd.DataFrame({
        "Date": df["Date"],
        "Home": df["HomeTeam"].str.strip(),
        "Away": df["AwayTeam"].str.strip(),
        "HG": df["FTHG"].astype(int),
        "AG": df["FTAG"].astype(int),
    })
    out["Result"] = np.select([out.HG > out.AG, out.HG == out.AG], [0, 1], default=2)

    # Probabilidades implícitas del mercado SIN margen (normalizadas a suma 1)
    mkt = np.full((len(out), 3), np.nan)
    for cols in ODDS_COLUMNS:
        if all(c in df.columns for c in cols):
            odds = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
            inv = 1.0 / odds
            probs = inv / inv.sum(axis=1, keepdims=True)
            fill = np.isnan(mkt[:, 0]) & ~np.isnan(probs[:, 0])
            mkt[fill] = probs[fill]
    out[["mkt_p1", "mkt_px", "mkt_p2"]] = mkt
    return out


def load_history(n_seasons: int = 2, leagues: tuple[str, ...] = ("SP1", "SP2")) -> pd.DataFrame:
    """
    Temporada actual (siempre re-descargada) + las n-1 anteriores, de Primera (SP1) y
    Segunda (SP2): La Quiniela mezcla ambas. Dixon-Coles las ajusta en un solo modelo;
    los equipos de cada liga forman grupos separados (nivel de goles propio por liga).
    """
    start = current_season_start()
    frames = []
    for league in leagues:
        for k in range(n_seasons):
            year = start - k
            try:
                frames.append(clean_matches(download_season(year, league, force=(k == 0))))
            except Exception as exc:  # temporada sin fichero aún, sin red, etc.
                warnings.warn(f"No se pudo cargar {league} {year}/{year + 1}: {exc}")
    if not frames:
        raise RuntimeError("Sin datos: revisa la conexión o descarga los CSV a mano en data_cache/.")
    return pd.concat(frames).sort_values("Date").reset_index(drop=True)


# Formatos de CSV admitidos para histórico extra (fútbol femenino u otras competiciones)
_EXTRA_FORMATS = [
    ("Date", "Home", "Away", "HG", "AG"),                      # formato propio
    ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"),          # formato football-data.co.uk
    ("date_GMT", "home_team_name", "away_team_name",           # formato FootyStats ("CSV de Partidos")
     "home_team_goal_count", "away_team_goal_count"),
]


def parse_extra_csv(source, femenino: bool = True) -> pd.DataFrame:
    """
    Lee un CSV de partidos (ruta o archivo subido) y lo devuelve en el formato interno.
    Con femenino=True los equipos se guardan como 'F:Nombre' para no mezclarlos con los masculinos
    (p. ej. Real Madrid y Real Madrid Femenino). Solo se usan partidos ya jugados.
    """
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    raw = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    fmt = next((f for f in _EXTRA_FORMATS if all(c in raw.columns for c in f)), None)
    if fmt is None:
        raise ValueError("No reconozco las columnas. Necesito: Date, Home, Away, HG, AG "
                         "(o las de football-data / FootyStats).")
    dcol, hcol, acol, hg, ag = fmt
    if "status" in raw.columns:
        raw = raw[raw["status"].astype(str).str.lower().isin(["complete", "completed", "finished", "ft"])]
    d = raw[dcol].astype(str)
    dates = pd.to_datetime(d.str.split(" - ").str[0], format="%b %d %Y", errors="coerce")
    if dates.isna().mean() > 0.5:
        dates = pd.to_datetime(d, dayfirst=True, errors="coerce")
    out = pd.DataFrame({"Date": dates, "Home": raw[hcol].astype(str).str.strip(),
                        "Away": raw[acol].astype(str).str.strip(),
                        "HG": pd.to_numeric(raw[hg], errors="coerce"), "AG": pd.to_numeric(raw[ag], errors="coerce")})
    out = out.dropna().reset_index(drop=True)
    out[["HG", "AG"]] = out[["HG", "AG"]].astype(int)
    out["Result"] = np.select([out.HG > out.AG, out.HG == out.AG], [0, 1], default=2)
    out[["mkt_p1", "mkt_px", "mkt_p2"]] = np.nan
    if femenino:
        out["Home"], out["Away"] = "F:" + out["Home"], "F:" + out["Away"]
    return out


# ======================================================================================
# MÓDULO 2 · DIXON-COLES
# ======================================================================================
def _tau(x, y, lam, mu, rho):
    """Corrección de Dixon-Coles para marcadores bajos (0-0, 0-1, 1-0, 1-1)."""
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    """
    Goles local ~ Poisson(lam), visitante ~ Poisson(mu):
        lam = exp(home_adv + ataque[local] - defensa[visitante])
        mu  = exp(ataque[visitante] - defensa[local])
    Los partidos antiguos pesan menos: peso = 0.5 ** (días / half_life_days).
    """

    def __init__(self, half_life_days: float = 365.0, reg: float = 0.5, max_goals: int = 10):
        self.half_life_days = half_life_days
        self.reg = reg              # regularización ridge (estabiliza equipos con pocos partidos)
        self.max_goals = max_goals

    # ---------------------------------------------------------------- entrenamiento
    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> "DixonColes":
        ref_date = ref_date or df["Date"].max()
        self.teams_ = sorted(set(df.Home) | set(df.Away))
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        h = df.Home.map(idx).to_numpy()
        a = df.Away.map(idx).to_numpy()
        x = df.HG.to_numpy()
        y = df.AG.to_numpy()
        days = (ref_date - df.Date).dt.days.clip(lower=0).to_numpy()
        w = 0.5 ** (days / self.half_life_days)

        def unpack(p):
            att = p[:n] - p[:n].mean()          # media 0 -> identificabilidad
            dfn = p[n:2 * n] - p[n:2 * n].mean()
            return att, dfn, p[2 * n], p[2 * n + 1]

        def neg_loglik(p):
            """Devuelve (valor, gradiente analítico) de -log-verosimilitud ponderada."""
            att, dfn, home_adv, rho = unpack(p)
            lam = np.exp(home_adv + att[h] - dfn[a])
            mu = np.exp(att[a] - dfn[h])
            tau = np.maximum(_tau(x, y, lam, mu, rho), 1e-10)
            ll = x * np.log(lam) - lam + y * np.log(mu) - mu + np.log(tau)
            f = -(w * ll).sum() + self.reg * ((att ** 2).sum() + (dfn ** 2).sum())
            # derivadas de log(tau) respecto a log(lam), log(mu) y rho
            c00, c01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
            c10, c11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
            t_lam = np.where(c00, -lam * mu * rho, np.where(c01, lam * rho, 0.0)) / tau
            t_mu = np.where(c00, -lam * mu * rho, np.where(c10, mu * rho, 0.0)) / tau
            t_rho = np.where(c00, -lam * mu, np.where(c01, lam, np.where(c10, mu, np.where(c11, -1.0, 0.0)))) / tau
            g_lam = w * (x - lam + t_lam)
            g_mu = w * (y - mu + t_mu)
            ga = -(np.bincount(h, g_lam, n) + np.bincount(a, g_mu, n)) + 2 * self.reg * att
            gd = (np.bincount(a, g_lam, n) + np.bincount(h, g_mu, n)) + 2 * self.reg * dfn
            grad = np.empty_like(p)
            grad[:n] = ga - ga.mean()            # el centrado de los parámetros se propaga al gradiente
            grad[n:2 * n] = gd - gd.mean()
            grad[2 * n] = -g_lam.sum()
            grad[2 * n + 1] = -(w * t_rho).sum()
            return f, grad

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.25                          # ventaja de campo inicial
        bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]   # rho acotado
        res = minimize(neg_loglik, p0, jac=True, method="L-BFGS-B", bounds=bounds)
        att, dfn, self.home_adv_, self.rho_ = unpack(res.x)
        self.attack_ = dict(zip(self.teams_, att))
        self.defence_ = dict(zip(self.teams_, dfn))
        return self

    # ---------------------------------------------------------------- predicción
    def _rates(self, home: str, away: str):
        for t in (home, away):
            if t not in self.attack_:
                warnings.warn(f"Equipo sin histórico: '{t}' (se usa nivel medio). "
                              f"Revisa que el nombre coincida con football-data.")
        ah, dh = self.attack_.get(home, 0.0), self.defence_.get(home, 0.0)
        aa, da = self.attack_.get(away, 0.0), self.defence_.get(away, 0.0)
        return np.exp(self.home_adv_ + ah - da), np.exp(aa - dh)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """M[i, j] = P(local marca i, visitante marca j)."""
        lam, mu = self._rates(home, away)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        r = self.rho_
        m[0, 0] *= 1 - lam * mu * r
        m[0, 1] *= 1 + lam * r
        m[1, 0] *= 1 + mu * r
        m[1, 1] *= 1 - r
        return m / m.sum()

    def predict_1x2(self, home: str, away: str) -> np.ndarray:
        m = self.score_matrix(home, away)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def dc_walk_forward(df: pd.DataFrame, min_train: int = 150, step_days: int = 14, **dc_kwargs) -> np.ndarray:
    """
    Probabilidades Dixon-Coles 'fuera de muestra': cada partido se predice con un modelo
    entrenado SOLO con partidos anteriores (reajuste cada `step_days` días).
    Es lo que alimenta a la capa XGBoost sin fugas de información.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    out = np.full((len(df), 3), np.nan)
    first = df.Date.iloc[min_train]
    starts = list(pd.date_range(first, df.Date.max(), freq=f"{step_days}D"))
    ends = starts[1:] + [df.Date.max() + pd.Timedelta(days=1)]
    for b0, b1 in zip(starts, ends):
        test_idx = df.index[(df.Date >= b0) & (df.Date < b1)]
        if len(test_idx) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DixonColes(**dc_kwargs).fit(df[df.Date < b0], ref_date=b0)
            for i in test_idx:
                out[i] = model.predict_1x2(df.at[i, "Home"], df.at[i, "Away"])
    return out


# ======================================================================================
# MÓDULO 3 · CAPA XGBOOST (Dixon-Coles + contexto) CON CALIBRACIÓN
# ======================================================================================
FEATURES = ["dc_p1", "dc_px", "dc_p2", "rest_diff", "form_diff", "absences_diff", "euro_diff"]


def add_rest_and_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descanso (días desde el último partido DE LIGA) y forma (media de puntos en los
    últimos 5 partidos). Solo usa información anterior a cada partido.
    NOTA: no ve Copa ni Europa; para eso está `add_context`.
    """
    last_date: dict[str, pd.Timestamp] = {}
    pts: dict[str, list[int]] = {}
    rows = []
    for r in df.itertuples():
        def snap(team):
            rest = min((r.Date - last_date[team]).days, 14) if team in last_date else 14
            form = np.mean(pts[team][-5:]) if pts.get(team) else 1.3
            return rest, form
        (rest_h, form_h), (rest_a, form_a) = snap(r.Home), snap(r.Away)
        rows.append((rest_h, rest_a, form_h, form_a))
        ph, pa = (3, 0) if r.Result == 0 else (1, 1) if r.Result == 1 else (0, 3)
        pts.setdefault(r.Home, []).append(ph)
        pts.setdefault(r.Away, []).append(pa)
        last_date[r.Home] = last_date[r.Away] = r.Date
    out = df.copy()
    out[["rest_h", "rest_a", "form_h", "form_a"]] = rows
    return out


def add_context(df: pd.DataFrame, context_csv: str | None = None, seed: int = 0) -> pd.DataFrame:
    """
    Añade bajas y fatiga europea por equipo:
        absences  : puntuación de bajas (0 = plantilla completa; sube con titulares ausentes)
        euro      : 1 si jugó competición europea en los 4 días previos
    - Con `context_csv` (columnas: Date, Team, absences, euro) usa datos REALES.
    - Sin él, los SIMULA aleatoriamente: sirve para probar la estructura, pero
      el modelo no aprenderá nada útil de ellos (son ruido).
    """
    out = df.copy()
    if context_csv and Path(context_csv).exists():
        ctx = pd.read_csv(context_csv, parse_dates=["Date"])
        for side in ("Home", "Away"):
            tag = "h" if side == "Home" else "a"
            m = ctx.rename(columns={"Team": side, "absences": f"absences_{tag}", "euro": f"euro_{tag}"})
            out = out.merge(m[["Date", side, f"absences_{tag}", f"euro_{tag}"]], on=["Date", side], how="left")
        out = out.fillna({"absences_h": 0, "absences_a": 0, "euro_h": 0, "euro_a": 0})
    else:
        warnings.warn("Contexto SIMULADO (bajas/fatiga aleatorias). Sustitúyelo por datos reales.")
        rng = np.random.default_rng(seed)
        n = len(out)
        out["absences_h"], out["absences_a"] = rng.poisson(1.0, n), rng.poisson(1.0, n)
        out["euro_h"], out["euro_a"] = rng.binomial(1, 0.12, n), rng.binomial(1, 0.12, n)
    return out


def build_features(df: pd.DataFrame, dc_probs: np.ndarray) -> pd.DataFrame:
    """Une todo en la matriz de variables. Se usan DIFERENCIAS local-visitante (menos columnas)."""
    f = df.copy()
    f[["dc_p1", "dc_px", "dc_p2"]] = dc_probs
    f["rest_diff"] = f.rest_h - f.rest_a
    f["form_diff"] = f.form_h - f.form_a
    f["absences_diff"] = f.absences_h - f.absences_a
    f["euro_diff"] = f.euro_h - f.euro_a
    return f


class QuinielaModel:
    """XGBoost poco profundo (para no sobreajustar con ~700 partidos) + calibración."""

    def __init__(self, calibration: str = "sigmoid"):   # 'isotonic' necesita más datos
        self.calibration = calibration

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuinielaModel":
        base = XGBClassifier(
            objective="multi:softprob", n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
            eval_metric="mlogloss", random_state=42,
        )
        # TimeSeriesSplit: la calibración también respeta el orden temporal
        self.model_ = CalibratedClassifierCV(base, method=self.calibration, cv=TimeSeriesSplit(n_splits=3))
        self.model_.fit(X[FEATURES], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[FEATURES])


# ======================================================================================
# MÓDULO 4 · EVALUACIÓN Y EJECUCIÓN
# ======================================================================================
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def evaluate(y: np.ndarray, models: dict[str, np.ndarray]) -> pd.DataFrame:
    """Log-loss y Brier (menor = mejor). Si no mejoras al mercado, no hay ventaja."""
    rows = {k: {"log_loss": log_loss(y, p, labels=[0, 1, 2]), "brier": brier(y, p)} for k, p in models.items()}
    return pd.DataFrame(rows).T.round(4)


def fixture_features(hist: pd.DataFrame, fixtures: pd.DataFrame, dc: DixonColes) -> pd.DataFrame:
    """
    Variables para la jornada a predecir. `fixtures` necesita: Home, Away
    y opcionales: Date, absences_h/a, euro_h/a, pct_1/pct_X/pct_2 (escrutinio, para el paso 3).
    """
    fx = fixtures.copy()
    fx["Date"] = pd.to_datetime(fx.get("Date", pd.Timestamp.today().normalize()))
    last_date, forms = {}, {}
    for r in hist.itertuples():                      # fecha del último partido de cada equipo
        last_date[r.Home] = last_date[r.Away] = r.Date
    for team in set(hist.Home) | set(hist.Away):     # forma: puntos medios en los últimos 5
        games = hist[(hist.Home == team) | (hist.Away == team)].tail(5)
        pts = [3 if (g.Result == 0 and g.Home == team) or (g.Result == 2 and g.Away == team)
               else 1 if g.Result == 1 else 0 for g in games.itertuples()]
        forms[team] = np.mean(pts) if pts else 1.3
    for side, tag in (("Home", "h"), ("Away", "a")):
        fx[f"rest_{tag}"] = [min((d - last_date[t]).days, 14) if t in last_date else 14
                             for t, d in zip(fx[side], fx["Date"])]
        fx[f"form_{tag}"] = [forms.get(t, 1.3) for t in fx[side]]
        for col in (f"absences_{tag}", f"euro_{tag}"):
            if col not in fx:
                fx[col] = 0
    probs = np.array([dc.predict_1x2(a, b) for a, b in zip(fx.Home, fx.Away)])
    return build_features(fx, probs)


def train_pipeline(hist: pd.DataFrame, progress=None) -> dict:
    """
    Parte pesada (conviene cachearla): Dixon-Coles fuera de muestra + XGBoost calibrado
    + evaluación temporal (80 % antiguo / 20 % reciente) contra Dixon-Coles y mercado.
    """
    say = progress or (lambda msg: None)
    say("Dixon-Coles fuera de muestra…")
    dc_oos = dc_walk_forward(hist, min_train=150, step_days=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = build_features(add_context(add_rest_and_form(hist)), dc_oos)
    data = data.dropna(subset=["dc_p1"]).reset_index(drop=True)

    say("Validando el modelo…")
    cut = int(len(data) * 0.8)
    train, test = data.iloc[:cut], data.iloc[cut:]
    tmp = QuinielaModel().fit(train, train.Result.to_numpy())
    comparison = {"Dixon-Coles": test[["dc_p1", "dc_px", "dc_p2"]].to_numpy(),
                  "DC + XGBoost calibrado": tmp.predict_proba(test)}
    mk = test[["mkt_p1", "mkt_px", "mkt_p2"]].to_numpy()
    if not np.isnan(mk).any():
        comparison["Mercado (sin margen)"] = mk
    evaluation = evaluate(test.Result.to_numpy(), comparison)

    say("Entrenando el modelo final…")
    final = QuinielaModel().fit(data, data.Result.to_numpy())
    return {"final": final, "dc": DixonColes().fit(hist), "evaluation": evaluation, "n_matches": len(hist)}


def predict_fixtures(hist: pd.DataFrame, fixtures: pd.DataFrame, trained: dict) -> pd.DataFrame:
    """
    Probabilidades 1X2 para la jornada. `fixtures`: Home, Away (nombres football-data)
    y opcionalmente Date, pct_1/pct_X/pct_2. Equipos sin histórico (Primera
    Federación, Copa...): se usan las probabilidades de las casas de apuestas (mkt_1/X/2) si
    existen y, si no, el % del público (sin ventaja).
    """
    dc = trained["dc"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fx = fixture_features(hist, fixtures, dc)
    result = fixtures.copy()
    result[["P1", "PX", "P2"]] = trained["final"].predict_proba(fx).round(3)
    result[["DC_1", "DC_X", "DC_2"]] = fx[["dc_p1", "dc_px", "dc_p2"]].round(3).to_numpy()
    known = (fixtures.Home.isin(dc.attack_) & fixtures.Away.isin(dc.attack_)).to_numpy()
    result["sin_historico"] = ~known
    pub_cols, mkt_cols = ["pct_1", "pct_X", "pct_2"], ["mkt_1", "mkt_X", "mkt_2"]
    if not known.all() and set(pub_cols) <= set(fixtures.columns):
        pub = fixtures[pub_cols].to_numpy(float)
        pub = pub / pub.sum(axis=1, keepdims=True)
        mk = (fixtures[mkt_cols].fillna(0).to_numpy(float) if set(mkt_cols) <= set(fixtures.columns)
              else np.zeros_like(pub))
        use_mk = mk.sum(axis=1) > 0
        fallback = np.where(use_mk[:, None], mk / np.where(use_mk, mk.sum(axis=1), 1)[:, None], pub)
        result.loc[~known, ["P1", "PX", "P2"]] = fallback[~known].round(3)
        result["fuente_sin_hist"] = np.where(~known, np.where(use_mk, "cuotas", "público"), "")
    return result


def pleno_probs(trained: dict, home: str, away: str):
    """
    Pleno al 15: matriz 4x4 con P(local marca 0/1/2/M, visitante marca 0/1/2/M), M = 3 o más.
    Sale de la matriz de marcadores de Dixon-Coles. None si algún equipo no tiene histórico.
    """
    dc = trained["dc"]
    if home not in dc.attack_ or away not in dc.attack_:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = dc.score_matrix(home, away)               # filas = goles local
    rows = np.vstack([mat[0], mat[1], mat[2], mat[3:].sum(axis=0)])
    return np.column_stack([rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3:].sum(axis=1)])


def main(hist: pd.DataFrame | None = None, fixtures_csv: str = "quiniela_jornada.csv"):
    """Ejecución por consola (la web app usa train_pipeline / predict_fixtures directamente)."""
    hist = load_history(n_seasons=2) if hist is None else hist
    print(f"Partidos cargados: {len(hist)}  ({hist.Date.min().date()} → {hist.Date.max().date()})")
    trained = train_pipeline(hist, progress=print)
    print("\nEvaluación en el 20 % más reciente (menor = mejor):")
    print(trained["evaluation"])

    if not Path(fixtures_csv).exists():
        pd.DataFrame({"Home": ["Real Madrid", "Ath Madrid"], "Away": ["Sevilla", "Barcelona"],
                      "pct_1": [70, 28], "pct_X": [20, 30], "pct_2": [10, 42]}).to_csv(fixtures_csv, index=False)
        print(f"\nCreé una plantilla de jornada en '{fixtures_csv}': edítala con tus partidos.")
    fixtures = pd.read_csv(fixtures_csv)
    result = predict_fixtures(hist, fixtures, trained)
    if result.sin_historico.any():
        print("\n⚠ Sin histórico (se usa el % del público, sin ventaja):",
              [f"{h}-{a}" for h, a in zip(result.Home[result.sin_historico], result.Away[result.sin_historico])])
    result.to_csv("predicciones_jornada.csv", index=False)
    print("\nPredicciones (guardadas en predicciones_jornada.csv, entrada del paso 3):")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    main()
