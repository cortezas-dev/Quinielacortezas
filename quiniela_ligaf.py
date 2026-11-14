"""
quiniela_ligaf.py  ·  Histórico GRATUITO de la Liga F (Primera División Femenina) desde Wikipedia

Lee la página de cada temporada (p. ej. https://en.wikipedia.org/wiki/2025–26_Liga_F) y saca los
resultados de dos formas posibles:
  A) Tablas de partidos (Home team | Results | Away team | ... | Date): traen fecha real.
  B) Matriz de resultados "Home \\ Away" (equipos en filas y columnas): NO trae fechas.
     Se reparten fechas aproximadas siguiendo un calendario de liguilla (todos contra todos)
     entre el inicio de la temporada y su final (o ayer, si la temporada sigue en marcha).

LIMITACIONES (importante)
  · Con la matriz las fechas son aproximadas: sirven para el peso de los partidos recientes, pero
    no son las reales (el descanso entre partidos de esos equipos será ruido).
  · Depende del formato de Wikipedia. Está probado con tablas de ejemplo, no con una descarga real.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/{y}%E2%80%93{yy}_Liga_F"
HEADERS = {"User-Agent": "QuinielaPersonalApp/1.0 (uso personal; analisis de resultados)"}
_SCORE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)")


def _clean(name) -> str:
    return re.sub(r"\[.*?\]|\(.*?\)", "", str(name)).replace("\xa0", " ").strip()


def _score(cell):
    m = _SCORE.search(str(cell))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[-1]) for c in df.columns]
    return df


# --------------------------------------------------------------------------------------
# A) tablas de partidos con fecha
# --------------------------------------------------------------------------------------
def _from_match_tables(tables: list[pd.DataFrame]) -> list[tuple]:
    out = []
    for df in tables:
        df = _flat(df)
        cols = {str(c).strip().lower(): c for c in df.columns}
        if not {"home team", "away team", "results"} <= set(cols):
            continue
        dcol = next((cols[c] for c in cols if c.startswith("date")), None)
        if dcol is None:
            continue
        for _, r in df.iterrows():
            sc = _score(r[cols["results"]])
            d = re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", str(r[dcol]))
            if sc is None or d is None:
                continue
            date = pd.to_datetime(d.group(0), dayfirst=True, errors="coerce")
            if pd.notna(date):
                out.append((date, _clean(r[cols["home team"]]), _clean(r[cols["away team"]]), sc[0], sc[1]))
    return out


# --------------------------------------------------------------------------------------
# B) matriz "Home \ Away" sin fechas
# --------------------------------------------------------------------------------------
def _round_robin(n: int) -> dict[tuple[int, int], int]:
    """Calendario de liguilla (método del círculo): ronda de cada partido (local i, visitante j)."""
    m = n + (n % 2)                                  # equipo ficticio si n es impar
    arr, rounds = list(range(m)), {}
    for r in range(m - 1):
        for k in range(m // 2):
            a, b = arr[k], arr[m - 1 - k]
            home, away = (a, b) if (r + k) % 2 == 0 else (b, a)
            rounds[(home, away)] = r + 1
            rounds[(away, home)] = r + 1 + (m - 1)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    return rounds


def _from_matrix(tables: list[pd.DataFrame], start_year: int) -> list[tuple]:
    for df in tables:
        df = _flat(df)
        cols = [str(c).strip() for c in df.columns]
        codes = [c for c in cols[1:] if re.fullmatch(r"[A-Z]{3}", c)]
        if len(codes) < 8 or len(codes) != len(cols) - 1:
            continue
        n = len(codes)
        teams = [_clean(x) for x in df.iloc[:n, 0]]
        if len(teams) < n:
            continue
        found = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    sc = _score(df.iloc[i, j + 1])
                    if sc is not None:
                        found.append((i, j, sc))
        if not found:
            continue
        rounds = _round_robin(n)
        start = pd.Timestamp(year=start_year, month=8, day=30)
        end = pd.Timestamp(year=start_year + 1, month=5, day=31)
        if len(found) < 0.95 * n * (n - 1):            # temporada en curso: hasta ayer
            end = min(end, pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
        r_max = max(rounds[(i, j)] for i, j, _ in found)
        span = max((end - start).days, 1)
        out = []
        for i, j, sc in found:
            frac = (rounds[(i, j)] - 1) / max(r_max - 1, 1)
            out.append((start + pd.Timedelta(days=int(round(frac * span))), teams[i], teams[j], sc[0], sc[1]))
        return out
    return []


def parse_season_html(html: str, start_year: int) -> pd.DataFrame:
    """Devuelve Date, Home, Away, HG, AG de una temporada (nombres tal como salen en Wikipedia)."""
    tables = pd.read_html(io.StringIO(html))
    boxes = _from_match_tables(tables)
    matrix = _from_matrix(tables, start_year)
    rows = boxes if len(boxes) >= len(matrix) else matrix
    if not rows:
        raise RuntimeError(f"No encontré resultados en la página de la temporada {start_year}/{start_year + 1}.")
    df = pd.DataFrame(rows, columns=["Date", "Home", "Away", "HG", "AG"])
    return df.drop_duplicates(["Date", "Home", "Away"]).sort_values("Date").reset_index(drop=True)


def fetch_season(start_year: int, timeout: int = 25) -> pd.DataFrame:
    url = URL.format(y=start_year, yy=f"{(start_year + 1) % 100:02d}")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_season_html(resp.text, start_year)


def load_liga_f(n_seasons: int = 2) -> pd.DataFrame:
    """Temporada actual y anteriores, en el formato interno del modelo (equipos como 'F:Nombre')."""
    today = pd.Timestamp.today()
    start = today.year if today.month >= 7 else today.year - 1
    frames, errors = [], []
    for k in range(n_seasons):
        try:
            frames.append(fetch_season(start - k))
        except Exception as exc:                       # temporada sin página, sin red, formato distinto…
            errors.append(f"{start - k}/{start - k + 1}: {exc}")
    if not frames:
        raise RuntimeError("No pude leer la Liga F de Wikipedia. " + " | ".join(errors))
    out = pd.concat(frames).sort_values("Date").reset_index(drop=True)
    out["Result"] = np.select([out.HG > out.AG, out.HG == out.AG], [0, 1], default=2)
    out[["mkt_p1", "mkt_px", "mkt_p2"]] = np.nan
    out["Home"], out["Away"] = "F:" + out["Home"], "F:" + out["Away"]
    return out
