"""
quiniela_optimizer.py  ·  Paso 3: valor esperado, filtros y columnas recomendadas

ENTRADA   predicciones_jornada.csv (lo genera quiniela_model.py). Necesita las columnas:
          Home, Away, P1, PX, P2        -> probabilidades de TU modelo
          pct_1, pct_X, pct_2           -> % de apuestas del público (escrutinio), en % o en 0-1
SALIDA    quiniela_optimizada.csv y quiniela_optimizada.txt

EJECUCIÓN
    python quiniela_optimizer.py --k 10
    python quiniela_optimizer.py --k 10 --bote 300000 --n-columnas 5000000   # con EV en euros
  o desde Python / al final de quiniela_model.main():
    from quiniela_optimizer import run; run(k=10)

LÓGICA (quiniela = apuesta parimutuel: el premio se REPARTE entre los acertantes)
    EV(columna) = P(columna) * Bote / (1 + N * S(columna))
        P(columna) = producto de las probabilidades de tu modelo, signo a signo
        S(columna) = producto de los % del público (cuánta gente juega esa misma columna)
        N          = nº total de columnas jugadas en la jornada
        Bote       = dinero de la categoría que persigues (p. ej. 14 aciertos + bote acumulado)
    Sin Bote ni N se ordena por el índice de valor P/S (equivale a N muy grande).

SUPUESTOS A TENER PRESENTES
    · Independencia entre partidos (para P y para S). El público en realidad correlaciona
      sus signos (todos ponen favoritos juntos), así que S de columnas "de favoritos" está
      algo infravalorado.
    · Solo se modela la categoría alta (todos los aciertos). Las categorías menores no.
    · Se trata la fila 15 (pleno al 15) como un 1X2 más; en la quiniela real es por goles.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SIGNS = np.array(["1", "X", "2"])
PLENO = ["0", "1", "2", "M"]          # goles de cada equipo en el Pleno al 15 (M = 3 o más)


# ======================================================================================
# 1) CARGA Y TABLA DE SESGO POR SIGNO
# ======================================================================================
def prepare_arrays(df: pd.DataFrame, peso_modelo: float = 0.8):
    """
    df con: P1, PX, P2 (modelo) y pct_1, pct_X, pct_2 (público). Devuelve (Pa, S):
        Pa : probabilidades ajustadas = peso_modelo*modelo + (1-peso_modelo)*público.
             Encoger hacia el público protege contra un modelo sobreconfiado
             (peso_modelo=1.0 usa el modelo puro).
        S  : porcentajes del público normalizados a suma 1 por partido.
    """
    need = {"P1", "PX", "P2", "pct_1", "pct_X", "pct_2"}
    if need - set(df.columns):
        raise ValueError(f"Faltan columnas: {sorted(need - set(df.columns))}")
    if not 2 <= len(df) <= 15:
        raise ValueError(f"La jornada debe tener entre 2 y 15 partidos (tiene {len(df)}).")
    P = df[["P1", "PX", "P2"]].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)
    S = np.clip(df[["pct_1", "pct_X", "pct_2"]].to_numpy(float), 1e-4, None)
    S = S / S.sum(axis=1, keepdims=True)
    Pa = peso_modelo * P + (1 - peso_modelo) * S
    return Pa / Pa.sum(axis=1, keepdims=True), S


def load_jornada(path: str = "predicciones_jornada.csv", peso_modelo: float = 0.8):
    """Lee el CSV de predicciones y devuelve (df, Pa, S)."""
    df = pd.read_csv(path)
    if {"Home", "Away"} - set(df.columns):
        raise ValueError(f"Faltan columnas Home/Away en {path}")
    Pa, S = prepare_arrays(df, peso_modelo)
    return df, Pa, S


def sign_table(df: pd.DataFrame, Pa: np.ndarray, S: np.ndarray, min_prob_signo: float = 0.10) -> pd.DataFrame:
    """
    Tabla de 'sesgo del apostante' por signo.
        ventaja = P_modelo - %público        (positivo: el público lo infravalora)
        ratio   = P_modelo / %público        (>1: el signo paga más de lo que debería)
    ★ marca los signos con ratio >= 1.3 y probabilidad real suficiente (no quimeras).
    """
    rows = []
    for i, r in enumerate(df.itertuples()):
        for j, s in enumerate(SIGNS):
            ratio = Pa[i, j] / S[i, j]
            rows.append({
                "partido": f"{i + 1:02d} {r.Home}-{r.Away}", "signo": s,
                "P_modelo": round(Pa[i, j], 3), "%público": round(S[i, j], 3),
                "ventaja": round(Pa[i, j] - S[i, j], 3), "ratio": round(ratio, 2),
                "valor": "★" if ratio >= 1.3 and Pa[i, j] >= min_prob_signo else "",
            })
    return pd.DataFrame(rows)


# ======================================================================================
# 2) GENERADOR DE COLUMNAS (enumeración exacta + filtros)
# ======================================================================================
def _enumerate_half(logP, logS, fav):
    """Todas las combinaciones de un bloque de partidos (3^m) con sus log-probabilidades."""
    m = logP.shape[0]
    codes = np.array(list(itertools.product(range(3), repeat=m)), dtype=np.int8).reshape(-1, m)
    rows = np.arange(m)
    lp = logP[rows, codes].sum(axis=1).astype(np.float32)
    ls = logS[rows, codes].sum(axis=1).astype(np.float32)
    ns = (codes != fav[rows]).sum(axis=1).astype(np.int8)   # nº de "sorpresas" (no favorito)
    nx = (codes == 1).sum(axis=1).astype(np.int8)           # nº de empates (X)
    return codes, lp, ls, ns, nx


# ======================================================================================
# COBERTURA: repartir el riesgo entre columnas (simulación de resultados)
# ======================================================================================
def _simular(Pa: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    """Resultados simulados (n_sims x n partidos) con 0 = '1', 1 = 'X', 2 = '2', según las probabilidades Pa."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(Pa, axis=1)[:, :2]
    u = rng.random((n_sims, Pa.shape[0]))
    return (u[:, :, None] > cdf[None, :, :]).sum(axis=2).astype(np.int8)


def _hits(cand: np.ndarray, sims: np.ndarray) -> np.ndarray:
    """Aciertos de cada columna (filas) en cada resultado simulado (columnas)."""
    hits = np.zeros((cand.shape[0], sims.shape[0]), dtype=np.int8)
    for m in range(cand.shape[1]):
        hits += cand[:, m][:, None] == sims[:, m][None, :]
    return hits


def _greedy_cobertura(cand, p_exact, sims, k, cobertura):
    """
    Elige k columnas de la lista candidata que se COMPLEMENTAN: en cada paso añade la que más
    aumenta el valor esperado de la mejor columna del conjunto, donde
        14 aciertos vale 1 y 13 aciertos vale c = 0,3 * cobertura (12 -> c/4, 11 -> c/16 ...).
    Con cobertura alta se prefieren columnas que cubren los fallos de las demás (p. ej. el otro
    signo en el partido más dudoso); con cobertura 0 sería simplemente "las más probables".
    El término de 14 aciertos usa la probabilidad exacta (en simulación casi nunca aparece).
    """
    M, n = cand.shape
    hits = _hits(cand, sims)
    c = 0.3 * cobertura
    w = np.zeros(n + 1, dtype=np.float32)
    w[n] = 1.0
    for j in range(1, 5):
        w[n - j] = c / (4 ** (j - 1))
    best = np.zeros(sims.shape[0], dtype=np.int8)
    chosen: list[int] = []
    for _ in range(min(k, M)):
        gain = (w[np.maximum(hits, best[None, :])] - w[best][None, :]).mean(axis=1) + p_exact
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        best = np.maximum(best, hits[j])
    return chosen


def optimizar(
    Pa: np.ndarray,
    S: np.ndarray,
    k: int = 10,
    min_prob_signo: float = 0.10,   # descarta signos con menos de 10 % de probabilidad
    min_prob_rel: float = 0.05,     # columna >= 5 % de la prob. de la columna más probable
    min_prob_abs: float = 0.0,      # umbral absoluto opcional (p. ej. 1e-6)
    max_sorpresas: int = 4,         # máx. de signos que NO son el favorito del modelo
    min_empates: int = 0,           # mín. de empates (X) por columna (en una jornada salen ~3-4)
    min_distancia: int = 2,         # las columnas elegidas difieren en >= 2 signos
    bote: float | None = None,      # € de la categoría objetivo (opcional)
    n_columnas: float | None = None,  # columnas totales jugadas en la jornada (opcional)
    coste_columna: float = 0.75,    # € por columna (VERIFICA el precio vigente)
    n_candidatos: int = 20000,
    cobertura: float = 0.0,         # 0 = las más probables; hasta 1 = reparte el riesgo entre columnas
    n_sims: int = 10000,
    seed: int = 7,
    pleno: tuple | None = None,     # (Pp, Sp): matrices 4x4 modelo y público del partido 15
) -> pd.DataFrame:
    n = Pa.shape[0]
    fav = Pa.argmax(axis=1)
    allowed = Pa >= min_prob_signo
    allowed[np.arange(n), fav] = True                       # el favorito siempre es válido
    logP = np.where(allowed, np.log(Pa), -np.inf)
    logS = np.log(S)

    # Enumeración exacta por mitades: 3^15 ≈ 14 M combinaciones en una matriz A x B
    h = n // 2
    cA, lpA, lsA, nsA, nxA = _enumerate_half(logP[:h], logS[:h], fav[:h])
    cB, lpB, lsB, nsB, nxB = _enumerate_half(logP[h:], logS[h:], fav[h:])
    LP = lpA[:, None] + lpB[None, :]
    LS = lsA[:, None] + lsB[None, :]
    NS = nsA[:, None] + nsB[None, :]
    NX = nxA[:, None] + nxB[None, :]

    # FILTRO DE PROBABILIDAD MÍNIMA (anti-quimeras) + límite de sorpresas
    reglas = (NS <= max_sorpresas) & (NX >= min_empates)
    if not reglas.any():
        raise ValueError("Ninguna columna cumple las reglas: sube las sorpresas permitidas o baja el mínimo de empates.")
    mask = (LP >= LP[reglas].max() + np.log(min_prob_rel)) & reglas
    if min_prob_abs > 0:
        mask &= LP >= np.log(min_prob_abs)
    n_validas = int(mask.sum())
    if n_validas == 0:
        raise ValueError("Ninguna columna supera los filtros: relájalos (min_prob_rel, max_sorpresas...).")

    # PUNTUACIÓN: EV en euros si hay datos de bote y N; si no, índice de valor P/S (log)
    ev_mode = bote is not None and n_columnas is not None
    score = (np.exp(LP) * bote / (1.0 + n_columnas * np.exp(LS))) if ev_mode else (LP - LS)
    score = np.where(mask, score, -np.inf).ravel()

    m = min(n_candidatos, n_validas)
    top = np.argpartition(score, -m)[-m:]
    top = top[np.argsort(-score[top])]

    nB = LP.shape[1]
    chosen_idx, chosen_codes = [], []
    if cobertura > 0:
        # SELECCIÓN POR COBERTURA: entre las mejores candidatas (ya ordenadas por valor), las que se complementan
        pool = top[:min(len(top), 1000)]
        ia_p, ib_p = pool // nB, pool % nB
        cand = np.hstack([cA[ia_p], cB[ib_p]])
        p_exact = np.exp(LP.reshape(-1)[pool]).astype(np.float32)
        picks = _greedy_cobertura(cand, p_exact, _simular(Pa, n_sims, seed), k, cobertura)
        chosen_idx = [(int(ia_p[i]), int(ib_p[i])) for i in picks]
        chosen_codes = [cand[i] for i in picks]
    else:
        # SELECCIÓN CON DIVERSIDAD: evita columnas casi idénticas
        for t in top:
            ia, ib = divmod(int(t), nB)
            code = np.concatenate([cA[ia], cB[ib]])
            if all((code != c).sum() >= min_distancia for c in chosen_codes):
                chosen_idx.append((ia, ib))
                chosen_codes.append(code)
                if len(chosen_codes) == k:
                    break

    out = []
    for (ia, ib), code in zip(chosen_idx, chosen_codes):
        p, s = float(np.exp(LP[ia, ib])), float(np.exp(LS[ia, ib]))
        row = {
            "columna": "".join(SIGNS[code]),
            "P_conjunta": p,
            "1_de_cada": round(1 / p),
            "share_publico": s,
            "indice_valor": round(p / s, 2),   # >1 = juegas lo que el público infravalora
            "sorpresas": int((code != fav).sum()),
            "empates": int((code == 1).sum()),
        }
        p15 = s15 = 1.0
        if pleno is not None:        # mejor marcador del partido 15 para esta columna
            Pp, Sp = pleno
            crit = Pp / (1 + n_columnas * s * Sp) if ev_mode else Pp
            i, j = np.unravel_index(np.argmax(crit), crit.shape)
            p15, s15 = float(Pp[i, j]), float(Sp[i, j])
            row["pleno"] = f"{PLENO[i]}-{PLENO[j]}"
            row["P_pleno"] = round(p15, 3)
        if ev_mode:
            row["EV_euros"] = round(p * p15 * bote / (1 + n_columnas * s * s15), 4)
            row["EV_sobre_coste"] = round(row["EV_euros"] / coste_columna, 3)
        out.append(row)
    res = pd.DataFrame(out)
    codes = np.array(chosen_codes)
    ev_hits = _hits(codes, _simular(Pa, 40000, seed + 1))          # aciertos de tus K columnas en 40.000 jornadas simuladas
    res.attrs["p_14"] = float(res.P_conjunta.sum())                # probabilidad exacta de 14 con alguna de las columnas
    res.attrs["p_13mas"] = float((ev_hits.max(axis=0) >= n - 1).mean())
    res.attrs["reparto"] = np.array([[int((codes[:, m] == s).sum()) for s in range(3)] for m in range(n)])
    res.attrs["n_validas"] = n_validas
    res.attrs["ev_mode"] = ev_mode
    return res


# ======================================================================================
# 3) EXPORTACIÓN
# ======================================================================================
def to_text(cols: pd.DataFrame, df: pd.DataFrame) -> str:
    """Texto listo para rellenar el boleto."""
    lines = ["QUINIELA OPTIMIZADA", "=" * 60, "", "PARTIDOS"]
    lines += [f"  {i + 1:02d}. {r.Home} - {r.Away}" for i, r in enumerate(df.itertuples())]
    lines += ["", "COLUMNAS RECOMENDADAS (rellena una por columna del boleto)",
              "  Nº  " + " ".join(f"{i + 1:>2}" for i in range(len(df))) + "   P conjunta   índice valor"]
    for i, r in enumerate(cols.itertuples(), 1):
        signs = " ".join(f"{s:>2}" for s in r.columna)
        extra = f"   Pleno15: {r.pleno}" if "pleno" in cols.columns else ""
        lines.append(f"  {i:02d}  {signs}   {r.P_conjunta:.2e}   {r.indice_valor:>6}{extra}")
    if "pleno" in cols.columns:
        lines += ["", "PLENO AL 15: goles del local - goles del visitante (0, 1, 2 o M = 3 o más)"]
    lines += ["", "AVISO: la quiniela retiene una parte de la recaudación; incluso con ventaja sobre el",
              "público, el retorno esperado puede seguir por debajo del coste. Juega con cabeza."]
    return "\n".join(lines)


def exportar(cols: pd.DataFrame, df: pd.DataFrame, csv_path="quiniela_optimizada.csv",
             txt_path="quiniela_optimizada.txt") -> None:
    cols.to_csv(csv_path, index=False)
    Path(txt_path).write_text(to_text(cols, df), encoding="utf-8")


# ======================================================================================
# 4) EJECUCIÓN
# ======================================================================================
def run(pred_csv="predicciones_jornada.csv", k=10, peso_modelo=0.8, **kwargs) -> pd.DataFrame:
    df, Pa, S = load_jornada(pred_csv, peso_modelo)
    print("\nSESGO DEL APOSTANTE POR SIGNO (★ = el público lo infravalora y tiene opciones reales)")
    print(sign_table(df, Pa, S, kwargs.get("min_prob_signo", 0.10)).to_string(index=False))
    cols = optimizar(Pa, S, k=k, **kwargs)
    exportar(cols, df)
    print(f"\nColumnas que superan los filtros: {cols.attrs['n_validas']:,}")
    print("TOP columnas (guardadas en quiniela_optimizada.csv / .txt):")
    print(cols.to_string(index=False))
    if not cols.attrs["ev_mode"]:
        print("\n(Sin --bote y --n-columnas se ordena por índice de valor, no por euros.)")
    return cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optimizador de columnas de La Quiniela")
    ap.add_argument("--pred", default="predicciones_jornada.csv")
    ap.add_argument("--k", type=int, default=10, help="nº de columnas a recomendar")
    ap.add_argument("--peso-modelo", type=float, default=0.8)
    ap.add_argument("--min-prob-signo", type=float, default=0.10)
    ap.add_argument("--min-prob-rel", type=float, default=0.05)
    ap.add_argument("--min-prob-abs", type=float, default=0.0)
    ap.add_argument("--max-sorpresas", type=int, default=4)
    ap.add_argument("--min-empates", type=int, default=0)
    ap.add_argument("--min-distancia", type=int, default=2)
    ap.add_argument("--bote", type=float, default=None)
    ap.add_argument("--n-columnas", type=float, default=None)
    ap.add_argument("--coste-columna", type=float, default=0.75)
    a = ap.parse_args()
    run(a.pred, a.k, a.peso_modelo, min_prob_signo=a.min_prob_signo, min_prob_rel=a.min_prob_rel,
        min_prob_abs=a.min_prob_abs, max_sorpresas=a.max_sorpresas, min_empates=a.min_empates, min_distancia=a.min_distancia,
        bote=a.bote, n_columnas=a.n_columnas, coste_columna=a.coste_columna)
