"""
app.py  ·  Quiniela 1X2 — web app (Streamlit)

    streamlit run app.py

Flujo: descarga histórico (football-data) -> entrena Dixon-Coles + XGBoost (cacheado)
       -> lee la jornada y los % de LAE de la web -> optimiza columnas -> las muestra para copiar.
Archivos necesarios en la misma carpeta: quiniela_model.py, quiniela_optimizer.py,
quiniela_scraper.py, requirements.txt
"""
import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import quiniela_model as qm
import quiniela_ligaf as ql
import quiniela_optimizer as qo
import quiniela_scraper as qs

st.set_page_config(page_title="Quiniela 1X2", page_icon="⚽", layout="centered")
SIN_HIST = "(sin histórico)"


# --------------------------------------------------------------------------------------
# Acceso opcional con contraseña (Settings > Secrets:  APP_PASSWORD = "tu_clave")
# --------------------------------------------------------------------------------------
def pedir_clave():
    try:
        clave = st.secrets["APP_PASSWORD"]
    except Exception:
        return                                   # sin secreto configurado: acceso libre
    if st.session_state.get("autorizado"):
        return
    txt = st.text_input("Contraseña", type="password")
    if txt and hmac.compare_digest(txt, str(clave)):
        st.session_state["autorizado"] = True
        st.rerun()
    elif txt:
        st.error("Contraseña incorrecta")
    st.stop()


# --------------------------------------------------------------------------------------
# Carga cacheada (lo pesado solo se hace la primera vez y se refresca cada 6 h)
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cargar_historico() -> pd.DataFrame:
    return qm.load_history(n_seasons=2)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def entrenar(n_partidos: int, ultima_fecha: str, _hist: pd.DataFrame) -> dict:
    return qm.train_pipeline(_hist)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cargar_liga_f() -> pd.DataFrame:
    return ql.load_liga_f(n_seasons=2)


@st.cache_data(ttl=600, show_spinner=False)
def cargar_jornada() -> dict:
    return qs.fetch_jornada()


def euros(x) -> str:
    return f"{x:,.0f} €".replace(",", ".")


pedir_clave()
st.title("⚽ Quiniela 1X2")
st.caption("Dixon-Coles + XGBoost frente al reparto de apuestas del público. Herramienta de análisis, no una garantía.")

# --------------------------------------------------------------------------------------
# 1) Modelo
# --------------------------------------------------------------------------------------
with st.expander("➕ Histórico extra: Liga F u otras competiciones"):
    auto_f = st.checkbox("Descargar la Liga F automáticamente (Wikipedia, gratis)", value=True,
                         help="Resultados de las dos últimas temporadas. Si Wikipedia no da las fechas, se reparten de "
                              "forma aproximada a lo largo de la temporada.")
    st.caption("Opcional: sube tu propio CSV de partidos (Date, Home, Away, HG, AG; también vale el formato de "
               "football-data). Para que quede fijo, súbelo a GitHub como `liga_f.csv` junto a app.py.")
    subido = st.file_uploader("CSV de partidos", type="csv")
    es_fem = st.checkbox("Son partidos de fútbol femenino", value=True)

try:
    with st.spinner("Descargando datos y entrenando el modelo (solo la primera vez, puede tardar ~1 min)…"):
        hist = cargar_historico()
        extras = []
        if auto_f:
            try:
                extras.append(cargar_liga_f())
            except Exception as exc:
                st.warning(f"No pude descargar la Liga F de Wikipedia ({exc}). Los partidos femeninos usarán las "
                           "cuotas de las casas; también puedes subir un CSV.")
        fijo = Path(__file__).parent / "liga_f.csv"
        if fijo.exists():
            extras.append(qm.parse_extra_csv(fijo, femenino=True))
        if subido is not None:
            extras.append(qm.parse_extra_csv(subido, femenino=es_fem))
        if extras:
            hist = pd.concat([hist] + extras).sort_values("Date").reset_index(drop=True)
            st.caption(f"Histórico extra cargado: {sum(len(e) for e in extras)} partidos.")
        trained = entrenar(len(hist), str(hist.Date.max().date()), hist)
except Exception as exc:
    st.error(f"No pude preparar el modelo: {exc}")
    st.stop()
known = sorted(set(hist.Home) | set(hist.Away))

# --------------------------------------------------------------------------------------
# 2) Jornada (automática, con respaldo manual)
# --------------------------------------------------------------------------------------
st.subheader("1 · Jornada")
if st.button("🔄 Recargar jornada y porcentajes", width="stretch"):
    cargar_jornada.clear()
    st.rerun()

jornada, error = None, None
try:
    jornada = cargar_jornada()
except qs.ScrapeError as exc:
    error = str(exc)
except Exception as exc:  # cualquier imprevisto de red
    error = f"Error inesperado: {exc}"

fuente = st.radio("Porcentajes del público", ["% LAE (Loterías y Apuestas del Estado)", "% Jugados (web de Losilla)"],
                  horizontal=True)
pref = "lae" if fuente.startswith("% LAE") else "jug"

if jornada:
    p = jornada["partidos"]
    resumen = f"Jornada {jornada['jornada']}" if jornada["jornada"] else "Jornada actual"
    if jornada["bote"]:
        resumen += f" · Bote {euros(jornada['bote'])}"
    st.success(resumen + f" · {len(p)} partidos leídos")
    if jornada["cerrada"]:
        st.warning("Esta jornada figura como **cerrada** (ya no se puede jugar). Para la siguiente, "
                   "pulsa «Recargar» cuando se publique.")
    base = pd.DataFrame({
        "Partido": p.Local_quiniela + " - " + p.Visitante_quiniela,
        "Local": [qs.map_team(n, known) or SIN_HIST for n in p.Local_quiniela],
        "Visitante": [qs.map_team(n, known) or SIN_HIST for n in p.Visitante_quiniela],
        "pct_1": p[f"{pref}_1"], "pct_X": p[f"{pref}_X"], "pct_2": p[f"{pref}_2"],
        "mkt_1": p.prob_1, "mkt_X": p.prob_X, "mkt_2": p.prob_2,
    })
else:
    st.error(f"No pude leer la jornada automáticamente: {error}")
    st.info("Puedes rellenar la tabla a mano: elige los equipos (nombres de football-data) y los % del público.")
    base = pd.DataFrame({"Partido": [f"Partido {i}" for i in range(1, 15)], "Local": SIN_HIST, "Visitante": SIN_HIST,
                         "pct_1": 40, "pct_X": 30, "pct_2": 30, "mkt_1": 0, "mkt_X": 0, "mkt_2": 0})

sin = int(((base.Local == SIN_HIST) | (base.Visitante == SIN_HIST)).sum())
with st.expander(f"📋 Revisar partidos y porcentajes ({len(base)} partidos · {sin} sin histórico)"):
    st.caption("Los partidos «sin histórico» usan las probabilidades de las casas de apuestas («Cuotas %») frente al "
               "% del público; no se apoyan en tu modelo. Puedes corregir equipos y porcentajes.")
    opciones = [SIN_HIST] + known
    editada = st.data_editor(
        base, hide_index=True, width="stretch", num_rows="fixed", key="tabla_jornada",
        column_config={
            "Partido": st.column_config.TextColumn(disabled=True),
            "Local": st.column_config.SelectboxColumn(options=opciones, required=True),
            "Visitante": st.column_config.SelectboxColumn(options=opciones, required=True),
            "pct_1": st.column_config.NumberColumn("1 %", min_value=0, max_value=100),
            "pct_X": st.column_config.NumberColumn("X %", min_value=0, max_value=100),
            "pct_2": st.column_config.NumberColumn("2 %", min_value=0, max_value=100),
            "mkt_1": st.column_config.NumberColumn("Cuotas 1 %", min_value=0, max_value=100),
            "mkt_X": st.column_config.NumberColumn("Cuotas X %", min_value=0, max_value=100),
            "mkt_2": st.column_config.NumberColumn("Cuotas 2 %", min_value=0, max_value=100),
        })

pl = (jornada or {}).get("pleno")
with st.expander("⚽ Partido 15 · Pleno al 15 (goles de cada equipo)"):
    st.caption("Cada equipo marca 0, 1, 2 o M (3 o más). Revisa los equipos y el % del público de cada opción.")
    if pl:
        st.write(f"**{pl['Local_quiniela']} - {pl['Visitante_quiniela']}**")
    loc0 = (qs.map_team(pl["Local_quiniela"], known) if pl and pl["Local_quiniela"] else None) or SIN_HIST
    vis0 = (qs.map_team(pl["Visitante_quiniela"], known) if pl and pl["Visitante_quiniela"] else None) or SIN_HIST
    local15 = st.selectbox("Local", opciones, index=opciones.index(loc0), key="l15")
    visit15 = st.selectbox("Visitante", opciones, index=opciones.index(vis0), key="v15")
    hv = pl[f"{pref}_H"] if pl else [25, 25, 25, 25]
    av = pl[f"{pref}_A"] if pl else [25, 25, 25, 25]
    pl_df = st.data_editor(
        pd.DataFrame({"0": [hv[0], av[0]], "1": [hv[1], av[1]], "2": [hv[2], av[2]], "M": [hv[3], av[3]]},
                     index=["Local", "Visitante"]),
        key="tabla_pleno", width="stretch", num_rows="fixed",
        column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=100) for c in ["0", "1", "2", "M"]})

# --------------------------------------------------------------------------------------
# 3) Ajustes visuales
# --------------------------------------------------------------------------------------
st.subheader("2 · Ajustes")
k = st.slider("Columnas a generar (K)", 1, 30, 10)
with st.expander("💶 Bote y reparto (para estimar el valor en euros)"):
    bote = st.number_input("Bote / premio máximo en juego (€)", min_value=0, step=100_000,
                           value=int((jornada or {}).get("premio_max") or 1_000_000))
    recaud = st.number_input("Recaudación estimada de la jornada (€)", min_value=100_000, step=100_000, value=3_500_000,
                             help="Orden de magnitud real: ~3,5 M€ (jornada del 11-oct-2026). Mira la última nota de prensa de LAE.")
    precio = st.number_input("Precio por apuesta (€)", min_value=0.10, step=0.05, value=0.75)
    st.caption("El bote solo escala el EV en euros; el orden de las columnas depende sobre todo de la recaudación "
               "(cuánta gente compite por el premio). Se asume que el premio entero va a quien acierte los 14 signos.")
with st.expander("🛡️ Filtros anti-quimeras"):
    min_signo = st.slider("Probabilidad mínima por signo (%)", 0, 30, 10) / 100
    min_rel = st.slider("Probabilidad mínima de la columna vs. la más probable (%)", 1, 30, 5) / 100
    max_sorp = st.slider("Máx. de sorpresas por columna (signos que no son el favorito)", 0, 8, 4)
    min_emp = st.slider("Mínimo de empates (X) por columna", 0, 6, 2,
                        help="En una jornada de 14 partidos suelen salir 3 o 4 empates. Con 0, las columnas "
                             "casi nunca llevan X porque el empate rara vez es el signo más probable de un partido.")
    min_dist = st.slider("Distancia mínima entre columnas (signos distintos)", 1, 8, 3,
                         help="Más distancia = columnas más distintas entre sí (no fallan todas a la vez).")
    peso = st.slider("Peso del modelo frente al público", 50, 100, 80, help="100 = solo modelo; menos = más cerca del público "
                     "(protege si el modelo está sobreconfiado)") / 100

# --------------------------------------------------------------------------------------
# 4) Cálculo y resultados
# --------------------------------------------------------------------------------------
st.subheader("3 · Tus columnas")
if len(editada) < 2:
    st.stop()
fixtures = pd.DataFrame({
    "Home": [h if h != SIN_HIST else f"{r.Partido}|L" for h, r in zip(editada.Local, editada.itertuples())],
    "Away": [a if a != SIN_HIST else f"{r.Partido}|V" for a, r in zip(editada.Visitante, editada.itertuples())],
    "Date": pd.Timestamp.today().normalize(),
    "pct_1": editada.pct_1.astype(float), "pct_X": editada.pct_X.astype(float), "pct_2": editada.pct_2.astype(float),
    "mkt_1": editada.mkt_1.astype(float), "mkt_X": editada.mkt_X.astype(float), "mkt_2": editada.mkt_2.astype(float),
})
if (fixtures[["pct_1", "pct_X", "pct_2"]].sum(axis=1) <= 0).any():
    st.warning("Hay partidos con los tres porcentajes a 0. Revisa la tabla.")
    st.stop()

pleno_arg = None
pl_vals = pl_df.fillna(0).to_numpy(float)
if pl_vals[0].sum() > 0 and pl_vals[1].sum() > 0:
    sh, sa = np.clip(pl_vals[0], 1e-4, None), np.clip(pl_vals[1], 1e-4, None)
    Sp = np.outer(sh / sh.sum(), sa / sa.sum())
    pp = None if SIN_HIST in (local15, visit15) else qm.pleno_probs(trained, local15, visit15)
    Pp = Sp if pp is None else peso * pp + (1 - peso) * Sp
    pleno_arg = (Pp / Pp.sum(), Sp)

try:
    result = qm.predict_fixtures(hist, fixtures, trained)
    Pa, S = qo.prepare_arrays(result, peso)
    cols = qo.optimizar(Pa, S, k=k, min_prob_signo=min_signo, min_prob_rel=min_rel, max_sorpresas=max_sorp, min_empates=min_emp,
                        min_distancia=min_dist, bote=float(bote), n_columnas=recaud / precio, coste_columna=precio,
                        pleno=pleno_arg)
except ValueError as exc:
    st.warning(f"{exc}")
    st.stop()

n = len(fixtures)
c1, c2, c3 = st.columns(3)
c1.metric("Columnas válidas", f"{cols.attrs['n_validas']:,}".replace(",", "."))
c2.metric("Mejor columna: 1 de cada", f"{int(cols.P_conjunta.iloc[0] ** -1):,}".replace(",", "."))
ev_total = cols.EV_sobre_coste.mean()
c3.metric("Retorno esperado / coste", f"{ev_total:.2f}")
if ev_total > 1.5:
    st.warning("Un retorno esperado muy por encima de 1 casi siempre significa que el modelo discrepa mucho del "
               "público (posible sobreconfianza o datos mal casados), no que haya una ganga. Revisa la tabla de "
               "partidos y baja el «peso del modelo».")
elif ev_total < 1:
    st.info(f"Con estos supuestos recuperarías de media ~{ev_total * 100:.0f} % de lo que pagas por columna. "
            "Un valor claramente por debajo de 1 significa que no hay ventaja esperada, aunque las columnas "
            "sean las mejores posibles.")

st.markdown("**Copia los signos al boleto** (icono de copiar arriba a la derecha del recuadro):")
filas = [f"{i:02d}  " + " ".join(c) + (f"   | 15: {pl_}" if "pleno" in cols else "")
         for i, (c, pl_) in enumerate(zip(cols.columna, cols.get("pleno", [""] * len(cols))), 1)]
st.code("\n".join(filas), language="text")

tabla = pd.DataFrame([list(c) for c in cols.columna], columns=[str(j) for j in range(1, n + 1)])
tabla.insert(0, "#", range(1, len(tabla) + 1))
if "pleno" in cols:
    tabla["Pleno 15"] = cols["pleno"]
tabla["1 de cada"] = cols["1_de_cada"].map(lambda v: f"{v:,}".replace(",", "."))
tabla["Valor"] = cols["indice_valor"]
tabla["EV/coste"] = cols["EV_sobre_coste"]
st.dataframe(tabla, hide_index=True, width="stretch")
st.caption("Valor = probabilidad de tu modelo ÷ cuota de público de esa columna (>1: juegas lo que el público infravalora). "
           "Pleno 15 = goles del local - goles del visitante (M = 3 o más); se elige el de mejor valor para cada columna.")

d1, d2 = st.columns(2)
d1.download_button("⬇️ TXT para el boleto", qo.to_text(cols, result), "quiniela_optimizada.txt", width="stretch")
d2.download_button("⬇️ CSV con métricas", cols.to_csv(index=False), "quiniela_optimizada.csv", width="stretch")

with st.expander("🔎 Análisis por signo (sesgo del público)"):
    st.dataframe(qo.sign_table(result, Pa, S, min_signo), hide_index=True, width="stretch")
    st.caption("★ = el público infravalora el signo (ratio ≥ 1,3) y tiene probabilidad real suficiente.")
if result.sin_historico.any():
    st.caption("Sin histórico (usan cuotas de las casas o, si no hay, el % del público): " + ", ".join(
        f"{h.split('|')[0]}" for h in result.Home[result.sin_historico]))
with st.expander("📈 Calidad del modelo"):
    st.dataframe(trained["evaluation"], width="stretch")
    st.caption(f"Evaluado en el 20 % más reciente de {trained['n_matches']} partidos (menor = mejor). "
               "Las bajas y la fatiga europea están SIMULADAS: el XGBoost no puede aprender de ellas todavía.")
