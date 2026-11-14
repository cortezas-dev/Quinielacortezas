"""
app.py  ·  Quiniela 1X2 — web app (Streamlit)

    streamlit run app.py

Flujo: descarga histórico (football-data) -> entrena Dixon-Coles + XGBoost (cacheado)
       -> lee la jornada y los % de LAE de la web -> optimiza columnas -> las muestra para copiar.
Archivos necesarios en la misma carpeta: quiniela_model.py, quiniela_optimizer.py,
quiniela_scraper.py, requirements.txt
"""
import hmac

import numpy as np
import pandas as pd
import streamlit as st

import quiniela_model as qm
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
try:
    with st.spinner("Descargando datos y entrenando el modelo (solo la primera vez, puede tardar ~1 min)…"):
        hist = cargar_historico()
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
    })
else:
    st.error(f"No pude leer la jornada automáticamente: {error}")
    st.info("Puedes rellenar la tabla a mano: elige los equipos (nombres de football-data) y los % del público.")
    base = pd.DataFrame({"Partido": [f"Partido {i}" for i in range(1, 15)], "Local": SIN_HIST, "Visitante": SIN_HIST,
                         "pct_1": 40, "pct_X": 30, "pct_2": 30})

sin = int(((base.Local == SIN_HIST) | (base.Visitante == SIN_HIST)).sum())
with st.expander(f"📋 Revisar partidos y porcentajes ({len(base)} partidos · {sin} sin histórico)"):
    st.caption("Los partidos «sin histórico» (femenino, otras categorías…) usan el % del público: no aportan ventaja. "
               "Puedes corregir equipos y porcentajes.")
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
        })

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
    min_dist = st.slider("Distancia mínima entre columnas (signos distintos)", 1, 4, 2)
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
})
if (fixtures[["pct_1", "pct_X", "pct_2"]].sum(axis=1) <= 0).any():
    st.warning("Hay partidos con los tres porcentajes a 0. Revisa la tabla.")
    st.stop()

try:
    result = qm.predict_fixtures(hist, fixtures, trained)
    Pa, S = qo.prepare_arrays(result, peso)
    cols = qo.optimizar(Pa, S, k=k, min_prob_signo=min_signo, min_prob_rel=min_rel, max_sorpresas=max_sorp,
                        min_distancia=min_dist, bote=float(bote), n_columnas=recaud / precio, coste_columna=precio)
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
filas = [f"{i:02d}  " + " ".join(c) for i, c in enumerate(cols.columna, 1)]
st.code("\n".join(filas), language="text")

tabla = pd.DataFrame([list(c) for c in cols.columna], columns=[str(j) for j in range(1, n + 1)])
tabla.insert(0, "#", range(1, len(tabla) + 1))
tabla["1 de cada"] = cols["1_de_cada"].map(lambda v: f"{v:,}".replace(",", "."))
tabla["Valor"] = cols["indice_valor"]
tabla["EV/coste"] = cols["EV_sobre_coste"]
st.dataframe(tabla, hide_index=True, width="stretch")
st.caption("Valor = probabilidad de tu modelo ÷ cuota de público de esa columna (>1: juegas lo que el público infravalora). "
           "El Pleno al 15 (goles) no está incluido: rellénalo tú.")

d1, d2 = st.columns(2)
d1.download_button("⬇️ TXT para el boleto", qo.to_text(cols, result), "quiniela_optimizada.txt", width="stretch")
d2.download_button("⬇️ CSV con métricas", cols.to_csv(index=False), "quiniela_optimizada.csv", width="stretch")

with st.expander("🔎 Análisis por signo (sesgo del público)"):
    st.dataframe(qo.sign_table(result, Pa, S, min_signo), hide_index=True, width="stretch")
    st.caption("★ = el público infravalora el signo (ratio ≥ 1,3) y tiene probabilidad real suficiente.")
if result.sin_historico.any():
    st.caption("Sin histórico (usan el % del público): " + ", ".join(
        f"{h.split('|')[0]}" for h in result.Home[result.sin_historico]))
with st.expander("📈 Calidad del modelo"):
    st.dataframe(trained["evaluation"], width="stretch")
    st.caption(f"Evaluado en el 20 % más reciente de {trained['n_matches']} partidos (menor = mejor). "
               "Las bajas y la fatiga europea están SIMULADAS: el XGBoost no puede aprender de ellas todavía.")
