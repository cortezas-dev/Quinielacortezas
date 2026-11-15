"""
app.py  ·  Quiniela inteligente (Streamlit)

    streamlit run app.py

Qué hace, de principio a fin:
  1. Descarga históricos gratuitos (Primera/Segunda, Liga F, selecciones) y entrena el modelo (cacheado).
  2. Lee la jornada de La Quiniela y los % del público (LAE) y de las casas de apuestas.
  3. Identifica la competición de cada partido y combina modelo + casas de apuestas.
  4. Elige las mejores columnas, repartiendo el riesgo entre ellas, y las muestra listas para copiar.

Archivos necesarios en la misma carpeta: quiniela_model.py, quiniela_optimizer.py, quiniela_scraper.py,
quiniela_ligaf.py, quiniela_selecciones.py, requirements.txt
"""
import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import quiniela_ligaf as ql
import quiniela_model as qm
import quiniela_optimizer as qo
import quiniela_scraper as qs
import quiniela_selecciones as qsel

st.set_page_config(page_title="Quiniela inteligente", page_icon="⚽", layout="centered",
                   initial_sidebar_state="collapsed")
SIN_HIST = "(sin histórico)"
SIGNOS = ["1", "X", "2"]

st.markdown("""
<style>
.chips{display:flex;flex-wrap:wrap;gap:4px}
.qchip{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:34px;height:44px;
  border-radius:9px;color:#fff;font-weight:700;line-height:1}
.qchip small{font-size:9px;font-weight:500;opacity:.85;margin-bottom:3px}
.qchip span{font-size:19px}
.c1{background:#2E7DF6}.cX{background:#7B8794}.c2{background:#F2994A}
.qchip.alt{outline:3px solid #FFD43B;outline-offset:-1px}
.qchip.pleno{background:#2F9E44;width:auto;min-width:48px;padding:0 10px}
.qcol{border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:10px 12px;margin-bottom:10px}
.qcol h4{margin:0 0 8px 0;font-size:15px}
.muted{opacity:.65;font-weight:400;font-size:12px}
.mcard{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:9px 12px;margin-bottom:8px}
.mtop{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.mnum{background:rgba(128,128,128,.28);border-radius:50%;width:24px;height:24px;display:inline-flex;
  align-items:center;justify-content:center;font-size:12px;font-weight:700;flex:none}
.mname{font-weight:700;font-size:14px}
.tag{font-size:10px;padding:2px 8px;border-radius:10px;color:#fff;font-weight:600}
.t-Primera{background:#1C7ED6}.t-Segunda{background:#0CA678}.t-LigaF{background:#D6336C}
.t-Selección{background:#37B24D}.t-Sindatos{background:#868E96}.t-Otra{background:#868E96}
.src{font-size:10px;opacity:.7}
.bar{display:flex;height:24px;border-radius:7px;overflow:hidden;font-size:11px;color:#fff;font-weight:700}
.bar div{display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden;min-width:0}
.pub{font-size:11px;opacity:.75;margin-top:5px}
.star{font-weight:700;color:#F59F00}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Acceso opcional con contraseña (Settings > Secrets:  APP_PASSWORD = "tu_clave")
# --------------------------------------------------------------------------------------
def pedir_clave():
    try:
        clave = st.secrets["APP_PASSWORD"]
    except Exception:
        return
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
# Cargas cacheadas
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cargar_historico() -> pd.DataFrame:
    return qm.load_history(n_seasons=2)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cargar_liga_f() -> pd.DataFrame:
    return ql.load_liga_f(n_seasons=2)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cargar_selecciones() -> pd.DataFrame:
    return qsel.load_selecciones(years=3)


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def entrenar(clave: str, _hist: pd.DataFrame) -> dict:
    return qm.train_pipeline(_hist)


@st.cache_data(ttl=600, show_spinner=False)
def cargar_jornada() -> dict:
    return qs.fetch_jornada()


def euros(x) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def pct(x, d=0) -> str:
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------------------
# Piezas visuales
# --------------------------------------------------------------------------------------
def chip(num: int, signo: str, alt: bool = False) -> str:
    cls = {"1": "c1", "X": "cX", "2": "c2"}[signo]
    return f'<div class="qchip {cls}{" alt" if alt else ""}"><small>{num}</small><span>{signo}</span></div>'


def columna_html(n_col: int, code: str, fav: np.ndarray, pleno, detalle: str) -> str:
    chips = "".join(chip(i + 1, s, SIGNOS.index(s) != fav[i]) for i, s in enumerate(code))
    if pleno:
        chips += f'<div class="qchip pleno"><small>15</small><span>{pleno}</span></div>'
    return (f'<div class="qcol"><h4>Columna {n_col} <span class="muted">· {detalle}</span></h4>'
            f'<div class="chips">{chips}</div></div>')


def partido_html(i: int, nombre: str, comp: str, P: np.ndarray, S: np.ndarray, fuente: str) -> str:
    seg = "".join(f'<div class="c{s}" style="flex:{max(p, 0.08):.3f}">{s} {pct(p)}</div>'
                  for s, p in zip(SIGNOS, P))
    estrellas = [f'{s} ({pct(P[j])} vs público {pct(S[j])})' for j, s in enumerate(SIGNOS)
                 if P[j] / S[j] >= 1.3 and P[j] >= 0.10]
    valor = f' · <span class="star">★ valor en {", ".join(estrellas)}</span>' if estrellas else ""
    css = comp.replace(" ", "")
    return (f'<div class="mcard"><div class="mtop"><span class="mnum">{i}</span><span class="mname">{nombre}</span>'
            f'<span class="tag t-{css}">{comp}</span><span class="src">{fuente}</span></div>'
            f'<div class="bar">{seg}</div>'
            f'<div class="pub">Público: 1 {pct(S[0])} · X {pct(S[1])} · 2 {pct(S[2])}{valor}</div></div>')


pedir_clave()
st.title("⚽ Quiniela inteligente")
st.caption("Modelo estadístico + casas de apuestas frente a lo que juega el público. "
           "Te propone columnas y reparte el riesgo entre ellas.")

# La jornada se lee primero (no depende del modelo): así el bote se propone solo
jornada, error = None, None
try:
    jornada = cargar_jornada()
except qs.ScrapeError as exc:
    error = str(exc)
except Exception as exc:
    error = f"Error inesperado: {exc}"

banner = st.container()
tab_cols, tab_part, tab_aj, tab_ayuda = st.tabs(["🎟️ Mis columnas", "📋 Partidos", "⚙️ Ajustes", "📖 Ayuda"])

# --------------------------------------------------------------------------------------
# AJUSTES (se definen antes de calcular)
# --------------------------------------------------------------------------------------
with tab_aj:
    st.markdown("#### 💶 Bote y reparto")
    bote = st.number_input("Bote / premio máximo en juego (€)", min_value=0, step=100_000,
                           value=int((jornada or {}).get("premio_max") or 1_000_000),
                           help="Solo escala el valor esperado en euros. Se propone el premio que anuncia la web.")
    recaud = st.number_input("Recaudación estimada de la jornada (€)", min_value=100_000, step=100_000, value=3_500_000,
                             help="Orden de magnitud real: ~3,5 M€ (jornada del 11-oct-2026). Mira la última nota de prensa de LAE.")
    precio = st.number_input("Precio por columna (€)", min_value=0.10, step=0.05, value=0.75)
    st.markdown("#### 🧠 Cómo combino las fuentes")
    alpha = st.slider("Peso de mi modelo frente a las casas de apuestas (%)", 0, 100, 50,
                      help="Las casas suelen afinar mejor que un modelo casero; mezclar ambos reduce errores. "
                           "100 = solo modelo, 0 = solo casas.") / 100
    peso = st.slider("Peso de esa estimación frente al público (%)", 50, 100, 90,
                     help="Encoge ligeramente la estimación hacia el público para no fiarte de diferencias exageradas.") / 100
    st.markdown("#### 🛡️ Filtros anti-quimeras")
    min_signo = st.slider("Probabilidad mínima por signo (%)", 0, 30, 10) / 100
    min_rel = st.slider("Probabilidad mínima de una columna vs. la más probable (%)", 1, 30, 5) / 100
    max_sorp = st.slider("Máx. de apuestas contra el favorito por columna", 0, 8, 4)
    min_emp = st.slider("Mínimo de empates (X) por columna", 0, 6, 2,
                        help="En 14 partidos suelen salir 3 o 4 empates; sin este mínimo casi nunca llevarían X.")
    min_dist = st.slider("Distancia mínima entre columnas (solo con cobertura 0 %)", 1, 8, 3)
    st.markdown("#### 🗂️ Datos")
    auto_f = st.checkbox("Liga F automática (Wikipedia)", value=True)
    auto_sel = st.checkbox("Selecciones automáticas (resultados internacionales)", value=True)
    st.caption("Opcional: sube tu propio CSV de partidos (Date, Home, Away, HG, AG). "
               "Para dejarlo fijo, súbelo a GitHub como `liga_f.csv`.")
    subido = st.file_uploader("CSV de partidos", type="csv")
    es_fem = st.checkbox("Son partidos de fútbol femenino", value=True)
    if st.button("🔄 Recargar jornada y datos", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------------------
# DATOS Y MODELO
# --------------------------------------------------------------------------------------
avisos: list[str] = []
with banner:
    with st.status("Preparando datos y modelo…", expanded=False) as estado:
        try:
            hist = cargar_historico()
        except Exception as exc:
            estado.update(label="No se pudo cargar el histórico", state="error")
            st.error(f"No pude descargar los resultados de Primera/Segunda: {exc}")
            st.stop()
        extras = []
        if auto_f:
            try:
                extras.append(cargar_liga_f())
            except Exception as exc:
                avisos.append(f"Liga F: no pude descargarla ({exc}). Esos partidos usarán las cuotas de las casas.")
        if auto_sel:
            try:
                extras.append(cargar_selecciones())
            except Exception as exc:
                avisos.append(f"Selecciones: no pude descargarlas ({exc}). Esos partidos usarán las cuotas de las casas.")
        try:
            fijo = Path(__file__).parent / "liga_f.csv"
            if fijo.exists():
                extras.append(qm.parse_extra_csv(fijo, femenino=True))
            if subido is not None:
                extras.append(qm.parse_extra_csv(subido, femenino=es_fem))
        except Exception as exc:
            avisos.append(f"CSV extra: {exc}")
        if extras:
            hist = pd.concat([hist] + extras).sort_values("Date").reset_index(drop=True)
        estado.update(label="Entrenando el modelo (solo la primera vez, ~1-2 min)…")
        trained = entrenar(f"{len(hist)}-{hist.Date.max()}", hist)
        estado.update(label=f"Modelo listo · {len(hist):,} partidos en el histórico".replace(",", "."), state="complete")
    for a in avisos:
        st.warning(a)
    if jornada:
        txt = f"**Jornada {jornada['jornada']}**" if jornada["jornada"] else "**Jornada actual**"
        if jornada["bote"]:
            txt += f" · Bote {euros(jornada['bote'])}"
        st.success(f"{txt} · {len(jornada['partidos'])} partidos + Pleno al 15 leídos automáticamente")
        if jornada["cerrada"]:
            st.warning("Esta jornada figura como **cerrada**: ya no se puede jugar. Cuando se publique la siguiente, "
                       "abre Ajustes y pulsa «Recargar».")
    else:
        st.error(f"No pude leer la jornada automáticamente: {error}")
        st.info("Puedes rellenar la tabla a mano en la pestaña «Partidos».")

known = sorted(set(hist.Home) | set(hist.Away))
comp_de = qm.team_competitions(hist)
opciones = [SIN_HIST] + known

# --------------------------------------------------------------------------------------
# PARTIDOS (tarjetas + corrección manual)
# --------------------------------------------------------------------------------------
with tab_part:
    cards_box = st.container()
    st.radio("Porcentajes del público", ["lae", "jug"], horizontal=True, key="fuente_pub",
             format_func=lambda x: "% LAE (oficial)" if x == "lae" else "% Jugados (web de Losilla)")
    pref = st.session_state["fuente_pub"]

    if jornada:
        p = jornada["partidos"]
        mapas = [qs.map_match(a, b, known) for a, b in zip(p.Local_quiniela, p.Visitante_quiniela)]
        base = pd.DataFrame({
            "Partido": p.Local_quiniela + " - " + p.Visitante_quiniela,
            "Local": [m[0] or SIN_HIST for m in mapas], "Visitante": [m[1] or SIN_HIST for m in mapas],
            "pct_1": p[f"{pref}_1"], "pct_X": p[f"{pref}_X"], "pct_2": p[f"{pref}_2"],
            "mkt_1": p.prob_1, "mkt_X": p.prob_X, "mkt_2": p.prob_2,
        })
    else:
        base = pd.DataFrame({"Partido": [f"Partido {i}" for i in range(1, 15)], "Local": SIN_HIST, "Visitante": SIN_HIST,
                             "pct_1": 40, "pct_X": 30, "pct_2": 30, "mkt_1": 0, "mkt_X": 0, "mkt_2": 0})
    sin = int(((base.Local == SIN_HIST) | (base.Visitante == SIN_HIST)).sum())
    with st.expander(f"✏️ Revisar y corregir equipos y porcentajes ({sin} sin identificar)"):
        st.caption("Si un equipo no se identificó, elígelo en el desplegable. «Cuotas %» son las probabilidades de las "
                   "casas de apuestas; «1/X/2 %» es lo que juega el público.")
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
        st.caption("Cada equipo marca 0, 1, 2 o M (3 o más). El Pleno solo cuenta para el bote: "
                   "hay que acertar los 14 signos y este marcador.")
        loc0 = vis0 = None
        if pl:
            st.write(f"**{pl['Local_quiniela']} - {pl['Visitante_quiniela']}**")
            if pl["Local_quiniela"] and pl["Visitante_quiniela"]:
                loc0, vis0 = qs.map_match(pl["Local_quiniela"], pl["Visitante_quiniela"], known)
        local15 = st.selectbox("Local", opciones, index=opciones.index(loc0 or SIN_HIST), key="l15")
        visit15 = st.selectbox("Visitante", opciones, index=opciones.index(vis0 or SIN_HIST), key="v15")
        hv = pl[f"{pref}_H"] if pl else [25, 25, 25, 25]
        av = pl[f"{pref}_A"] if pl else [25, 25, 25, 25]
        pl_df = st.data_editor(
            pd.DataFrame({"0": [hv[0], av[0]], "1": [hv[1], av[1]], "2": [hv[2], av[2]], "M": [hv[3], av[3]]},
                         index=["Local", "Visitante"]),
            key="tabla_pleno", width="stretch", num_rows="fixed",
            column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=100) for c in ["0", "1", "2", "M"]})

# --------------------------------------------------------------------------------------
# CONTROLES PRINCIPALES
# --------------------------------------------------------------------------------------
with tab_cols:
    c_k, c_cob = st.columns(2)
    k = c_k.slider("¿Cuántas columnas?", 1, 30, 4)
    cobertura = c_cob.slider("Cobertura (%)", 0, 100, 50, help=(
        "0 % = las columnas de mayor valor, que suelen parecerse y fallar juntas. Más cobertura = columnas que se "
        "complementan (p. ej. el otro signo en el partido más dudoso) para no fallar todas en el mismo partido.")) / 100
    st.caption(f"Coste: {k} × {precio:.2f} € = **{k * precio:.2f} €**")
    resultados = st.container()

# --------------------------------------------------------------------------------------
# CÁLCULO
# --------------------------------------------------------------------------------------
fixtures = pd.DataFrame({
    "Home": [h if h != SIN_HIST else f"{r.Partido}|L" for h, r in zip(editada.Local, editada.itertuples())],
    "Away": [a if a != SIN_HIST else f"{r.Partido}|V" for a, r in zip(editada.Visitante, editada.itertuples())],
    "Date": pd.Timestamp.today().normalize(),
    "pct_1": editada.pct_1.astype(float), "pct_X": editada.pct_X.astype(float), "pct_2": editada.pct_2.astype(float),
    "mkt_1": editada.mkt_1.astype(float), "mkt_X": editada.mkt_X.astype(float), "mkt_2": editada.mkt_2.astype(float),
})
nombres = list(editada.Partido)

if (fixtures[["pct_1", "pct_X", "pct_2"]].sum(axis=1) <= 0).any():
    with resultados:
        st.warning("Hay partidos con los tres porcentajes del público a 0. Revísalos en la pestaña «Partidos».")
    st.stop()

with st.spinner("Calculando…"):
    result = qm.predict_fixtures(hist, fixtures, trained, alpha=alpha)
    Pa, S = qo.prepare_arrays(result, peso)
    fav = Pa.argmax(axis=1)

    pleno_arg = None
    pl_vals = pl_df.fillna(0).to_numpy(float)
    if pl_vals[0].sum() > 0 and pl_vals[1].sum() > 0:
        sh, sa = np.clip(pl_vals[0], 1e-4, None), np.clip(pl_vals[1], 1e-4, None)
        Sp = np.outer(sh / sh.sum(), sa / sa.sum())
        pp = None if SIN_HIST in (local15, visit15) else qm.pleno_probs(trained, local15, visit15)
        Pp = Sp if pp is None else peso * pp + (1 - peso) * Sp
        pleno_arg = (Pp / Pp.sum(), Sp)

    try:
        cols = qo.optimizar(Pa, S, k=k, min_prob_signo=min_signo, min_prob_rel=min_rel, max_sorpresas=max_sorp,
                            min_empates=min_emp, min_distancia=min_dist, bote=float(bote), n_columnas=recaud / precio,
                            coste_columna=precio, pleno=pleno_arg, cobertura=cobertura)
        error_opt = None
    except ValueError as exc:
        cols, error_opt = None, str(exc)

comp_part = [comp_de.get(h) or comp_de.get(a) or "Sin datos" for h, a in zip(fixtures.Home, fixtures.Away)]

# --------------------------------------------------------------------------------------
# TARJETAS DE PARTIDOS
# --------------------------------------------------------------------------------------
with cards_box:
    resumen = pd.Series(comp_part).value_counts()
    st.markdown("**La jornada:** " + " · ".join(f"{v} {k_}" for k_, v in resumen.items()) + " · + Pleno al 15")
    st.caption("Barra = probabilidad estimada de 1 / X / 2. ★ = el público infravalora ese signo. "
               "«modelo + casas» = mezcla de tu modelo y las cuotas; «casas» o «público» = sin histórico propio.")
    st.markdown("".join(partido_html(i + 1, nombres[i], comp_part[i], Pa[i], S[i], result.fuente.iloc[i])
                        for i in range(len(nombres))), unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# RESULTADOS
# --------------------------------------------------------------------------------------
with resultados:
    if cols is None:
        st.warning(f"{error_opt}")
    else:
        ev_total = cols.EV_sobre_coste.mean()
        c1, c2 = st.columns(2)
        c1.metric("Prob. de 14 aciertos", f"{cols.attrs['p_14'] * 100:.4f} %", help="Con alguna de tus columnas.")
        c2.metric("Prob. de 13 o más", f"{cols.attrs['p_13mas'] * 100:.2f} %", help="Simulada con 40.000 jornadas.")
        c3, c4 = st.columns(2)
        c3.metric("Retorno esperado / coste", f"{ev_total:.2f}",
                  help="1,00 = recuperas de media lo que pagas. Casi siempre es menor: la Quiniela retiene parte de la recaudación.")
        c4.metric("Columnas válidas tras filtros", f"{cols.attrs['n_validas']:,}".replace(",", "."))
        if ev_total > 1.5:
            st.warning("Un retorno esperado muy por encima de 1 casi siempre indica que el modelo discrepa mucho del público "
                       "(posible sobreconfianza o datos mal casados). Revisa la pestaña «Partidos» y baja el peso del modelo en Ajustes.")
        elif ev_total < 1:
            st.info(f"Con estos supuestos recuperarías de media ~{ev_total * 100:.0f} % de lo que pagas. "
                    "Es lo habitual en una lotería con retención: las columnas son las mejores posibles, pero no hay garantía.")

        st.markdown("##### Tus columnas")
        st.caption("🔵 1 · ⚪ X · 🟠 2 · **contorno amarillo** = apuesta contra el favorito (así se reparte el riesgo). "
                   "El número pequeño es el partido.")
        html = ""
        for i, r in enumerate(cols.itertuples(), 1):
            det = f"1 de cada {int(r.P_conjunta ** -1):,}".replace(",", ".")
            html += columna_html(i, r.columna, fav, getattr(r, "pleno", None), det)
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("**Copia los signos al boleto** (icono de copiar arriba a la derecha):")
        filas = [f"{i:02d}  " + " ".join(c) + (f"   | 15: {pl_}" if "pleno" in cols else "")
                 for i, (c, pl_) in enumerate(zip(cols.columna, cols.get("pleno", [""] * len(cols))), 1)]
        st.code("\n".join(filas), language="text")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ TXT", qo.to_text(cols, result), "quiniela_optimizada.txt", width="stretch")
        d2.download_button("⬇️ CSV", cols.to_csv(index=False), "quiniela_optimizada.csv", width="stretch")

        with st.expander("🧩 Reparto de signos por partido"):
            rep = pd.DataFrame(cols.attrs["reparto"], columns=SIGNOS)
            rep.insert(0, "Partido", [f"{i + 1}. {t}" for i, t in enumerate(nombres)])
            st.dataframe(rep, hide_index=True, width="stretch")
            st.caption("Cuántas de tus columnas llevan cada signo. Si un partido tiene todas con el mismo signo y ese signo falla, "
                       "todas fallan a la vez: sube la cobertura para repartirlo.")
        with st.expander("🔎 Valor de cada signo (público frente a estimación)"):
            st.dataframe(qo.sign_table(result.assign(Home=nombres, Away=""), Pa, S, min_signo),
                         hide_index=True, width="stretch")
            st.caption("ratio = probabilidad estimada ÷ % del público. ★ = ratio ≥ 1,3 con probabilidad suficiente.")

# --------------------------------------------------------------------------------------
# AYUDA
# --------------------------------------------------------------------------------------
with tab_ayuda:
    st.markdown("""
#### Cómo usarla en 3 pasos
1. **Abre la app** el día de la jornada: lee sola los partidos, los % del público y las cuotas de las casas.
2. **Elige cuántas columnas** quieres (pestaña «Mis columnas») y cuánta cobertura.
3. **Copia los signos** al boleto. Si un equipo no se identificó, corrígelo en «Partidos».

#### Qué significa cada cosa
- **Probabilidad estimada** (barras de «Partidos»): mezcla tu modelo estadístico con las cuotas de las casas de apuestas.
  Si un equipo no tiene histórico, usa solo las cuotas; si tampoco hay cuotas, el % del público (sin ventaja).
- **★ Valor**: el público juega ese signo menos de lo que debería según la estimación. Si aciertas, el premio se reparte entre menos gente.
- **Cobertura**: reparte el riesgo. Sin ella, las columnas se parecen y fallan juntas (como si todas llevan el mismo 1 en el partido 1).
  Con cobertura, algunas llevan el otro signo en los partidos más dudosos.
- **Contorno amarillo**: esa columna apuesta contra el favorito en ese partido.
- **Pleno al 15**: goles de cada equipo (0, 1, 2 o M = 3 o más). Solo cuenta para el bote.
- **Retorno esperado / coste**: cuánto recuperarías de media por cada euro. 1,00 sería quedar en paz; casi siempre es menor.

#### De dónde salen los datos
Resultados de Primera y Segunda (football-data.co.uk), Liga F (Wikipedia), selecciones (resultados internacionales en GitHub),
jornada y porcentajes (eduardolosilla.es). Todo gratuito; si una fuente falla, la app te lo avisa y sigue con lo que tenga.

#### Límites que conviene conocer
- El modelo **no está validado con jornadas pasadas** y las bajas/fatiga están simuladas. Úsalo como una ayuda más.
- La Quiniela retiene parte de la recaudación: incluso con ventaja sobre el público, lo normal es perder dinero a largo plazo.
  **Juega solo lo que puedas permitirte perder.**
- En la Liga F, las fechas históricas de las temporadas cerradas son aproximadas.
""")
    st.markdown("#### 📈 Calidad del modelo")
    ev1 = trained.get("evaluation")
    if ev1 is not None:
        st.caption(f"20 % más reciente del histórico ({trained.get('n_test', '?')} partidos). Menor = mejor.")
        st.dataframe(ev1, width="stretch")
    ev2 = trained.get("evaluation_market")
    if ev2 is not None:
        st.caption("Solo partidos con cuotas (Primera y Segunda): así se compara con las casas de apuestas.")
        st.dataframe(ev2, width="stretch")
    st.caption("Si «Mercado» aparece por debajo de tu modelo, fíate más de las casas (baja el peso del modelo en Ajustes).")
