"""
app.py  ·  Quiniela inteligente (Streamlit)

    streamlit run app.py

Qué hace, de principio a fin:
  1. Descarga históricos gratuitos (Primera/Segunda, Liga F, selecciones) y entrena el modelo (cacheado).
  2. Lee la jornada de La Quiniela: partidos, día y hora, % del público (LAE o Losilla) y cuotas de las casas.
  3. Identifica la competición de cada partido y combina modelo + casas de apuestas.
  4. Elige las mejores columnas, repartiendo el riesgo entre ellas, y las muestra listas para copiar.

Archivos necesarios en la misma carpeta: quiniela_model.py, quiniela_optimizer.py, quiniela_scraper.py,
quiniela_ligaf.py, quiniela_selecciones.py, requirements.txt
"""
import hmac
import html as _html
import re
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
DIAS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

st.markdown("""
<style>
/* ---- ayudas al pasar el ratón (o tocar) ---- */
.tip{position:relative;cursor:help}
.tipbox{display:none;position:absolute;z-index:1000;left:50%;bottom:calc(100% + 8px);transform:translateX(-50%);
  width:max-content;max-width:240px;background:#101828;color:#fff;font-size:12px;font-weight:400;line-height:1.4;
  padding:7px 10px;border-radius:8px;text-align:left;white-space:normal;box-shadow:0 4px 14px rgba(0,0,0,.4);font-style:normal}
.tip:hover .tipbox{display:block}
/* ---- cabecera de la jornada ---- */
.hero{background:linear-gradient(135deg,#1c7ed6 0%,#7048e8 55%,#d6336c 100%);border-radius:18px;padding:16px 18px;
  color:#fff;margin:4px 0 12px 0;box-shadow:0 6px 18px rgba(80,60,200,.25)}
.hero-j{font-size:30px;font-weight:900;letter-spacing:.5px;line-height:1.1}
.hero-sub{font-size:14px;opacity:.95;margin-top:2px}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pill{background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:3px 11px;
  font-size:12px;font-weight:600}
.pill.tip:hover{background:rgba(255,255,255,.32)}
.steps{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px 0}
.step{flex:1;min-width:130px;border-radius:12px;padding:7px 10px;font-size:12px;font-weight:600;color:#fff}
.s1{background:#1c7ed6}.s2{background:#7048e8}.s3{background:#2f9e44}
/* ---- títulos de sección ---- */
.sec{font-size:18px;font-weight:800;border-left:6px solid #7048e8;padding-left:10px;margin:16px 0 8px 0}
.sec.verde{border-color:#2f9e44}.sec.azul{border-color:#1c7ed6}.sec.rosa{border-color:#d6336c}.sec.naranja{border-color:#f59f00}
/* ---- indicadores ---- */
.kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 10px 0}
.kpi{border-radius:14px;padding:10px 13px;color:#fff;position:relative}
.kpi small{display:block;font-size:11px;opacity:.9;font-weight:600}
.kpi b{font-size:22px;font-weight:900}
.k1{background:linear-gradient(135deg,#2f9e44,#69db7c)}.k2{background:linear-gradient(135deg,#1c7ed6,#4dabf7)}
.k3{background:linear-gradient(135deg,#7048e8,#9775fa)}.k4{background:linear-gradient(135deg,#f76707,#ffa94d)}
/* ---- columnas ---- */
.chips{display:flex;flex-wrap:wrap;gap:5px}
.qchip{position:relative;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:36px;height:46px;
  border-radius:10px;color:#fff;font-weight:800;line-height:1;cursor:help}
.qchip small{font-size:9px;font-weight:600;opacity:.9;margin-bottom:3px}
.qchip b{font-size:20px}
.qchip:hover .tipbox{display:block}
.c1{background:#2E7DF6}.cX{background:#7B8794}.c2{background:#F2994A}
.qchip.alt{outline:3px solid #FFD43B;outline-offset:-1px}
.qchip.pleno{background:#2F9E44;width:auto;min-width:52px;padding:0 10px}
.qcol{border:1px solid rgba(128,128,128,.35);border-radius:14px;padding:10px 12px;margin-bottom:10px;
  background:linear-gradient(180deg,rgba(112,72,232,.07),rgba(112,72,232,0))}
.qcol h4{margin:0 0 8px 0;font-size:15px}
.muted{opacity:.7;font-weight:400;font-size:12px}
/* ---- tarjetas de partido ---- */
.mcard{border:1px solid rgba(128,128,128,.3);border-left:7px solid #868e96;border-radius:14px;padding:10px 13px;margin-bottom:10px}
.mcard.Primera{border-left-color:#1c7ed6}.mcard.Segunda{border-left-color:#0ca678}.mcard.LigaF{border-left-color:#d6336c}
.mcard.Selección{border-left-color:#37b24d}.mcard.Pleno{border-left-color:#f59f00}
.mtop{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:7px}
.mnum{background:#7048e8;color:#fff;border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;
  justify-content:center;font-size:13px;font-weight:800;flex:none}
.when{background:rgba(245,159,0,.18);border:1px solid rgba(245,159,0,.55);border-radius:8px;padding:1px 8px;font-size:12px;font-weight:700}
.tag{font-size:10px;padding:2px 9px;border-radius:10px;color:#fff;font-weight:700;background:#868e96}
.t-Primera{background:#1c7ed6}.t-Segunda{background:#0ca678}.t-LigaF{background:#d6336c}.t-Selección{background:#37b24d}
.src{font-size:10px;padding:2px 8px;border-radius:10px;border:1px dashed rgba(128,128,128,.7)}
.teams{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:end;margin-bottom:6px}
.team{font-weight:800;font-size:14px;line-height:1.2}
.team.r{text-align:right}.vs{font-size:11px;opacity:.6;font-weight:700}
.bar{display:flex;height:26px;border-radius:8px;overflow:hidden;font-size:12px;color:#fff;font-weight:800}
.bar div{display:flex;align-items:center;justify-content:center;white-space:nowrap;overflow:hidden;min-width:0}
.pcts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:6px;font-size:12px}
.pcts div{border-radius:9px;padding:4px 8px;color:#fff;font-weight:700;position:relative;cursor:help}
.pcts div:hover .tipbox{display:block}
.pcts .a{background:#2E7DF6}.pcts .b{background:#7B8794;text-align:center}.pcts .c{background:#F2994A;text-align:right}
.pcts i{font-style:normal;display:block;font-size:10px;opacity:.9;font-weight:600}
.pub{font-size:11px;opacity:.85;margin-top:7px}
.star{font-weight:800;color:#f59f00}
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


# --------------------------------------------------------------------------------------
# Utilidades de texto y piezas visuales
# --------------------------------------------------------------------------------------
def euros(x) -> str:
    return f"{x:,.0f} €".replace(",", ".")


def pct(x, d=0) -> str:
    return f"{x * 100:.{d}f}%"


def esc(x) -> str:
    return _html.escape(str(x), quote=True)


def bonito(nombre: str) -> str:
    """'R.SOCIEDAD B' -> 'R.Sociedad B' (la web los da en mayúsculas)."""
    out = str(nombre).strip().title()
    return re.sub(r"\(f\)", "(F)", out)


def cuando(fecha, dia: str, hora: str) -> str:
    if pd.notna(fecha):
        d = f"{DIAS_ES[pd.Timestamp(fecha).weekday()]} {pd.Timestamp(fecha):%d/%m}"
    else:
        d = (dia or "").title()
    return " · ".join(x for x in (d, hora) if x) or "Sin hora"


def tip(texto: str) -> str:
    """Caja de ayuda que aparece al pasar el ratón (o tocar)."""
    return f'<em class="tipbox">{esc(texto)}</em>'


def clase(comp: str) -> str:
    return re.sub(r"[^A-Za-zÁ-úñ]", "", comp)


FUENTES = {
    "modelo + casas": "Mezcla tu modelo estadístico con las probabilidades de las casas de apuestas. Es la estimación más fiable.",
    "modelo": "Solo tu modelo estadístico: este partido no tiene cuotas de casas de apuestas.",
    "casas": "No hay histórico de alguno de los equipos: uso las probabilidades de las casas de apuestas.",
    "público": "No hay histórico ni cuotas: uso lo que juega el público. No puede haber ventaja en este partido.",
}
COMPS = {
    "Primera": "Primera División. Histórico de resultados de football-data.co.uk.",
    "Segunda": "Segunda División. Histórico de resultados de football-data.co.uk.",
    "Liga F": "Primera División femenina. Histórico de Wikipedia (fechas aproximadas en temporadas cerradas).",
    "Selección": "Selecciones nacionales. Histórico de resultados internacionales (partidos en campo propio).",
}


def chip(num: int, signo: str, alt: bool, ayuda: str) -> str:
    cls = {"1": "c1", "X": "cX", "2": "c2"}[signo]
    return (f'<div class="qchip {cls}{" alt" if alt else ""}"><small>{num}</small><b>{signo}</b>{tip(ayuda)}</div>')


def columna_html(n_col, code, fav, pleno, detalle, loc, vis, Pa, pleno_ayuda="") -> str:
    chips = ""
    for i, s in enumerate(code):
        j = SIGNOS.index(s)
        que = {"1": f"gana {loc[i]}", "X": "empate", "2": f"gana {vis[i]}"}[s]
        ayuda = (f"Partido {i + 1}: {loc[i]} – {vis[i]}. Esta columna apuesta a que {que} "
                 f"(estimación: {pct(Pa[i][j])})." + (" Va contra el favorito." if j != fav[i] else " Es el favorito."))
        chips += chip(i + 1, s, j != fav[i], ayuda)
    if pleno:
        chips += (f'<div class="qchip pleno"><small>15</small><b>{pleno}</b>'
                  f'{tip(pleno_ayuda or "Pleno al 15: goles del local – goles del visitante (M = 3 o más). Solo cuenta para el bote.")}</div>')
    cab = (f'<span class="tip">1 de cada {detalle}{tip("Probabilidad de que ESTA columna acierte los 14 signos (sin contar el pleno).")}</span>')
    return f'<div class="qcol"><h4>Columna {n_col} <span class="muted">· {cab}</span></h4><div class="chips">{chips}</div></div>'


def partido_html(i, loc, vis, comp, cuando_txt, P, S, fuente) -> str:
    seg = "".join(f'<div class="c{s}" style="flex:{max(p, 0.08):.3f}">{pct(p)}</div>' for s, p in zip(SIGNOS, P))
    estrellas = [f"{s} ({pct(P[j])} frente a {pct(S[j])} del público)" for j, s in enumerate(SIGNOS)
                 if P[j] / S[j] >= 1.3 and P[j] >= 0.10]
    valor = ""
    if estrellas:
        valor = (f' · <span class="star tip">★ Valor en {", ".join(estrellas)}'
                 f'{tip("Valor: la estimación supera en un 30 % o más lo que juega el público. Si acierta, el premio se reparte entre menos gente.")}</span>')
    comp_tip = COMPS.get(comp, "Sin histórico propio para este partido: no se puede estimar con el modelo.")
    return (
        f'<div class="mcard {clase(comp)}">'
        f'<div class="mtop"><span class="mnum tip">{i}{tip("Número del partido en el boleto de La Quiniela.")}</span>'
        f'<span class="when tip">🕒 {esc(cuando_txt)}{tip("Día y hora del partido según la web de La Quiniela.")}</span>'
        f'<span class="tag t-{clase(comp)} tip">{esc(comp)}{tip(comp_tip)}</span>'
        f'<span class="src tip">{esc(fuente)}{tip(FUENTES.get(fuente, ""))}</span></div>'
        f'<div class="teams"><div class="team">{esc(loc)}</div><div class="vs">VS</div><div class="team r">{esc(vis)}</div></div>'
        f'<div class="bar">{seg}</div>'
        f'<div class="pcts">'
        f'<div class="a"><i>Gana {esc(loc)}</i>{pct(P[0])}{tip(f"Probabilidad estimada de que gane {loc} (signo 1).")}</div>'
        f'<div class="b"><i>Empate</i>{pct(P[1])}{tip("Probabilidad estimada de empate (signo X).")}</div>'
        f'<div class="c"><i>Gana {esc(vis)}</i>{pct(P[2])}{tip(f"Probabilidad estimada de que gane {vis} (signo 2).")}</div></div>'
        f'<div class="pub tip">Lo que juega el público: 1 → {pct(S[0])} · X → {pct(S[1])} · 2 → {pct(S[2])}{valor}'
        f'{tip("Porcentaje de quinielistas que pone cada signo. Si tu estimación es mucho mayor que el público, ese signo tiene valor.")}</div>'
        f'</div>')


def pleno_html(loc, vis, cuando_txt, Pp, Sp) -> str:
    if Pp is None:
        return ""
    etiquetas = ["0", "1", "2", "M"]
    flat = sorted(((Pp[a, b], a, b) for a in range(4) for b in range(4)), reverse=True)[:3]
    top = " · ".join(f"<b>{etiquetas[a]}–{etiquetas[b]}</b> ({pct(p)})" for p, a, b in flat)
    gl = " · ".join(f"{e}: {pct(v)}" for e, v in zip(etiquetas, Pp.sum(axis=1)))
    gv = " · ".join(f"{e}: {pct(v)}" for e, v in zip(etiquetas, Pp.sum(axis=0)))
    return (
        f'<div class="mcard Pleno"><div class="mtop"><span class="mnum tip" style="background:#f59f00">15{tip("Partido 15: Pleno al 15.")}</span>'
        f'<span class="when tip">🕒 {esc(cuando_txt)}{tip("Día y hora del partido según la web de La Quiniela.")}</span>'
        f'<span class="tag tip" style="background:#f59f00">Pleno al 15{tip("Hay que acertar cuántos goles marca cada equipo (0, 1, 2 o M = 3 o más). Solo cuenta para el bote junto con los 14 signos.")}</span></div>'
        f'<div class="teams"><div class="team">{esc(loc)}</div><div class="vs">VS</div><div class="team r">{esc(vis)}</div></div>'
        f'<div class="pub tip">Marcadores más probables: {top}{tip("Los tres marcadores (goles local – goles visitante) con más probabilidad estimada.")}</div>'
        f'<div class="pub">Goles de {esc(loc)} → {gl}</div><div class="pub">Goles de {esc(vis)} → {gv}</div></div>')


def kpi(clase_css, titulo, valor, ayuda) -> str:
    return f'<div class="kpi {clase_css} tip"><small>{titulo}</small><b>{valor}</b>{tip(ayuda)}</div>'


pedir_clave()
st.title("⚽ Quiniela inteligente")
st.caption("Modelo estadístico + casas de apuestas frente a lo que juega el público. "
           "Te propone columnas y reparte el riesgo entre ellas. Pasa el ratón (o toca) sobre cualquier etiqueta para ver para qué sirve.")

# La jornada se lee primero (no depende del modelo)
jornada, error = None, None
try:
    jornada = cargar_jornada()
except qs.ScrapeError as exc:
    error = str(exc)
except Exception as exc:
    error = f"Error inesperado: {exc}"

lae_pendiente = bool(jornada) and bool(jornada["partidos"][["lae_1", "lae_X", "lae_2"]].isna().any(axis=1).all())
hero_box = st.container()
banner = st.container()
tab_cols, tab_part, tab_aj, tab_ayuda = st.tabs(["🎟️ Mis columnas", "📋 Partidos", "⚙️ Ajustes", "📖 Ayuda"])

# --------------------------------------------------------------------------------------
# AJUSTES (se definen antes de calcular)
# --------------------------------------------------------------------------------------
with tab_aj:
    st.markdown('<div class="sec verde">💶 Bote y reparto</div>', unsafe_allow_html=True)
    bote = st.number_input("Bote / premio máximo en juego (€)", min_value=0, step=100_000,
                           value=int((jornada or {}).get("premio_max") or 1_000_000),
                           help="Solo escala el valor esperado en euros. Se propone el premio que anuncia la web.")
    recaud = st.number_input("Recaudación estimada de la jornada (€)", min_value=100_000, step=100_000, value=3_500_000,
                             help="Orden de magnitud real: ~3,5 M€ (jornada del 11-oct-2026). Mira la última nota de prensa de LAE.")
    precio = st.number_input("Precio por columna (€)", min_value=0.10, step=0.05, value=0.75)
    st.markdown('<div class="sec azul">🧠 Cómo combino las fuentes</div>', unsafe_allow_html=True)
    alpha = st.slider("Peso de mi modelo frente a las casas de apuestas (%)", 0, 100, 50,
                      help="Las casas suelen afinar mejor que un modelo casero; mezclar ambos reduce errores. "
                           "100 = solo modelo, 0 = solo casas.") / 100
    peso = st.slider("Peso de esa estimación frente al público (%)", 50, 100, 90,
                     help="Encoge ligeramente la estimación hacia el público para no fiarte de diferencias exageradas.") / 100
    st.markdown('<div class="sec rosa">🛡️ Filtros anti-quimeras</div>', unsafe_allow_html=True)
    min_signo = st.slider("Probabilidad mínima por signo (%)", 0, 30, 10,
                          help="Descarta signos con menos de esta probabilidad estimada.") / 100
    min_rel = st.slider("Probabilidad mínima de una columna vs. la más probable (%)", 1, 30, 5,
                        help="Descarta columnas muchísimo menos probables que la mejor.") / 100
    max_sorp = st.slider("Máx. de apuestas contra el favorito por columna", 0, 8, 4,
                         help="Cuántos partidos puede apostar una columna contra el favorito.")
    min_emp = st.slider("Mínimo de empates (X) por columna", 0, 6, 2,
                        help="En 14 partidos suelen salir 3 o 4 empates; sin este mínimo casi nunca llevarían X.")
    min_dist = st.slider("Distancia mínima entre columnas (solo con cobertura 0 %)", 1, 8, 3,
                         help="Signos en los que deben diferir dos columnas.")
    pleno_modo = st.radio("Pleno al 15: elegir el marcador…", ["probable", "valor"], horizontal=True,
                          format_func=lambda x: "más probable" if x == "probable" else "de mejor valor (poco jugado)",
                          help="«Más probable» da más opciones de acertarlo. «De mejor valor» busca marcadores que juega poca gente: "
                               "si aciertas, el bote se reparte entre menos acertantes, pero es menos probable acertarlo.")
    st.markdown('<div class="sec naranja">🗂️ Datos</div>', unsafe_allow_html=True)
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
    if lae_pendiente:
        st.warning("⚠️ **Los porcentajes oficiales de LAE aún no están publicados.** Uso los de la web de Losilla, que son "
                   "menos representativos del público de toda España: el «valor» (★) y el retorno esperado son provisionales. "
                   "Vuelve a abrir la app más tarde, antes de que cierre la jornada, y se actualizan solos.")
    if jornada and jornada["cerrada"]:
        st.warning("Esta jornada figura como **cerrada**: ya no se puede jugar. Cuando se publique la siguiente, "
                   "abre Ajustes y pulsa «Recargar».")
    if not jornada:
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
    st.radio("Porcentajes del público", ["auto", "jug"], horizontal=True, key="fuente_pub",
             format_func=lambda x: "LAE (oficial) si está publicado" if x == "auto" else "Siempre % de la web de Losilla",
             help="LAE publica sus porcentajes más tarde. Mientras no estén, se usan los de la web de Losilla (quinielistas reales).")
    pref = st.session_state["fuente_pub"]

    meta = pd.DataFrame({"Local": [f"Partido {i}" for i in range(1, 15)], "Visit": "", "Dia": "", "Hora": "", "Fecha": pd.NaT})
    nota_pub = ""
    if jornada:
        p = jornada["partidos"].reset_index(drop=True)
        lae = p[["lae_1", "lae_X", "lae_2"]].to_numpy(float)
        jug = p[["jug_1", "jug_X", "jug_2"]].to_numpy(float)
        usar_lae = (pref == "auto") & ~np.isnan(lae).any(axis=1)
        publico = np.where(usar_lae[:, None], lae, jug)
        if not usar_lae.all():
            nota_pub = ("Los % oficiales de LAE aún no están publicados: uso los de la web de Losilla (quinielistas reales)."
                        if not usar_lae.any() else "Algunos partidos usan los % de Losilla porque LAE aún no los publica.")
        mapas = [qs.map_match(a, b, known) for a, b in zip(p.Local_quiniela, p.Visitante_quiniela)]
        meta = pd.DataFrame({"Local": p.Local_quiniela.map(bonito), "Visit": p.Visitante_quiniela.map(bonito),
                             "Dia": p.Dia, "Hora": p.Hora, "Fecha": p.Fecha})
        base = pd.DataFrame({
            "Partido": p.Local_quiniela + " - " + p.Visitante_quiniela,
            "Día": [cuando(f, d, "") for f, d in zip(p.Fecha, p.Dia)], "Hora": p.Hora,
            "Local": [m[0] or SIN_HIST for m in mapas], "Visitante": [m[1] or SIN_HIST for m in mapas],
            "pct_1": publico[:, 0], "pct_X": publico[:, 1], "pct_2": publico[:, 2],
            "mkt_1": p.prob_1.fillna(0), "mkt_X": p.prob_X.fillna(0), "mkt_2": p.prob_2.fillna(0),
        })
    else:
        base = pd.DataFrame({"Partido": [f"Partido {i}" for i in range(1, 15)], "Día": "", "Hora": "",
                             "Local": SIN_HIST, "Visitante": SIN_HIST,
                             "pct_1": 40.0, "pct_X": 30.0, "pct_2": 30.0, "mkt_1": 0.0, "mkt_X": 0.0, "mkt_2": 0.0})
    if nota_pub:
        st.caption("ℹ️ " + nota_pub)
    sin = int(((base.Local == SIN_HIST) | (base.Visitante == SIN_HIST)).sum())
    with st.expander(f"✏️ Revisar y corregir equipos y porcentajes ({sin} sin identificar)"):
        st.caption("Si un equipo no se identificó, elígelo en el desplegable. «Cuotas %» son las probabilidades de las "
                   "casas de apuestas; «1/X/2 %» es lo que juega el público.")
        editada = st.data_editor(
            base, hide_index=True, width="stretch", num_rows="fixed", key="tabla_jornada",
            column_config={
                "Partido": st.column_config.TextColumn(disabled=True),
                "Día": st.column_config.TextColumn(disabled=True), "Hora": st.column_config.TextColumn(disabled=True),
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
            st.write(f"**{bonito(pl['Local_quiniela'])} - {bonito(pl['Visitante_quiniela'])}**")
            if pl["Local_quiniela"] and pl["Visitante_quiniela"]:
                loc0, vis0 = qs.map_match(pl["Local_quiniela"], pl["Visitante_quiniela"], known)
        local15 = st.selectbox("Local", opciones, index=opciones.index(loc0 or SIN_HIST), key="l15")
        visit15 = st.selectbox("Visitante", opciones, index=opciones.index(vis0 or SIN_HIST), key="v15")
        if pl:
            lae_ok = not (np.isnan(pl["lae_H"]).any() or np.isnan(pl["lae_A"]).any())
            hv = pl["lae_H"] if (lae_ok and pref == "auto") else pl["jug_H"]
            av = pl["lae_A"] if (lae_ok and pref == "auto") else pl["jug_A"]
        else:
            hv = av = [25, 25, 25, 25]
        pl_df = st.data_editor(
            pd.DataFrame({"0": [hv[0], av[0]], "1": [hv[1], av[1]], "2": [hv[2], av[2]], "M": [hv[3], av[3]]},
                         index=["Local", "Visitante"]),
            key="tabla_pleno", width="stretch", num_rows="fixed",
            column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=100) for c in ["0", "1", "2", "M"]})

# --------------------------------------------------------------------------------------
# CONTROLES PRINCIPALES
# --------------------------------------------------------------------------------------
with tab_cols:
    st.markdown('<div class="steps"><div class="step s1">① Mira los partidos y su probabilidad (pestaña Partidos)</div>'
                '<div class="step s2">② Elige cuántas columnas y la cobertura</div>'
                '<div class="step s3">③ Copia los signos al boleto</div></div>', unsafe_allow_html=True)
    c_k, c_cob = st.columns(2)
    k = c_k.slider("¿Cuántas columnas?", 1, 30, 4, help="Cada columna cuesta lo que indica Ajustes (0,75 € por defecto).")
    cobertura = c_cob.slider("Cobertura (%)", 0, 100, 50, help=(
        "0 % = las columnas de mayor valor, que suelen parecerse y fallar juntas. Más cobertura = columnas que se "
        "complementan (p. ej. el otro signo en el partido más dudoso) para no fallar todas en el mismo partido.")) / 100
    st.caption(f"Coste: {k} × {precio:.2f} € = **{k * precio:.2f} €**")
    resultados = st.container()

# --------------------------------------------------------------------------------------
# CÁLCULO
# --------------------------------------------------------------------------------------
fechas = pd.to_datetime(meta["Fecha"]).fillna(pd.Timestamp.today().normalize()).reset_index(drop=True)
fixtures = pd.DataFrame({
    "Home": [h if h != SIN_HIST else f"{r.Partido}|L" for h, r in zip(editada.Local, editada.itertuples())],
    "Away": [a if a != SIN_HIST else f"{r.Partido}|V" for a, r in zip(editada.Visitante, editada.itertuples())],
    "Date": fechas.iloc[:len(editada)].to_numpy(),
    "pct_1": editada.pct_1.astype(float), "pct_X": editada.pct_X.astype(float), "pct_2": editada.pct_2.astype(float),
    "mkt_1": editada.mkt_1.astype(float), "mkt_X": editada.mkt_X.astype(float), "mkt_2": editada.mkt_2.astype(float),
})
n = len(editada)
locales = [bonito(x) for x in meta["Local"].iloc[:n]]
visitantes = [bonito(x) for x in meta["Visit"].iloc[:n]]
cuandos = [cuando(f, d, h) for f, d, h in zip(meta["Fecha"].iloc[:n], meta["Dia"].iloc[:n], meta["Hora"].iloc[:n])]

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
                            coste_columna=precio, pleno=pleno_arg, pleno_probable=(pleno_modo == "probable"),
                            cobertura=cobertura)
        error_opt = None
    except ValueError as exc:
        cols, error_opt = None, str(exc)


def comp_partido(h, a, nombre):
    ch, ca = comp_de.get(h), comp_de.get(a)
    if ch and ca:
        return ch if ch == ca else f"{ch}/{ca}"
    return "Femenino (sin datos)" if "(F)" in nombre.upper() else "Sin datos"


comp_part = [comp_partido(h, a, f"{l} {v}") for h, a, l, v in zip(fixtures.Home, fixtures.Away, locales, visitantes)]

# --------------------------------------------------------------------------------------
# CABECERA DE LA JORNADA
# --------------------------------------------------------------------------------------
with hero_box:
    fmin, fmax = (pd.to_datetime(meta["Fecha"]).min(), pd.to_datetime(meta["Fecha"]).max()) if jornada else (pd.NaT, pd.NaT)
    titulo = f"JORNADA {jornada['jornada']}" if jornada and jornada["jornada"] else "JORNADA ACTUAL"
    sub = []
    if jornada and jornada.get("cierre"):
        c = jornada["cierre"]
        sub.append(f"Cierra el {c['dia'].lower()} {c['fecha']:%d/%m}" + (f" a las {c['hora']}" if c["hora"] else ""))
    if pd.notna(fmin):
        sub.append(f"Partidos del {DIAS_ES[fmin.weekday()]} {fmin:%d/%m}" + (f" al {DIAS_ES[fmax.weekday()]} {fmax:%d/%m}" if fmax != fmin else ""))
    pills = (f'<span class="pill tip">{n} partidos + Pleno al 15{tip("La Quiniela: 14 partidos con signo 1, X o 2, y el partido 15 (Pleno al 15) por goles.")}</span>')
    if jornada and jornada.get("premio_max"):
        pills += (f'<span class="pill tip">💰 Premio máx. {euros(jornada["premio_max"])}{tip("Lo que cobraría un único acertante de 14 + Pleno al 15 según la web (incluye el bote).")}</span>')
    for nombre, cnt in pd.Series(comp_part).value_counts().items():
        pills += f'<span class="pill tip">{cnt} {esc(nombre)}{tip(COMPS.get(nombre, "Partidos sin histórico propio: se estiman con las cuotas de las casas o el público."))}</span>'
    st.markdown(f'<div class="hero"><div class="hero-j tip">{titulo}{tip("Número de jornada de La Quiniela, leído de la web.")}</div>'
                f'<div class="hero-sub">{" · ".join(sub) if sub else "Rellena los partidos en la pestaña «Partidos»"}</div>'
                f'<div class="pills">{pills}</div></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# TARJETAS DE PARTIDOS
# --------------------------------------------------------------------------------------
with cards_box:
    st.markdown('<div class="sec azul">Los 14 partidos + el Pleno al 15</div>', unsafe_allow_html=True)
    st.caption("Cada tarjeta: día y hora, competición, de dónde sale la estimación, y la probabilidad de que gane cada equipo "
               "o empate. Debajo, lo que juega el público. ★ = el público infravalora ese signo.")
    html_cards = "".join(partido_html(i + 1, locales[i], visitantes[i], comp_part[i], cuandos[i], Pa[i], S[i],
                                      result.fuente.iloc[i]) for i in range(n))
    if pl and pleno_arg is not None:
        pl_cuando = cuando(pl.get("Fecha"), pl.get("Dia", ""), pl.get("Hora", ""))
        html_cards += pleno_html(bonito(pl["Local_quiniela"]), bonito(pl["Visitante_quiniela"]), pl_cuando,
                                 pleno_arg[0], pleno_arg[1])
    st.markdown(html_cards, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# RESULTADOS
# --------------------------------------------------------------------------------------
with resultados:
    if cols is None:
        st.warning(f"{error_opt}")
    else:
        ev_total = cols.EV_sobre_coste.mean()
        st.markdown('<div class="kpis">' +
                    kpi("k1", "Coste", f"{k * precio:.2f} €", "Lo que pagarías por todas las columnas.") +
                    kpi("k2", "Prob. de 14 aciertos", f"{cols.attrs['p_14'] * 100:.4f} %",
                        "Probabilidad de que alguna de tus columnas acierte los 14 signos.") +
                    kpi("k3", "Prob. de 13 o más", f"{cols.attrs['p_13mas'] * 100:.2f} %",
                        "Probabilidad de acertar al menos 13 signos con alguna columna (simulada con 40.000 jornadas).") +
                    kpi("k4", "Retorno / coste", f"{ev_total:.2f}",
                        "Lo que recuperarías de media por cada euro. 1,00 = quedar en paz. Casi siempre es menor: la Quiniela retiene parte de la recaudación.") +
                    '</div>', unsafe_allow_html=True)
        if ev_total > 1.5:
            st.warning("Un retorno esperado muy por encima de 1 casi siempre indica que el modelo discrepa mucho del público "
                       "(posible sobreconfianza o datos mal casados). Revisa la pestaña «Partidos» y baja el peso del modelo en Ajustes.")
        elif ev_total < 1:
            st.info(f"Con estos supuestos recuperarías de media ~{ev_total * 100:.0f} % de lo que pagas. "
                    "Es lo habitual en una lotería con retención: las columnas son las mejores posibles, pero no hay garantía.")

        st.markdown('<div class="sec">Tus columnas</div>', unsafe_allow_html=True)
        st.caption("🔵 1 (gana el local) · ⚪ X (empate) · 🟠 2 (gana el visitante) · **contorno amarillo** = apuesta contra el favorito. "
                   "Pasa el ratón por una ficha para ver el partido y la probabilidad.")
        html = ""
        for i, r in enumerate(cols.itertuples(), 1):
            det = f"{int(r.P_conjunta ** -1):,}".replace(",", ".")
            pl_lab = getattr(r, "pleno", None)
            ayuda_pl = ""
            if pl_lab and pleno_arg is not None:
                ia, ib = "012M".index(pl_lab[0]), "012M".index(pl_lab[2])
                ayuda_pl = (f"Pleno al 15: {pl_lab} (goles del local – goles del visitante; M = 3 o más). Probabilidad estimada "
                            f"{pct(pleno_arg[0][ia, ib], 1)}; lo juega el {pct(pleno_arg[1][ia, ib], 1)} del público"
                            f"{' (poco jugado: si aciertas, menos gente comparte el bote)' if pleno_arg[1][ia, ib] < 0.03 else ''}.")
            html += columna_html(i, r.columna, fav, pl_lab, det, locales, visitantes, Pa, ayuda_pl)
        st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div class="sec verde">Para copiar al boleto</div>', unsafe_allow_html=True)
        filas = [f"{i:02d}  " + " ".join(c) + (f"   | 15: {pl_}" if "pleno" in cols else "")
                 for i, (c, pl_) in enumerate(zip(cols.columna, cols.get("pleno", [""] * len(cols))), 1)]
        st.code("\n".join(filas), language="text")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ TXT", qo.to_text(cols, result), "quiniela_optimizada.txt", width="stretch")
        d2.download_button("⬇️ CSV", cols.to_csv(index=False), "quiniela_optimizada.csv", width="stretch")

        with st.expander("🧩 Reparto de signos por partido"):
            rep = pd.DataFrame(cols.attrs["reparto"], columns=SIGNOS)
            rep.insert(0, "Partido", [f"{i + 1}. {l} - {v}" for i, (l, v) in enumerate(zip(locales, visitantes))])
            st.dataframe(rep, hide_index=True, width="stretch")
            st.caption("Cuántas de tus columnas llevan cada signo. Si un partido tiene todas con el mismo signo y ese signo falla, "
                       "todas fallan a la vez: sube la cobertura para repartirlo.")
        with st.expander("🔎 Valor de cada signo (público frente a estimación)"):
            st.dataframe(qo.sign_table(result.assign(Home=[f"{l} - {v}" for l, v in zip(locales, visitantes)], Away=""),
                                       Pa, S, min_signo), hide_index=True, width="stretch")
            st.caption("ratio = probabilidad estimada ÷ % del público. ★ = ratio ≥ 1,3 con probabilidad suficiente.")

# --------------------------------------------------------------------------------------
# AYUDA
# --------------------------------------------------------------------------------------
with tab_ayuda:
    st.markdown("""
#### Cómo usarla en 3 pasos
1. **Abre la app** el día de la jornada: lee sola los partidos, su día y hora, los % del público y las cuotas de las casas.
2. **Elige cuántas columnas** quieres (pestaña «Mis columnas») y cuánta cobertura.
3. **Copia los signos** al boleto. Si un equipo no se identificó, corrígelo en «Partidos».

#### Qué significa cada cosa
- **Probabilidad estimada** (barras de «Partidos»): mezcla tu modelo estadístico con las cuotas de las casas de apuestas.
  Si un equipo no tiene histórico, usa solo las cuotas; si tampoco hay cuotas, el % del público (sin ventaja).
- **Lo que juega el público**: % de quinielistas que pone cada signo (LAE si ya está publicado; si no, los de la web de Losilla).
- **★ Valor**: el público juega ese signo menos de lo que debería según la estimación. Si aciertas, el premio se reparte entre menos gente.
- **Cobertura**: reparte el riesgo. Sin ella, las columnas se parecen y fallan juntas (como si todas llevan el mismo 1 en el partido 1).
- **Contorno amarillo**: esa columna apuesta contra el favorito en ese partido.
- **Pleno al 15**: goles de cada equipo (0, 1, 2 o M = 3 o más). Solo cuenta para el bote.
- **Retorno esperado / coste**: cuánto recuperarías de media por cada euro. 1,00 sería quedar en paz; casi siempre es menor.

#### De dónde salen los datos
Resultados de Primera y Segunda (football-data.co.uk), Liga F (Wikipedia), selecciones (resultados internacionales en GitHub),
jornada, horarios y porcentajes (eduardolosilla.es). Todo gratuito; si una fuente falla, la app te lo avisa y sigue con lo que tenga.

#### Límites que conviene conocer
- El modelo **no está validado con jornadas pasadas** y las bajas/fatiga están simuladas. Úsalo como una ayuda más.
- Los partidos de competiciones sin histórico (p. ej. Champions femenina) solo se estiman con cuotas o público: no hay ventaja posible.
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
