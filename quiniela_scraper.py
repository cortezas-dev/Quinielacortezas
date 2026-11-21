"""
quiniela_scraper.py  ·  Obtiene la jornada de La Quiniela y los % de apuestas automáticamente

FUENTE: https://www.eduardolosilla.es/  (portada de Quiniela). Para cada partido publica:
        % Jugados (su peña) · % LAE (porcentajes de Loterías y Apuestas del Estado) ·
        % Probables (cuotas de casas de apuestas) · y el estado del juego (abierto/cerrado).
        Usamos % LAE como "escrutinio popular".

IMPORTANTE
  · Es web scraping de una página privada: puede dejar de funcionar si cambian su diseño,
    y conviene revisar sus condiciones de uso. Aquí se hace UNA petición cuando pulsas el
    botón (y se cachea unos minutos). Si falla, la app te deja rellenar la tabla a mano.
  · No hay acceso a ese sitio desde donde desarrollé el código: el parser está probado con
    el texto de la página tal como se ve, no con una descarga real.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from quiniela_selecciones import is_seleccion_name, map_seleccion

URL = "https://www.eduardolosilla.es/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuinielaPersonalApp/1.0)", "Accept-Language": "es-ES,es;q=0.9"}


class ScrapeError(RuntimeError):
    """No se pudo leer la jornada (sin red, cambio de diseño, página dinámica...)."""


# ======================================================================================
# 1) DESCARGA Y PARSEO
# ======================================================================================
def _lines_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in soup.get_text("\n").split("\n")]
    return [ln for ln in lines if ln]


def _money(text: str, pattern: str) -> float | None:
    m = re.search(pattern + r"\s*:?\s*([\d\.]+(?:,\d+)?)\s*€", text, flags=re.I | re.S)
    return float(m.group(1).replace(".", "").replace(",", ".")) if m else None


_DIAS = {"LUN": 0, "MAR": 1, "MIE": 2, "JUE": 3, "VIE": 4, "SAB": 5, "DOM": 6}
_DASH = {"-", "–", "—"}


def _ascii_up(x: str) -> str:
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().upper().strip()


def _read_values(lines: list[str], limit: int) -> list[float]:
    """
    Lee hasta `limit` valores consecutivos: números (%) o '-' (= todavía no disponible -> NaN).
    La web pone '-' en los % de LAE antes de que se publiquen y en las cuotas de partidos sin cuotas.
    """
    out: list[float] = []
    for ln in lines:
        for tok in re.split(r"\s+", ln.replace("%", " ").strip()):
            if re.fullmatch(r"\d{1,3}", tok):
                out.append(float(tok))
            elif tok in _DASH:
                out.append(np.nan)
            if len(out) >= limit:
                return out
    return out


def _parse_cierre(text: str) -> dict | None:
    """'CIERRE MARTES 17/11' + '19:45H para Particulares' -> fecha y hora de cierre."""
    m = re.search(r"CIERRE\s+([A-ZÁÉÍÓÚ]+)\s+(\d{1,2})/(\d{1,2})", text)
    if not m:
        return None
    hora = re.search(r"(\d{1,2}:\d{2})\s*H\s+para\s+Particulares", text, flags=re.I)
    today = pd.Timestamp.today().normalize()
    try:
        fecha = pd.Timestamp(year=today.year, month=int(m.group(3)), day=int(m.group(2)))
    except ValueError:
        return None
    if fecha < today - pd.Timedelta(days=200):
        fecha = fecha + pd.DateOffset(years=1)
    return {"dia": m.group(1).title(), "fecha": fecha, "hora": hora.group(1) if hora else None}


def _fecha_partido(dia: str, cierre: dict | None):
    """Día de la semana ('MAR') -> fecha real, buscando desde el día de cierre de la jornada."""
    if not dia or not cierre or dia not in _DIAS:
        return pd.NaT
    for k in range(8):
        d = cierre["fecha"] + pd.Timedelta(days=k)
        if d.weekday() == _DIAS[dia]:
            return d
    return pd.NaT


def parse_lines(lines: list[str]) -> dict:
    """Convierte las líneas de texto de la página en un diccionario con jornada, partidos, pleno y cierre."""
    # Une "7" + "MÁLAGA - RACING S." si el número de partido viene en una línea aparte
    merged: list[str] = []
    i = 0
    while i < len(lines):
        if re.fullmatch(r"\d{1,2}", lines[i]) and i + 1 < len(lines) and re.search(r"\S\s-\s\S", lines[i + 1]) \
                and not re.search(r"\d{1,2}:\d{2}", lines[i + 1]) and not lines[i + 1][0].isdigit():
            merged.append(f"{lines[i]} {lines[i + 1]}")
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    lines = merged
    text = "\n".join(lines)

    header = re.compile(r"^(\d{1,2})\s+(.+?)\s+-\s+(.+)$")
    idx = [(k, header.match(ln)) for k, ln in enumerate(lines)]
    idx = [(k, m) for k, m in idx if m and 1 <= int(m.group(1)) <= 14
           and not re.search(r"\d{1,2}:\d{2}", m.group(0))]
    # Nos quedamos con la primera secuencia consecutiva 1..14
    seq, expected = [], 1
    for k, m in idx:
        if int(m.group(1)) == expected:
            seq.append((k, m))
            expected += 1
    if len(seq) < 8:
        raise ScrapeError("No encontré la tabla de partidos (¿la página carga los datos con JavaScript "
                          "o ha cambiado de diseño?).")

    cierre = _parse_cierre(text)
    rows = []
    for pos, (k, m) in enumerate(seq):
        end = seq[pos + 1][0] if pos + 1 < len(seq) else min(k + 45, len(lines))
        block = lines[k + 1:end]
        start = None
        for j, ln in enumerate(block):                      # marcador de los botones "1 X 2"
            if ln.replace(" ", "") == "1X2":
                start = j + 1
                break
            if ln == "1" and block[j + 1:j + 3] == ["X", "2"]:
                start = j + 3
                break
        if start is None:
            continue
        nums = _read_values(block[start:], 12)
        if len(nums) < 12:
            continue
        previo = [_ascii_up(x) for x in block[:start]]
        hora = next((x for x in previo if re.fullmatch(r"\d{1,2}:\d{2}", x)), "")
        dia = next((x for x in previo if x in _DIAS), "")
        rows.append({
            "Nº": int(m.group(1)), "Local_quiniela": m.group(2).strip(), "Visitante_quiniela": m.group(3).strip(),
            "Dia": dia, "Hora": hora, "Fecha": _fecha_partido(dia, cierre),
            "jug_1": nums[0], "jug_X": nums[1], "jug_2": nums[2],
            "lae_1": nums[3], "lae_X": nums[4], "lae_2": nums[5],
            "prob_1": nums[6], "prob_X": nums[7], "prob_2": nums[8],
        })
    if len(rows) < 8:
        raise ScrapeError("Encontré los partidos pero no los porcentajes (cambio de formato).")

    jornada = re.search(r"JORNADA\s+(\d+)", text)
    return {
        "jornada": int(jornada.group(1)) if jornada else None,
        "cerrada": bool(re.search(r"JUEGO\s+CERRADO", text, flags=re.I)),
        "cierre": cierre,
        "bote": _money(text, r"\bBote"),
        "premio_max": _money(text, r"puede\s+ganar"),
        "partidos": pd.DataFrame(rows),
        "pleno": _parse_pleno(lines, seq[-1][0], cierre),
    }


def _parse_pleno(lines: list[str], after: int, cierre: dict | None = None) -> dict | None:
    """
    Partido 15 (Pleno al 15): cada equipo marca 0, 1, 2 o M (3 o más). La página da, por grupo
    (jugados, LAE, probables, jugados), 4 % del local y 4 % del visitante. '-' = no disponible (NaN).
    Devuelve None si no se encuentra (la app permite rellenarlo a mano).
    """
    start = next((k for k in range(after + 1, len(lines))
                  if re.match(r"^15(\s+\S.*)?$", lines[k]) and not re.search(r"\d{1,2}:\d{2}", lines[k])), None)
    if start is None:
        return None
    block = lines[start:start + 80]
    names = []
    first = re.match(r"^15\s*(.*)$", block[0]).group(1).strip()
    if first:
        names += [p.strip() for p in re.split(r"\s+-\s+", first)]
    for ln in block[1:]:
        if len(names) >= 2:
            break
        if _ascii_up(ln) in _DIAS or re.search(r"\d{1,2}:\d{2}", ln) or re.fullmatch(r"[\d\sM]+", ln):
            break
        names.append(ln)
    previo = [_ascii_up(x) for x in block[:12]]
    dia = next((x for x in previo if x in _DIAS), "")
    hora = next((x for x in previo if re.fullmatch(r"\d{1,2}:\d{2}", x)), "")
    end = None                                       # último marcador "0 1 2 M"
    for j, ln in enumerate(block):
        if ln.replace(" ", "") == "012M":
            end = j
        elif ln == "M" and block[max(0, j - 3):j] == ["0", "1", "2"]:
            end = j
    if end is None:
        return None
    nums = _read_values(block[end + 1:], 32)
    if len(nums) < 16:
        return None
    return {
        "Local_quiniela": names[0].strip() if names else "", "Visitante_quiniela": names[1].strip() if len(names) > 1 else "",
        "Dia": dia, "Hora": hora, "Fecha": _fecha_partido(dia, cierre),
        "jug_H": nums[0:4], "jug_A": nums[4:8], "lae_H": nums[8:12], "lae_A": nums[12:16],
        "prob_H": nums[16:20] if len(nums) >= 24 else [np.nan] * 4,
        "prob_A": nums[20:24] if len(nums) >= 24 else [np.nan] * 4,
    }


def fetch_jornada(url: str = URL, timeout: int = 20) -> dict:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ScrapeError(f"No se pudo descargar la página: {exc}") from exc
    return parse_lines(_lines_from_html(resp.text))


# ======================================================================================
# 2) NOMBRES DE EQUIPO: Quiniela  ->  football-data.co.uk
# ======================================================================================
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


# nombre normalizado de la Quiniela -> posibles nombres en football-data (el primero que exista gana)
ALIASES = {
    "at madrid": ["Ath Madrid"], "atl madrid": ["Ath Madrid"],
    "ath club": ["Ath Bilbao"], "athletic": ["Ath Bilbao"],
    "r madrid": ["Real Madrid"], "r sociedad": ["Sociedad"], "real sociedad": ["Sociedad"],
    "rayo": ["Vallecano"], "rayo vallecano": ["Vallecano"],
    "r oviedo": ["Oviedo"], "real oviedo": ["Oviedo"],
    "racing s": ["Santander", "Racing Santander"], "r santander": ["Santander", "Racing Santander"],
    "sporting": ["Sp Gijon", "Sporting Gijon"], "espanyol": ["Espanol"], "alaves": ["Alaves"],
    "celta": ["Celta"], "betis": ["Betis"], "r valladolid": ["Valladolid"], "valladolid": ["Valladolid"],
    "las palmas": ["Las Palmas"], "deportivo": ["La Coruna", "Deportivo"], "r zaragoza": ["Zaragoza"],
    "cadiz": ["Cadiz"], "malaga": ["Malaga"], "almeria": ["Almeria"], "castellon": ["Castellon"],
    "leganes": ["Leganes"], "cordoba": ["Cordoba"], "mirandes": ["Mirandes"], "huesca": ["Huesca"],
}


_NOISE = {"f", "femenino", "fem", "women", "w", "cf", "fc", "cff", "club", "de", "futbol", "cd", "ud", "sad", "rcd"}


def _tokens(name: str) -> list[str]:
    return [t for t in _norm(name.replace("F:", "")).split() if t not in _NOISE]


def _map_fem(name: str, pool: list[str]) -> str | None:
    """Equipo femenino: compara palabras (admite abreviaturas: 'ath' ~ 'athletic', 'r' ~ 'real')."""
    q = _tokens(name)
    if not q or not pool:
        return None
    best, best_key = None, (0.0, 0)
    for cand in pool:
        c = _tokens(cand)
        hits = sum(any(ct.startswith(qt) for ct in c) for qt in q)
        score = hits / len(q)
        key = (score, -(len(c) - hits))             # a igualdad, el nombre más ajustado
        if score >= 0.99 and key > best_key:
            best, best_key = cand, key
    return best


_CLUB_PREFIX = {"cd", "ad", "fc", "ce", "ud", "cf", "sd", "rcd", "rc", "cp", "sad"}

ALIASES.update({
    "r sociedad b": ["Sociedad B"], "real sociedad b": ["Sociedad B"], "sociedad b": ["Sociedad B"],
    "sporting gijon": ["Sp Gijon", "Sporting Gijon"], "andorra": ["Andorra"], "castellon": ["Castellon"],
    "ceuta": ["Ceuta"], "eldense": ["Eldense"], "sabadell": ["Sabadell"], "oviedo": ["Oviedo"],
    "tenerife": ["Tenerife"], "granada": ["Granada"], "burgos": ["Burgos"], "albacete": ["Albacete"],
    "zaragoza": ["Zaragoza"], "cultural leonesa": ["Cultural Leonesa"], "r racing": ["Santander"],
    "racing": ["Santander", "Racing Santander"],
})


def _strip_prefix(n: str) -> str:
    return " ".join(t for t in n.split() if t not in _CLUB_PREFIX) or n


def map_team(name: str, known: list[str]) -> str | None:
    """
    Traduce el nombre de la Quiniela al del histórico:
      · "(F)" / "FEM." -> equipos femeninos ('F:...')
      · clubes de Primera y Segunda (admite CD, AD, FC, CE..., alias y nombres parecidos)
      · selecciones ('S:...'), con el nombre en español (BÉLGICA -> S:Belgium)
    Devuelve None si no hay histórico: el modelo usará las cuotas de las casas o el % del público.
    """
    up = name.upper().replace(".", " ")
    if "(F)" in up or re.search(r"\bFEM\b", up):
        return _map_fem(name, [k for k in known if k.startswith("F:")])
    clubs = [k for k in known if not k.startswith(("F:", "S:"))]
    n = re.sub(r"\s+", " ", _norm(name))
    n2 = _strip_prefix(n)
    by_norm: dict[str, str] = {}
    for k in clubs:
        by_norm[_norm(k)] = k
        by_norm.setdefault(_strip_prefix(_norm(k)), k)
    for key in (n, n2):
        for cand in ALIASES.get(key, []):
            if cand in clubs:
                return cand
    for key in (n, n2):
        if key in by_norm:
            return by_norm[key]
    sel = map_seleccion(name, known)
    if sel:
        return sel
    close = difflib.get_close_matches(n2, list(by_norm), n=1, cutoff=0.8)
    return by_norm[close[0]] if close else None


def map_match(local: str, visitante: str, known: list[str]) -> tuple[str | None, str | None]:
    """Mapea un partido entero. Si un equipo es selección, el otro también (p. ej. ANDORRA - ESPAÑA)."""
    a, b = map_team(local, known), map_team(visitante, known)
    if (a or "").startswith("S:") ^ (b or "").startswith("S:"):
        if not (a or "").startswith("S:") and is_seleccion_name(local):
            a = map_seleccion(local, known) or a
        if not (b or "").startswith("S:") and is_seleccion_name(visitante):
            b = map_seleccion(visitante, known) or b
    return a, b
