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

import pandas as pd
import requests
from bs4 import BeautifulSoup

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


def parse_lines(lines: list[str]) -> dict:
    """Convierte las líneas de texto de la página en un diccionario con jornada y partidos."""
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
        nums = []
        for ln in block[start:]:
            for tok in re.split(r"\s+", ln.replace("%", " ")):
                if re.fullmatch(r"\d{1,3}", tok):
                    nums.append(int(tok))
            if len(nums) >= 12:
                break
        if len(nums) < 12:
            continue
        hora = next((x for x in block[:start] if re.fullmatch(r"\d{1,2}:\d{2}", x)), "")
        dia = next((x for x in block[:start] if x in {"SAB", "DOM", "VIE", "LUN", "MAR", "MIE", "JUE"}), "")
        rows.append({
            "Nº": int(m.group(1)), "Local_quiniela": m.group(2).strip(), "Visitante_quiniela": m.group(3).strip(),
            "Dia": dia, "Hora": hora,
            "jug_1": nums[0], "jug_X": nums[1], "jug_2": nums[2],
            "lae_1": nums[3], "lae_X": nums[4], "lae_2": nums[5],
            "prob_1": nums[6], "prob_X": nums[7], "prob_2": nums[8],
        })
    if len(rows) < 8:
        raise ScrapeError("Encontré los partidos pero no los porcentajes (cambio de formato).")

    jornada = re.search(r"JORNADA\s+(\d+)", text)
    pleno = _parse_pleno(lines, seq[-1][0])
    return {
        "jornada": int(jornada.group(1)) if jornada else None,
        "cerrada": bool(re.search(r"JUEGO\s+CERRADO", text, flags=re.I)),
        "bote": _money(text, r"\bBote"),
        "premio_max": _money(text, r"puede\s+ganar"),
        "partidos": pd.DataFrame(rows),
        "pleno": pleno,
    }


_DIAS = {"SAB", "DOM", "VIE", "LUN", "MAR", "MIE", "JUE"}


def _parse_pleno(lines: list[str], after: int) -> dict | None:
    """
    Partido 15 (Pleno al 15): cada equipo marca 0, 1, 2 o M (3 o más). La página da, por
    grupo (jugados, LAE, probables, jugados), 4 % del local y 4 % del visitante.
    Devuelve None si no se encuentra (la app permite rellenarlo a mano).
    """
    start = next((k for k in range(after + 1, len(lines))
                  if re.match(r"^15(\s+\S.*)?$", lines[k]) and not re.search(r"\d{1,2}:\d{2}", lines[k])), None)
    if start is None:
        return None
    block = lines[start:start + 70]
    names = []
    first = re.match(r"^15\s*(.*)$", block[0]).group(1).strip()
    if first:
        names += [p.strip() for p in re.split(r"\s+-\s+", first)]
    for ln in block[1:]:
        if len(names) >= 2:
            break
        if ln in _DIAS or re.search(r"\d{1,2}:\d{2}", ln) or re.fullmatch(r"[\d\sM]+", ln):
            break
        names.append(ln)
    # último marcador "0 1 2 M" y los 32 números que le siguen
    end = None
    for j, ln in enumerate(block):
        if ln.replace(" ", "") == "012M":
            end = j
        elif ln == "M" and block[max(0, j - 3):j] == ["0", "1", "2"]:
            end = j
    if end is None:
        return None
    nums = []
    for ln in block[end + 1:]:
        for tok in re.split(r"\s+", ln.replace("%", " ")):
            if re.fullmatch(r"\d{1,3}", tok):
                nums.append(int(tok))
        if len(nums) >= 32:
            break
    if len(nums) < 16:
        return None
    return {
        "Local_quiniela": names[0] if names else "", "Visitante_quiniela": names[1] if len(names) > 1 else "",
        "jug_H": nums[0:4], "jug_A": nums[4:8], "lae_H": nums[8:12], "lae_A": nums[12:16],
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


def map_team(name: str, known: list[str]) -> str | None:
    """
    Traduce el nombre de la Quiniela al del histórico. Los equipos femeninos "(F)" se buscan entre
    los equipos 'F:...' del histórico extra. Devuelve None si no hay histórico: el modelo usará
    las cuotas del mercado o el % del público.
    """
    fem = "(F)" in name.upper()
    if fem:
        return _map_fem(name, [k for k in known if k.startswith("F:")])
    known = [k for k in known if not k.startswith("F:")]
    n = _norm(name)
    for cand in ALIASES.get(n, []):
        if cand in known:
            return cand
    by_norm = {_norm(k): k for k in known}
    if n in by_norm:
        return by_norm[n]
    close = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.8)
    return by_norm[close[0]] if close else None
