"""
quiniela_selecciones.py  ·  Histórico GRATUITO de selecciones nacionales

Fuente: https://github.com/martj42/international_results  (resultados de partidos internacionales,
archivo results.csv, actualizado con frecuencia). Se usan los partidos de los últimos años jugados
en campo de uno de los dos equipos (los neutrales se descartan: en ellos "local" es arbitrario).

Los equipos se guardan como 'S:Spain', 'S:Belgium'... para no mezclarlos con clubes.
"""
from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Nombre de la Quiniela (normalizado, sin tildes) -> nombre del dataset (inglés)
ES2EN = {
    "espana": "Spain", "inglaterra": "England", "francia": "France", "alemania": "Germany", "italia": "Italy",
    "portugal": "Portugal", "belgica": "Belgium", "paises bajos": "Netherlands", "holanda": "Netherlands",
    "croacia": "Croatia", "suiza": "Switzerland", "dinamarca": "Denmark", "suecia": "Sweden", "noruega": "Norway",
    "polonia": "Poland", "austria": "Austria", "chequia": "Czech Republic", "republica checa": "Czech Republic",
    "eslovaquia": "Slovakia", "hungria": "Hungary", "serbia": "Serbia", "turquia": "Turkey", "grecia": "Greece",
    "escocia": "Scotland", "gales": "Wales", "irlanda": "Republic of Ireland", "irlanda del norte": "Northern Ireland",
    "ucrania": "Ukraine", "rumania": "Romania", "bulgaria": "Bulgaria", "albania": "Albania",
    "bosnia": "Bosnia and Herzegovina", "bosnia herzegovina": "Bosnia and Herzegovina", "bosnia y herzegovina": "Bosnia and Herzegovina",
    "rep irlanda": "Republic of Ireland", "r irlanda": "Republic of Ireland", "irlanda n": "Northern Ireland",
    "irl norte": "Northern Ireland", "rep checa": "Czech Republic", "r checa": "Czech Republic", "macedonia n": "North Macedonia",
    "macedonia": "North Macedonia", "eslovenia": "Slovenia", "islandia": "Iceland", "finlandia": "Finland",
    "israel": "Israel", "georgia": "Georgia", "armenia": "Armenia", "kosovo": "Kosovo", "montenegro": "Montenegro",
    "macedonia del norte": "North Macedonia", "luxemburgo": "Luxembourg", "chipre": "Cyprus", "estonia": "Estonia",
    "letonia": "Latvia", "lituania": "Lithuania", "rusia": "Russia", "andorra": "Andorra", "malta": "Malta",
    "gibraltar": "Gibraltar", "kazajistan": "Kazakhstan", "azerbaiyan": "Azerbaijan", "bielorrusia": "Belarus",
    "moldavia": "Moldova", "islas feroe": "Faroe Islands", "liechtenstein": "Liechtenstein", "san marino": "San Marino",
    "argentina": "Argentina", "brasil": "Brazil", "uruguay": "Uruguay", "colombia": "Colombia", "chile": "Chile",
    "peru": "Peru", "ecuador": "Ecuador", "paraguay": "Paraguay", "venezuela": "Venezuela", "bolivia": "Bolivia",
    "mexico": "Mexico", "estados unidos": "United States", "eeuu": "United States", "canada": "Canada",
    "costa rica": "Costa Rica", "panama": "Panama", "jamaica": "Jamaica", "honduras": "Honduras",
    "el salvador": "El Salvador", "guatemala": "Guatemala", "haiti": "Haiti", "cuba": "Cuba",
    "marruecos": "Morocco", "senegal": "Senegal", "nigeria": "Nigeria", "egipto": "Egypt", "argelia": "Algeria",
    "tunez": "Tunisia", "camerun": "Cameroon", "ghana": "Ghana", "costa de marfil": "Ivory Coast",
    "sudafrica": "South Africa", "mali": "Mali", "burkina faso": "Burkina Faso", "cabo verde": "Cape Verde",
    "guinea ecuatorial": "Equatorial Guinea", "japon": "Japan", "corea del sur": "South Korea", "iran": "Iran",
    "arabia saudi": "Saudi Arabia", "arabia saudita": "Saudi Arabia", "qatar": "Qatar", "australia": "Australia",
    "nueva zelanda": "New Zealand", "china": "China PR", "irak": "Iraq", "uzbekistan": "Uzbekistan",
    "jordania": "Jordan", "emiratos arabes": "United Arab Emirates", "india": "India",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def map_seleccion(name: str, known: list[str]) -> str | None:
    """Nombre de la Quiniela ('BÉLGICA') -> nombre en el histórico ('S:Belgium'), o None."""
    n = re.sub(r"\s+", " ", _norm(name))
    en = ES2EN.get(n)
    if en and f"S:{en}" in known:
        return f"S:{en}"
    return None


def is_seleccion_name(name: str) -> bool:
    return re.sub(r"\s+", " ", _norm(name)) in ES2EN


def load_selecciones(years: float = 3.0, solo_no_neutrales: bool = True, timeout: int = 40) -> pd.DataFrame:
    """Partidos internacionales recientes en el formato interno del modelo (equipos como 'S:Nombre')."""
    resp = requests.get(URL, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"])
    raw = raw[raw["date"] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=int(365 * years))]
    if solo_no_neutrales:
        raw = raw[~raw["neutral"].astype(str).str.upper().eq("TRUE")]
    out = pd.DataFrame({
        "Date": raw["date"], "Home": "S:" + raw["home_team"].astype(str), "Away": "S:" + raw["away_team"].astype(str),
        "HG": raw["home_score"].astype(int), "AG": raw["away_score"].astype(int),
    }).reset_index(drop=True)
    out["Result"] = np.select([out.HG > out.AG, out.HG == out.AG], [0, 1], default=2)
    out[["mkt_p1", "mkt_px", "mkt_p2"]] = np.nan
    out["Liga"] = "Selección"
    return out
