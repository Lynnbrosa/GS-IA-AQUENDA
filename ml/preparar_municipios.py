"""Monta a tabela de "desertos de cuidado": uma linha por município do estado de São Paulo.

    python ml/preparar_municipios.py

Junta três fontes públicas:
- SINAN (DATASUS): notificações de violência interpessoal de 2022 a 2024, pelo município de residência
  da vítima, no total e só com vítima LGBTQIAPN+. Usa os .dbc que o preparar_dados.py já baixou.
- CNES (DATASUS), competência dez/2024: onde há CAPS, serviço habilitado no processo transexualizador
  e serviço especializado de atenção às pessoas em situação de violência sexual.
- IBGE: população e rendimento domiciliar per capita do Censo 2022 (SIDRA) e a malha municipal
  (para o centro de cada município e as distâncias).

Saída: ml/dados/municipios_sp_desertos_de_cuidado.csv
"""

from __future__ import annotations

import gzip
import json
import math
import urllib.request
from pathlib import Path

import pandas as pd
from dbfread import DBF
from pyreaddbc import dbc2dbf

from preparar_dados import BRUTOS, IDENTIDADE_TRANS, ORIENTACAO_LGB, baixar

PASTA = Path(__file__).resolve().parent / "dados"
SAIDA = PASTA / "municipios_sp_desertos_de_cuidado.csv"
ANOS_SINAN = (2022, 2023, 2024)
CNES = "ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/{tipo}/{tipo}SP2412.dbc"
SIDRA = "https://apisidra.ibge.gov.br/values/t/{t}/n6/in%20n3%2035/v/{v}/p/{p}"
MALHA = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&intrarregiao=municipio&qualidade=minima"

CAPS = "70"  # TP_UNID: Centro de Atenção Psicossocial
TRANS = {"3001", "3002", "3003"}  # habilitações do processo transexualizador (HB, SGRUPHAB)
VIOLENCIA_SEXUAL = "165"  # serviço especializado: atenção às pessoas em situação de violência sexual (SR, SERV_ESP)
CACHE_SINAN = BRUTOS / "sinan_por_municipio_2022_2024.csv"  # a leitura do SINAN leva minutos


def _json(url: str):
    with urllib.request.urlopen(url, timeout=300) as r:
        corpo = r.read()
    if corpo[:2] == b"\x1f\x8b":  # a API de malhas do IBGE responde em gzip mesmo sem pedir
        corpo = gzip.decompress(corpo)
    return json.loads(corpo)


def _dbf(dbc: Path) -> Path:
    dbf = BRUTOS / "dbf" / dbc.with_suffix(".dbf").name
    dbf.parent.mkdir(parents=True, exist_ok=True)
    if not dbf.exists():
        dbc2dbf(str(dbc), str(dbf))
    return dbf


def violencia_por_municipio() -> pd.DataFrame:
    """Notificações de violência interpessoal (sem lesão autoprovocada) por município de residência em SP."""
    if CACHE_SINAN.exists():
        return pd.read_csv(CACHE_SINAN, dtype={"cod6": str})
    total: dict[str, int] = {}
    lgbt: dict[str, int] = {}
    for ano in ANOS_SINAN:
        for r in DBF(_dbf(baixar(ano)), encoding="latin-1", load=False, char_decode_errors="replace"):
            mun = str(r.get("ID_MN_RESI") or "")
            if not mun.startswith("35") or str(r.get("LES_AUTOP")) == "1":
                continue
            total[mun] = total.get(mun, 0) + 1
            if str(r.get("ORIENT_SEX")) in ORIENTACAO_LGB or str(r.get("IDENT_GEN")) in IDENTIDADE_TRANS:
                lgbt[mun] = lgbt.get(mun, 0) + 1
        print(f"SINAN {ano} lido")
    df = pd.DataFrame({"cod6": list(total), "violencia_total": list(total.values()),
                       "violencia_lgbt": [lgbt.get(m, 0) for m in total]})
    df.to_csv(CACHE_SINAN, index=False)
    return df


def servicos_por_municipio() -> pd.DataFrame:
    def baixar_cnes(tipo: str) -> Path:
        destino = BRUTOS / f"{tipo}SP2412.dbc"
        if not destino.exists():
            print(f"baixando {destino.name}…")
            urllib.request.urlretrieve(CNES.format(tipo=tipo), destino)
        return destino

    caps: set[str] = set()
    for r in DBF(_dbf(baixar_cnes("ST")), encoding="latin-1", load=False):
        if str(r.get("TP_UNID")) == CAPS:
            caps.add(str(r["CODUFMUN"]))
    trans = {str(r["CODUFMUN"]) for r in DBF(_dbf(baixar_cnes("HB")), encoding="latin-1", load=False)
             if str(r.get("SGRUPHAB")) in TRANS}
    vsexual = {str(r["CODUFMUN"]) for r in DBF(_dbf(baixar_cnes("SR")), encoding="latin-1", load=False)
               if str(r.get("SERV_ESP")) == VIOLENCIA_SEXUAL}
    return pd.DataFrame({"cod6": sorted(caps | trans | vsexual)}).assign(
        tem_caps=lambda d: d.cod6.isin(caps), tem_trans=lambda d: d.cod6.isin(trans),
        tem_violencia_sexual=lambda d: d.cod6.isin(vsexual))


def ibge() -> pd.DataFrame:
    def sidra(t: int, v: int, p: str, nome: str) -> pd.DataFrame:
        linhas = _json(SIDRA.format(t=t, v=v, p=p))[1:]
        return pd.DataFrame({"cod7": [x["D1C"] for x in linhas], "municipio": [x["D1N"].removesuffix(" - SP") for x in linhas],
                             nome: pd.to_numeric([x["V"] for x in linhas], errors="coerce")})

    pop = sidra(4709, 93, "last", "populacao")
    renda = sidra(10295, 13431, "all", "renda_per_capita").drop(columns="municipio")
    centros = []
    for f in _json(MALHA)["features"]:
        geo = f["geometry"]
        aneis = [p[0] for p in geo["coordinates"]] if geo["type"] == "MultiPolygon" else [geo["coordinates"][0]]
        anel = max(aneis, key=len)  # o maior pedaço do município
        lon, lat = _centroide(anel)
        centros.append({"cod7": f["properties"]["codarea"], "lat": lat, "lon": lon})
    return pop.merge(renda, on="cod7").merge(pd.DataFrame(centros), on="cod7")


def _centroide(anel: list[list[float]]) -> tuple[float, float]:
    """Centroide de um polígono simples (fórmula do shoelace)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(anel, anel[1:] + anel[:1]):
        c = x0 * y1 - x1 * y0
        a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    a /= 2
    return cx / (6 * a), cy / (6 * a)


def _km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def distancia_ate(base: pd.DataFrame, tem: pd.Series) -> list[float]:
    """Distância em linha reta do centro de cada município ao centro do município mais próximo com o serviço (0 se tem)."""
    alvos = base[tem.to_numpy()]
    return [0.0 if t else min(_km(r.lat, r.lon, a.lat, a.lon) for a in alvos.itertuples())
            for r, t in zip(base.itertuples(), tem)]


def main() -> None:
    base = ibge()
    base["cod6"] = base["cod7"].str[:6]
    base = base.merge(servicos_por_municipio(), on="cod6", how="left").merge(violencia_por_municipio(), on="cod6", how="left")
    for c in ["tem_caps", "tem_trans", "tem_violencia_sexual"]:
        base[c] = base[c].fillna(False).astype(bool)
    for c in ["violencia_total", "violencia_lgbt"]:
        base[c] = base[c].fillna(0).astype(int)
    anos = len(ANOS_SINAN)
    base["taxa_violencia_100mil"] = base["violencia_total"] / anos / base["populacao"] * 1e5
    base["taxa_violencia_lgbt_100mil"] = base["violencia_lgbt"] / anos / base["populacao"] * 1e5
    base["dist_caps_km"] = distancia_ate(base, base["tem_caps"])
    base["dist_trans_km"] = distancia_ate(base, base["tem_trans"])
    base["dist_violencia_sexual_km"] = distancia_ate(base, base["tem_violencia_sexual"])
    base = base.drop(columns="cod6").round(3)
    base.to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"{len(base)} municípios em {SAIDA}")
    print(base.describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
