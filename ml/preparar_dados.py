"""Baixa e filtra as notificações de violência do SINAN (DATASUS) para o recorte do Aquenda.

Recorte: pessoas LGBTQIAPN+ (orientação homossexual ou bissexual, ou identidade travesti,
mulher trans ou homem trans) atendidas na cidade de São Paulo, de 2015 a 2024.

    python ml/preparar_dados.py            # baixa o que faltar e gera o CSV filtrado
    python ml/preparar_dados.py --anos 2023 2024

Fonte: ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/FINAIS/VIOLBRaa.dbc
Os arquivos brutos (o Brasil inteiro, ~240 MB) ficam em ml/dados/brutos/ e não vão para o Git.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from dbfread import DBF
from pyreaddbc import dbc2dbf

PASTA = Path(__file__).resolve().parent / "dados"
BRUTOS = PASTA / "brutos"
SAIDA = PASTA / "sinan_violencia_lgbt_sao_paulo_2015_2024.csv"
URL = "ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/FINAIS/VIOLBR{aa}.dbc"

MUNICIPIO_SP = "355030"  # código IBGE de 6 dígitos usado pelo SINAN
ORIENTACAO_LGB = {"2", "3"}  # 2 homossexual (gay/lésbica), 3 bissexual
IDENTIDADE_TRANS = {"1", "2", "3"}  # 1 travesti, 2 mulher transexual, 3 homem transexual

# Texto livre ("especifique") e datas de processamento não entram: não ajudam o modelo
# e o texto livre pode trazer detalhe que identifica alguém.
DESCARTAR = lambda c: c.endswith("_ESPEC") or c.startswith("DT_TRANS") or c in {"DT_DIGITA", "LOCAL_ESPE", "LESAO_ESPE", "DEF_ESPEC", "ID_OCUPA_N", "NDUPLIC"}


def baixar(ano: int) -> Path:
    BRUTOS.mkdir(parents=True, exist_ok=True)
    destino = BRUTOS / f"VIOLBR{ano % 100:02d}.dbc"
    if not destino.exists():
        print(f"baixando {destino.name}…")
        urllib.request.urlretrieve(URL.format(aa=f"{ano % 100:02d}"), destino)
    return destino


def ler_recorte(dbc: Path) -> pd.DataFrame:
    dbf = BRUTOS / "dbf" / dbc.with_suffix(".dbf").name
    dbf.parent.mkdir(exist_ok=True)
    if not dbf.exists():
        dbc2dbf(str(dbc), str(dbf))
    linhas = []
    for r in DBF(dbf, encoding="latin-1", load=False, char_decode_errors="replace"):
        if not str(r.get("ID_MUNICIP", "")).startswith(MUNICIPIO_SP):
            continue
        if str(r.get("ORIENT_SEX", "")) in ORIENTACAO_LGB or str(r.get("IDENT_GEN", "")) in IDENTIDADE_TRANS:
            linhas.append({k: v for k, v in r.items() if not DESCARTAR(k)})
    return pd.DataFrame(linhas)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anos", type=int, nargs="*", default=list(range(2015, 2025)))
    a = p.parse_args()
    partes = []
    for ano in a.anos:
        df = ler_recorte(baixar(ano))
        print(f"{ano}: {len(df)} notificações no recorte")
        partes.append(df)
    tudo = pd.concat(partes, ignore_index=True)
    tudo.to_csv(SAIDA, index=False, encoding="utf-8")
    print(f"{len(tudo)} linhas em {SAIDA}")


if __name__ == "__main__":
    main()
