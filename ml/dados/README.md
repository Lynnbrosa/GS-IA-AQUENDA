# Dados

Os dados são todos públicos e reais. Há duas tabelas, uma para cada parte do trabalho:

| Arquivo | Linha = | Usado em |
|---|---|---|
| `sinan_violencia_lgbt_sao_paulo_2015_2024.csv` | uma notificação de violência | parte 1, violência de repetição (XGBoost) |
| `municipios_sp_desertos_de_cuidado.csv` | um município do estado de SP | parte 2, desertos de cuidado (KMeans + PCA) |

# Parte 1 · Notificações de violência

## Fonte

**SINAN: Violência Interpessoal/Autoprovocada**, do Ministério da Saúde. É a ficha que qualquer serviço de saúde preenche quando atende alguém em situação de violência. Os microdados são públicos e não têm nome nem documento de ninguém.

- Arquivos: `VIOLBR15.dbc` a `VIOLBR24.dbc` (Brasil inteiro, um por ano), dados finais.
- Endereço: ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/FINAIS/
- Período: 2015 a 2024, pelo ano da notificação.
- Baixados em 06/11/2026.

Os campos de orientação sexual (`ORIENT_SEX`) e identidade de gênero (`IDENT_GEN`) entraram na ficha em 2014. Por isso o recorte começa em 2015, o primeiro ano completo com eles.

## Recorte

`sinan_violencia_lgbt_sao_paulo_2015_2024.csv` tem as notificações que atendem às duas condições:

1. **Notificadas na cidade de São Paulo:** `ID_MUNICIP` = 355030, o código IBGE de 6 dígitos que o SINAN usa.
2. **A pessoa atendida é LGBTQIAPN+**, por pelo menos um destes campos:
   - `ORIENT_SEX` = 2 (homossexual: gay ou lésbica) ou 3 (bissexual);
   - `IDENT_GEN` = 1 (travesti), 2 (mulher transexual) ou 3 (homem transexual).

| Ano | Notificações no recorte |
|---|---|
| 2015 | 205 |
| 2016 | 533 |
| 2017 | 702 |
| 2018 | 1.064 |
| 2019 | 1.268 |
| 2020 | 1.287 |
| 2021 | 1.921 |
| 2022 | 2.320 |
| 2023 | 3.451 |
| 2024 | 3.427 |
| **Total** | **16.178** |

O crescimento ao longo dos anos é, em boa parte, mais preenchimento desses campos e mais notificação, e não só mais violência.

Saem do CSV, já na preparação:

- os campos de texto livre ("especifique"), que podem trazer detalhe que identifica alguém;
- a ocupação;
- as datas de digitação e de transferência entre sistemas.

## Do CSV ao conjunto do modelo

No notebook:

| Passo | Notificações |
|---|---|
| Recorte acima | 16.178 |
| Sem lesão autoprovocada (`LES_AUTOP` ≠ 1): a pergunta é sobre violência de outra pessoa | 10.934 |
| Com resposta em `OUT_VEZES` (1 sim, 2 não), sem as 1.037 fichas com "ignorado" e as 41 em branco | **9.856** |
| Violência de repetição (`OUT_VEZES` = 1) | 4.493 (45,6%) |

O arquivo de 2020 vem com `NU_ANO` em branco. O ano sai de `DT_NOTIFIC`.

## Campos usados

Os campos de sim/não seguem o código do SINAN: 1 = sim, 2 = não, 9 = ignorado. Cada um vira duas colunas, "sim" e "ignorado", para não perder a informação de que o campo ficou sem resposta.

| Grupo | Campos | Como entram |
|---|---|---|
| Alvo | `OUT_VEZES`: "ocorreu outras vezes?" | 1 = repetição, 2 = não |
| Pessoa | `NU_IDADE_N`, `CS_ESCOL_N`, `CS_RACA`, `CS_SEXO`, `SIT_CONJUG`, `DEF_TRANS`, `ORIENT_SEX`, `IDENT_GEN` | idade em anos (código 4xxx); escolaridade de 0 a 8 (9 e 10 viram faltante); categorias em colunas 0/1 |
| Ocorrência | `HORA_OCOR`, `LOCAL_OCOR` | hora (0–23) e "à noite"; local em colunas 0/1 (residência, via pública, escola, bar…) |
| Tipo de violência | `VIOL_FISIC`, `VIOL_PSICO`, `VIOL_TORT`, `VIOL_SEXU`, `VIOL_FINAN`, `VIOL_NEGLI`, `VIOL_LEGAL`, `VIOL_OUTR` | sim/ignorado, mais a contagem de tipos |
| Meio | `AG_FORCA`, `AG_ENFOR`, `AG_OBJETO`, `AG_CORTE`, `AG_QUENTE`, `AG_ENVEN`, `AG_FOGO`, `AG_AMEACA`, `AG_OUTROS` | sim/ignorado |
| Provável autor | `REL_*` (pai, mãe, padrasto, madrasta, cônjuge, ex, namorado(a), ex, filho, irmão, conhecido, desconhecido, cuidador, patrão, institucional, policial, outros), `AUTOR_SEXO`, `AUTOR_ALCO`, `NUM_ENVOLV`, `CICL_VID` | sim/ignorado; resumos "autor da família" e "autor parceiro(a) ou ex" |
| Motivação | `VIOL_MOTIV` (sexismo, LGBTfobia, racismo, …) | colunas 0/1 |

São 124 colunas no total.

**Fora do modelo de propósito:** os encaminhamentos (`ENC_*`, `REDE_SAU`, `ASSIST_SOC`, `DELEG*`, `CONS_TUTEL`, `DEFEN_PUBL`…), a evolução (`EVOLUCAO`, `DT_OBITO`) e a classificação final. Eles são decididos depois do atendimento, às vezes justamente porque a violência se repete. Usar esses campos seria dar a resposta ao modelo (vazamento).

## Como gerar de novo

```bash
python ml/preparar_dados.py
```

O script baixa os `.dbc` que faltarem para `ml/dados/brutos/` (cerca de 240 MB, fora do Git), converte para `.dbf` com o `pyreaddbc`, filtra o recorte e grava o CSV. Leva uns 10 minutos, porque cada ano tem centenas de milhares de fichas do Brasil inteiro.

# Parte 2 · Municípios e desertos de cuidado

`municipios_sp_desertos_de_cuidado.csv` tem uma linha para cada um dos **645 municípios** do estado de São Paulo. É gerado por `preparar_municipios.py`.

## Fontes

| Fonte | O que vem | Link | Período |
|---|---|---|---|
| SINAN (DATASUS) | notificações de violência interpessoal pelo município de **residência** da vítima, no total e só com vítima LGBTQIAPN+ (mesmo critério da parte 1) | ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/FINAIS/ | 2022 a 2024 |
| CNES (DATASUS): estabelecimentos (`STSP2412`) | municípios com **CAPS** (`TP_UNID` = 70) | ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/ST/ | dez/2024 |
| CNES: habilitações (`HBSP2412`) | municípios com serviço habilitado no **processo transexualizador** (`SGRUPHAB` 3001, 3002, 3003) | ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/HB/ | dez/2024 |
| CNES: serviços especializados (`SRSP2412`) | municípios com **atenção às pessoas em situação de violência sexual** (`SERV_ESP` 165) | ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/SR/ | dez/2024 |
| IBGE, Censo 2022 (SIDRA, tabela 4709) | população residente | https://apisidra.ibge.gov.br/values/t/4709/n6/in%20n3%2035/v/93/p/last | 2022 |
| IBGE, Censo 2022 (SIDRA, tabela 10295, variável 13431) | rendimento nominal médio mensal domiciliar per capita | https://apisidra.ibge.gov.br/values/t/10295/n6/in%20n3%2035/v/13431/p/all | 2022 |
| IBGE, malha municipal | contorno de cada município, para achar o centro | https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&intrarregiao=municipio | atual |

Os códigos do CNES foram conferidos nas tabelas auxiliares do DATASUS (`TAB_CNES.zip`: `HABILITA.DBF` e `SRN_ORD_N.dbf`).

## Colunas

| Coluna | Como é calculada |
|---|---|
| `cod7`, `municipio`, `populacao`, `renda_per_capita` | IBGE |
| `lat`, `lon` | centro do maior polígono do município (fórmula do shoelace) |
| `tem_caps`, `tem_trans`, `tem_violencia_sexual` | se o município tem o serviço no CNES |
| `violencia_total`, `violencia_lgbt` | notificações de violência interpessoal (sem lesão autoprovocada) de 2022 a 2024 |
| `taxa_violencia_100mil`, `taxa_violencia_lgbt_100mil` | notificações por ano por 100 mil moradores |
| `dist_caps_km`, `dist_trans_km`, `dist_violencia_sexual_km` | distância em linha reta do centro do município ao centro do município mais próximo com o serviço; 0 se o próprio município tem |

O que a tabela mostra sobre o estado:

- 285 dos 645 municípios têm CAPS;
- 49 têm serviço de atenção à violência sexual;
- **só 2 têm serviço habilitado no processo transexualizador pelo SUS:** São Paulo e São José do Rio Preto. A distância mediana até um deles é de 149 km, e a maior chega a 405 km.

Ambulatórios trans municipais ou estaduais sem habilitação federal não aparecem no CNES com esse código, e esse é um limite da medida.

## Como gerar de novo

```bash
python ml/preparar_municipios.py
```

O script reaproveita os `.dbc` do SINAN que o `preparar_dados.py` baixou, baixa os três arquivos do CNES e consulta as APIs do IBGE. A contagem do SINAN por município fica em cache em `brutos/`, fora do Git.
