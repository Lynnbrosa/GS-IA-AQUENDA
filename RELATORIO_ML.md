# Relatório de Machine Learning · Aquenda

**Onde falta cuidado e quem precisa de retorno: violência contra pessoas LGBTQIAPN+ em São Paulo**

Global Solution 2026.2 · CitizenTech · Inteligência Artificial & Machine Learning

Giovanne Charelli Zaniboni Silva (RM 556223) · Gustavo Oliveira de Moura (RM 555827) · Lynn Bueno Rosa (RM 551102)

Notebooks executados: [`ml/violencia_de_repeticao.ipynb`](ml/violencia_de_repeticao.ipynb) e [`ml/desertos_de_cuidado.ipynb`](ml/desertos_de_cuidado.ipynb). Dados: [`ml/dados/README.md`](ml/dados/README.md). Todos os gráficos: [`README.md`](README.md).

## 1. Problema e dados reais

O Aquenda é um app para a comunidade LGBTQIAPN+ de São Paulo, com casas de acolhimento, salas de conversa e um mapa de serviços. Ele precisa saber duas coisas, e cada uma virou uma pergunta de ML:

| | Parte 1 · Quem precisa de retorno | Parte 2 · Onde falta cuidado |
|---|---|---|
| Pergunta | A violência sofrida por uma pessoa LGBTQIAPN+ **já tinha acontecido antes**? | Quais municípios formam **desertos de cuidado**? |
| Dados | 9.856 notificações de violência interpessoal do SINAN, cidade de SP, 2015–2024, vítima homossexual, bissexual, travesti ou trans | 645 municípios de SP: violência notificada (SINAN 2022–2024), distância até CAPS, serviço trans habilitado e atenção à violência sexual (CNES dez/2024), renda per capita (IBGE, Censo 2022) |
| Alvo | `OUT_VEZES` (1 sim, 2 não); 45,6% são repetição | não há rótulo |

Todas as fontes são públicas, baixadas do FTP do DATASUS e das APIs do IBGE pelos scripts `preparar_dados.py` e `preparar_municipios.py`. Nada é simulado. Um dado já diz muito: **só 2 municípios do estado têm serviço habilitado pelo SUS no processo transexualizador**, São Paulo e São José do Rio Preto. A distância mediana até um deles é de 149 km.

## 2. Escolha dos modelos

**Parte 1: XGBoost.**

- A ficha é tabular e quase toda de sim/não.
- O efeito vem de combinações: violência psicológica de um parceiro, dentro de casa, pesa diferente da mesma violência vinda de um desconhecido na rua.
- O XGBoost trata faltantes (a hora falta em 40% das fichas) e mostra a contribuição de cada variável.
- Não há sequência no tempo para justificar uma LSTM, e o volume é pequeno para uma MLP.
- Comparamos com dois modelos de referência: um que chuta sempre a classe mais comum e uma regressão logística.

**Parte 2: KMeans + PCA.**

- Ninguém marcou antes quais municípios são desertos. A pergunta é não supervisionada.
- O KMeans agrupa municípios parecidos, e o centro de cada grupo é um perfil que dá para ler.
- O PCA mede a redundância entre as variáveis (as três distâncias andam juntas) e desenha os grupos num plano.

## 3. Preparação e divisão treino/teste

**Parte 1.**

- **Variáveis.** As 124 variáveis vêm da ficha preenchida no atendimento: quem é a pessoa, onde foi, que violência foi e quem é o provável autor. O código 9 ("ignorado") vira uma coluna própria, para o modelo saber que o campo ficou em branco.
- **Vazamento.** Os encaminhamentos (delegacia, conselho tutelar, Defensoria) e a evolução do caso ficaram de fora. Eles são decididos depois e às vezes por causa da repetição, então entregariam a resposta ao modelo.
- **Divisão pelo tempo.** O treino vai de 2015 a 2023 (7.743 fichas) e o teste é 2024 (2.113 fichas). O modelo seria usado em casos novos, e uma divisão aleatória daria uma nota otimista.

**Parte 2.**

- **Transformação.** As seis variáveis passam por `log(1 + x)`, porque a capital tem 11 milhões de moradores e há municípios com taxa zero. Depois são padronizadas.
- **Divisão.** 20% dos municípios (129) ficam para teste, sorteados. Padronização, PCA e KMeans são ajustados só nos outros 516. Os municípios do teste são distribuídos no fim, nos grupos que o modelo já tinha.

**Nas duas partes,** a escolha dos hiperparâmetros usa validação cruzada de 5 partes **só no treino**. O teste é usado uma única vez, no fim.

## 4. Treino com diferentes hiperparâmetros

As duas grades têm **36 combinações de 3 hiperparâmetros**:

| | Hiperparâmetros | Métrica na validação | Regra de escolha |
|---|---|---|---|
| XGBoost | `max_depth` {2, 3, 4, 6} × `learning_rate` {0,03; 0,1; 0,3} × `n_estimators` {100, 300, 600} | ROC-AUC | a mais simples a até 1 desvio-padrão da melhor |
| KMeans | `n_clusters` {2 a 7} × componentes do PCA {2, 3, sem redução} × `init` {k-means++, random} | silhueta na parte de fora, mais a estabilidade entre partes (ARI) | k ≥ 3 (precisamos de ao menos "bem servido", "intermediário" e "deserto"), estabilidade ≥ 0,8 e maior silhueta |

**XGBoost.**

- O melhor da grade (`max_depth` 6, `learning_rate` 0,03, 300 árvores) chegou a AUC 0,811 na validação, mas tinha 0,898 no treino.
- Ficou **`max_depth` 2, `learning_rate` 0,3, 100 árvores**: AUC 0,804, a apenas 0,026 do treino. Rende quase igual e decora muito menos.
- Passo alto com muitas árvores profundas foi o pior: AUC 0,769.

**KMeans.**

- Reduzir para 2 ou 3 componentes piorou a silhueta (0,218 e 0,225, contra 0,265 sem redução). O PCA ficou para a visualização.
- Com k = 7, a estabilidade cai para 0,74.
- Ficou **k = 6, sem redução, k-means++**: silhueta 0,265 na validação e estabilidade 0,89.

## 5. Resultados em treino e teste

| Parte 1 · XGBoost | ROC-AUC | PR-AUC | Acurácia balanceada | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|
| Treino | 0,830 | 0,812 | 0,742 | 0,670 | 0,840 | 0,745 |
| Validação cruzada | 0,804 | 0,780 | 0,727 | 0,658 | 0,821 | 0,730 |
| **Teste (2024)** | **0,798** | **0,754** | **0,719** | **0,617** | **0,833** | **0,709** |
| Regressão logística · teste | 0,789 | 0,745 | 0,703 | 0,600 | 0,824 | 0,695 |

| Parte 2 · KMeans | Silhueta | Davies-Bouldin (menor é melhor) | Inércia por município |
|---|---|---|---|
| Treino (516) | 0,274 | 1,307 | 2,522 |
| **Teste (129)** | **0,225** | **1,393** | **2,677** |

**Parte 1.**

- Em 2024, o modelo encontrou **762 dos 915 casos de repetição (83%)**, com 473 alarmes falsos em 1.198 fichas sem repetição. O limiar de 0,35 foi escolhido no treino.
- A queda do treino para o teste é de 0,03, e a validação cruzada já tinha previsto quase isso.
- A AUC por ano fica entre 0,75 e 0,83, sem tendência de queda.
- A curva de profundidade mostra onde o overfitting começaria. Com `max_depth` 10, o treino chega a 0,996 e o teste cai para 0,750.

![Curva de profundidade](ml/figuras/07_curva_profundidade.png)

**Parte 2.**

- Os grupos se sustentam em municípios que o modelo não viu: a silhueta cai 0,05 no teste.
- A silhueta é moderada: os municípios formam um contínuo, e os grupos são faixas de perfil, não blocos isolados.
- Overfitting: a inércia do treino cai sempre que k aumenta, mas a distância entre treino e teste sobe de 0,04–0,06 (k = 3 e 4) para 0,16–0,25 (k ≥ 5).

![Treino e teste do KMeans](ml/figuras/13_treino_vs_teste_kmeans.png)

**O que os modelos mostram:**

| Parte 1 · situação na ficha | % de repetição | | Parte 2 · grupo | Municípios | Moradores |
|---|---|---|---|---|---|
| autor da família | 72% | | polo de serviços | 55 | 19,8 mi |
| autor parceiro(a) ou ex | 66% | | rede regional | 44 | 10,8 mi |
| ocorreu na residência | 63% | | intermediário | 162 | 9,6 mi |
| ocorreu na via pública | 27% | | **deserto de cuidado** | **177** | **1,7 mi** |
| autor desconhecido | 20% | | longe e sem notificação (2 grupos) | 207 | 2,4 mi |

![Grupos no PCA e no mapa](ml/figuras/15_grupos_pca_e_mapa.png)

## 6. Conclusão para o cidadão

- **Na cidade de São Paulo, a violência que se repete vem de casa.** Com autor da família, 7 em cada 10 casos já tinham acontecido antes. Com um desconhecido na rua, 1 em cada 5.
- **No estado, ser LGBTQIAPN+ fora dos polos é estar longe do cuidado.** O deserto de cuidado reúne 177 municípios e 1,7 milhão de pessoas, com 573 notificações de violência contra pessoas LGBTQIAPN+ em 2022–2024. Nenhum desses municípios tem CAPS, a renda é baixa e o serviço trans habilitado fica, na mediana, a 148 km. Exemplos são Birigui, Mongaguá, Tremembé e Piquete.
- **Silêncio não é ausência.** Nos grupos "longe e sem notificação", quase não há registro, provavelmente por falta de onde notificar.

O que isso muda no Aquenda:

- casas de acolhimento e rede de apoio para quem sofre em casa;
- um mapa honesto, que mostra o serviço mais próximo de verdade e oferece teleatendimento, ONG ou Defensoria onde não há nada perto;
- uma lista de municípios (`ml/modelo/municipios_por_grupo.csv`) para priorizar parcerias.

O XGBoost **não serve para decidir o atendimento de uma pessoa**: ele erra 17% dos casos de repetição. Serve para a rede planejar retorno e acompanhamento.

**Limites.**

- Só aparece quem chegou a um serviço de saúde e teve a ficha preenchida, e orientação e identidade ficam como "ignorado" em parte das fichas.
- A distância é em linha reta, entre os centros dos municípios.
- O CNES só mostra a habilitação federal.
