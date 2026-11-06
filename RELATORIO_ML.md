# Relatório de Machine Learning · Aquenda

**Onde falta cuidado e quem precisa de retorno: violência contra pessoas LGBTQIAPN+ em São Paulo**

Global Solution 2026.2 · CitizenTech · Inteligência Artificial & Machine Learning

Giovanne Charelli Zaniboni Silva (RM 556223) · Gustavo Oliveira de Moura (RM 555827) · Lynn Bueno Rosa (RM 551102)

Notebooks executados: [`ml/violencia_de_repeticao.ipynb`](ml/violencia_de_repeticao.ipynb) (parte 1) e [`ml/desertos_de_cuidado.ipynb`](ml/desertos_de_cuidado.ipynb) (parte 2) · Dados: [`ml/dados/README.md`](ml/dados/README.md)

O trabalho responde a duas perguntas complementares, com os mesmos dados de violência do SINAN:

| | Parte 1 · Quem precisa de retorno | Parte 2 · Onde falta cuidado |
|---|---|---|
| Pergunta | a violência sofrida por uma pessoa LGBTQIAPN+ já tinha acontecido antes? | quais municípios de SP formam desertos de cuidado? |
| Unidade | uma notificação (9.856, cidade de SP) | um município (645, estado de SP) |
| Tipo | supervisionado (classificação) | não supervisionado (agrupamento) |
| Modelo | XGBoost | KMeans + PCA |

---

# Parte 1 · Quem precisa de retorno: violência de repetição (XGBoost)

## 1. Problema e dados reais

O Aquenda é um app para a comunidade LGBTQIAPN+ de São Paulo. Ele tem casas de acolhimento com vagas em tempo real, salas de conversa moderadas e um mapa de serviços (CRAS, CAPS, ambulatório trans, Defensoria). A violência mais difícil de sair é a que se repete: quase sempre vem de alguém que mora junto ou é próximo. É para essa pessoa que acolhimento e rede de apoio fazem mais diferença.

**Pergunta de ML.** Quando uma pessoa LGBTQIAPN+ é atendida por violência na cidade de São Paulo, dá para saber, só com o que a ficha de notificação registra, se essa violência **já aconteceu outras vezes**? É uma classificação binária.

**Dados.** Usamos as notificações de violência interpessoal do **SINAN** (Ministério da Saúde, DATASUS), de 2015 a 2024. São os arquivos `VIOLBRaa.dbc` do FTP público do DATASUS, todos reais e sem nada simulado. O recorte:

- fichas notificadas na cidade de São Paulo;
- com pessoa homossexual ou bissexual, ou travesti, mulher trans ou homem trans: são 16.178 fichas;
- tirando a lesão autoprovocada (outra pergunta, outra rede de cuidado), sobram 10.934;
- tirando as fichas sem resposta no campo "ocorreu outras vezes?", ficam **9.856 notificações**.

Dessas, **45,6% são violência de repetição**. As classes estão quase equilibradas.

![Alvo por ano](ml/figuras/01_alvo_por_ano.png)

O número de fichas cresce muito no período, de 158 com resposta em 2015 para mais de 2 mil por ano a partir de 2023. Isso é mais preenchimento dos campos de orientação e identidade e mais notificação, e não só mais violência. A taxa de repetição também muda: 30% em 2015, 57% em 2021 e 43% em 2024.

## 2. Escolha do modelo

Escolhemos o **XGBoost**, gradient boosting de árvores, pelos seguintes motivos:

- **Dados tabulares, quase todos binários.** A ficha é uma lista de sim/não: tipo de violência, meio, quem é o autor. Nesse formato, as árvores com boosting costumam ser o modelo mais forte.
- **Interações.** "Violência psicológica" pesa diferente quando vem de um parceiro dentro de casa e quando vem de um desconhecido na rua. As árvores aprendem essas combinações sem precisarmos escrevê-las.
- **Valores faltantes.** A hora falta em 40% das fichas e a escolaridade em 15%. O XGBoost aprende para que lado mandar o faltante, sem inventar valor.
- **Leitura.** A contribuição de cada variável em cada previsão (`pred_contribs`) mostra o que pesou.

Os outros modelos da lista não se encaixam:

- **LSTM:** não há sequência no tempo da mesma pessoa.
- **MLP:** pede mais dados e explica menos.
- **KMeans + PCA:** não responde a uma pergunta supervisionada.

Para saber se o XGBoost vale a pena, comparamos com duas referências:

- um modelo que chuta sempre a classe mais comum: AUC 0,50;
- uma regressão logística: AUC 0,79 na validação cruzada.

## 3. Preparação e divisão treino/teste

**Variáveis.** Todas vêm do que a ficha registra no atendimento:

- quem é a pessoa: idade, escolaridade, raça/cor, deficiência, orientação, identidade;
- onde e quando: local, hora, se foi à noite;
- que violência foi: tipos e meios;
- quem é o provável autor: vínculo, sexo, uso de álcool, quantos;
- a motivação.

Os campos sim/não (1 sim, 2 não, 9 ignorado) viram duas colunas cada, "sim" e "ignorado". Assim, o modelo sabe quando o campo ficou em branco. Criamos dois resumos: "autor da família" e "autor parceiro(a) ou ex". São **124 variáveis** no total.

**Vazamento.** Os encaminhamentos (delegacia, conselho tutelar, rede de saúde, Defensoria) e a evolução do caso ficaram de fora. Eles são decididos depois do atendimento, muitas vezes justamente porque a violência se repete. Com eles, o modelo pareceria ótimo sem servir para nada.

**Divisão pelo tempo.**

| | Notificações | % repetição |
|---|---|---|
| Treino: 2015 a 2023 | 7.743 | 46,2% |
| Teste: 2024 | 2.113 | 43,3% |

O modelo seria usado em fichas novas, então o teste é o ano mais recente. Uma divisão aleatória misturaria os anos e daria uma nota otimista. O teste só foi usado uma vez, no fim. Toda a escolha de hiperparâmetros e do limiar foi feita com validação cruzada estratificada de 5 partes **só no treino**.

![Divisão temporal](ml/figuras/02_divisao_temporal.png)

## 4. Treino com diferentes hiperparâmetros

Testamos **36 combinações** de três hiperparâmetros, cada uma com validação cruzada de 5 partes no treino:

| Hiperparâmetro | O que controla | Valores |
|---|---|---|
| `max_depth` | complexidade de cada árvore | 2, 3, 4, 6 |
| `learning_rate` | tamanho do passo de cada árvore nova | 0,03 · 0,1 · 0,3 |
| `n_estimators` | número de árvores | 100, 300, 600 |

O resto ficou fixo: `subsample` 0,8, `colsample_bytree` 0,8 e `min_child_weight` 5.

![Grade de hiperparâmetros](ml/figuras/03_grade_auc_validacao.png)

O que a grade mostra:

- **O pior é passo alto com muitas árvores.** Com `learning_rate` 0,3, a AUC de validação cai à medida que entram mais árvores e mais profundidade: de 0,804 para 0,769 com 600 árvores e profundidade 6. É o modelo corrigindo ruído do treino.
- **O melhor é passo baixo.** `learning_rate` 0,03, com profundidade 4 a 6 e 300 árvores, chega a **0,811**.
- **A distância de 0,804 a 0,811 é pequena.** É do tamanho do desvio-padrão entre as partes da validação (0,007).

**Regra de escolha:** a combinação mais simples cuja AUC fica a menos de um desvio-padrão da melhor. Pela regra, ficou **`max_depth` 2, `learning_rate` 0,3 e 100 árvores**, com AUC de validação 0,804. A melhor da grade (profundidade 6, 300 árvores) tinha AUC de treino 0,898 contra 0,811 na validação, uma distância de 0,087. A escolhida tem distância de 0,026: rende quase igual e decora muito menos. Na validação cruzada, ela ganha da regressão logística (0,804 contra 0,790).

O limiar de decisão também saiu do treino. É o que maximiza o F1 nas previsões da validação cruzada: **0,35**. Um limiar abaixo de 0,5 favorece achar os casos de repetição (recall), que é o erro mais caro aqui.

## 5. Resultados em treino e teste

| | ROC-AUC | PR-AUC | Acurácia balanceada | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|
| XGBoost · treino | 0,830 | 0,812 | 0,742 | 0,670 | 0,840 | 0,745 |
| XGBoost · validação cruzada | 0,804 | 0,780 | 0,727 | 0,658 | 0,821 | 0,730 |
| **XGBoost · teste (2024)** | **0,798** | **0,754** | **0,719** | **0,617** | **0,833** | **0,709** |
| Regressão logística · teste | 0,789 | 0,745 | 0,703 | 0,600 | 0,824 | 0,695 |

![Treino e teste lado a lado](ml/figuras/05_treino_vs_teste.png)

![ROC e matriz de confusão](ml/figuras/06_roc_e_confusao.png)

Em 2024, o modelo encontrou **762 dos 915 casos de repetição (83%)**. Em troca, deu alarme em 473 das 1.198 fichas que não eram repetição. Precisão de 62% quer dizer que, de cada 5 fichas marcadas, 3 são de fato repetição.

**Overfitting.** A queda do treino (0,830) para o teste (0,798) é de 0,03, e a validação cruzada (0,804) já tinha previsto quase isso. A curva por profundidade mostra onde começaria o problema:

![Curva de profundidade](ml/figuras/07_curva_profundidade.png)

| `max_depth` | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| treino | 0,803 | 0,834 | 0,864 | 0,897 | 0,955 | 0,986 | 0,996 |
| validação | 0,794 | 0,804 | 0,802 | 0,798 | 0,789 | 0,781 | 0,783 |
| teste | 0,790 | 0,798 | 0,798 | 0,783 | 0,770 | 0,756 | 0,750 |

Com profundidade 10, o treino chega a 0,996: o modelo praticamente decora as fichas. Mas o teste cai para 0,750. A profundidade 2 está no ponto em que validação e teste são máximos.

**Mudança no tempo.** A AUC por ano (validação cruzada nos anos de treino, modelo final em 2024) fica entre 0,75 e 0,83, sem tendência de queda. Os anos mais fracos são 2015 a 2017, com poucas fichas. O padrão aprendido continua valendo em 2024.

![AUC por ano](ml/figuras/08_auc_por_ano.png)

**O que pesa na previsão:**

![Importância](ml/figuras/09_importancia.png)

| Situação na ficha | % de repetição |
|---|---|
| autor da família | 72% (1.667 fichas) |
| autor parceiro(a) ou ex | 66% (2.390) |
| ocorreu na residência | 63% (4.577) |
| violência psicológica | 61% (2.720) |
| motivação LGBTfobia | 50% (1.508) |
| ocorreu na via pública | 27% (2.517) |
| autor desconhecido | 20% (2.870) |

## 6. Conclusão para o cidadão (parte 1)

**A violência que se repete contra pessoas LGBTQIAPN+ em São Paulo vem de dentro de casa.** Quando o autor é da família, 7 em cada 10 casos já tinham acontecido antes. Com parceiro(a) ou ex, 2 em cada 3. Com um desconhecido na rua, 1 em cada 5. Para quem é LGBTQIAPN+, o lugar mais perigoso pode ser onde mora, e com quem mora. Isso reforça três decisões do Aquenda:

- as casas de acolhimento com vaga em tempo real;
- a sala "família" nas conversas;
- o mapa que leva até CRAS, CREAS e Defensoria.

O modelo acerta bem mais do que o acaso (AUC 0,80) e manteve o desempenho em 2024, um ano que nunca tinha visto. Mesmo assim, **ele não serve para decidir o atendimento de uma pessoa**. Ele erra 17% dos casos de repetição e dá alarme falso em 4 de cada 10 fichas sem repetição. Serve para planejar:

- a rede de saúde e de assistência pode priorizar retorno e acompanhamento nas fichas com perfil de repetição;
- o app pode mostrar primeiro acolhimento e apoio para quem conta estar nessa situação.

**Limites.**

- É dado de notificação: só aparece quem chegou a um serviço de saúde e teve a ficha preenchida.
- Muita violência contra pessoas LGBTQIAPN+ nunca é notificada.
- Orientação e identidade ficam como "ignorado" em parte das fichas, então o recorte é menor do que a realidade.
- O campo "ocorreu outras vezes?" depende do que a pessoa conta no atendimento.


---

# Parte 2 · Onde falta cuidado: desertos de cuidado (KMeans + PCA)

## 1. Problema e dados reais

O pitch do Aquenda promete mostrar "serviços perto da pessoa". Fora da capital, perto pode não existir. **Pergunta:** quais municípios do estado de São Paulo se parecem quando olhamos violência notificada, distância até o cuidado e renda? Quais formam **desertos de cuidado**, com violência contra pessoas LGBTQIAPN+, cuidado longe e pouca renda para chegar até ele?

Montamos uma linha para cada um dos **645 municípios** a partir de três fontes públicas:

- **SINAN** (2022–2024): notificações de violência interpessoal por 100 mil moradores, no total e com vítima LGBTQIAPN+, pelo município de residência;
- **CNES** (dez/2024): distância até o município mais próximo com CAPS, com serviço habilitado no processo transexualizador e com atenção à violência sexual;
- **IBGE, Censo 2022:** população e renda domiciliar per capita.

Um dado já diz muito: **só 2 municípios do estado têm serviço habilitado pelo SUS no processo transexualizador**, São Paulo e São José do Rio Preto. A distância mediana até um deles é de 149 km, e a maior chega a 405 km.

## 2. Escolha do modelo

A pergunta **não tem rótulo**: ninguém marcou antes quais municípios são desertos. Por isso o modelo é o **KMeans**, que agrupa municípios parecidos e entrega o "perfil" de cada grupo, que dá para ler.

O **PCA** entra para tirar redundância entre as variáveis, porque as três distâncias andam juntas, e para desenhar os grupos num plano. Os 2 primeiros componentes explicam 59% da variação, e os 3 primeiros, 73%. XGBoost, Random Forest, MLP e LSTM precisariam de um rótulo que aqui não existe.

## 3. Preparação e divisão treino/teste

- **Transformação.** As seis variáveis são muito assimétricas: a capital tem 11 milhões de moradores, e há municípios com taxa zero. Todas passam por `log(1 + x)` e por padronização, ajustada só no treino. Sem isso, o KMeans, que mede distância, ficaria refém da variável de maior escala.
- **Teste.** Mesmo sem rótulo, separamos **20% dos municípios** (129) ao acaso. Padronização, PCA e KMeans foram ajustados nos outros 516. Os do teste só entram no fim, distribuídos nos grupos já formados.
- **Escolha dos hiperparâmetros.** Foi feita com validação cruzada de 5 partes **só no treino**. Em cada parte, o modelo ajustado em 4/5 do treino distribui o 1/5 que ficou de fora, e medimos a silhueta desse 1/5.

## 4. Treino com diferentes hiperparâmetros

São **36 combinações**: `n_clusters` (2 a 7) × componentes do PCA (2, 3 ou sem redução) × `init` (`k-means++` ou `random`). Para cada uma medimos duas coisas:

- a **silhueta** na parte de validação, sempre no mesmo espaço das 6 variáveis padronizadas, para comparar com e sem PCA;
- a **estabilidade**: o ARI entre os agrupamentos feitos em partes diferentes do treino.

![Grade KMeans](ml/figuras/12_grade_silhueta.png)

- **Reduzir para 2 ou 3 componentes piorou o agrupamento.** A melhor silhueta foi 0,218 com 2 componentes, 0,225 com 3 e **0,265 sem redução**. O que separa os grupos está espalhado em mais de 3 direções. O PCA ficou para a visualização e para mostrar o que cada direção carrega.
- **k = 7 já passa do ponto.** A silhueta de validação cai para 0,23 e a estabilidade para 0,74–0,76: os grupos mudam conforme a amostra.

**Regra de escolha,** fixada antes do teste:

- k de 3 para cima, porque o objetivo exige ao menos "bem servido", "intermediário" e "deserto";
- estabilidade de pelo menos 0,8;
- entre essas, a maior silhueta.

Ficou **k = 6, sem redução, `k-means++`**, com silhueta de validação 0,265 e estabilidade 0,89.

## 5. Resultados em treino e teste

| | Silhueta | Davies-Bouldin (menor é melhor) | Inércia por município |
|---|---|---|---|
| Treino (516) | 0,274 | 1,307 | 2,522 |
| **Teste (129)** | **0,225** | **1,393** | **2,677** |

![Treino e teste do KMeans](ml/figuras/13_treino_vs_teste_kmeans.png)

**Os grupos se sustentam em municípios que o modelo não viu.** No teste, a silhueta cai 0,05, o Davies-Bouldin sobe 0,09 e a inércia sobe 6%. A silhueta de 0,23–0,27 é moderada: os municípios formam um contínuo, sem blocos isolados. Os grupos são faixas de perfil, não categorias rígidas.

**Overfitting.** A inércia do treino cai sempre que k aumenta, de 4,58 com k = 2 para 1,87 com k = 10. A distância entre treino e teste, porém, fica em 0,04–0,06 com k = 3 e 4 e sobe para 0,16–0,25 de k = 5 em diante. Somado à queda de estabilidade em k = 7, esse é o sinal de que grupos a mais passam a descrever a amostra, e não o estado. Com k = 6, a silhueta de validação foi a mais alta, e foi isso que pesou na escolha.

![Grupos no PCA e no mapa](ml/figuras/15_grupos_pca_e_mapa.png)

![Perfil dos grupos](ml/figuras/16_perfil_dos_grupos.png)

| Grupo | Municípios | Moradores | Perfil (mediana) |
|---|---|---|---|
| polo de serviços | 55 | 19,8 milhões | CAPS no município, serviço trans a 31 km |
| rede regional | 44 | 10,8 milhões | CAPS e atenção à violência sexual no município, renda alta (R$ 1.989) |
| intermediário | 162 | 9,6 milhões | CAPS no município, serviço trans a 142 km |
| **deserto de cuidado** | **177** | **1,7 milhão** | **a maior taxa de violência LGBT notificada (9,3 por 100 mil), sem CAPS no município (20 km), serviço trans a 148 km, renda baixa (R$ 1.429)** |
| longe e sem notificação (sem CAPS) | 147 | 0,9 milhão | quase nenhuma notificação LGBT, sem CAPS, serviço trans a 175 km, a menor renda (R$ 1.372) |
| longe e sem notificação (com CAPS) | 60 | 1,4 milhão | quase nenhuma notificação de violência, serviço trans a 195 km |

## 6. Conclusão para o cidadão (parte 2)

**Ser LGBTQIAPN+ fora da capital é estar longe do cuidado.** No grupo deserto de cuidado moram 1,7 milhão de pessoas, em 177 municípios. Entre 2022 e 2024, eles somaram 573 notificações de violência contra pessoas LGBTQIAPN+. Nenhum desses municípios tem CAPS, e o ambulatório trans habilitado mais próximo fica, na mediana, a 148 km. Exemplos são Birigui, Mongaguá, Tremembé e Piquete.

O grupo **longe e sem notificação** preocupa de outro jeito. Ali quase não há registro de violência contra pessoas LGBTQIAPN+, e isso não quer dizer que ela não aconteça: falta onde notificar e quem pergunte.

Para o Aquenda, a lista de municípios por grupo (`ml/modelo/municipios_por_grupo.csv`) diz onde priorizar:

- mapear ONGs e serviços parceiros;
- oferecer orientação por teleatendimento;
- indicar a Defensoria e o transporte até a consulta.

Para o poder público, ela diz onde habilitar o próximo serviço.

**Limites.**

- A distância é em linha reta, entre os centros dos municípios.
- O CNES só mostra habilitação federal. Ambulatórios trans municipais sem esse código ficam de fora.
- Notificação baixa pode querer dizer pouca violência ou pouco serviço para notificar.

## Conclusão geral

As duas partes contam a mesma história de dois ângulos.

- **Na cidade de São Paulo, a violência que se repete vem de casa:** família, parceiro(a) ou ex. É quem mais precisa de acolhimento e acompanhamento.
- **No estado, quem mora fora dos polos** está a dezenas ou centenas de quilômetros do cuidado especializado.

O Aquenda junta as duas respostas: casas de acolhimento e rede de apoio para quem sofre em casa, e um mapa honesto que mostra o serviço mais próximo de verdade e a alternativa (teleatendimento, ONG, Defensoria) onde não há nada perto.
