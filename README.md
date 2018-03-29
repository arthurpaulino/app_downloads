# app downloads

## overview

o projeto é composto por 3 módulos principais:

* `process_data.py` é responsável pelo feature engineering a partir dos dados originais localizados na pasta `input`, gerando dados processados na pasta `intermediary`

* `train_model.py` utiliza o conjunto de dados processados da pasta `intermediary` e gera um modelo xgboost treinado e o seu gráfico de fscores das features, também na pasta `intermediary`

* `make_predictions.py` carrega o modelo da pasta `intermediary` e realiza as predições para os dados de teste processados, gerando um arquivo `submission_X.csv` na pasta `output` a ser submetido à avaliação competitiva

## data story

contar a história dos dados é adentrar no módulo `process_data.py`.

primeiramente, lê-se os dados de treinamento e de teste. para o conjunto de treinamento, os dados considerados são os mais próximos ao final do dataset.

em seguida, os dados são combinados em um único dataframe para a realização do processamento de dados.

1. re-indexação das features categóricas `app`, `os`, `device`, `channel`

    cada id recebe um novo nome, que é a posição daquele id no ranking das probabilidades de download

    o score de um id é calculado da seguinte forma:

    * caso aquele id tenha downloads associados a ele, o seu score é a quantidade de downloads dividida pela quantidade de clicks

    * caso aquele id não tenha download associado a ele, o seu score é `-N`, onde `N` é a quantidade de clicks naquele id

2. marcação da hora do dia

    uma coluna `moment` é criada demarcando a quantidade de minutos corridos no dia até o instante do click. a intuição é que a probabilidade de downloads pode variar ao longo do dia

3. contagens temporais e globais

    segundo os agrupamentos definidos na função `transform`, novas colunas serão criadas contendo as quantidades de clicks realizados para cada combinação de elementos do agrupamento

    se a contagem envolve uma série temporal, são contabilizados os clicks apenas para cada intervalo de tempo característico da série temporal do agrupamento

    se a contagem não envolve uma série temporal, são contabilizados os clicks de todo o dataset considerado

## parâmetros reguláveis

* `process_data.py` possui as variáveis `data_perc` e `use_supplement`

    * `data_perc` especifica a fração do conjunto de treinamento que será utilizada para o processamento

    * `use_supplement` indica se o conjunto de teste suplementar será utilizado no processamento de dados, o que teoricamente propiciaria maior grau de verdade aos metadados mas consome bem mais memória

* `train_model.py` possui as variáveis `data_perc` e `use_gpu`

    * `data_perc` especifica a fração do conjunto de treinamento **processado** que será utilizada para o treinamento do modelo

    * `use_gpu` indica se o treinamento será realizado com o auxílio de uma *gpu*
