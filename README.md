# app downloads

## overview

o projeto é composto por 3 módulos principais:

* `process_data.py` é responsável pelo feature engineering a partir dos dados originais localizados na pasta `input`, gerando dados processados na pasta `intermediary`

* `train_model.py` utiliza o conjunto de dados processados da pasta `intermediary` e gera um modelo xgboost treinado e o seu gráfico de fscores das features, também na pasta `intermediary`

* `make_predictions.py` carrega o modelo da pasta `intermediary` e realiza as predições para os dados de teste processados, gerando um arquivo `submission_X.csv` a ser submetido à avaliação competitiva