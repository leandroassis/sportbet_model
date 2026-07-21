# Redes Neurais - Modelos para Previsão de Futebol (Sports Betting)

Este repositório contém a implementação de diversos modelos de Redes Neurais e Machine Learning utilizados para previsão de resultados esportivos e análise financeira de estratégias de apostas. 

## Pré-requisitos e Instalação

Recomenda-se utilizar um ambiente virtual (como `venv` ou `conda`) para isolar as dependências do projeto.

### Instalação das bibliotecas

O projeto possui um arquivo `requirements.txt` com todas as dependências necessárias (PyTorch, Pandas, Scikit-Learn, CatBoost, etc.). Para instalá-las, execute:

```bash
pip install -r requirements.txt
```

## Como executar os scripts

Os principais scripts estão localizados no diretório `models/`: `train.py` (para treinar os modelos) e `evaluate.py` (para rodar a avaliação de performance, ensembles/Monte Carlo e métricas financeiras).

### Treinando um modelo (`train.py`)

Para executar o treinamento com as configurações padrão, você pode rodar:

```bash
python models/train.py
```

**Principais argumentos de linha de comando (`train.py`):**
- `--arch`: Arquitetura do modelo. Escolha entre `legacy`, `mlp`, `siamese`, `hybrid`. (Padrão: `mlp`)
- `--model_type`: Tipo do modelo. Escolha entre `classifier` ou `regressor`. (Padrão: `classifier`)
- `--epochs`: Número de épocas de treinamento. (Padrão: `150`)
- `--batch_size`: Tamanho do lote. (Padrão: `128`)
- `--seq_len`: Tamanho da sequência de entrada, útil para modelos temporais. (Padrão: `5`)
- `--lr`: Taxa de aprendizado. (Padrão: `1e-4`)
- `--hidden_size`: Tamanho da camada oculta. (Padrão: `64`)
- `--num_layers`: Número de camadas (para arquiteturas como LSTM). (Padrão: `1`)
- `--dropout`: Probabilidade de dropout. (Padrão: `0.25`)
- `--early_stopping_patience`: Paciência para parada antecipada. (Padrão: `2`)
- `--online_learning`: Flag para ativar o fine-tuning Walk-Forward durante a validação.
- `--catboost_iterations`: Número máximo de iterações (se usar o ensemble Hybrid com CatBoost).

**Exemplo prático de treinamento:**
```bash
python models/train.py --arch siamese --model_type classifier --epochs 200 --batch_size 64 --lr 0.001
```

### Avaliando um modelo (`evaluate.py`)

O script `evaluate.py` é responsável por rodar os modelos em modo de avaliação, podendo inclusive gerar métricas de estratégia financeira e ensembles (via múltiplas iterações).

```bash
python models/evaluate.py
```

**Principais argumentos de linha de comando (`evaluate.py`):**
- `--n_iterations`: Número de iterações do ensemble/Monte Carlo. (Padrão: `5`)
- `--arch`: Arquitetura do modelo (mesmas opções do treino). (Padrão: `mlp`)
- `--model_type`: `classifier`. (Padrão: `classifier`)
- `--epochs`: Número de épocas. (Padrão: `100`)
- `--financial_strategy`: Estratégia de aposta financeira, opções: `flat` ou `ev`. (Padrão: `ev`)
- `--ev_threshold`: Limiar mínimo de EV (Expected Value) para a estratégia `ev`. (Padrão: `0.1`)
- `--betting_unit`: Valor unitário da aposta. (Padrão: `10.0`)
- `--temperature`: Temperatura para calibrar probabilidades (softmax). (Padrão: `1.0`)
- `--verbose`: Exibe detalhes adicionais da avaliação financeira.
- `--online_learning`: Ativa o fine-tuning.

**Exemplo prático de avaliação:**
```bash
python models/evaluate.py --arch siamese --n_iterations 10 --financial_strategy ev --ev_threshold 0.15 --verbose
```

## Estrutura do Projeto
- `models/`: Contém as implementações dos modelos (`LSTM.py`, `TabularMLP.py`, `SiameseLSTM.py`, `SiameseHybrid.py`), dataloaders e scripts principais (`train.py`, `evaluate.py`).
- `data/`: Diretório para colocar o CSV de dados pré-processados (`dataset_preprocessed.csv`).
- `articles/`: Documentos de suporte e referências.
- `requirements.txt`: Dependências do projeto.
