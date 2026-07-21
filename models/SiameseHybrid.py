import torch
import torch.nn as nn
import numpy as np
from LSTM import TabularFeatureExtractor, TemporalAttention
from SiameseLSTM import SiameseLSTMBackbone, SiameseClassifier, SiameseRegressor

class SiameseHybridClassifier(nn.Module):
    """
    Abordagem Wide & Deep Nativa do PyTorch.
    Resolve o problema do CatBoost criando uma 'Skip Connection'. 
    As features numéricas do jogo atual (que contêm as Odds limpas) são injetadas 
    DIRETAMENTE na camada final de decisão, não sendo diluídas pela LSTM.
    """
    def __init__(self, backbone: SiameseLSTMBackbone, hidden_size: int, numerical_input_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        
        # O tamanho híbrido recebe o Embedding extraído (hidden_size) + as features originais (numerical_input_size)
        hybrid_size = hidden_size + numerical_input_size
        
        self.classification_head = nn.Sequential(
            nn.Linear(hybrid_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )
        initial_bias = torch.tensor([0.27, 0.49, 0.24])
        self.classification_head[-1].bias.data = torch.log(initial_bias)

    def forward(self, h_num, h_cat, a_num, a_cat, m_num, m_cat) -> torch.Tensor:
        # 1. O cérebro profundo (Scout): Lê a fase dos times
        backbone_output = self.backbone(h_num, h_cat, a_num, a_cat, m_num, m_cat)
        
        # 2. O cérebro Racional: Concatena DIRETAMENTE as features do dia (que possuem as Odds)
        hybrid_features = torch.cat([backbone_output, m_num], dim=-1)
        
        # 3. Decisão final ancorada nos dados de mercado
        return self.classification_head(hybrid_features)


class SiameseEmbeddingExtractor(nn.Module):
    """
    Extrator puro para uso futuro com CatBoost externo.
    Caso decida treinar o CatBoost fora do loop do PyTorch, esta classe
    apenas congela o backbone e retorna os embeddings de 128 dimensões.
    """
    def __init__(self, backbone: SiameseLSTMBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        # Congela a rede para não consumir memória nem quebrar gradientes na inferência
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, h_num, h_cat, a_num, a_cat, m_num, m_cat) -> torch.Tensor:
        return self.backbone(h_num, h_cat, a_num, a_cat, m_num, m_cat)

class CatBoostOrdinalWrapper:
    """
    Wrapper Frank-Hall para Classificação Ordinal.
    Finge ser um CatBoost nativo para que o FinancialAnalyzer consiga extrair
    parâmetros e realizar o Online Learning sem quebrar.
    """
    def __init__(self, model_m1, model_m2):
        self.model_m1 = model_m1
        self.model_m2 = model_m2

    @property
    def feature_names_(self):
        # Engana a Inteligência Dinâmica do Analyzer para o alinhamento de colunas
        return self.model_m1.feature_names_

    def get_param(self, param_name):
        # Retorna o parâmetro do M1, assumindo que M2 é simétrico
        if param_name == 'loss_function':
            return 'Logloss' # Força Logloss binária
        return self.model_m1.get_param(param_name)

    def predict_proba(self, X):
        # M1 prevê a probabilidade de ser Mandante
        p1 = self.model_m1.predict_proba(X)[:, 1]
        
        # M2 prevê a probabilidade de ser Mandante ou Empate
        p2 = self.model_m2.predict_proba(X)[:, 1]

        # Garantia matemática de monotonicidade (P2 nunca pode ser menor que P1)
        p1 = np.clip(p1, 0.0, 1.0)
        p2 = np.maximum(p1, p2)
        p2 = np.clip(p2, 0.0, 1.0)

        # Reconstrução das 3 classes ordinais
        prob_mandante = p1
        prob_empate = p2 - p1
        prob_visitante = 1.0 - p2

        # Retorna na ordem exata: [Empate(0), Mandante(1), Visitante(2)]
        return np.column_stack((prob_empate, prob_mandante, prob_visitante))

    def fit(self, X, y, init_model=None, iterations=1, verbose=False):
        """
        Método invocado pelo Online Learning.
        Recebe o target real do jogo (0, 1, ou 2) e o converte em tempo real 
        para os dois alvos binários das árvores.
        """
        # Se y vier como One-Hot encoding, converte para índice
        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        # Conversão Frank-Hall
        y_m1 = (y == 1).astype(int)           # 1 se Mandante, 0 caso contrário
        y_m2 = np.isin(y, [0, 1]).astype(int) # 1 se Mandante ou Empate, 0 se Visitante

        # Atualiza os dois modelos com a nova linha de dados
        self.model_m1.fit(X, y_m1, init_model=self.model_m1, iterations=iterations, verbose=verbose)
        self.model_m2.fit(X, y_m2, init_model=self.model_m2, iterations=iterations, verbose=verbose)