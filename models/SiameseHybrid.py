import torch
import torch.nn as nn
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

