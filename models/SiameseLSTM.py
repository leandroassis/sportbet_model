import torch
import torch.nn as nn
from LSTM import TabularFeatureExtractor, TemporalAttention

class SiameseLSTMBackbone(nn.Module):
    def __init__(
        self,
        numerical_input_size: int,
        embedding_settings,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings, dim)
            for name, (num_embeddings, dim) in embedding_settings.dimensions.items()
        })
        self.embedding_dropout = nn.Dropout(dropout * 0.5)
        
        raw_input_size = numerical_input_size + sum(dim for _, dim in embedding_settings.dimensions.values())
        self.input_ln = nn.LayerNorm(raw_input_size)
        
        # Extrator Tabular Compartilhado (mastiga as features de cada jogo isolado)
        self.feature_extractor = TabularFeatureExtractor(
            input_dim=raw_input_size,
            output_dim=hidden_size,
            dropout=dropout
        )
        
        # LSTM Compartilhada (entende a sequência de qualquer time)
        self.shared_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        
        # Camada de fusão: Contexto Mandante + Contexto Visitante + Confronto Atual
        # Tamanho: hidden_size (Mandante) + hidden_size (Visitante) + hidden_size (Dados do Jogo Atual)
        fusion_size = hidden_size * 3
        
        self.fusion_dense = nn.Linear(fusion_size, hidden_size)
        self.fusion_ln = nn.LayerNorm(hidden_size)
        self.fusion_act = nn.ReLU()
        self.fusion_drop = nn.Dropout(dropout)

    def _process_sequence(self, num_seq: torch.Tensor, cat_seq: torch.Tensor) -> torch.Tensor:
        # Extrai embeddings para a sequência
        embeddings = [
            self.embedding_layers[name](cat_seq[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        cat_emb = self.embedding_dropout(torch.cat(embeddings, dim=-1))
        
        combined = torch.cat([num_seq, cat_emb], dim=-1)
        combined = self.input_ln(combined)
        
        # Processa cada timestep com a MLP Residual
        refined = self.feature_extractor(combined)
        
        # Passa pela LSTM e extrai contexto com Atenção
        outputs, _ = self.shared_lstm(refined)
        context = self.attention(outputs)
        return context

    def forward(self, 
                home_num_seq: torch.Tensor, home_cat_seq: torch.Tensor,
                away_num_seq: torch.Tensor, away_cat_seq: torch.Tensor,
                match_num: torch.Tensor, match_cat: torch.Tensor) -> torch.Tensor:
        
        # 1. Extrai a "fase" de cada time
        home_context = self._process_sequence(home_num_seq, home_cat_seq)
        away_context = self._process_sequence(away_num_seq, away_cat_seq)
        
        # 2. Processa as características do jogo de hoje (sem LSTM)
        match_embeddings = [
            self.embedding_layers[name](match_cat[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        match_cat_emb = self.embedding_dropout(torch.cat(match_embeddings, dim=-1))
        match_combined = torch.cat([match_num, match_cat_emb], dim=-1)
        match_combined = self.input_ln(match_combined)
        match_features = self.feature_extractor(match_combined)
        
        # 3. Funde tudo para a decisão
        fused = torch.cat([home_context, away_context, match_features], dim=-1)
        x = self.fusion_dense(fused)
        x = self.fusion_ln(x)
        x = self.fusion_act(x)
        x = self.fusion_drop(x)
        
        return x

class SiameseClassifier(nn.Module):
    def __init__(self, backbone: SiameseLSTMBackbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )
        initial_bias = torch.tensor([0.27, 0.49, 0.24]) 
        self.classification_head[-1].bias.data = torch.log(initial_bias)

    def forward(self, h_num, h_cat, a_num, a_cat, m_num, m_cat) -> torch.Tensor:
        backbone_output = self.backbone(h_num, h_cat, a_num, a_cat, m_num, m_cat)
        return self.classification_head(backbone_output)

class SiameseRegressor(nn.Module):
    def __init__(self, backbone: SiameseLSTMBackbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 2),
            nn.Softplus() 
        )

    def forward(self, h_num, h_cat, a_num, a_cat, m_num, m_cat) -> torch.Tensor:
        backbone_output = self.backbone(h_num, h_cat, a_num, a_cat, m_num, m_cat)
        return self.regression_head(backbone_output)