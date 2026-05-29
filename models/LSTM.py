import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Calcula o peso de cada timestep da LSTM para formar um vetor de contexto focado,
    dando mais importância a rodadas chave na sequência histórica.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(lstm_outputs)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector


class TabularFeatureExtractor(nn.Module):
    """
    Residual MLP com ativação GELU para pré-processar dados tabulares ruidosos.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        
        hidden_dim = output_dim * 2 
        
        self.feature_crossing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.feature_crossing(x)
        return transformed + self.skip_connection(x)


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        numerical_input_size: int,
        embedding_settings,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings, dim)
            for name, (num_embeddings, dim) in embedding_settings.dimensions.items()
        })
        
        self.embedding_dropout = nn.Dropout(dropout * 0.8)
        
        # Tamanho total da entrada bruta (numéricos + embeddings concatenados)
        raw_input_size = numerical_input_size + sum(dim for _, dim in embedding_settings.dimensions.values())
        self.input_ln = nn.LayerNorm(raw_input_size)
        
        # --------------------------------------------------------------------
        # INSERÇÃO DO FEATURE EXTRACTOR
        # Ele recebe o dado bruto e mastiga para o tamanho esperado pela LSTM
        # --------------------------------------------------------------------
        self.feature_extractor = TabularFeatureExtractor(
            input_dim=raw_input_size,
            output_dim=hidden_size,
            dropout=dropout
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size, # Agora a LSTM recebe a saída do extrator
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.attention = TemporalAttention(hidden_size)
        
        self.latent_dense1 = nn.Linear(hidden_size, hidden_size)
        self.latent_ln1 = nn.LayerNorm(hidden_size)
        self.latent_act1 = nn.ReLU()
        self.latent_drop1 = nn.Dropout(dropout)
        
        self.latent_dense2 = nn.Linear(hidden_size, hidden_size)
        self.latent_ln2 = nn.LayerNorm(hidden_size)
        self.latent_act2 = nn.ReLU()
        self.latent_drop2 = nn.Dropout(dropout)

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        embeddings = [
            self.embedding_layers[name](categorical_inputs[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        
        cat_embeddings = self.embedding_dropout(torch.cat(embeddings, dim=-1))
        combined_input = torch.cat([numerical_inputs, cat_embeddings], dim=-1)
        
        # 1. Normaliza a entrada bruta
        combined_input = self.input_ln(combined_input)
        
        # 2. Passa pela MLP Residual para cruzar as features
        refined_features = self.feature_extractor(combined_input)
        
        # 3. Alimenta a LSTM com as features limpas
        outputs, _ = self.lstm(refined_features)
        
        # 4. Captura o contexto via Atenção
        context = self.attention(outputs)
        
        # 5. Processamento Latente Final
        x = self.latent_dense1(context)
        x = self.latent_ln1(x)
        x = self.latent_act1(x)
        x = self.latent_drop1(x)
        
        residual = x
        x = self.latent_dense2(x)
        x = self.latent_ln2(x)
        x = x + residual
        x = self.latent_act2(x)
        x = self.latent_drop2(x)
        
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, backbone: LSTMBackbone, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 3)
        )
        # Distribuição: Empate (~0.27), Mandante (~0.49), Visitante (~0.24)
        initial_bias = torch.tensor([0.27, 0.49, 0.24]) 
        # Usando o logaritmo das probabilidades
        self.classification_head[-1].bias.data = torch.log(initial_bias)

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(numerical_inputs, categorical_inputs)
        return self.classification_head(backbone_output)


class LSTMRegressor(nn.Module):
    def __init__(self, backbone: LSTMBackbone, hidden_size: int, dropout: float) -> None:
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

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(numerical_inputs, categorical_inputs)
        return self.regression_head(backbone_output)
