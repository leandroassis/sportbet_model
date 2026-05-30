import torch
import torch.nn as nn
from LSTM import TabularFeatureExtractor

class TabularMLPBackbone(nn.Module):
    def __init__(
        self,
        numerical_input_size: int,
        embedding_settings,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.embedding_layers = nn.ModuleDict({
            name: nn.Embedding(num_embeddings, dim)
            for name, (num_embeddings, dim) in embedding_settings.dimensions.items()
        })
        
        self.embedding_dropout = nn.Dropout(dropout * 0.8)
        
        raw_input_size = numerical_input_size + sum(dim for _, dim in embedding_settings.dimensions.values())
        self.input_ln = nn.LayerNorm(raw_input_size)
        
        self.feature_extractor = TabularFeatureExtractor(
            input_dim=raw_input_size,
            output_dim=hidden_size,
            dropout=dropout
        )
        
        # Como não há LSTM, adicionamos mais profundidade na MLP para compensar
        self.latent_dense1 = nn.Linear(hidden_size, hidden_size)
        self.latent_ln1 = nn.LayerNorm(hidden_size)
        self.latent_act1 = nn.ReLU()
        self.latent_drop1 = nn.Dropout(dropout)

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        embeddings = [
            self.embedding_layers[name](categorical_inputs[..., i])
            for i, name in enumerate(self.embedding_layers)
        ]
        
        cat_embeddings = self.embedding_dropout(torch.cat(embeddings, dim=-1))
        combined_input = torch.cat([numerical_inputs, cat_embeddings], dim=-1)
        
        combined_input = self.input_ln(combined_input)
        refined_features = self.feature_extractor(combined_input)
        
        x = self.latent_dense1(refined_features)
        x = self.latent_ln1(x)
        x = self.latent_act1(x)
        x = self.latent_drop1(x)
        
        return x

class TabularClassifier(nn.Module):
    def __init__(self, backbone: TabularMLPBackbone, hidden_size: int, dropout: float) -> None:
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

    def forward(self, numerical_inputs: torch.Tensor, categorical_inputs: torch.Tensor) -> torch.Tensor:
        backbone_output = self.backbone(numerical_inputs, categorical_inputs)
        return self.classification_head(backbone_output)

class TabularRegressor(nn.Module):
    def __init__(self, backbone: TabularMLPBackbone, hidden_size: int, dropout: float) -> None:
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