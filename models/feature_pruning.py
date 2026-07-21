import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

# Importa a função de carregamento do seu próprio dataloader
from dataloader import load_match_dataframe, FEATURE_COLUMNS, YEAR_COLUMN

def run_xgboost_pruning(csv_path: str):
    print("Carregando e preparando dados...")
    df = load_match_dataframe(csv_path)
    
    # 1. Definir Features (Numéricas + Categóricas)
    categorical_columns = [
        "arbitro", "estadio", "tecnico_mandante", "tecnico_visitante",
        "time_mandante", "time_visitante", 'missing_colocacao_mandante',
        'missing_colocacao_visitante', 'missing_rodada'
    ]
    
    # Converte tipos categóricos na base inteira para garantir que todas as categorias sejam conhecidas
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Filtra apenas as features que realmente existem no dataframe
    features_to_use = [col for col in FEATURE_COLUMNS + categorical_columns if col in df.columns]
    
    # ==========================================
    # IMPLEMENTAÇÃO DO MÉTODO DA VARIÁVEL SOMBRA
    # ==========================================
    print("Injetando variável sombra (ruído aleatório)...")
    # Cria uma coluna de ruído baseada em uma distribuição normal padrão
    shadow_feature_name = 'random_noise'
    df[shadow_feature_name] = np.random.randn(len(df))
    
    # Adiciona a variável sombra à lista de features que o XGBoost vai treinar
    features_to_use.append(shadow_feature_name)
    # ==========================================
    
    # 2. Criar Target Único para Classificação Multiclasse (0: Empate, 1: Mandante, 2: Visitante)
    target_cols = ['resultado_empate', 'resultado_vitoria_mandante', 'resultado_vitoria_visitante']
    df['target_class'] = np.argmax(df[target_cols].values, axis=1)
    
    # 3. Divisão Temporal
    train_mask = df[YEAR_COLUMN] <= 2025
    test_mask = df[YEAR_COLUMN] >= 2026
    
    X_train = df.loc[train_mask, features_to_use]
    y_train = df.loc[train_mask, 'target_class']
    
    X_test = df.loc[test_mask, features_to_use]
    y_test = df.loc[test_mask, 'target_class']

    print(f"Treinando XGBoost com {len(features_to_use)} features (incluindo sombra)...")
    
    # 4. Configuração e Treinamento do Modelo
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        enable_categorical=True,
        random_state=42,
        objective='multi:softprob'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50 
    )
    
    # 5. Avaliação Rápida
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAcurácia XGBoost no Teste (Baseline Tabular): {acc * 100:.2f}%\n")
    
    # 6. Extração e Plotagem do Feature Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features_to_use,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("--- TOP 15 FEATURES MAIS IMPORTANTES ---")
    print(importance_df.head(15).to_string(index=False))
    
    # ==========================================
    # ANÁLISE DO CORTE PELA VARIÁVEL SOMBRA
    # ==========================================
    print("\n" + "="*50)
    print("ANÁLISE DE CORTE (SHADOW FEATURE METHOD)")
    print("="*50)
    
    # Encontra o índice (posição no ranking) da variável sombra
    shadow_index = importance_df[importance_df['Feature'] == shadow_feature_name].index[0]
    shadow_importance = importance_df.loc[shadow_index, 'Importance']
    
    print(f"A variável sombra '{shadow_feature_name}' ficou na posição: {shadow_index + 1} de {len(importance_df)}")
    print(f"Importância da sombra: {shadow_importance:.6f}\n")
    
    # Separa as features que ficaram abaixo do ruído
    features_below_noise = importance_df.iloc[shadow_index + 1:]
    
    if len(features_below_noise) > 0:
        print(f"⚠️ ATENÇÃO: Encontradas {len(features_below_noise)} features com desempenho INFERIOR ao ruído puro.")
        print("Estas são as candidatas ideais para a lista FEATURES_TO_DROP no dataloader:")
        print(features_below_noise['Feature'].tolist())
    else:
        print("✅ Excelente! Nenhuma feature performou pior que o ruído aleatório. O dataset já está limpo.")
    print("="*50 + "\n")
    # ==========================================
    
    # Plotagem Visual com destaque para a variável sombra
    plt.figure(figsize=(14, 6))
    # Define as cores: destaca a sombra de vermelho
    colors = ['red' if feat == shadow_feature_name else 'royalblue' for feat in importance_df['Feature']]

    # Plota barras verticais: eixo x -> Features, eixo y -> Importance
    sns.barplot(
        x='Feature',
        y='Importance',
        data=importance_df,
        palette=colors,
    )
    plt.title('Feature Importance (Variável Sombra em Vermelho)')
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/xgboost_feature_importance.png')
    print("\nGráfico salvo como 'plots/xgboost_feature_importance.png'")

if __name__ == "__main__":
    # Substitua pelo caminho do seu dataset real
    run_xgboost_pruning("../data/dataset_preprocessed.csv")