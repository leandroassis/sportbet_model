import pandas as pd
import numpy as np
from pathlib import Path

# 2. Parâmetros e Constantes do Modelo
ELO_INICIAL = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 10.0
REGRESSAO_MEDIA_FATOR = 0.2

def apply_elo_system(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função que itera sobre o DataFrame histórico de partidas e preenche as colunas
    'elo_mandante' e 'elo_visitante' simulando o sistema de Rating ELO sem data leakage.
    O DataFrame precisa estar temporalmente ordenado.
    """
    
    # 3. Controle de Estado (Memória)
    elo_tracker = {}
    ano_vigente = None
    
    elo_mandante_list = []
    elo_visitante_list = []
    
    # Fazemos uma cópia para não alterar o DataFrame original caso seja passado por referência
    df_result = df.copy()
    
    # Vamos garantir a ordenação antes caso existam essas colunas, para não ferir a lógica do loop
    if all(col in df_result.columns for col in ['ano_campeonato', 'data', 'rodada']):
        df_result = df_result.sort_values(by=['ano_campeonato', 'data', 'rodada']).reset_index(drop=True)
    
    # 4. Algoritmo de Iteração e Lógica Passo a Passo
    for row in df_result.itertuples():
        ano_atual = row.ano_campeonato
        
        # Passo 4.1: Checagem de Virada de Temporada
        if ano_vigente is not None and ano_atual != ano_vigente:
            for time in elo_tracker:
                elo_tracker[time] = (elo_tracker[time] * (1 - REGRESSAO_MEDIA_FATOR)) + (ELO_INICIAL * REGRESSAO_MEDIA_FATOR)
        
        ano_vigente = ano_atual
        
        time_mandante = row.time_mandante
        time_visitante = row.time_visitante
        gols_mandante = float(row.gols_mandante)
        gols_visitante = float(row.gols_visitante)
        
        # Passo 4.2: Inicialização Lazy (Casos de Borda)
        if time_mandante not in elo_tracker:
            elo_tracker[time_mandante] = ELO_INICIAL
        if time_visitante not in elo_tracker:
            elo_tracker[time_visitante] = ELO_INICIAL
            
        # Passo 4.3: Extração de Features
        elo_mandante_atual = elo_tracker[time_mandante]
        elo_visitante_atual = elo_tracker[time_visitante]
        
        elo_mandante_list.append(elo_mandante_atual)
        elo_visitante_list.append(elo_visitante_atual)
        
        # Passo 4.4: Cálculos de Atualização Pós-Jogo
        # A. Cálculo da Expectativa (Probabilidade de Vitória)
        expectativa_mandante = 1 / (1 + 10 ** ((elo_visitante_atual - (elo_mandante_atual + HOME_ADVANTAGE)) / 400.0))
        expectativa_visitante = 1 - expectativa_mandante
        
        # B. Multiplicador de Margem de Gols (G)
        margem_gols = np.sqrt(1 + abs(gols_mandante - gols_visitante))
        
        # C. Definição do Resultado Real (S)
        if gols_mandante > gols_visitante:
            S_M, S_V = 1.0, 0.0
        elif gols_mandante < gols_visitante:
            S_M, S_V = 0.0, 1.0
        else:
            S_M, S_V = 0.5, 0.5
            
        # D. Atualização do Tracker
        novo_elo_mand = elo_mandante_atual + (K_FACTOR * margem_gols * (S_M - expectativa_mandante))
        novo_elo_vis = elo_visitante_atual + (K_FACTOR * margem_gols * (S_V - expectativa_visitante))
        
        elo_tracker[time_mandante] = novo_elo_mand
        elo_tracker[time_visitante] = novo_elo_vis
        
    df_result['elo_mandante'] = elo_mandante_list
    df_result['elo_visitante'] = elo_visitante_list
    
    return df_result

if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "dataset_preprocessed.csv"
    print(f"Lendo dataset: {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print("Processando histórico e inserindo ELO em", len(df), "partidas...")
    
    df_com_elo = apply_elo_system(df)
    
    df_com_elo.to_csv(csv_path, index=False)
    print(f"Features 'elo_mandante' e 'elo_visitante' adicionadas e salvas com sucesso em: {csv_path}")
