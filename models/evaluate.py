"""
evaluate_robustness.py
Realiza N iterações de treino e análise financeira do modelo escolhido.
Calcula e extrai a média de desempenho e P&L para estabilizar a variância.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Importações do ecossistema do seu projeto
from dataloader import load_match_dataframe, split_match_dataframe
from financial_analyzer import FinancialAnalyzer
from train import train_model, build_catboost_dataset, get_device

try:
    from SiameseHybrid import SiameseEmbeddingExtractor
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Avaliação de Robustez Financeira (N Iterações)")
    
    # Parâmetro principal da robustez
    parser.add_argument("--n_iterations", type=int, default=5, help="Número de iterações do ensemble/Monte Carlo.")
    
    # Repasse dos parâmetros do train.py
    parser.add_argument("--arch", choices=["legacy", "mlp", "siamese", "hybrid"], default="mlp")
    parser.add_argument("--model_type", choices=["classifier"], default="classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--csv_path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "dataset_preprocessed.csv"))
    parser.add_argument("--financial_strategy", choices=["flat", "ev"], default="ev")
    parser.add_argument("--ev_threshold", type=float, default=0.1)
    parser.add_argument("--betting_unit", type=float, default=10.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--verbose', action='store_true', help="Exibe detalhes adicionais durante a avaliação financeira.")

    args = parser.parse_args()

    # Preparações
    df_full = load_match_dataframe(Path(args.csv_path))
    device = get_device(None)

    all_results = []
    all_df_preds = []

    print("\n" + "="*80)
    print(f"INICIANDO AVALIAÇÃO DE ROBUSTEZ ({args.n_iterations} ITERAÇÕES)")
    print(f"Arquitetura: {args.arch.upper()} | Estratégia: {args.financial_strategy.upper()} | Limiar EV: {args.ev_threshold}")
    print("="*80)

    for i in range(args.n_iterations):
        print(f"Treinando iteração {i+1}/{args.n_iterations} (Modo Silencioso)...", end=" ")
        
        # 1. Treinamento da Rede Base
        output = train_model(
            csv_path=Path(args.csv_path),
            arch=args.arch,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.lr,
            num_workers=args.workers,
            save_path=None,
            best_model_path=None,
            validation_predictions_path=Path(__file__).resolve().parents[0] / f"results/respostas_iter_{i}.csv",
            early_stopping_patience=args.early_stopping_patience,
            verbose=False  # <-- SUPRIME O TERMINAL DURANTE O TREINO
        )
        
        pytorch_model_final = output["model"]
        catboost_model_final = None

        # 2. Tratamento do CatBoost se a Arquitetura for Hybrid
        if args.arch == 'hybrid' and CATBOOST_AVAILABLE:
            extractor = SiameseEmbeddingExtractor(output["model"].backbone).to(device)
            pytorch_model_final = extractor
            
            X_train, y_train = build_catboost_dataset(output["bundle"].train_loader, extractor, device, one_hot_encoded=False)
            X_val, y_val = build_catboost_dataset(output["bundle"].validation_loader, extractor, device, one_hot_encoded=False)
            
            catboost_model_final = CatBoostClassifier(
                iterations=3000, learning_rate=0.01, depth=3,
                loss_function='MultiCrossEntropy', eval_metric='MultiCrossEntropy',
                use_best_model=True, od_type='Iter', od_wait=150, 
                verbose=0  # <-- CATBOOST SILENCIOSO
            )
            catboost_model_final.fit(X_train, y_train, eval_set=(X_val, y_val))

        # 3. Restaura as Odds originais para a matriz de Validação Financeira
        original_odds = df_full[['AvgCH', 'AvgCD', 'AvgCA']].copy()
        validation_split = split_match_dataframe(df_full)
        validation_dataframe = validation_split.validation.copy()
        validation_dataframe['AvgCH'] = original_odds.loc[validation_dataframe.index, 'AvgCH']
        validation_dataframe['AvgCD'] = original_odds.loc[validation_dataframe.index, 'AvgCD']
        validation_dataframe['AvgCA'] = original_odds.loc[validation_dataframe.index, 'AvgCA']

        # 4. Avaliação Financeira
        analyzer = FinancialAnalyzer(
            validation_dataframe=validation_dataframe,
            model=pytorch_model_final,
            catboost_model=catboost_model_final,
            arch=args.arch,
            model_type=args.model_type,
            numerical_feature_columns=output["bundle"].numerical_feature_columns,
            categorical_feature_columns=output["bundle"].categorical_feature_columns,
            sequence_length=args.seq_len,
            validation_loader=output["bundle"].validation_loader,
            device=device,
            financial_strategy=args.financial_strategy,
            ev_threshold=args.ev_threshold,
            betting_unit=args.betting_unit,
            temperature=args.temperature,
            verbose=args.verbose
        )
        
        results = analyzer.run()
        all_results.append(results)
        all_df_preds.append(analyzer.df_predictions)
        
        # Feedback unitário
        print(f"✓ Concluído! P&L do Modelo: {results['Modelo'].final_balance:+.2f}")

    # =========================================================================
    # AGREGAÇÃO E CONSTRUÇÃO DO CSV E RELATÓRIOS
    # =========================================================================
    print("\n" + "="*80)
    print("CONSTRUINDO DADOS MÉDIOS E RELATÓRIO FINAL")
    print("="*80)
    
    # Inicializa DataFrame base com a classe e Odds do primeiro resultado
    final_df = all_df_preds[0][['AvgCD', 'AvgCH', 'AvgCA', 'classe_real']].copy()
    
    # Acumuladores de média
    avg_cumsum_model = np.zeros_like(all_results[0]['Modelo'].cumsum_pl)
    avg_cumsum_base = np.zeros_like(all_results[0]['Baseline'].cumsum_pl)
    avg_ev_model = 0.0
    avg_yield = 0.0
    avg_win_rate = 0.0

    for i in range(args.n_iterations):
        res = all_results[i]['Modelo']
        df_i = all_df_preds[i]
        
        # Colunas com probabilidades e lucros específicos de cada fold
        final_df[f'Iter_{i}_Prob_Empate'] = df_i['pred_resultado_empate']
        final_df[f'Iter_{i}_Prob_Mandante'] = df_i['pred_resultado_vitoria_mandante']
        final_df[f'Iter_{i}_Prob_Visitante'] = df_i['pred_resultado_vitoria_visitante']
        final_df[f'Iter_{i}_PL_Aposta'] = res.pl_series
        
        # Soma para médias
        avg_cumsum_model += res.cumsum_pl
        avg_cumsum_base += all_results[i]['Baseline'].cumsum_pl
        avg_ev_model += res.avg_ev
        avg_yield += res.yield_roi
        avg_win_rate += (res.bets_won / res.bets_placed * 100) if res.bets_placed > 0 else 0.0

    # Fechamento das médias
    avg_cumsum_model /= args.n_iterations
    avg_cumsum_base /= args.n_iterations
    avg_ev_model /= args.n_iterations
    avg_yield /= args.n_iterations
    avg_win_rate /= args.n_iterations
    
    # PL e Probabilidades Médias consolidadas por jogo
    final_df['Media_Prob_Empate'] = final_df[[f'Iter_{i}_Prob_Empate' for i in range(args.n_iterations)]].mean(axis=1)
    final_df['Media_Prob_Mandante'] = final_df[[f'Iter_{i}_Prob_Mandante' for i in range(args.n_iterations)]].mean(axis=1)
    final_df['Media_Prob_Visitante'] = final_df[[f'Iter_{i}_Prob_Visitante' for i in range(args.n_iterations)]].mean(axis=1)
    final_df['Media_PL_Aposta_Jogo'] = final_df[[f'Iter_{i}_PL_Aposta' for i in range(args.n_iterations)]].mean(axis=1)

    # Exporta para CSV
    csv_filename = f"data/{args.arch}_financial_records.csv"
    final_df.to_csv(csv_filename, index=False)
    print(f"✓ Registros completos salvos em: {csv_filename}")

    # Plot Gráfico Médio
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_cumsum_base, label=f"Baseline Média (Final: {avg_cumsum_base[-1]:.2f})", color='#d62728', linewidth=2.5)
    ax.plot(avg_cumsum_model, label=f"Modelo Média (Final: {avg_cumsum_model[-1]:.2f})", color='#2ca02c', linewidth=2.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Número de Partidas de Validação', fontweight='bold')
    ax.set_ylabel(f'P&L Médio Acumulado (Unidades de {args.betting_unit})', fontweight='bold')
    ax.set_title(f'Evolução Financeira Média - {args.n_iterations} Iterações ({args.arch.upper()})', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plot_filename = f"plots/evolucao_media_{args.arch}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"✓ Gráfico médio salvo em: {plot_filename}\n")

    # Sumário Final Terminal
    print("="*80)
    print(f"SUMÁRIO FINANCEIRO GLOBAL ({args.n_iterations} ITERAÇÕES) - ARCH: {args.arch.upper()}")
    print("="*80)
    print(f"P&L Final Médio: {avg_cumsum_model[-1]:+.2f} unidades")
    print(f"EV Médio Global: {avg_ev_model:.4f}")
    print(f"Win Rate Médio:  {avg_win_rate:.2f}%")
    print(f"Yield/ROI Médio: {avg_yield:+.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()