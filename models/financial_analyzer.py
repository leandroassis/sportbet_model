"""
Módulo de Análise Financeira Integrada
Executa simulação de apostas com as previsões do modelo treinado em tempo real.
Suporta duas estratégias: Flat Betting (sempre aposta) e Value Betting (apenas +EV).
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

import numpy as np
from sklearn.metrics import classification_report


@dataclass
class BettingResult:
    """Resultado financeiro de uma aposta."""
    strategy_name: str
    predictions: np.ndarray
    pl_series: np.ndarray
    cumsum_pl: np.ndarray
    final_balance: float
    avg_ev: float
    bets_placed: int = 0
    bets_won: int = 0
    yield_roi: float = 0.0


@dataclass
class StrategyMetrics:
    """Métricas de desempenho de uma estratégia."""
    name: str
    accuracy: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    confusion_matrix: np.ndarray


class FinancialAnalyzer:
    """Análise financeira integrada com suporte a múltiplas estratégias."""

    CLASS_NAMES = ['Empate (0)', 'Mandante (1)', 'Visitante (2)']
    ODDS_COLS = ['AvgCD', 'AvgCH', 'AvgCA']  # ordem: empate, mandante, visitante
    TARGET_COLS = ['resultado_empate', 'resultado_vitoria_mandante', 'resultado_vitoria_visitante']
    PRED_COL_NAMES = ['pred_resultado_empate', 'pred_resultado_vitoria_mandante', 'pred_resultado_vitoria_visitante']

    def __init__(
        self,
        validation_dataframe: pd.DataFrame,
        model: torch.nn.Module,
        arch: str,
        model_type: str,
        numerical_feature_columns: list[str],
        categorical_feature_columns: list[str],
        sequence_length: int,
        validation_loader: DataLoader,
        device: torch.device,
        output_dir: Path = None,
        financial_strategy: str = "flat",
        ev_threshold: float = 0.0,
        betting_unit: float = 1.0,
        temperature: float = 1.0,
        catboost_model = None,
        verbose: bool = True,
        # --- PARÂMETROS DE ONLINE LEARNING ---
        online_learning: bool = False,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = None
    ):
        self.validation_dataframe = validation_dataframe
        self.model = model
        self.catboost_model = catboost_model
        self.arch = arch
        self.model_type = model_type
        self.numerical_feature_columns = numerical_feature_columns
        self.categorical_feature_columns = categorical_feature_columns
        self.sequence_length = sequence_length
        self.validation_loader = validation_loader
        self.device = device
        self.verbose = verbose

        self.financial_strategy = financial_strategy
        self.ev_threshold = ev_threshold
        self.betting_unit = betting_unit
        self.temperature = temperature
        
        self.online_learning = online_learning
        self.optimizer = optimizer
        self.criterion = criterion

        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent
        self.data_dir = self.output_dir / 'data'
        self.plots_dir = self.output_dir / 'plots'

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.df_predictions = None

        self._validate_odds_sanity()

    def _validate_odds_sanity(self) -> None:
        odds_values = self.validation_dataframe[self.ODDS_COLS].values.flatten()
        odds_values = odds_values[~np.isnan(odds_values)]

        avg_odds = np.mean(odds_values)
        min_odds = np.min(odds_values)

        if avg_odds < 1.01:
            raise ValueError(
                f"CRITICAL: Odds appear to be normalized/scaled (avg={avg_odds:.4f}). "
                "Expected raw odds from validation dataframe, not scaled data."
            )

        if self.verbose:
            print(f"✓ Odds validation passed: mean={avg_odds:.4f}, min={min_odds:.4f}")
            print(f"✓ Estratégia: {self.financial_strategy.upper()}")
            if self.financial_strategy == "ev":
                print(f"✓ Threshold de EV: {self.ev_threshold:.4f}")

    def generate_predictions(self) -> pd.DataFrame:
        df_with_preds = self.validation_dataframe.reset_index(drop=True).copy()

        for col in self.PRED_COL_NAMES:
            df_with_preds[col] = np.nan

        all_outputs_list = []
        
        if self.online_learning and self.verbose:
            print("Modo Walk-Forward (Online Learning) Ativado: O modelo aprenderá com os resultados em tempo real.")

        for batch in self.validation_loader:
            
            # ========================================================
            # FASE 1: PREVISÃO (CÉGO PARA O RESULTADO)
            # ========================================================
            self.model.eval()
            with torch.no_grad():
                if self.arch in ['siamese', 'hybrid']:
                    (h_num, h_cat, a_num, a_cat, m_num, m_cat), targets, _ = batch
                    outputs = self.model(
                        h_num.to(self.device), h_cat.to(self.device),
                        a_num.to(self.device), a_cat.to(self.device),
                        m_num.to(self.device), m_cat.to(self.device)
                    )
                    
                    if self.arch == 'hybrid' and self.catboost_model is not None:
                        emb_np = outputs.cpu().numpy()
                        try:
                            expected_features = len(self.catboost_model.feature_names_)
                        except Exception:
                            expected_features = emb_np.shape[1]
                            
                        if expected_features > emb_np.shape[1]:
                            m_num_np = m_num.cpu().numpy()
                            batch_features = np.hstack([m_num_np, emb_np])
                        else:
                            batch_features = emb_np
                            
                        batch_probs = self.catboost_model.predict_proba(batch_features)
                        all_outputs_list.append(batch_probs)
                        
                    else:
                        if self.model_type == 'classifier':
                            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                            all_outputs_list.append(probs)
                else:
                    numerical_feat, categorical_feat, targets, _ = batch
                    outputs = self.model(numerical_feat.to(self.device), categorical_feat.to(self.device))
                    if self.model_type == 'classifier':
                        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        all_outputs_list.append(probs)

            # ========================================================
            # FASE 2: ONLINE LEARNING (ATUALIZAÇÃO COM O RESULTADO REAL)
            # ========================================================
            if self.online_learning:
                # 2A. Atualiza a Rede PyTorch (Siamese, MLP, Legacy)
                # Bloqueio crítico: A arquitetura 'hybrid' usa um extrator congelado (128 dims), ignoramos o RPSLoss.
                if self.optimizer is not None and self.criterion is not None and self.arch != 'hybrid':
                    self.model.train()
                    self.optimizer.zero_grad()
                    
                    if self.arch == 'siamese':
                        out_update = self.model(
                            h_num.to(self.device), h_cat.to(self.device),
                            a_num.to(self.device), a_cat.to(self.device),
                            m_num.to(self.device), m_cat.to(self.device)
                        )
                    else:
                        out_update = self.model(numerical_feat.to(self.device), categorical_feat.to(self.device))
                        
                    loss = self.criterion(out_update, targets.to(self.device))
                    loss.backward()
                    self.optimizer.step()
                
                # 2B. Atualiza a Árvore CatBoost (Hybrid)
                if self.arch == 'hybrid' and self.catboost_model is not None:
                    try:
                        y_cat = targets.cpu().numpy()
                        loss_fn = self.catboost_model.get_param('loss_function')
                        
                        if 'MultiClass' in loss_fn and y_cat.ndim > 1:
                            y_cat = np.argmax(y_cat, axis=1)
                            
                        # Micro-ajuste Walk-Forward
                        self.catboost_model.fit(
                            batch_features, y_cat,
                            init_model=self.catboost_model,
                            iterations=1, 
                            verbose=False
                        )
                    except Exception:
                        pass # Suprime falhas em lotes residuais mínimos

        final_probabilities = np.vstack(all_outputs_list)

        if len(final_probabilities) < len(df_with_preds):
            diff = len(df_with_preds) - len(final_probabilities)
            if self.verbose: print(f"Alinhamento ativado: Removendo {diff} partidas de warm-up...")
            try:
                from dataloader import YEAR_COLUMN
                valid_indices = []
                ordered = df_with_preds.sort_values([YEAR_COLUMN])
                for _, season_frame in ordered.groupby(YEAR_COLUMN, sort=True):
                    if len(season_frame) < self.sequence_length:
                        continue
                    for end_index in range(self.sequence_length - 1, len(season_frame)):
                        valid_indices.append(season_frame.index[end_index])
                if len(valid_indices) == len(final_probabilities):
                    df_with_preds = df_with_preds.loc[valid_indices].reset_index(drop=True)
                else:
                    df_with_preds = df_with_preds.iloc[diff:].reset_index(drop=True)
            except ImportError:
                df_with_preds = df_with_preds.iloc[diff:].reset_index(drop=True)

        for i, col in enumerate(self.PRED_COL_NAMES):
            df_with_preds[col] = final_probabilities[:, i]

        if self.temperature != 1.0:
            if self.verbose: print(f"Calibrando probabilidades com Temperature Scaling (T={self.temperature})...")
            probs_calibradas = aplicar_temperature_scaling(df_with_preds, self.PRED_COL_NAMES, temperature=self.temperature)
            df_with_preds['pred_resultado_empate'] = probs_calibradas[:, 0]
            df_with_preds['pred_resultado_vitoria_mandante'] = probs_calibradas[:, 1]
            df_with_preds['pred_resultado_vitoria_visitante'] = probs_calibradas[:, 2]

        return df_with_preds

    def calculate_baseline_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Calcula predições baseline: sempre aposta no menor odd (favorito do mercado)."""
        predictions = np.zeros(len(df), dtype=int)

        for idx, (_, row) in enumerate(df.iterrows()):
            odds = [
                row.get('AvgCD', np.nan),  # empate (0)
                row.get('AvgCH', np.nan),  # mandante (1)
                row.get('AvgCA', np.nan),  # visitante (2)
            ]

            if any(np.isnan(v) for v in odds):
                predictions[idx] = 1  # Default: mandante
            else:
                predictions[idx] = np.argmin(odds)

        return predictions

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> StrategyMetrics:
        """Calcula métricas de classificação."""
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            output_dict=True, zero_division=0
        )

        precision = np.array([report['0']['precision'], report['1']['precision'], report['2']['precision']])
        recall = np.array([report['0']['recall'], report['1']['recall'], report['2']['recall']])
        f1 = np.array([report['0']['f1-score'], report['1']['f1-score'], report['2']['f1-score']])
        support = np.array([report['0']['support'], report['1']['support'], report['2']['support']], dtype=int)

        return StrategyMetrics(
            name=name,
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            confusion_matrix=cm
        )

    def simulate_flat_betting(self, df: pd.DataFrame, predictions: np.ndarray, predictions_prob: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, float, int, int, float]:
        """Simula flat betting (aposta na opção com Maior EV)."""
        pl_series = np.zeros(len(df))
        ev_series = np.zeros(len(df))
        bets_won = 0

        classe_real = (df[self.TARGET_COLS].values @ np.array([0, 1, 2])).astype(int)

        for idx in range(len(df)):
            if predictions_prob is not None:
                # ESTRATÉGIA DO MODELO: Buscar o MAIOR EV entre as 3 opções (0, 1, 2)
                evs = []
                for c in range(3):
                    odd_c = df.iloc[idx].get(self.ODDS_COLS[c], np.nan)
                    if np.isnan(odd_c) or odd_c <= 1.0:
                        evs.append(-np.inf)
                    else:
                        evs.append((predictions_prob[idx][c] * odd_c) - 1.0)
                
                # A classe escolhida passa a ser a que dá mais lucro a longo prazo!
                classe_predita = np.argmax(evs)
                odd_aposta = df.iloc[idx].get(self.ODDS_COLS[classe_predita], np.nan)
                ev_calc = evs[classe_predita]
            else:
                # ESTRATÉGIA BASELINE: Segue a predição da casa (geralmente o favorito/menor odd)
                classe_predita = predictions[idx]
                odd_aposta = df.iloc[idx].get(self.ODDS_COLS[classe_predita], np.nan)
                
                # EV teórico da Baseline (usamos 1/Odd como proxy de probabilidade)
                prob_implicita = 1.0 / odd_aposta if not np.isnan(odd_aposta) and odd_aposta > 0 else 0.0
                ev_calc = (prob_implicita * odd_aposta) - 1.0 if not np.isnan(odd_aposta) else 0.0

            if np.isnan(odd_aposta) or odd_aposta <= 1.0 or ev_calc == -np.inf:
                pl_series[idx] = 0.0
                ev_series[idx] = 0.0
            else:
                ev_series[idx] = ev_calc
                
                # Verificação de lucro ou prejuízo da aposta escolhida
                if classe_predita == classe_real[idx]:
                    pl_series[idx] = self.betting_unit * (odd_aposta - 1.0)
                    bets_won += 1
                else:
                    pl_series[idx] = -self.betting_unit

        cumsum_pl = np.cumsum(pl_series)
        final_balance = cumsum_pl[-1] if len(cumsum_pl) > 0 else 0.0
        avg_ev = np.mean(ev_series[~np.isnan(ev_series)])
        bets_placed = len(df)
        yield_roi = (final_balance / (bets_placed * self.betting_unit) * 100) if bets_placed > 0 else 0.0

        return pl_series, cumsum_pl, final_balance, avg_ev, bets_placed, bets_won, yield_roi


    def simulate_value_betting(self, df: pd.DataFrame, predictions: np.ndarray, predictions_prob: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, float, int, int, float]:
        """Simula value betting: aposta na opção com Maior EV apenas quando EV > threshold."""
        pl_series = np.zeros(len(df))
        ev_apostas_realizadas = []
        bets_placed = 0
        bets_won = 0

        classe_real = (df[self.TARGET_COLS].values @ np.array([0, 1, 2])).astype(int)

        for idx in range(len(df)):
            if predictions_prob is not None:
                # ESTRATÉGIA DO MODELO: Buscar o MAIOR EV entre as 3 opções
                evs = []
                for c in range(3):
                    odd_c = df.iloc[idx].get(self.ODDS_COLS[c], np.nan)
                    if np.isnan(odd_c) or odd_c <= 1.0:
                        evs.append(-np.inf)
                    else:
                        evs.append((predictions_prob[idx][c] * odd_c) - 1.0)
                
                classe_predita = np.argmax(evs)
                odd_aposta = df.iloc[idx].get(self.ODDS_COLS[classe_predita], np.nan)
                ev_calc = evs[classe_predita]
            else:
                # ESTRATÉGIA BASELINE: Segue a predição da casa
                classe_predita = predictions[idx]
                odd_aposta = df.iloc[idx].get(self.ODDS_COLS[classe_predita], np.nan)
                
                prob_implicita = 1.0 / odd_aposta if not np.isnan(odd_aposta) and odd_aposta > 0 else 0.0
                ev_calc = (prob_implicita * odd_aposta) - 1.0 if not np.isnan(odd_aposta) else 0.0

            if np.isnan(odd_aposta) or odd_aposta <= 1.0 or ev_calc == -np.inf:
                pl_series[idx] = 0.0
            else:
                # Filtro principal do Value Betting
                should_bet = (ev_calc >= self.ev_threshold)

                if should_bet:
                    bets_placed += 1
                    ev_apostas_realizadas.append(ev_calc)

                    if classe_predita == classe_real[idx]:
                        pl_series[idx] = self.betting_unit * (odd_aposta - 1.0)
                        bets_won += 1
                    else:
                        pl_series[idx] = -self.betting_unit
                else:
                    pl_series[idx] = 0.0

        cumsum_pl = np.cumsum(pl_series)
        final_balance = cumsum_pl[-1] if len(cumsum_pl) > 0 else 0.0
        avg_ev = np.mean(ev_apostas_realizadas) if ev_apostas_realizadas else 0.0
        yield_roi = (final_balance / (bets_placed * self.betting_unit) * 100) if bets_placed > 0 else 0.0

        return pl_series, cumsum_pl, final_balance, avg_ev, bets_placed, bets_won, yield_roi

    def plot_confusion_matrix(self, metrics_baseline: StrategyMetrics, metrics_model: StrategyMetrics) -> None:
        """Gera dashboard com matrizes de confusão."""
        fig = plt.figure(figsize=(16, 6))

        for idx, metrics in enumerate([metrics_baseline, metrics_model]):
            gs = fig.add_gridspec(1, 2, left=0.05 + idx*0.5, right=0.45 + idx*0.5, wspace=0.3)
            ax_hm = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])
            ax_table.axis('off')

            sns.heatmap(
                metrics.confusion_matrix,
                annot=True, fmt='d', cmap='Blues', ax=ax_hm,
                xticklabels=self.CLASS_NAMES,
                yticklabels=self.CLASS_NAMES,
                cbar=False
            )
            ax_hm.set_title(f'Matriz de Confusão - {metrics.name}', fontsize=12, fontweight='bold')
            ax_hm.set_ylabel('Verdadeiro')
            ax_hm.set_xlabel('Predito')

            table_data = [['Classe', 'Precision', 'Recall', 'F1-Score', 'Support']]
            for i, class_name in enumerate(self.CLASS_NAMES):
                table_data.append([
                    class_name,
                    f"{metrics.precision[i]:.3f}",
                    f"{metrics.recall[i]:.3f}",
                    f"{metrics.f1[i]:.3f}",
                    str(metrics.support[i])
                ])
            table_data.append(['Acurácia', f"{metrics.accuracy:.3f}", '', '', ''])

            table = ax_table.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for i in range(len(table_data)):
                if i == 0:
                    table[(i, 0)].set_facecolor('#40466e')
                    for j in range(5):
                        table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    for j in range(5):
                        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        plt.suptitle('Comparação de Desempenho: Baseline vs Modelo', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plots_dir / 'matriz_confusao.png', dpi=300, bbox_inches='tight')
        print("✓ Matriz de Confusão salva")
        plt.close()

    def plot_financial_evolution(self, results: Dict[str, BettingResult]) -> None:
        """Gera gráfico de evolução financeira."""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {'baseline': '#d62728', 'model': '#2ca02c'}

        for strategy_name, result in results.items():
            color = colors.get('baseline' if 'baseline' in strategy_name.lower() else 'model')
            ax.plot(
                range(len(result.cumsum_pl)),
                result.cumsum_pl,
                label=f"{strategy_name} (Final: {result.final_balance:.2f})",
                linewidth=2.5,
                color=color,
                alpha=0.8
            )

            final_idx = len(result.cumsum_pl) - 1
            ax.text(
                final_idx, result.cumsum_pl[final_idx],
                f" {result.final_balance:.2f}",
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2)
            )

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Break-even')
        ax.set_xlabel('Número de Partidas', fontsize=12, fontweight='bold')
        ax.set_ylabel('P&L Acumulado (Unidades)', fontsize=12, fontweight='bold')
        ax.set_title(f'Evolução Financeira: {self.financial_strategy.upper()} Betting', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        plt.tight_layout()
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plots_dir / 'evolucao_financeira.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de evolução financeira salvo")
        plt.close()

    def run(self) -> Dict[str, BettingResult]:
        if self.verbose:
            print("\n" + "="*80)
            print("ANÁLISE FINANCEIRA INTEGRADA")
            print("="*80)
            print("\nGerando predições do modelo...")
            
        df_predictions = self.generate_predictions()
        self.df_predictions = df_predictions  # GUARDA OS DADOS EXPORTÁVEIS
        df_predictions['classe_real'] = (df_predictions[self.TARGET_COLS].values @ np.array([0, 1, 2])).astype(int)

        if self.verbose: print("Calculando estratégia Baseline...")
        baseline_pred = self.calculate_baseline_predictions(df_predictions)

        if self.verbose: print("Calculando estratégia do Modelo...")
        model_pred = np.argmax(df_predictions[self.PRED_COL_NAMES].values, axis=1)
        model_probs = df_predictions[self.PRED_COL_NAMES].values

        if self.verbose: print("Computando métricas...")
        metrics_baseline = self.calculate_metrics(df_predictions['classe_real'].values, baseline_pred, 'Baseline (Odds)')
        metrics_model = self.calculate_metrics(df_predictions['classe_real'].values, model_pred, 'Modelo Preditivo')

        if self.verbose: print(f"Simulando apostas ({self.financial_strategy.upper()})...")
        if self.financial_strategy == 'flat':
            pl_base, cumsum_base, final_base, ev_base, bets_base, wins_base, yield_base = self.simulate_flat_betting(df_predictions, baseline_pred)
            pl_model, cumsum_model, final_model, ev_model, bets_model, wins_model, yield_model = self.simulate_flat_betting(df_predictions, model_pred, predictions_prob=model_probs)
        else:
            pl_base, cumsum_base, final_base, ev_base, bets_base, wins_base, yield_base = self.simulate_flat_betting(df_predictions, baseline_pred)
            pl_model, cumsum_model, final_model, ev_model, bets_model, wins_model, yield_model = self.simulate_value_betting(df_predictions, model_pred, predictions_prob=model_probs)

        results = {
            'Baseline': BettingResult(
                strategy_name='Baseline', predictions=baseline_pred, pl_series=pl_base,
                cumsum_pl=cumsum_base, final_balance=final_base, avg_ev=ev_base,
                bets_placed=bets_base, bets_won=wins_base, yield_roi=yield_base
            ),
            'Modelo': BettingResult(
                strategy_name='Modelo', predictions=model_pred, pl_series=pl_model,
                cumsum_pl=cumsum_model, final_balance=final_model, avg_ev=ev_model,
                bets_placed=bets_model, bets_won=wins_model, yield_roi=yield_model
            )
        }

        if self.verbose:
            print("Gerando visualizações...")
            self.plot_confusion_matrix(metrics_baseline, metrics_model)
            self.plot_financial_evolution(results)

            self.data_dir.mkdir(parents=True, exist_ok=True)
            df_predictions.to_csv(self.data_dir / f'analise_financeira_{self.arch}.csv', index=False)
            print(f"✓ Previsões salvas em data/analise_financeira_{self.arch}.csv")

            self._print_summary(df_predictions, metrics_baseline, metrics_model, results)

        return results

    def _print_summary(self, df: pd.DataFrame, metrics_baseline: StrategyMetrics, metrics_model: StrategyMetrics,
                      results: Dict[str, BettingResult]) -> None:
        """Imprime resumo executivo."""
        print("\n" + "="*80)
        print("RESUMO EXECUTIVO - ANÁLISE FINANCEIRA E DE CLASSIFICAÇÃO")
        print("="*80)

        print(f"\nDataset de Validação: {len(df)} partidas analisadas")

        # Financial Integrity Check
        print("\n" + "-"*80)
        print("VALIDAÇÃO DE INTEGRIDADE FINANCEIRA")
        print("-"*80)
        odds_values = df[self.ODDS_COLS].values.flatten()
        odds_values = odds_values[~np.isnan(odds_values)]
        print(f"  Odds - Média: {np.mean(odds_values):.4f}, Min: {np.min(odds_values):.4f}, Max: {np.max(odds_values):.4f}")
        print(f"  ✓ Odds estão em escala real (não escaladas)")
        print(f"  ✓ Fórmula de lucro: Lucro = (Odd - 1.0) × {self.betting_unit}")

        print("\n" + "-"*80)
        print("MÉTRICAS DE CLASSIFICAÇÃO")
        print("-"*80)
        print(f"\nBaseline (Odds do Mercado):")
        print(f"  Acurácia: {metrics_baseline.accuracy:.4f}")
        print(f'\n{"Classe":<20} {"Precision":>10} {"Recall":>10} {"F1-Score":>10} {"Support":>10}')
        for i, class_name in enumerate(self.CLASS_NAMES):
            print(f"{class_name:<20} {metrics_baseline.precision[i]:>10.3f} {metrics_baseline.recall[i]:>10.3f} {metrics_baseline.f1[i]:>10.3f} {metrics_baseline.support[i]:>10d}")

        print(f"\nModelo Preditivo:")
        print(f"  Acurácia: {metrics_model.accuracy:.4f}")
        print(f"  Melhoria: {(metrics_model.accuracy - metrics_baseline.accuracy):+.4f}")
        print(f'\n{"Classe":<20} {"Precision":>10} {"Recall":>10} {"F1-Score":>10} {"Support":>10}')
        for i, class_name in enumerate(self.CLASS_NAMES):
            print(f"{class_name:<20} {metrics_model.precision[i]:>10.3f} {metrics_model.recall[i]:>10.3f} {metrics_model.f1[i]:>10.3f} {metrics_model.support[i]:>10d}")

        print("\n" + "-"*80)
        print(f"SIMULAÇÃO FINANCEIRA ({self.financial_strategy.upper()} BETTING)")
        print("-"*80)

        for strategy_name, result in results.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  P&L Final: {result.final_balance:+.2f} unidades")
            print(f"  EV Médio: {result.avg_ev:.4f}")
            print(f"  Volume de Apostas: {result.bets_placed} de {len(df)} partidas ({(result.bets_placed/len(df))*100:.1f}%)")
            win_rate = (result.bets_won / result.bets_placed) * 100 if result.bets_placed > 0 else 0
            print(f"  Win Rate: {win_rate:.2f}% ({result.bets_won}/{result.bets_placed})")
            print(f"  Yield/ROI: {result.yield_roi:+.2f}%")

        print("\n" + "="*80 + "\n")

def aplicar_temperature_scaling(df, colunas_probs, temperature=1.5):
    """
    Aplica Temperature Scaling nas probabilidades pós-softmax.
    Matematicamente: p_novo = p_original^(1/T) / sum(p_original^(1/T))
    """
    # Extrai a matriz de probabilidades atual
    p = df[colunas_probs].values
    
    # Clip de segurança para evitar 0 absoluto antes da exponenciação
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    
    # Aplica o fator de temperatura
    p_scaled = np.power(p, 1.0 / temperature)
    
    # Re-normaliza para garantir que a soma das 3 classes feche em 100% (1.0)
    p_calibrada = p_scaled / np.sum(p_scaled, axis=1, keepdims=True)
    
    return p_calibrada