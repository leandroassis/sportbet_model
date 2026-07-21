from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


CLASS_LABELS = {
    0: "Empate",
    1: "Mandante",
    2: "Visitante",
}

PRED_COLUMNS = [
    "pred_resultado_empate",
    "pred_resultado_vitoria_mandante",
    "pred_resultado_vitoria_visitante",
]

MODEL_FILES = {
    "Legacy": "analise_financeira_legacy.csv",
    "MLP": "analise_financeira_mlp.csv",
    "Hybrid": "analise_financeira_hybrid.csv",
    "Siamese": "analise_financeira_siamese.csv",
}


def _compute_roc_curves(y_true: np.ndarray, y_score: np.ndarray) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, float]], np.ndarray, np.ndarray, float]:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    per_class: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for class_idx in range(3):
        # Some splits may miss a class entirely; skip invalid ROC in that case.
        if len(np.unique(y_true_bin[:, class_idx])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
        per_class[class_idx] = (fpr, tpr, auc(fpr, tpr))

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    return per_class, fpr_micro, tpr_micro, auc_micro


def _plot_model_roc(ax: plt.Axes, model_name: str, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    missing = [col for col in ["classe_real", *PRED_COLUMNS] if col not in df.columns]
    if missing:
        ax.text(
            0.5,
            0.5,
            f"Colunas ausentes:\n{', '.join(missing)}",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_title(model_name)
        ax.set_axis_off()
        return

    y_true = df["classe_real"].to_numpy(dtype=int)
    y_score = df[PRED_COLUMNS].to_numpy(dtype=float)

    per_class, fpr_micro, tpr_micro, auc_micro = _compute_roc_curves(y_true, y_score)

    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    class_aucs = []

    for class_idx in range(3):
        if class_idx not in per_class:
            continue

        fpr, tpr, roc_auc = per_class[class_idx]
        class_aucs.append(roc_auc)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=colors[class_idx],
            label=f"{CLASS_LABELS[class_idx]} (AUC={roc_auc:.3f})",
        )

    ax.plot(fpr_micro, tpr_micro, color="tab:red", lw=1.8, linestyle="--", label=f"Micro média (AUC={auc_micro:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")

    macro_auc = float(np.mean(class_aucs)) if class_aucs else np.nan
    title_suffix = f"Macro AUC={macro_auc:.3f}" if not np.isnan(macro_auc) else "Macro AUC=NA"
    ax.set_title(f"{model_name} | {title_suffix}", fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, (model_name, filename) in enumerate(MODEL_FILES.items()):
        csv_path = data_dir / filename
        ax = axes[idx]

        if not csv_path.exists():
            ax.text(
                0.5,
                0.5,
                f"Arquivo nao encontrado:\n{csv_path.name}",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )
            ax.set_title(model_name)
            ax.set_axis_off()
            continue

        _plot_model_roc(ax, model_name, csv_path)

    fig.suptitle("ROC Multiclasse (OVR) e AUC por Modelo", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_path = plots_dir / "roc_multiclasse_4_modelos.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {out_path}")


if __name__ == "__main__":
    main()
