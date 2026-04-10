import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pasta onde as imagens geradas serão salvas
OUTPUT_DIR = os.path.join("results", "graphs")

# Cores fixas por modelo para manter consistência entre gráficos
MODEL_COLORS = {
    "Gemini":   "#4285F4",
    "Claude":   "#CC785C",
    "OpenAI":   "#10A37F",
    "DeepSeek": "#7B5EA7",
}

# Nomes amigáveis para exibição nos gráficos
METRIC_NAMES = {
    "precision": "Precision",
    "recall":    "Recall",
    "F1":        "F1-Score",
}

STAT_NAMES = {
    "media":   "Média",
    "mediana": "Mediana",
    "desvio":  "Desvio",
}


def _get_color(model, index):
    """Retorna a cor do modelo ou uma cor padrão pelo índice."""
    fallback = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    return MODEL_COLORS.get(model, fallback[index % len(fallback)])


def _save(fig, filename):
    """Cria a pasta de saída (se necessário) e salva o gráfico."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# GRÁFICO 1 — Barras agrupadas

def plot_bar(df, metrics, stats):
    models = df["model"].tolist()
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.7 / n_models

    fig, axes = plt.subplots(1, len(stats), figsize=(10 * len(stats), 6), squeeze=False)

    for col, stat in enumerate(stats):
        ax = axes[0][col]
        for i, model in enumerate(models):
            offset = (i - n_models / 2 + 0.5) * width
            values = [df.loc[df["model"] == model, f"bert_{m}_{stat}"].values[0] for m in metrics]

            bars = ax.bar(x + offset, values, width * 0.9,
                          label=model, color=_get_color(model, i))

            # Anota o valor em cima de cada barra
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_NAMES[m] for m in metrics], fontsize=12)
        ax.set_ylabel(f"Score ({STAT_NAMES[stat]})")
        ax.set_title(f"Bar Chart — {STAT_NAMES[stat]}")
        ax.legend(title="Model")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("BERT Metrics — Grouped Bar Chart", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "bar_chart.png")


# GRÁFICO 2 — Box plot
def plot_box(df, metrics, stats):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6), sharey=False)

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, (_, row) in enumerate(df.iterrows()):
            model = row["model"]
            mean  = row[f"bert_{metric}_media"]
            std   = row[f"bert_{metric}_desvio"]
            med   = row[f"bert_{metric}_mediana"]
            color = _get_color(model, i)

            # Caixa mean ± std (sempre desenhada como base do box plot)
            ax.bar(i, 2 * std, bottom=mean - std, width=0.5,
                   color=color, alpha=0.5, edgecolor=color, linewidth=1.5)

            # Traço da mediana (exibido se "mediana" foi selecionada)
            if "mediana" in stats:
                ax.plot([i - 0.25, i + 0.25], [med, med], color=color, linewidth=2.5)

            # Ponto da média (exibido se "media" foi selecionada)
            if "media" in stats:
                ax.scatter([i], [mean], color="white", edgecolors=color, s=60, zorder=5)

            # Whiskers ±2σ (exibidos se "desvio" foi selecionado)
            if "desvio" in stats:
                ax.plot([i, i], [mean - 2 * std, mean - std], color=color, lw=1.2, ls="--")
                ax.plot([i, i], [mean + std, mean + 2 * std], color=color, lw=1.2, ls="--")

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["model"].tolist(), rotation=20, fontsize=10)
        ax.set_title(METRIC_NAMES[metric])
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legenda descrevendo o que está visível
    shown = []
    if "media"   in stats: shown.append("○ = Mean")
    if "mediana" in stats: shown.append("— = Median")
    if "desvio"  in stats: shown.append("box = ±SD  |  whiskers = ±2SD")
    fig.suptitle("BERT Metrics — Box Plot\n" + "   ".join(shown), fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "box_plot.png")


# GRÁFICO 3 — Radar
def plot_radar(df, metrics, stats):
    if len(metrics) < 3:
        print("  [Aviso] Radar requer pelo menos 3 métricas. Pulando...")
        return

    labels = [METRIC_NAMES[m] for m in metrics]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(stats), figsize=(7 * len(stats), 7),
                             subplot_kw={"polar": True})

    if len(stats) == 1:
        axes = [axes]

    for ax, stat in zip(axes, stats):
        for i, (_, row) in enumerate(df.iterrows()):
            model  = row["model"]
            values = [row[f"bert_{m}_{stat}"] for m in metrics]
            values += values[:1]
            color  = _get_color(model, i)

            ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
        ax.set_title(f"Radar — {STAT_NAMES[stat]}", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    fig.suptitle("BERT Metrics — Radar Chart", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "radar_chart.png")


# GRÁFICO 4 — Heatmap
def plot_heatmap(df, metrics, stats):
    col_labels = [f"{METRIC_NAMES[m]} {STAT_NAMES[s]}" for m in metrics for s in stats]
    data = []
    for _, row in df.iterrows():
        data.append([row[f"bert_{m}_{s}"] for m in metrics for s in stats])

    matrix = np.array(data)
    models = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.3), max(4, len(models) * 1.3)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Anota o valor em cada célula
    for r in range(len(models)):
        for c in range(len(col_labels)):
            ax.text(c, r, f"{matrix[r, c]:.4f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if matrix[r, c] > 0.85 else "black")

    ax.set_title("BERT Metrics — Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "heatmap.png")


# GRÁFICO 5 — Line chart
def plot_line(df, metrics, stats):
    if len(metrics) < 2:
        print("  [Aviso] Line chart requer pelo menos 2 métricas. Pulando...")
        return

    # Estilos de linha para diferenciar as estatísticas visualmente
    stat_styles = {"media": "-", "mediana": "--", "desvio": ":"}

    labels = [METRIC_NAMES[m] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        model = row["model"]
        color = _get_color(model, i)

        for stat in stats:
            values = [row[f"bert_{m}_{stat}"] for m in metrics]
            ls     = stat_styles[stat]
            # Rótulo só na primeira estatística para não duplicar na legenda
            label = model if stat == stats[0] else None

            ax.plot(range(len(metrics)), values, "o" + ls,
                    label=label, color=color, linewidth=2, markersize=7, alpha=0.85)

            # Anota o valor e o nome da estatística em cada ponto
            for xi, v in enumerate(values):
                ax.annotate(f"{v:.4f}\n({STAT_NAMES[stat]})",
                            (xi, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color, alpha=0.8)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("BERT Metrics — Line Chart")
    ax.legend(title="Model")
    ax.grid(linestyle="--", alpha=0.4)

    # Explica os estilos de linha no eixo X
    style_legend = "  ".join([f"{stat_styles[s]} = {STAT_NAMES[s]}" for s in stats])
    ax.set_xlabel(f"Line style:  {style_legend}", fontsize=9)

    fig.tight_layout()
    _save(fig, "line_chart.png")


# ──────────────────────────────────────────────────────────
# DISPATCHER — chamado pelo menu da main
# ──────────────────────────────────────────────────────────

CHART_OPTIONS = {
    1: ("Bar Chart",   plot_bar),
    2: ("Box Plot",    plot_box),
    3: ("Radar Chart", plot_radar),
    4: ("Heatmap",     plot_heatmap),
    5: ("Line Chart",  plot_line),
}

METRIC_OPTIONS = {
    1: ["precision", "recall", "F1"],
    2: ["precision"],
    3: ["recall"],
    4: ["F1"],
    5: ["precision", "F1"],
    6: ["recall", "F1"],
}

STAT_OPTIONS = {
    1: ["media", "mediana", "desvio"],  # todas
    2: ["media"],
    3: ["mediana"],
    4: ["desvio"],
    5: ["media", "mediana"],
    6: ["media", "desvio"],
    7: ["mediana", "desvio"],
}


def run_charts(stats_csv_path):
    """
    Exibe o submenu de gráficos e gera o gráfico escolhido.
    Recebe o caminho do CSV de estatísticas gerado por generate_statistics().
    """
    # Verifica se o CSV de estatísticas existe antes de continuar
    if not os.path.exists(stats_csv_path):
        print(f"\n[Erro] Arquivo de estatísticas não encontrado: {stats_csv_path}")
        print("Execute a opção 4 primeiro para gerar as estatísticas.\n")
        return

    df = pd.read_csv(stats_csv_path)

    # ── Seleção de métricas ──
    print("\n--- Métricas ---")
    print("1 - Todas (Precision, Recall, F1)")
    print("2 - Apenas Precision")
    print("3 - Apenas Recall")
    print("4 - Apenas F1")
    print("5 - Precision + F1")
    print("6 - Recall + F1")
    m_op    = int(input("Escolha as métricas: "))
    metrics = METRIC_OPTIONS.get(m_op, METRIC_OPTIONS[1])

    # ── Seleção de estatísticas ──
    print("\n--- Estatísticas ---")
    print("1 - Todas (Média, Mediana, Desvio Padrão)")
    print("2 - Apenas Média")
    print("3 - Apenas Mediana")
    print("4 - Apenas Desvio Padrão")
    print("5 - Média + Mediana")
    print("6 - Média + Desvio Padrão")
    print("7 - Mediana + Desvio Padrão")
    s_op  = int(input("Escolha as estatísticas: "))
    stats = STAT_OPTIONS.get(s_op, STAT_OPTIONS[1])

    # ── Seleção de gráfico ──
    print("\n--- Tipo de Gráfico ---")
    for key, (name, _) in CHART_OPTIONS.items():
        print(f"{key} - {name}")
    print("6 - Todos os gráficos")
    c_op = int(input("Escolha o gráfico: "))

    print()
    if c_op == 6:
        for _, (name, func) in CHART_OPTIONS.items():
            print(f"Gerando {name}...")
            func(df, metrics, stats)
    elif c_op in CHART_OPTIONS:
        name, func = CHART_OPTIONS[c_op]
        print(f"Gerando {name}...")
        func(df, metrics, stats)
    else:
        print("Opção inválida.")