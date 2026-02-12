"""
Generate paper-ready results: tables, figures, and qualitative examples.

Reads experiment results JSON files and produces:
- LaTeX-formatted tables
- Publication-quality figures (PNG)
- Qualitative example tables

Usage:
    python analysis/generate_results.py --results-dir results/ --output figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import seaborn as sns

# Style for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sns.set_palette("deep")


# ═══════════════════════════════════════════════════════
# Experiment 1 Figures
# ═══════════════════════════════════════════════════════

def plot_ocr_degradation(exp1_results: dict, output_dir: Path):
    """Figure 1: nDCG@10 degradation across OCR quality tiers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    tiers = ["high", "medium", "low"]
    tier_labels = ["High\n(>90%)", "Medium\n(70–90%)", "Low\n(<70%)"]
    
    bm25_scores = []
    dense_scores = []
    
    bm25_data = exp1_results["models"]["bm25"]["by_ocr_tier"]
    dense_data = exp1_results["models"]["dense"]["by_ocr_tier"]
    
    for tier in tiers:
        bm25_scores.append(bm25_data.get(tier, {}).get("nDCG@10", {}).get("mean", 0))
        dense_scores.append(dense_data.get(tier, {}).get("nDCG@10", {}).get("mean", 0))
    
    x = np.arange(len(tiers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bm25_scores, width, label='BM25', 
                   color='#2196F3', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, dense_scores, width, label='Dense (BGE)', 
                   color='#FF5722', alpha=0.85, edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('OCR Quality Tier')
    ax.set_ylabel('nDCG@10')
    ax.set_title('Retrieval Performance Degradation Under OCR Noise')
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add degradation annotation
    if bm25_scores[0] > 0 and bm25_scores[-1] < bm25_scores[0]:
        drop = (bm25_scores[0] - bm25_scores[-1]) / bm25_scores[0] * 100
        ax.annotate(
            f'{drop:.0f}% drop',
            xy=(2, bm25_scores[-1]),
            xytext=(1.5, bm25_scores[0] * 0.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=11, color='red', fontweight='bold',
        )
    
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_ocr_degradation.png")
    plt.close(fig)
    print(f"  Saved: fig1_ocr_degradation.png")


def plot_retrieval_comparison(exp1_results: dict, output_dir: Path):
    """Figure: Multi-metric comparison between BM25 and Dense by OCR tier."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    tiers = ["high", "medium", "low"]
    metrics = ["nDCG@10", "MRR", "Recall@10"]
    
    for ax, metric in zip(axes, metrics):
        bm25_vals = []
        dense_vals = []
        bm25_stds = []
        dense_stds = []
        
        for tier in tiers:
            bm25_tier = exp1_results["models"]["bm25"]["by_ocr_tier"].get(tier, {})
            dense_tier = exp1_results["models"]["dense"]["by_ocr_tier"].get(tier, {})
            
            bm25_vals.append(bm25_tier.get(metric, {}).get("mean", 0))
            bm25_stds.append(bm25_tier.get(metric, {}).get("std", 0))
            dense_vals.append(dense_tier.get(metric, {}).get("mean", 0))
            dense_stds.append(dense_tier.get(metric, {}).get("std", 0))
        
        x = np.arange(len(tiers))
        width = 0.35
        
        ax.bar(x - width/2, bm25_vals, width, yerr=bm25_stds,
               label='BM25', color='#2196F3', alpha=0.85, capsize=3)
        ax.bar(x + width/2, dense_vals, width, yerr=dense_stds,
               label='Dense', color='#FF5722', alpha=0.85, capsize=3)
        
        ax.set_xlabel('OCR Tier')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(["High", "Med", "Low"])
        ax.set_ylim(0, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if ax == axes[0]:
            ax.legend()
    
    fig.suptitle('Retrieval Metrics by OCR Quality Tier', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "fig1b_retrieval_comparison.png")
    plt.close(fig)
    print(f"  Saved: fig1b_retrieval_comparison.png")


# ═══════════════════════════════════════════════════════
# Experiment 2 Figures
# ═══════════════════════════════════════════════════════

def plot_hallucination_vs_ocr(exp2_results: dict, output_dir: Path):
    """Figure 2: Hallucination rate vs OCR accuracy scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Collect data points from detailed results
    for config_key, config_data in exp2_results.items():
        detailed = config_data.get("detailed_results", [])
        
        if not detailed:
            # Use tier-level summary data
            by_tier = config_data.get("by_ocr_tier", {})
            tier_centers = {"high": 0.95, "medium": 0.80, "low": 0.60}
            
            ocr_values = []
            hallu_values = []
            for tier, center in tier_centers.items():
                if tier in by_tier:
                    ocr_values.append(center)
                    hallu_values.append(by_tier[tier].get("hallucination_rate", 0))
            
            if ocr_values:
                model_label = config_key.replace("_", " / ")
                ax.scatter(ocr_values, [h * 100 for h in hallu_values], 
                          s=100, label=model_label, alpha=0.8, zorder=5)
                
                # Trend line
                if len(ocr_values) >= 2:
                    z = np.polyfit(ocr_values, [h * 100 for h in hallu_values], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(ocr_values) - 0.05, max(ocr_values) + 0.05, 100)
                    ax.plot(x_line, p(x_line), '--', alpha=0.5)
        else:
            # Use individual question results
            ocr_values = [r.get("ocr_quality", 0.5) for r in detailed if r.get("ocr_quality")]
            hallu_values = [1 if r["eval"]["is_hallucination"] else 0 for r in detailed if r.get("ocr_quality")]
            
            if ocr_values:
                model_label = config_key.replace("_", " / ")
                ax.scatter(ocr_values, [h * 100 for h in hallu_values],
                          s=30, alpha=0.5, label=model_label)
    
    ax.set_xlabel('OCR Quality Score')
    ax.set_ylabel('Hallucination Rate (%)')
    ax.set_title('RAG Hallucination Rate vs. OCR Quality')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(-5, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Reference annotation
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.42, 52, 'random baseline', fontsize=8, color='gray')
    
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_hallucination_vs_ocr.png")
    plt.close(fig)
    print(f"  Saved: fig2_hallucination_vs_ocr.png")


def plot_confidence_gap(exp2_results: dict, output_dir: Path):
    """Figure: Confidence scores for correct vs hallucinated answers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    configs = []
    correct_confs = []
    hallu_confs = []
    
    for config_key, config_data in exp2_results.items():
        overall = config_data.get("overall", {})
        if "correct_mean_confidence" in overall and "hallucinated_mean_confidence" in overall:
            configs.append(config_key.replace("_", "\n"))
            correct_confs.append(overall["correct_mean_confidence"])
            hallu_confs.append(overall["hallucinated_mean_confidence"])
    
    if not configs:
        print("  ⚠ No confidence data available for gap plot")
        return
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, correct_confs, width, label='Correct Answers',
           color='#4CAF50', alpha=0.85)
    ax.bar(x + width/2, hallu_confs, width, label='Hallucinated Answers',
           color='#F44336', alpha=0.85)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean Confidence Score')
    ax.set_title('Confidence-Hallucination Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(output_dir / "fig2b_confidence_gap.png")
    plt.close(fig)
    print(f"  Saved: fig2b_confidence_gap.png")


# ═══════════════════════════════════════════════════════
# Experiment 3 Figures
# ═══════════════════════════════════════════════════════

def plot_length_impact(exp3_results: dict, output_dir: Path):
    """Figure 3: Impact of document length on retrieval strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    comp = exp3_results.get("strategy_comparison", {})
    percentiles = ["P25", "P50", "P90", "P99"]
    available = [p for p in percentiles if p in comp]
    
    if not available:
        print("  ⚠ No strategy comparison data available")
        return
    
    # Left: Coverage and chunk count
    word_counts = [comp[p]["mean_word_count"] for p in available]
    n_chunks = [comp[p]["mean_n_chunks"] for p in available]
    coverage = [comp[p]["mean_coverage_at_5"] * 100 for p in available]
    
    ax1_twin = ax1.twinx()
    
    bars = ax1.bar(available, coverage, color='#2196F3', alpha=0.7, label='Coverage@5 (%)')
    line = ax1_twin.plot(available, n_chunks, 'o-', color='#FF5722', 
                         linewidth=2, markersize=8, label='# Chunks')
    
    ax1.set_xlabel('Document Length Percentile')
    ax1.set_ylabel('Coverage@5 (%)', color='#2196F3')
    ax1_twin.set_ylabel('Number of Chunks', color='#FF5722')
    ax1.set_title('Chunking Coverage vs. Document Length')
    
    # Add word count labels
    for bar, wc in zip(bars, word_counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{wc:,.0f}w', ha='center', fontsize=8, color='gray')
    
    ax1.set_ylim(0, 110)
    ax1.spines['top'].set_visible(False)
    
    # Right: Inter-chunk coherence
    inter_sim = [comp[p].get("mean_inter_chunk_sim", 0) for p in available]
    chunk_doc_sim = [comp[p].get("mean_chunk_doc_sim", 0) for p in available]
    
    ax2.plot(available, inter_sim, 'o-', color='#9C27B0', linewidth=2,
             markersize=8, label='Inter-chunk similarity')
    ax2.plot(available, chunk_doc_sim, 's-', color='#FF9800', linewidth=2,
             markersize=8, label='Chunk-to-doc similarity')
    
    ax2.set_xlabel('Document Length Percentile')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Coherence Loss with Chunking')
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle('Document Length Impact on Retrieval', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_length_impact.png")
    plt.close(fig)
    print(f"  Saved: fig3_length_impact.png")


# ═══════════════════════════════════════════════════════
# LaTeX Tables
# ═══════════════════════════════════════════════════════

def generate_latex_table1(exp1_results: dict) -> str:
    """Table 1: Retrieval metrics by OCR tier."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval performance stratified by OCR quality tier. "
        r"Higher OCR noise correlates with significant metric degradation.}",
        r"\label{tab:retrieval_ocr}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & OCR Tier & nDCG@10 & MRR & Recall@10 \\",
        r"\midrule",
    ]
    
    for model_key, model_name in [("bm25", "BM25"), ("dense", "Dense (BGE)")]:
        model_data = exp1_results["models"][model_key]["by_ocr_tier"]
        first = True
        for tier in ["high", "medium", "low"]:
            if tier not in model_data:
                continue
            d = model_data[tier]
            name_col = model_name if first else ""
            first = False
            ndcg = f"{d['nDCG@10']['mean']:.3f}"
            mrr_val = f"{d['MRR']['mean']:.3f}"
            recall = f"{d['Recall@10']['mean']:.3f}"
            lines.append(f"{name_col} & {tier.capitalize()} & {ndcg} & {mrr_val} & {recall} \\\\")
        lines.append(r"\midrule")
    
    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_latex_summary_table(exp1: dict, exp2: dict, exp3: dict) -> str:
    """Summary table combining key findings from all experiments."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary of key findings across all experiments.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{p{2.5cm}p{4.5cm}p{5cm}}",
        r"\toprule",
        r"Challenge & Finding & Implication \\",
        r"\midrule",
    ]
    
    # OCR noise finding
    if exp1:
        bm25_tiers = exp1.get("models", {}).get("bm25", {}).get("by_ocr_tier", {})
        if "high" in bm25_tiers and "low" in bm25_tiers:
            high = bm25_tiers["high"]["nDCG@10"]["mean"]
            low = bm25_tiers["low"]["nDCG@10"]["mean"]
            drop = (high - low) / high * 100 if high > 0 else 0
            lines.append(
                f"OCR Noise & {drop:.0f}\\% drop in nDCG@10 between high/low OCR tiers & "
                f"Need for noise-robust retrieval models \\\\"
            )
    
    # RAG hallucination finding
    if exp2:
        for key, val in exp2.items():
            by_tier = val.get("by_ocr_tier", {})
            if "high" in by_tier and "low" in by_tier:
                high_h = by_tier["high"].get("hallucination_rate", 0)
                low_h = by_tier["low"].get("hallucination_rate", 0)
                increase = (low_h - high_h) / max(high_h, 0.01) * 100
                lines.append(
                    f"RAG Hallucin. & {increase:.0f}\\% higher error rate on low-OCR docs & "
                    f"Critical for deployment safety \\\\"
                )
                break
    
    # Document length finding
    if exp3:
        comp = exp3.get("strategy_comparison", {})
        if "P99" in comp:
            cov = comp["P99"].get("mean_coverage_at_5", 0) * 100
            lines.append(
                f"Doc. Length & Standard chunking covers only {cov:.0f}\\% of P99 docs & "
                f"Motivates hierarchical retrieval \\\\"
            )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate paper results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating paper-ready results...")
    print(f"  Results: {results_dir}")
    print(f"  Output: {output_dir}")
    
    exp1 = exp2 = exp3 = None
    
    # Experiment 1
    exp1_path = results_dir / "experiment1_results.json"
    if exp1_path.exists():
        with open(exp1_path) as f:
            exp1 = json.load(f)
        print("\n  Experiment 1: Retrieval")
        plot_ocr_degradation(exp1, output_dir)
        plot_retrieval_comparison(exp1, output_dir)
    else:
        print(f"\n  ⚠ {exp1_path} not found")
    
    # Experiment 2
    exp2_path = results_dir / "experiment2_results.json"
    if exp2_path.exists():
        with open(exp2_path) as f:
            exp2 = json.load(f)
        print("\n  Experiment 2: RAG")
        plot_hallucination_vs_ocr(exp2, output_dir)
        plot_confidence_gap(exp2, output_dir)
    else:
        print(f"\n  ⚠ {exp2_path} not found")
    
    # Experiment 3
    exp3_path = results_dir / "experiment3_results.json"
    if exp3_path.exists():
        with open(exp3_path) as f:
            exp3 = json.load(f)
        print("\n  Experiment 3: Length")
        plot_length_impact(exp3, output_dir)
    else:
        print(f"\n  ⚠ {exp3_path} not found")
    
    # LaTeX tables
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(exist_ok=True)
    
    if exp1:
        table1 = generate_latex_table1(exp1)
        with open(latex_dir / "table1_retrieval.tex", "w") as f:
            f.write(table1)
        print(f"\n  Saved: latex/table1_retrieval.tex")
    
    if exp1 or exp2 or exp3:
        summary = generate_latex_summary_table(exp1, exp2, exp3)
        with open(latex_dir / "table_summary.tex", "w") as f:
            f.write(summary)
        print(f"  Saved: latex/table_summary.tex")
    
    print(f"\n✓ All results generated in {output_dir}")


if __name__ == "__main__":
    main()
