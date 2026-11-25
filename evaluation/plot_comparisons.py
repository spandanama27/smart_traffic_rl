"""
Comparison Plotting and Visualization Module.
Generates publication-quality Matplotlib figures comparing:
- Average Vehicle Waiting Time
- Average & Peak Queue Lengths
- Total Throughput
- Congestion Index
- Time-series Queue Trajectories
"""
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_kpi_comparison(df_summary: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """
    Creates a 4-panel comparison bar chart across controllers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0F172A")

    controllers = df_summary["controller"].tolist()
    colors = ["#EF4444", "#3B82F6", "#10B981"][: len(controllers)]

    def _style_ax(ax, title, ylabel):
        ax.set_facecolor("#1E293B")
        ax.set_title(title, color="#F8FAFC", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color="#94A3B8", fontsize=10)
        ax.tick_params(colors="#94A3B8", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # 1. Average Waiting Time
    ax = axes[0, 0]
    bars = ax.bar(controllers, df_summary["avg_waiting_time"], color=colors, width=0.55, edgecolor="#0F172A")
    ax.bar_label(bars, fmt="%.1fs", padding=3, color="#F8FAFC", fontsize=9, fontweight="bold")
    _style_ax(ax, f"Average Vehicle Waiting Time {title_suffix}", "Seconds (lower is better)")

    # 2. Average Queue Length
    ax = axes[0, 1]
    bars = ax.bar(controllers, df_summary["avg_queue_length"], color=colors, width=0.55, edgecolor="#0F172A")
    ax.bar_label(bars, fmt="%.1f", padding=3, color="#F8FAFC", fontsize=9, fontweight="bold")
    _style_ax(ax, f"Average Queue Length {title_suffix}", "Vehicles (lower is better)")

    # 3. Total Throughput
    ax = axes[1, 0]
    bars = ax.bar(controllers, df_summary["total_throughput"], color=colors, width=0.55, edgecolor="#0F172A")
    ax.bar_label(bars, fmt="%d", padding=3, color="#F8FAFC", fontsize=9, fontweight="bold")
    _style_ax(ax, f"Total Vehicles Cleared (Throughput) {title_suffix}", "Vehicles (higher is better)")

    # 4. Congestion Index
    ax = axes[1, 1]
    bars = ax.bar(controllers, df_summary["avg_congestion_index"], color=colors, width=0.55, edgecolor="#0F172A")
    ax.bar_label(bars, fmt="%.2f", padding=3, color="#F8FAFC", fontsize=9, fontweight="bold")
    _style_ax(ax, f"Average Congestion Index {title_suffix}", "Index 0.0 - 1.0 (lower is better)")

    plt.tight_layout()
    return fig


def plot_queue_trajectories(telemetries: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Plots dynamic second-by-second total queue trajectories over simulation time.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    palette = {"Fixed-Time Controller": "#EF4444", "Q-Learning Agent": "#3B82F6", "DQN Agent": "#10B981"}

    for name, df in telemetries.items():
        color = palette.get(name, "#F59E0B")
        # Moving average for smooth visual comparison
        smooth = df["total_queue"].rolling(window=7, min_periods=1).mean()
        ax.plot(df["step"], smooth, label=name, color=color, linewidth=2.2)

    ax.set_title("Queue Length Dynamics Over Simulation Horizon", color="#F8FAFC", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Simulation Ticks / Seconds", color="#94A3B8", fontsize=10)
    ax.set_ylabel("Total Queued Vehicles", color="#94A3B8", fontsize=10)
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    leg = ax.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=10)
    plt.tight_layout()
    return fig
