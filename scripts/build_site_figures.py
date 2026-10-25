"""Draw the website's focused overview from frozen daily NAVs; no backtest changes."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".build/mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.font_manager import FontProperties, findfont, fontManager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def build_overview():
    font_path = os.environ.get("MICROCAP_FONT")
    if not font_path:
        for family in ["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC", "Heiti SC", "STHeiti"]:
            try:
                font_path = findfont(FontProperties(family=family), fallback_to_default=False)
                break
            except ValueError:
                pass
    if not font_path and Path('/System/Library/Fonts/STHeiti Light.ttc').is_file():
        font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    if not font_path:
        raise RuntimeError("Install a CJK font or set MICROCAP_FONT before regenerating the figure.")
    fontManager.addfont(font_path)
    font = FontProperties(fname=font_path)
    plt.rcParams.update({
        "font.family": font.get_name(), "font.size": 10,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cbd7e5",
        "axes.labelcolor": "#4d6078", "xtick.color": "#4d6078", "ytick.color": "#4d6078",
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.hashsalt": "microcap-overview",
    })
    winner = json.loads((ROOT / "results/selection.json").read_text())["winner"]
    daily = pd.read_csv(ROOT / "results/daily_nav.csv")
    daily['date'] = pd.to_datetime(daily.date, format='mixed')
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1], "hspace": .16})
    for strategy, label, color, ls in [
        (winner, "1—100名 · 冻结赢家", "#155bb5", "-"),
        ("CSI1000", "中证1000 · 价格指数", "#75879c", "--"),
    ]:
        full = daily[daily.strategy.eq(strategy)].sort_values("date")
        previous = full[full.date.lt("2024-01-01")].iloc[-1]
        period = full[full.date.between("2024-01-01", "2025-12-31")]
        nav = np.r_[1.0, period.nav.to_numpy() / previous.nav]
        dates = [previous.date, *period.date]
        axes[0].plot(dates, nav, color=color, ls=ls, lw=1.8, label=label)
        drawdown = nav / np.maximum.accumulate(nav) - 1
        axes[1].plot(dates, drawdown, color=color, ls=ls, lw=1.5)
        if strategy == winner:
            axes[1].fill_between(dates, drawdown, 0, color=color, alpha=.07)
    axes[0].set_ylabel("归一化净值")
    axes[0].legend(loc="upper left", frameon=False, prop=font, ncol=2)
    axes[1].set_ylabel("日度回撤")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_xticks(pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2025-12-31"]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    for ax in axes:
        ax.grid(axis="y", color="#e5ebf3", lw=.7)
        ax.set_axisbelow(True)
        ax.margins(x=.015)
        for label in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontproperties(font)
    fig.subplots_adjust(left=.095, right=.98, bottom=.09, top=.97)
    output = ROOT / "docs/assets/overview_nav.svg"
    fig.savefig(output, metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    print(f"Built {output.relative_to(ROOT)}")


if __name__ == "__main__":
    build_overview()
