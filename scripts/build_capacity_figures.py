"""Draw completed capacity results without rerunning or changing the strategy.

Use the research Python environment with matplotlib. Word authoring is a
separate step using the bundled artifact runtime. All eight scenarios must have
finished, although each chart deliberately shows only a focused comparison.
"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.build/mpl_cache'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report_capacity import COMPARISONS, load_capacity, number, sha


def build_figures(root=ROOT, output_dir=None):
    data = load_capacity(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont, fontManager
    from matplotlib.ticker import PercentFormatter, FixedLocator, FuncFormatter

    root = Path(root)
    output_dir = Path(output_dir) if output_dir else root / 'results/word_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = os.environ.get('MICROCAP_FONT')
    if not font_path:
        for family in ['Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'STHeiti']:
            try:
                font_path = findfont(FontProperties(family=family), fallback_to_default=False)
                break
            except ValueError:
                pass
    if not font_path and Path('/System/Library/Fonts/STHeiti Light.ttc').is_file():
        font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    if not font_path:
        raise RuntimeError('Install a CJK font or set MICROCAP_FONT before drawing.')
    fontManager.addfont(font_path)
    font = FontProperties(fname=font_path)
    plt.rcParams.update({
        'font.family': font.get_name(), 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#cbd7e5', 'axes.labelcolor': '#4d6078',
        'xtick.color': '#4d6078', 'ytick.color': '#4d6078',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.hashsalt': 'microcap-capacity',
    })
    capitals = sorted(data['protocol']['capitals_yuan'])
    x = [c / 10000 for c in capitals]
    scenarios = [('MAIN_1P', '主压力 1%', '#155bb5', '-'),
                 ('TIGHT_0P5', '更紧 0.5%', '#102b50', '-.'),
                 ('LOOSE_5P', '更宽 5%', '#6fa8da', ':')]

    def y(period, scenario, metric):
        return [number(data['indexed'][(period, scenario, c)], metric) for c in capitals]

    def style_axis(ax, percent=True):
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val:,.0f}'))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.tick_params(axis='x', labelrotation=35, labelsize=8.5)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis='y', color='#e5ebf3', lw=.7)
        ax.set_axisbelow(True)
        ax.set_xlabel('期初资金（万元，对数刻度）')
        ax.margins(x=.03)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for scenario, label, color, ls in scenarios:
        for ax, metric in zip(axes, ['cagr', 'attempt_completion']):
            ax.plot(x, y('TEST', scenario, metric), label=label, color=color,
                    linestyle=ls, marker='o', markersize=3.2, linewidth=1.7)
    axes[0].plot(x, y('TEST', 'OPEN_BASE', 'cagr'), color='#75879c',
                 ls='--', lw=1.5, label='原开盘基准')
    axes[0].set_title('2024—2025  成本后年化收益', loc='left', color='#102b50', fontsize=11)
    axes[1].set_title('每日尝试金额完成率', loc='left', color='#102b50', fontsize=11)
    axes[0].set_ylabel('复合年化收益')
    axes[1].set_ylabel('实际金额 / 计划金额')
    axes[1].set_ylim(0, 1.04)
    for ax in axes:
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.01),
               frameon=False, ncol=4, fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.81, bottom=.24, wspace=.26)
    for suffix in ['png', 'svg']:
        kwargs = {'dpi': 230} if suffix == 'png' else {'metadata': {'Date': None}}
        fig.savefig(output_dir / f'capacity_stress.{suffix}', bbox_inches='tight', **kwargs)
    plt.close(fig)

    # Separate panels preserve the decomposition without hiding the relatively
    # small volatility-coverage effect beneath the larger execution changes.
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0), sharex=True)
    for ax, (scenario, reference, label) in zip(axes.flat, COMPARISONS):
        for period, name, color, ls in [('TRAIN', '2015—2023', '#6fa8da', '--'),
                                       ('TEST', '2024—2025', '#155bb5', '-')]:
            values = [100 * number(data['differences'][(period, c, scenario, reference)], 'cagr_difference')
                      for c in capitals]
            ax.plot(x, values, label=name, color=color, ls=ls, marker='o', ms=3.1, lw=1.6)
        ax.axhline(0, color='#cbd7e5', lw=.8)
        ax.set_title(f'{label}\n{scenario} − {reference}', loc='left',
                     color='#102b50', fontsize=10, linespacing=1.5)
        ax.set_ylabel('年化收益差（百分点）')
        style_axis(ax, percent=False)
    for ax in axes[0]:
        ax.set_xlabel('')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', frameon=False, ncol=2)
    fig.subplots_adjust(left=.08, right=.985, top=.85, bottom=.12, hspace=.62, wspace=.28)
    for suffix in ['png', 'svg']:
        kwargs = {'dpi': 230} if suffix == 'png' else {'metadata': {'Date': None}}
        fig.savefig(output_dir / f'capacity_decomposition.{suffix}', bbox_inches='tight', **kwargs)
    plt.close(fig)
    manifest = {'capacity_run_id': data['manifest']['run_id'],
                'full_predeclared_grid': True,
                'sources': {name: sha(root / 'results/capacity' / name)
                            for name in ['protocol.json', 'manifest.json', 'performance.csv', 'performance_differences.csv']},
                'outputs': {f'{name}.{ext}': sha(output_dir / f'{name}.{ext}')
                            for name in ['capacity_stress', 'capacity_decomposition'] for ext in ['png', 'svg']}}
    (root / 'results/capacity/figure_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Built capacity_stress and capacity_decomposition PNG/SVG from the full frozen grid.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    build_figures(args.root, args.output_dir)
