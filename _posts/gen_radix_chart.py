import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
scenarios = [
    "shared_prefix\n_basic",
    "high_reuse\n_many_req",
    "branching\n_conversation",
    "multi_prefix\n_groups",
    "low_reuse\n_control",
    "eviction\n_pressure",
]

flat_hit    = [77.8, 85.2, 84.0, 76.9, 0.0,  0.0]
radix_hit   = [87.5, 95.8, 87.5, 83.3, 0.0, 79.2]

flat_tput   = [0.94, 0.77, 0.87, 0.70, 0.87, 0.76]
radix_tput  = [1.02, 0.82, 1.10, 0.91, 0.99, 0.91]

flat_evict  = [0, 0, 0, 0, 0, 136]
radix_evict = [0, 0, 0, 0, 0,   0]

flat_cached  = [112, 552, 168, 320,   0,   0]
radix_cached = [112, 552, 168, 320,   0, 288]

# ── Colors (matching KV cache benchmark) ──────────────────────────────────────
ORANGE   = '#F28C28'
BLUE     = '#2196F3'
GREEN    = '#4CAF50'
RED      = '#E53935'
DARK_BG  = '#FAFAFA'
GRID_CLR = '#E0E0E0'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.facecolor': 'white',
    'figure.facecolor': DARK_BG,
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.color': GRID_CLR,
    'grid.alpha': 0.5,
    'grid.linewidth': 0.5,
})

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Radix Tree Prefix Caching Benchmark Results',
             fontsize=18, fontweight='bold', y=0.98)
fig.text(0.5, 0.955,
         '0.056769M parameter model | block_size=4 | CPU | higher hit rate & throughput is better',
         ha='center', fontsize=9, color='#666666')

gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                       left=0.07, right=0.97, top=0.92, bottom=0.06)

x = np.arange(len(scenarios))
w = 0.32

# ── Panel 1: Hit Rate ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars_f = ax1.bar(x - w/2, flat_hit,  w, color=ORANGE, label='flat_cache', zorder=3)
bars_r = ax1.bar(x + w/2, radix_hit, w, color=BLUE,   label='radix_tree', zorder=3)
ax1.set_ylabel('Hit Rate (%)', fontweight='bold')
ax1.set_title('Cache Hit Rate by Scenario', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=8)
ax1.set_ylim(0, 110)
ax1.legend(loc='upper right', fontsize=9)

for i in range(len(scenarios)):
    diff = radix_hit[i] - flat_hit[i]
    if diff > 0:
        ax1.text(x[i] + w/2, radix_hit[i] + 2,
                 f'+{diff:.1f}%', ha='center', fontsize=8,
                 fontweight='bold', color=GREEN)
    elif radix_hit[i] == 0 and flat_hit[i] == 0:
        ax1.text(x[i], 5, '—', ha='center', fontsize=9, color='#999999')

# highlight the eviction_pressure win
ax1.annotate('', xy=(5 + w/2, radix_hit[5] + 8), xytext=(5 + w/2, radix_hit[5] + 20),
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
ax1.text(5 + w/2, radix_hit[5] + 22, '79.2% vs 0%!', ha='center',
         fontsize=9, fontweight='bold', color=GREEN)

# ── Panel 2: Throughput ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars_f2 = ax2.bar(x - w/2, flat_tput,  w, color=ORANGE, label='flat_cache', zorder=3)
bars_r2 = ax2.bar(x + w/2, radix_tput, w, color=BLUE,   label='radix_tree', zorder=3)
ax2.axhline(y=1.0, color='#999999', linestyle='--', linewidth=1, zorder=2, label='no_cache baseline')
ax2.set_ylabel('Throughput (×)', fontweight='bold')
ax2.set_title('Throughput vs No-Cache Baseline', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=8)
ax2.set_ylim(0, 1.35)
ax2.legend(loc='upper right', fontsize=9)

for i in range(len(scenarios)):
    clr = GREEN if radix_tput[i] >= 1.0 else RED
    ax2.text(x[i] + w/2, radix_tput[i] + 0.02,
             f'{radix_tput[i]:.2f}x', ha='center', fontsize=8,
             fontweight='bold', color=clr)

# ── Panel 3: Evictions ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bars_f3 = ax3.bar(x - w/2, flat_evict,  w, color=ORANGE, label='flat_cache', zorder=3)
bars_r3 = ax3.bar(x + w/2, radix_evict, w, color=BLUE,   label='radix_tree', zorder=3)
ax3.set_ylabel('Evictions', fontweight='bold')
ax3.set_title('Cache Evictions by Scenario', fontweight='bold', fontsize=13)
ax3.set_xticks(x)
ax3.set_xticklabels(scenarios, fontsize=8)
ax3.legend(loc='upper left', fontsize=9)

ax3.text(5 - w/2, flat_evict[5] + 3, '136', ha='center', fontsize=9,
         fontweight='bold', color=RED)
ax3.text(5 + w/2, radix_evict[5] + 3, '0', ha='center', fontsize=9,
         fontweight='bold', color=GREEN)

# ── Panel 4: Cached Tokens ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
bars_f4 = ax4.bar(x - w/2, flat_cached,  w, color=ORANGE, label='flat_cache', zorder=3)
bars_r4 = ax4.bar(x + w/2, radix_cached, w, color=BLUE,   label='radix_tree', zorder=3)
ax4.set_ylabel('Cached Tokens', fontweight='bold')
ax4.set_title('Tokens Served from Cache', fontweight='bold', fontsize=13)
ax4.set_xticks(x)
ax4.set_xticklabels(scenarios, fontsize=8)
ax4.legend(loc='upper right', fontsize=9)

for i in range(len(scenarios)):
    diff = radix_cached[i] - flat_cached[i]
    if diff > 0:
        ax4.text(x[i] + w/2, radix_cached[i] + 8,
                 f'+{diff}', ha='center', fontsize=8,
                 fontweight='bold', color=GREEN)

# highlight eviction_pressure cached tokens win
ax4.annotate('', xy=(5 + w/2, radix_cached[5] + 15), xytext=(5 + w/2, radix_cached[5] + 50),
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))
ax4.text(5 + w/2, radix_cached[5] + 55, '288 vs 0!', ha='center',
         fontsize=9, fontweight='bold', color=GREEN)

# ── Footer ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         'Interpretation: blue bars above orange bars indicate radix tree outperforming flat cache; '
         'green labels mark improvements.',
         ha='center', fontsize=8, color=ORANGE, style='italic')

plt.savefig('/home/colin-zhou/blog/images/radix_tree_benchmark.png', dpi=180, bbox_inches='tight')
print("Saved to /home/colin-zhou/blog/images/radix_tree_benchmark.png")
