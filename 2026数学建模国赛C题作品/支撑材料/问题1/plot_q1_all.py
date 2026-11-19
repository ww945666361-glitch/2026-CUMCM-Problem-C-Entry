# -*- coding: utf-8 -*-
"""
问题1 最终方案绘图脚本（合并版）
=====================================================================
- 复用 solve_q1 的建模/求解函数，保证图片与 Excel 结果严格同源；
- 一次生成4张论文可直接使用的图（文件名与旧版一致，避免论文换图）：
    1) Q1_LP_vs_MILP_论文版.png   LP vs MILP 三面板对比（论文风格）
    2) Q1_时序图.png              电价 / 负荷与光伏 / SOC 时序
    3) Q1_充放电图.png            购电功率 + 储能充放电功率
    4) Q1_供需堆叠图.png          光伏+储能放电+外网购电 供需堆叠

运行：python plot_q1_all.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_annex1
from solve_q1 import build_model, solve_and_summarize, PERIODS, E_MIN, E_MAX, E_INIT

# ============ 中文字体 ============
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 1.0 / 6.0

# ============ 数据与求解（与 solve_q1 同一套口径） ============
a1 = load_annex1()
PRICE = a1["price"].to_numpy(float)
LOAD = a1["load_kw"].to_numpy(float)
PV = a1["pv_pred_kw"].to_numpy(float)

lp = solve_and_summarize(
    build_model(use_milp=False, price=PRICE, load=LOAD, pv=PV),
    use_milp=False, price=PRICE, load=LOAD, pv=PV)
milp = solve_and_summarize(
    build_model(use_milp=True, price=PRICE, load=LOAD, pv=PV),
    use_milp=True, price=PRICE, load=LOAD, pv=PV)
print(f"LP 成本 = {lp['cost']:.4f} 元, MILP 成本 = {milp['cost']:.4f} 元, "
      f"差值 = {abs(lp['cost'] - milp['cost']):.6f} 元")

t = np.arange(PERIODS)
tick_pos = np.arange(0, PERIODS + 1, 24)
tick_lab = [f'{int(h)}:00' for h in np.arange(0, 25, 4)]

# ============ 图1：LP vs MILP 论文版三面板 ============
hours = t / 6.0
tick3 = np.arange(0, 25, 3)
fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
fig.subplots_adjust(hspace=0.12, left=0.10, right=0.97, top=0.95, bottom=0.08)

ax = axes[0]
ax.plot(hours, lp['g'], color='black', lw=1.2, label='LP')
ax.plot(hours, milp['g'], color='gray', lw=1.0, ls='--', label='MILP')
ax.set_ylabel('购电功率 (kW)'); ax.set_xlim(0, 24)
ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.3, 8500, '(a) 购电功率', fontsize=10, va='top')

ax = axes[1]
ax.plot(hours, lp['c'], color='#2ca02c', lw=1.2, label='LP 充电')
ax.plot(hours, milp['c'], color='#98df8a', lw=1.0, ls='--', label='MILP 充电')
ax.plot(hours, -lp['d'], color='#ff7f0e', lw=1.2, label='LP 放电')
ax.plot(hours, -milp['d'], color='#ffbb78', lw=1.0, ls='--', label='MILP 放电')
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylabel('充/放电功率 (kW)'); ax.set_xlim(0, 24)
ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.3, 4800, '(b) 储能充放电功率', fontsize=10, va='top')

ax = axes[2]
ax.plot(hours, lp['E'][:-1], color='#9467bd', lw=1.2, label='LP')
ax.plot(hours, milp['E'][:-1], color='#c994c7', lw=1.0, ls='--', label='MILP')
ax.axhline(y=E_MIN, color='gray', ls=':', lw=0.7)
ax.axhline(y=E_MAX, color='gray', ls=':', lw=0.7)
ax.set_ylabel('储能电量 (kWh)'); ax.set_xlabel('时间 (h)')
ax.set_xlim(0, 24); ax.set_xticks(tick3)
ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylim(0, 11500)
ax.text(0.3, 11000, '(c) 储能电量 SOC', fontsize=10, va='top')

plt.savefig(os.path.join(OUT_DIR, 'Q1_LP_vs_MILP_论文版.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============ 图2：时序图（电价 / 负荷与光伏 / SOC） ============
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
ax = axes[0]
ax.plot(t, PRICE, color='#d62728', linewidth=1.5)
ax.fill_between(t, PRICE, alpha=0.1, color='#d62728')
ax.set_ylabel('电价（元/kWh）')
ax.set_title('(a) 分时电价（MILP结果）', fontsize=12)
ax.grid(alpha=0.3)
ax.axhline(y=np.mean(PRICE), color='gray', ls='--', lw=0.8,
           label=f'平均电价 {np.mean(PRICE):.3f}')
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(t, LOAD, color='#1f77b4', linewidth=1.5, label='小区负荷')
ax.plot(t, PV, color='#2ca02c', linewidth=1.5, label='光伏发电')
ax.set_ylabel('功率（kW）')
ax.set_title('(b) 小区负荷与光伏发电', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(t, milp['E'][:-1], color='#ff7f0e', linewidth=2, label='储能电量SOC')
ax.axhline(y=E_MIN, color='gray', ls='--', lw=0.8, label=f'下限 {E_MIN:.0f}')
ax.axhline(y=E_MAX, color='gray', ls='--', lw=0.8, label=f'上限 {E_MAX:.0f}')
ax.axhline(y=E_INIT, color='purple', ls=':', lw=1, label=f'初始值 {E_INIT:.0f}')
ax.set_ylabel('储能电量（kWh）')
ax.set_title('(c) 储能电量SOC时序', fontsize=12)
ax.set_xlabel('时段（每10分钟一段，0~143对应0:00~24:00）')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 12000)
axes[2].set_xticks(tick_pos)
axes[2].set_xticklabels(tick_lab)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q1_时序图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============ 图3：充放电图（购电功率 + 充放电功率） ============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(t, milp['g'], color='#1f77b4', linewidth=1.3)
ax1.fill_between(t, milp['g'], alpha=0.15, color='#1f77b4')
ax1.set_ylabel('购电功率（kW）')
ax1.set_title('(a) 外网购电功率', fontsize=12)
ax1.grid(alpha=0.3)

ax2.plot(t, milp['c'], color='#2ca02c', linewidth=1.3, label='充电功率')
ax2.plot(t, -milp['d'], color='#d62728', linewidth=1.3, label='放电功率')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('充/放电功率（kW）')
ax2.set_title('(b) 储能充放电功率（正=充电，负=放电）', fontsize=12)
ax2.set_xlabel('时段（0:00~24:00）')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xticks(tick_pos)
ax2.set_xticklabels(tick_lab)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q1_充放电图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============ 图4：供需堆叠图 ============
fig2, ax = plt.subplots(figsize=(12, 5.5))
ax.stackplot(t, PV, milp['d'], milp['g'],
             labels=['光伏发电', '储能放电', '外网购电'],
             colors=['#2ca02c', '#ff7f0e', '#1f77b4'], alpha=0.8)
ax.plot(t, LOAD, color='#d62728', linewidth=1.8, label='小区负荷')
ax.plot(t, LOAD + milp['c'], color='#9467bd', linewidth=1.2, ls='--',
        label='负荷+储能充电')
ax.set_xlabel('时段（0:00~24:00）')
ax.set_ylabel('功率 (kW)')
ax.set_title('问题1 微网功率供需构成', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q1_供需堆叠图.png'), dpi=300, bbox_inches='tight')
plt.close()

print("4 张图已全部保存到:", OUT_DIR)
