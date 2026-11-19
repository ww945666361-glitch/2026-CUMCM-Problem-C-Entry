# -*- coding: utf-8 -*-
"""
Q4 论文图（升级版）
- 保持原4张子图
- 新增：预算鲁棒Γ参数对比子图
- 支持黑白打印：用线型/标记区分，不仅靠颜色
- 字体：微软雅黑，dpi=200
"""
import matplotlib, pandas as pd, numpy as np, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 0.8

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))   # 附件目录（Andy\附件）

# ===== 数据 =====
q2_fixed = 12252479.47
no_adapt = 12965651.45
adapt = 12839295.31

# 灵敏度曲线（确定性缩放）
sens = [(0.0, 15228375.95), (0.5, 14361720.32), (0.75, 13609998.53),
        (1.0, 12839295.31), (1.25, 11739997.25), (1.5, 10617477.43)]

# 预算鲁棒Γ对比（实际运行结果）
gamma_vals = [0, 12, 36, 72, 144]
gamma_costs = [12830486, 13269834, 13782659, 14124455, 14206334]

# 读result4-2典型日数据（由 solve_q4_2_fixed.py 生成于本目录）
df = pd.read_excel(os.path.join(CURRENT_DIR, "result4-2.xlsx"), "计划购电量")
df1 = pd.read_excel(os.path.join(DATA_DIR, "附件1.xlsx"))
price_fixed = df1['电价'].values
df4 = pd.read_excel(os.path.join(DATA_DIR, "附件4.xlsx"))
price_fluc = df4.iloc[:, 1:].values.astype(float)

i_715 = pd.to_datetime(df.iloc[:, 0]).tolist().index(pd.Timestamp('2025-07-15'))
idx4 = pd.to_datetime(df4.iloc[:, 0]).tolist().index(pd.Timestamp('2025-07-15'))

# ===== 绘图：2行3列 =====
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# --- (a) 方案对比柱状图 ---
ax = axes[0, 0]
cats = ['固定电价\n最优(Q2)', '固定计划→\n波动执行', '波动电价\n适应优化']
vals = [q2_fixed, no_adapt, adapt]
colors = ['#4CAF50', '#FF9800', '#2196F3']
bars = ax.bar(cats, [v/1e4 for v in vals], color=colors, width=0.5, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v/1e4, f'{v/1e4:.1f}万',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('(a) 固定与波动电价下全年成本对比', fontsize=12, fontweight='bold')
ax.set_ylabel('万元')
ax.set_ylim(1100, 1350)
ax.grid(axis='y', alpha=0.3)

# --- (b) 灵敏度曲线 ---
ax = axes[0, 1]
scales = [s for s, _ in sens]
costs = [c/1e4 for _, c in sens]
ax.plot(scales, costs, 'o-', color='#FF5722', lw=2, markersize=8,
        markerfacecolor='white', markeredgewidth=1.5, label='确定性缩放')
ax.axhline(y=q2_fixed/1e4, color='green', ls='--', label=f'固定电价 {q2_fixed/1e4:.1f}万')
ax.set_xlabel('波动幅度缩放系数')
ax.set_ylabel('全年成本（万元）')
ax.set_title('(b) 电价波动幅度灵敏度', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- (c) 典型日电价对比 ---
ax = axes[0, 2]
x = np.arange(144)/6
ax.plot(x, price_fixed, 'k-', lw=1.5, label='固定电价')
ax.plot(x, price_fluc[idx4], 'r--', lw=1.2, label='波动电价(7/15)')
ax.set_xlabel('时间(h)')
ax.set_ylabel('元/kWh')
ax.set_title('(c) 典型日电价对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 24)
ax.grid(alpha=0.3)

# --- (d) 典型日调度 ---
ax = axes[1, 0]
buy = df.iloc[i_715, 1:145].values.astype(float)
ax2 = ax.twinx()
ax.plot(x, buy, 'b-', lw=1.2, label='购电量(kWh/段)')
ax2.plot(x, price_fluc[idx4], 'r--', lw=0.8, alpha=0.5, label='波动电价')
ax.set_xlabel('时间(h)')
ax.set_ylabel('购电(kWh/段)', color='b')
ax2.set_ylabel('电价(元/kWh)', color='r')
ax.set_title('(d) 波动电价下典型日调度(7/15)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 24)

# --- (e) 预算鲁棒Γ对比（新增）---
ax = axes[1, 1]
ax.plot(gamma_vals, [c/1e4 for c in gamma_costs], 's-', color='#6A1B9A', lw=2,
        markersize=8, markerfacecolor='white', markeredgewidth=1.5)
ax.axhline(y=q2_fixed/1e4, color='green', ls='--', label=f'固定电价最优 {q2_fixed/1e4:.1f}万')
ax.set_xlabel('预算参数 Γ')
ax.set_ylabel('全年成本（万元）')
ax.set_title('(e) 预算鲁棒优化不同Γ对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# 标注含义
ax.annotate('Γ=0\n确定性', xy=(0, gamma_costs[0]/1e4), fontsize=8, ha='center')
ax.annotate('Γ=144\n最坏情况', xy=(144, gamma_costs[-1]/1e4), fontsize=8, ha='center')

# --- (f) 三种模型对比雷达/柱状 ---
ax = axes[1, 2]
model_names = ['两阶段\n随机规划', '预算鲁棒\n(Γ=36)', '确定性\n灵敏度']
model_costs = [2051, 1378, 1283]  # 万元（SP/RO(Γ=36)/确定性）
bars = ax.bar(model_names, model_costs, color=['#FF9800', '#9C27B0', '#4CAF50'],
              width=0.5, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, model_costs):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.0f}万',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('全年成本（万元）')
ax.set_title('(f) 三类模型成本对比', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_DIR, 'Q4_论文图.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Q4_论文图.png 已保存（2×3子图）")
