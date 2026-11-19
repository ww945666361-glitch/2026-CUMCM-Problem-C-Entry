# -*- coding: utf-8 -*-
"""
问题4-2：波动电价下的问题2（两阶段随机规划 + 紧急购电）
基于 solve_q2.py 修改，将附件1固定电价替换为附件4逐日波动电价
"""
import pulp
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2
from openpyxl import load_workbook
import warnings
import time
import os
import shutil

warnings.filterwarnings('ignore')

# ===== 参数 =====
N = 144
DT = 1.0 / 6.0
ETA = 0.9
S_MIN = 1200.0
S_MAX = 10800.0
P_MAX = 5000.0
S0 = 6000.0
N_SCEN = 5
EMERG_MULT = 5.0

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))       # 附件目录（Andy\附件）
TEMPLATE = os.path.join(BASE, "附件5", "result4-2.xlsx")              # 官方模板
OUT_DIR = CURRENT_DIR
OUT = os.path.join(OUT_DIR, "result4-2.xlsx")

# 复制模板到输出目录
shutil.copy(TEMPLATE, OUT)
print(f"模板已复制到: {OUT}")

# ===== 读数据 =====
print("=" * 60)
print("步骤1：读取数据")
print("=" * 60)

# 附件4：波动电价 (366天 x 144时段)
df_price = pd.read_excel(os.path.join(BASE, "附件4.xlsx"))
price_dates = pd.to_datetime(df_price.iloc[:, 0])
price_data = df_price.iloc[:, 1:].values.astype(float)
price_dict = {}
for i, d in enumerate(price_dates):
    price_dict[d.strftime('%Y-%m-%d')] = price_data[i]
print(f"波动电价: {len(price_dict)} 天, 每天 {price_data.shape[1]} 时段")

# 附件2：负载和光伏
df_load = pd.read_excel(os.path.join(BASE, "附件2.xlsx"), sheet_name="小区负载")
df_pv = pd.read_excel(os.path.join(BASE, "附件2.xlsx"), sheet_name="光伏发电实际功率")
dates_all = pd.to_datetime(df_load.iloc[:, 0])
load_all = df_load.iloc[:, 1:].values.astype(float)
pv_all = df_pv.iloc[:, 1:].values.astype(float)
print(f"负载: {load_all.shape}, 光伏: {pv_all.shape}")

# 只处理 2025.2.1 ~ 2025.12.31
mask = (dates_all >= '2025-02-01') & (dates_all <= '2025-12-31')
dates = dates_all[mask].reset_index(drop=True)
load_data = load_all[mask]
pv_data = pv_all[mask]
print(f"处理天数: {len(dates)} (2.1~12.31)")

# ===== k-means 场景生成 =====
print("\n" + "=" * 60)
print("步骤2：k-means 情景缩减")
print("=" * 60)
jan_mask = dates_all < '2025-02-01'
jan_load = load_all[jan_mask]
jan_pv = pv_all[jan_mask]
jan_combined = np.hstack([jan_load / jan_load.max(), jan_pv / (jan_pv.max() + 1e-6)])
centroids, labels = kmeans2(jan_combined, N_SCEN, minit='points', seed=42)
avg_load = jan_load.mean(axis=0)
avg_pv = jan_pv.mean(axis=0)
scen_prob = np.zeros(N_SCEN)
scen_dev_load = np.zeros((N_SCEN, N))
scen_dev_pv = np.zeros((N_SCEN, N))
for s in range(N_SCEN):
    idx = np.where(labels == s)[0]
    scen_prob[s] = len(idx) / len(jan_combined)
    typ_load = jan_load[idx].mean(axis=0)
    typ_pv = jan_pv[idx].mean(axis=0)
    scen_dev_load[s] = (typ_load - avg_load) / (avg_load + 1e-6)
    scen_dev_pv[s] = (typ_pv - avg_pv) / (avg_pv + 1e-6)
print(f"场景概率: {[f'{p:.3f}' for p in scen_prob]}")

# ===== 两阶段随机规划求解器 =====
def solve_two_stage(day_load, day_pv, price, scen_dev_load, scen_dev_pv, scen_prob):
    s_load = np.zeros((N_SCEN, N))
    s_pv = np.zeros((N_SCEN, N))
    for s in range(N_SCEN):
        s_load[s] = np.maximum(day_load * (1 + scen_dev_load[s]), 0)
        s_pv[s] = np.maximum(day_pv * (1 + scen_dev_pv[s]), 0)

    prob = pulp.LpProblem("Q2_TwoStage", pulp.LpMinimize)
    g = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(N)]
    c, d, e, E = {}, {}, {}, {}
    for s in range(N_SCEN):
        c[s] = [pulp.LpVariable(f"c_{s}_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
        d[s] = [pulp.LpVariable(f"d_{s}_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
        e[s] = [pulp.LpVariable(f"e_{s}_{t}", lowBound=0) for t in range(N)]
        E[s] = [pulp.LpVariable(f"E_{s}_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    obj = pulp.lpSum(price[t] * g[t] * DT for t in range(N))
    for s in range(N_SCEN):
        obj += scen_prob[s] * pulp.lpSum(EMERG_MULT * price[t] * e[s][t] * DT for t in range(N))
    prob += obj

    for s in range(N_SCEN):
        prob += E[s][0] == S0
        prob += E[s][N] == S0
        for t in range(N):
            prob += g[t] + s_pv[s][t] + d[s][t] + e[s][t] >= s_load[s][t] + c[s][t]
            prob += E[s][t+1] == E[s][t] + c[s][t] * DT * ETA - d[s][t] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    g_val = np.array([pulp.value(g[t]) for t in range(N)])
    return g_val

# ===== 实际执行 =====
def actual_execution(g_plan, day_load, day_pv, price, init_soc=S0):
    prob = pulp.LpProblem("Q2_Actual", pulp.LpMinimize)
    c = [pulp.LpVariable(f"ac_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    d = [pulp.LpVariable(f"ad_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    e = [pulp.LpVariable(f"ae_{t}", lowBound=0) for t in range(N)]
    E = [pulp.LpVariable(f"aE_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    prob += pulp.lpSum(EMERG_MULT * price[t] * e[t] * DT for t in range(N))
    prob += E[0] == init_soc
    prob += E[N] == init_soc

    for t in range(N):
        prob += g_plan[t] + day_pv[t] + d[t] + e[t] >= day_load[t] + c[t]
        prob += E[t+1] == E[t] + c[t] * DT * ETA - d[t] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

    c_val = np.array([pulp.value(c[t]) for t in range(N)]) * DT
    d_val = np.array([pulp.value(d[t]) for t in range(N)]) * DT
    e_val = np.array([pulp.value(e[t]) for t in range(N)]) * DT
    E_val = np.array([pulp.value(E[t]) for t in range(N+1)])

    plan_cost = sum(price[t] * g_plan[t] * DT for t in range(N))
    emerg_cost = sum(EMERG_MULT * price[t] * e_val[t] for t in range(N))
    total_cost = plan_cost + emerg_cost
    return total_cost, plan_cost, emerg_cost, c_val, d_val, e_val, E_val

# ===== 批量求解 =====
print("\n" + "=" * 60)
print("步骤3：批量求解（334天，波动电价）")
print("=" * 60)
results = []
t0 = time.time()

for i in range(len(dates)):
    day_key = dates[i].strftime('%Y-%m-%d')
    price = price_dict[day_key]  # 当天波动电价
    day_load = load_data[i]
    day_pv = pv_data[i]

    # 持续性预测：用前一天实际数据
    fc_idx = 30 + i
    fc_load = load_all[fc_idx]
    fc_pv = pv_all[fc_idx]

    g_plan = solve_two_stage(fc_load, fc_pv, price, scen_dev_load, scen_dev_pv, scen_prob)
    total_cost, plan_cost, emerg_cost, c_val, d_val, e_val, E_val = actual_execution(
        g_plan, day_load, day_pv, price)

    results.append({
        'date': dates[i],
        'g_plan': g_plan,
        'plan_cost': plan_cost,
        'emerg_cost': emerg_cost,
        'total_cost': total_cost,
        'c_val': c_val,
        'd_val': d_val,
        'e_val': e_val,
        'E_val': E_val,
        'emerg_total': e_val.sum(),
    })

    if (i+1) % 50 == 0 or i == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(dates)}] {day_key} 计划={plan_cost:.0f} 紧急={emerg_cost:.0f} "
              f"总计={total_cost:.0f} 紧急购电={e_val.sum():.1f}kWh 用时={elapsed:.0f}s")

print(f"\n全部完成，总用时={time.time()-t0:.0f}s")

# ===== 汇总统计 =====
total_plan = sum(r['plan_cost'] for r in results)
total_emerg = sum(r['emerg_cost'] for r in results)
total_all = sum(r['total_cost'] for r in results)
print(f"\n全年计划购电费: {total_plan:.2f} 元")
print(f"全年紧急购电费: {total_emerg:.2f} 元")
print(f"全年总购电费: {total_all:.2f} 元")
print(f"有紧急购电的天数: {sum(1 for r in results if r['emerg_total'] > 0.1)}")

# ===== 输出 result4-2.xlsx =====
print("\n" + "=" * 60)
print("步骤4：写入 result4-2.xlsx")
print("=" * 60)

# 时间段标签
time_labels = []
for i in range(N):
    start_min = (i + 1) * 10
    end_min = (i + 2) * 10
    sh, sm = divmod(start_min, 60)
    eh, em = divmod(end_min, 60)
    if eh >= 24:
        eh -= 24
        label = f"{sh}:{sm:02d}-{eh}:{em:02d}+1" if eh == 0 else f"{sh}:{sm:02d}-{eh}:{em:02d}"
    else:
        label = f"{sh}:{sm:02d}-{eh}:{em:02d}"
    time_labels.append(label)

wb = load_workbook(OUT)

# --- 计划购电量 ---
ws = wb['计划购电量']
for day_idx, r in enumerate(results):
    row = day_idx + 2
    g_kwh = r['g_plan'] * DT  # 转换为kWh
    for t in range(N):
        ws.cell(row=row, column=t + 2, value=round(float(g_kwh[t]), 4))
    ws.cell(row=row, column=N + 2, value=round(float(g_kwh.sum()), 2))
    ws.cell(row=row, column=N + 3, value=round(float(r['plan_cost']), 2))
print(f"  计划购电量: {len(results)} 行已写入")

# --- 充放电量（指定日期：3.20, 6.21, 9.23, 12.21）---
ws = wb['充放电量']
# 清空数据行（保留表头）
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for cell in row:
        cell.value = None

spec_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
blocks = [(0, 24, '0:00-4:00'), (24, 48, '4:00-8:00'), (48, 72, '8:00-12:00'),
          (72, 96, '12:00-16:00'), (96, 120, '16:00-20:00'), (120, 144, '20:00-24:00')]
row = 2
for sd in spec_dates:
    match = [r for r in results if r['date'].strftime('%Y-%m-%d') == sd]
    if not match:
        continue
    r = match[0]
    for bi, (s, e, name) in enumerate(blocks):
        chg = float(r['c_val'][s:e].sum())
        dis = float(r['d_val'][s:e].sum())
        ws.cell(row=row, column=1, value=r['date'] if bi == 0 else None)
        ws.cell(row=row, column=2, value=name)
        ws.cell(row=row, column=3, value=round(chg, 2))
        ws.cell(row=row, column=4, value=round(dis, 2))
        if bi == 0:
            ws.cell(row=row, column=5, value="0:00")
            ws.cell(row=row, column=6, value=round(float(r['E_val'][0]), 2))
        if bi == 5:
            ws.cell(row=row, column=5, value="24:00")
            ws.cell(row=row, column=6, value=round(float(r['E_val'][-1]), 2))
        row += 1
print(f"  充放电量: {len(spec_dates)} 个指定日期已写入")

# --- 紧急购电量 ---
ws = wb['紧急购电量']
for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
    for cell in row_cells:
        cell.value = None

row = 2
for r in results:
    if r['emerg_total'] > 0.1:
        e_arr = r['e_val']
        segments = []
        in_seg = False
        seg_start = 0
        for t in range(N):
            if e_arr[t] > 0.01 and not in_seg:
                seg_start = t
                in_seg = True
            elif e_arr[t] <= 0.01 and in_seg:
                segments.append((seg_start, t - 1))
                in_seg = False
        if in_seg:
            segments.append((seg_start, N - 1))
        for si, (s, e) in enumerate(segments):
            amount = float(e_arr[s:e + 1].sum())
            t_start = time_labels[s].split('-')[0]
            t_end = time_labels[e].split('-')[1] if '-' in time_labels[e] else time_labels[e]
            ws.cell(row=row, column=1, value=r['date'] if si == 0 else None)
            ws.cell(row=row, column=2, value=f"{t_start}-{t_end}")
            ws.cell(row=row, column=3, value=round(amount, 2))
            row += 1
print(f"  紧急购电量: {row - 2} 条记录已写入")

wb.save(OUT)
print(f"\n已保存: {OUT}")
print("完成！")
