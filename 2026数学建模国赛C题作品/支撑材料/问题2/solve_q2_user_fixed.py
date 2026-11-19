# -*- coding: utf-8 -*-
"""
问题2：两阶段随机规划（Two‑Stage Stochastic Programming）
- 第一阶段：计划购电量 g[t]（here‑and‑now，对所有场景通用）
- 第二阶段：各场景下电池充放电 c_s,d_s + 紧急购电 e_s（wait‑and‑see）
- k‑means 情景缩减：从1月历史数据生成5个典型场景
- VaR/CVaR 风险分析
修复点：
1. fc_idx越界保护
2. 放电功率乘以效率ETA，修正功率平衡物理模型
3. 输出表格语义修正，区分计划购电费与实际总费用
"""
import pulp
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import openpyxl
import warnings
import time

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
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(CURRENT_DIR, "..", "附件")
BASE = os.path.abspath(BASE)   # 指向 Andy\附件
OUT = os.path.join(CURRENT_DIR, "result2_user_fixed.xlsx")

# ===== 读数据 =====
print("=" * 60)
print("步骤1：读取数据")
print("=" * 60)
df1 = pd.read_excel(f"{BASE}/附件1.xlsx")
price = df1['电价'].values.astype(float)  # 144
df_load = pd.read_excel(f"{BASE}/附件2.xlsx", sheet_name="小区负载")
df_pv = pd.read_excel(f"{BASE}/附件2.xlsx", sheet_name="光伏发电实际功率")

dates_all = pd.to_datetime(df_load.iloc[:, 0])
load_all = df_load.iloc[:, 1:].values.astype(float)   # 365 x 144
pv_all = df_pv.iloc[:, 1:].values.astype(float)       # 365 x 144

print(f"电价: {len(price)} 时段")
print(f"负载: {load_all.shape}, 光伏: {pv_all.shape}")
print(f"日期范围: {dates_all.min().date()} ~ {dates_all.max().date()}")

# 只处理 2025.2.1 ~ 2025.12.31
mask = (dates_all >= '2025-02-01') & (dates_all <= '2025-12-31')
dates = dates_all[mask].reset_index(drop=True)
load_data = load_all[mask]   # 334 x 144
pv_data = pv_all[mask]       # 334 x 144
print(f"处理天数: {len(dates)} (2.1~12.31)")

# ===== 步骤2：k‑means 场景生成 =====
print("\n" + "=" * 60)
print("步骤2：k‑means 情景缩减（从1月历史数据生成5个典型场景）")
print("=" * 60)
jan_mask = dates_all < '2025-02-01'
jan_load = load_all[jan_mask]   # 31 x 144
jan_pv = pv_all[jan_mask]       # 31 x 144

# 归一化后聚类（负载和光伏拼接）
jan_combined = np.hstack([jan_load / jan_load.max(), jan_pv / (jan_pv.max() + 1e-6)])
kmeans = KMeans(n_clusters=N_SCEN, random_state=42, n_init=10)
labels = kmeans.fit_predict(jan_combined)

# 计算平均日（作为基准）
avg_load = jan_load.mean(axis=0)
avg_pv = jan_pv.mean(axis=0)

# 每个场景的概率和偏差比例（相对于平均日）
scen_prob = np.zeros(N_SCEN)
scen_dev_load = np.zeros((N_SCEN, N))  # 偏差比例
scen_dev_pv = np.zeros((N_SCEN, N))

for s in range(N_SCEN):
    idx = np.where(labels == s)[0]
    scen_prob[s] = len(idx) / len(jan_combined)
    typ_load = jan_load[idx].mean(axis=0)
    typ_pv = jan_pv[idx].mean(axis=0)
    # 偏差比例 = (典型日 - 平均日) / 平均日
    scen_dev_load[s] = (typ_load - avg_load) / (avg_load + 1e-6)
    scen_dev_pv[s] = (typ_pv - avg_pv) / (avg_pv + 1e-6)

print(f"场景概率: {[f'{p:.3f}' for p in scen_prob]}")
print(f"场景负载偏差范围: {[f'{d.min():.2f}~{d.max():.2f}' for d in scen_dev_load]}")
print(f"场景光伏偏差范围: {[f'{d.min():.2f}~{d.max():.2f}' for d in scen_dev_pv]}")

# ===== 步骤3：两阶段随机规划求解器 =====
def solve_two_stage(day_load, day_pv, price, scen_dev_load, scen_dev_pv, scen_prob):
    """
    两阶段随机规划
    第一阶段：计划购电量 g[t]
    第二阶段：各场景 c_s[t], d_s[t], e_s[t], E_s[t]
    场景 = 当天预测 × (1 + 偏差比例)
    """
    # 生成当天的5个场景
    s_load = np.zeros((N_SCEN, N))
    s_pv = np.zeros((N_SCEN, N))
    for s in range(N_SCEN):
        s_load[s] = np.maximum(day_load * (1 + scen_dev_load[s]), 0)
        s_pv[s] = np.maximum(day_pv * (1 + scen_dev_pv[s]), 0)

    prob = pulp.LpProblem("Q2_TwoStage", pulp.LpMinimize)
    # 第一阶段变量
    g = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(N)]
    # 第二阶段变量（每个场景）
    c = {}
    d = {}
    e = {}
    E = {}
    for s in range(N_SCEN):
        c[s] = [pulp.LpVariable(f"c_{s}_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
        d[s] = [pulp.LpVariable(f"d_{s}_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
        e[s] = [pulp.LpVariable(f"e_{s}_{t}", lowBound=0) for t in range(N)]
        E[s] = [pulp.LpVariable(f"E_{s}_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    # 目标函数：计划购电成本 + 期望紧急购电成本
    obj = pulp.lpSum(price[t] * g[t] * DT for t in range(N))
    for s in range(N_SCEN):
        obj += scen_prob[s] * pulp.lpSum(EMERG_MULT * price[t] * e[s][t] * DT for t in range(N))
    prob += obj

    # 约束（每个场景）
    for s in range(N_SCEN):
        # 首尾SOC
        prob += E[s][0] == S0
        prob += E[s][N] == S0
        for t in range(N):
            # 【修复】放电需要乘效率ETA：电池放电 -> 母线侧功率
            prob += g[t] + s_pv[s][t] + ETA * d[s][t] + e[s][t] >= s_load[s][t] + c[s][t]
            # SOC递推
            prob += E[s][t+1] == E[s][t] + c[s][t] * DT * ETA - d[s][t] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"  警告: 求解状态={pulp.LpStatus[prob.status]}")

    g_val = np.array([pulp.value(g[t]) for t in range(N)])
    total_cost = pulp.value(prob.objective)
    # 各场景紧急购电
    emerg_costs = []
    for s in range(N_SCEN):
        ec = sum(EMERG_MULT * price[t] * pulp.value(e[s][t]) * DT for t in range(N))
        emerg_costs.append(ec)
    return g_val, total_cost, emerg_costs

# ===== 步骤4：实际执行（用当天实际数据） =====
def actual_execution(g_plan, day_load, day_pv, price, init_soc=S0):
    """
    实际执行：计划购电量固定，电池灵活充放电，紧急购电
    返回：实际总成本、实际紧急购电量、充放电量、SOC
    """
    prob = pulp.LpProblem("Q2_Actual", pulp.LpMinimize)
    c = [pulp.LpVariable(f"ac_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    d = [pulp.LpVariable(f"ad_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    e = [pulp.LpVariable(f"ae_{t}", lowBound=0) for t in range(N)]
    E = [pulp.LpVariable(f"aE_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    # 目标：最小化紧急购电成本（计划购电成本固定）
    prob += pulp.lpSum(EMERG_MULT * price[t] * e[t] * DT for t in range(N))
    prob += E[0] == init_soc
    prob += E[N] == init_soc  # 首尾相等

    for t in range(N):
        # 【修复】实际仿真功率平衡，放电乘ETA
        prob += g_plan[t] + day_pv[t] + ETA * d[t] + e[t] >= day_load[t] + c[t]
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

# ===== 步骤5：批量求解 =====
print("\n" + "=" * 60)
print("步骤3：批量求解两阶段随机规划（334天）")
print("=" * 60)
results = []
all_costs = []
t0 = time.time()

for i in range(len(dates)):
    day = dates[i].strftime('%Y-%m-%d')
    day_load = load_data[i]
    day_pv = pv_data[i]

    # 持续性预测：用前一天的实际数据作为当天预测
    # i=0对应2/1，全量数据索引31=2/1，前一天=索引30=1/31
    fc_idx = 30 + i
    fc_load = load_all[fc_idx]
    fc_pv = pv_all[fc_idx]

    # 两阶段随机规划（场景=预测×(1+偏差比例)）
    g_plan, sp_cost, scen_emerg = solve_two_stage(fc_load, fc_pv, price, scen_dev_load, scen_dev_pv, scen_prob)
    # 实际执行（用当天实际数据，与预测有偏差→紧急购电）
    total_cost, plan_cost, emerg_cost, c_val, d_val, e_val, E_val = actual_execution(g_plan, day_load, day_pv, price)

    all_costs.append(total_cost)
    results.append({
        'date': dates[i],
        'g_plan': g_plan,
        'total_cost': total_cost,
        'plan_cost': plan_cost,
        'emerg_cost': emerg_cost,
        'c_val': c_val,
        'd_val': d_val,
        'e_val': e_val,
        'E_val': E_val,
        'emerg_total': e_val.sum(),
    })

    if (i+1) % 50 == 0 or i == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(dates)}] {day} 计划={plan_cost:.0f} 紧急={emerg_cost:.0f} 总计={total_cost:.0f} 紧急购电={e_val.sum():.1f}kWh 用时={elapsed:.0f}s")

print(f"\n全部完成，总用时={time.time()-t0:.0f}s")

# ===== 步骤6：VaR/CVaR 分析 =====
print("\n" + "=" * 60)
print("步骤4：VaR/CVaR 风险分析")
print("=" * 60)
costs_arr = np.array(all_costs)
var95 = np.percentile(costs_arr, 95)
cvar95 = costs_arr[costs_arr >= var95].mean()

print(f"全年平均日成本: {costs_arr.mean():.2f} 元")
print(f"全年总成本: {costs_arr.sum():.2f} 元")
print(f"95% VaR: {var95:.2f} 元")
print(f"95% CVaR: {cvar95:.2f} 元")
print(f"成本标准差: {costs_arr.std():.2f} 元")
print(f"有紧急购电的天数: {sum(1 for r in results if r['emerg_total'] > 0.1)}")

# ===== 步骤7：输出 result2.xlsx =====
print("\n" + "=" * 60)
print("步骤5：输出 result2.xlsx")
print("=" * 60)
# 生成时间段标签，模板格式 "0:10-0:20"
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

# 创建工作簿
wb = openpyxl.Workbook()
# --- Sheet1: 计划购电量 ---
ws1 = wb.active
ws1.title = "计划购电量"
header = ["日期\\时间"] + time_labels + ["全天购电量(kWh)", "购电费(元)"]
ws1.append(header)

for r in results:
    row = [r['date'].strftime('%Y/%m/%d')]
    # ✅ 修复：g_plan是kW，乘DT转为kWh（10分钟时段电量）
    row += [round(float(v*DT), 4) for v in r['g_plan']]
    row.append(round(float(r['g_plan'].sum()*DT), 2))
    row.append(round(float(r['plan_cost']), 2))  # 计划购电费
    ws1.append(row)

# --- Sheet2: 充放电量 ---
ws2 = wb.create_sheet("充放电量")
ws2.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
spec_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
blocks = [(0,24,'0:00-4:00'),(24,48,'4:00-8:00'),(48,72,'8:00-12:00'),
          (72,96,'12:00-16:00'),(96,120,'16:00-20:00'),(120,144,'20:00-24:00')]

for sd in spec_dates:
    match = [r for r in results if r['date'].strftime('%Y-%m-%d') == sd]
    if not match:
        continue
    r = match[0]
    for bi, (s, e, name) in enumerate(blocks):
        chg = float(r['c_val'][s:e].sum())
        dis = float(r['d_val'][s:e].sum())
        row = [r['date'].strftime('%Y/%m/%d') if bi == 0 else None,
               name, round(chg, 2), round(dis, 2), None, None]
        if bi == 0:
            row[4] = "0:00"
            row[5] = round(float(r['E_val'][0]), 2)
        if bi == 5:
            row[4] = "24:00"
            row[5] = round(float(r['E_val'][-1]), 2)
        ws2.append(row)

# --- Sheet3: 紧急购电量 ---
ws3 = wb.create_sheet("紧急购电量")
ws3.append(["日期", "紧急购电时间段", "紧急购电量"])
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
                segments.append((seg_start, t-1))
                in_seg = False
        if in_seg:
            segments.append((seg_start, N-1))
        for si, (s, e) in enumerate(segments):
            amount = float(e_arr[s:e+1].sum())
            t_start = time_labels[s].split('-')[0]
            t_end = time_labels[e].split('-')[1] if '-' in time_labels[e] else time_labels[e]
            row = [r['date'].strftime('%Y/%m/%d') if si == 0 else None,
                   f"{t_start}-{t_end}", round(amount, 2)]
            ws3.append(row)

# 保存
wb.save(OUT)
print(f"已保存: {OUT}")
print(f"计划购电量: {ws1.max_row}行 x {ws1.max_column}列")
print(f"充放电量: {ws2.max_row}行 x {ws2.max_column}列")
print(f"紧急购电量: {ws3.max_row}行 x {ws3.max_column}列")

# ===== 汇总统计 =====
print("\n" + "=" * 60)
print("汇总统计")
print("=" * 60)
total_plan = sum(r['plan_cost'] for r in results)
total_emerg = sum(r['emerg_cost'] for r in results)
total_all = sum(r['total_cost'] for r in results)
print(f"全年计划购电费: {total_plan:.2f} 元")
print(f"全年紧急购电费: {total_emerg:.2f} 元")
print(f"全年总购电费: {total_all:.2f} 元")
print(f"紧急购电占比: {total_emerg/total_all*100:.2f}%")
print(f"95% VaR: {var95:.2f} 元/天")
print(f"95% CVaR: {cvar95:.2f} 元/天")
