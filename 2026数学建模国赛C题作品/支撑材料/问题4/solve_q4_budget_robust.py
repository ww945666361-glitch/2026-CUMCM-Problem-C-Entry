# -*- coding: utf-8 -*-
"""
问题四-预算鲁棒优化（Bertsimas-Sim Budget of Uncertainty）
============================================================
【模型概念区分，避免论文混淆】
  - 两阶段随机规划（Q2/Q4-2）：需要场景概率，最小化期望成本
  - 预算鲁棒优化（本脚本）：不需要概率，仅知道价格偏差区间，
    通过预算参数Γ控制"同时最坏偏离的时段数"，Γ=0退化为确定性，Γ=N最坏情况
  - 电价缩放灵敏度（独立函数）：纯确定性单场景，仅做参数分析

【电价不确定集】
  对每个时段t，真实电价 p_t ∈ [p̄_t - δ_t, p̄_t + δ_t]
  其中 p̄_t = 附件4当天实际电价（作为名义值），
       δ_t = |p̄_t - 附件1固定电价|（价格偏差幅度）
  预算约束：Σ_t |p_t - p̄_t| / δ_t ≤ Γ

【Bertsimas-Sim线性化】
  对最小化问题，购电成本中价格不确定部分：
    max_{|u_t|≤1, Σ|u_t|≤Γ} Σ_t δ_t * g_t * u_t
  等价于（引入对偶变量 q≥0, r_t≥0）：
    min  Γ*q + Σ_t r_t
    s.t. q + r_t ≥ δ_t * g_t * DT  ∀t
         q ≥ 0, r_t ≥ 0

【修复的3个致命bug】
  ① 储能每日结束强制SOC闭环：s[N] == s[0]（首尾相等），禁止SOC逐日漂移
  ② 电价/负荷/光伏按日期key字典对齐，不用mask切片，避免行列错位
  ③ 变量名/注释不自称"鲁棒优化"指代灵敏度，明确区分两个模块

路径已基于脚本所在目录自动定位（BASE 指向上级目录的 附件）。
"""
import pulp, pandas as pd, numpy as np, os, time, warnings
import openpyxl
from openpyxl.styles import Font, Alignment
warnings.filterwarnings('ignore')

# ============ 参数 ============
N = 144
DT = 10.0 / 60.0       # 每时段小时数
ETA = 0.9              # 充放电效率
S_MIN = 1200.0         # 储能下限 kWh
S_MAX = 10800.0        # 储能上限 kWh
P_MAX = 5000.0         # 最大充放电功率 kW
S0 = 6000.0            # 初始SOC kWh
EM_MULT = 5.0          # 紧急购电倍数
TIME_LIMIT = 60        # CBC求解时间限制(秒)

# ============ 路径（基于脚本所在目录）============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))   # 附件目录（Andy\附件）
# ================================================

# ============ 步骤1：按日期key读取数据（不做mask切片）============
print("=" * 60)
print("步骤1：按日期key读取数据")
print("=" * 60)

# 附件1：固定电价 + 预测（144行）
df1 = pd.read_excel(os.path.join(BASE, "附件1.xlsx"))
price_fixed = df1['电价'].values.astype(float)
print(f"  固定电价: {len(price_fixed)} 时段, 均值={price_fixed.mean():.3f}")

# 附件2：全年实际负荷+光伏
df_load = pd.read_excel(os.path.join(BASE, "附件2.xlsx"), sheet_name="小区负载")
df_pv = pd.read_excel(os.path.join(BASE, "附件2.xlsx"), sheet_name="光伏发电实际功率")
load_dates = pd.to_datetime(df_load.iloc[:, 0])
load_vals = df_load.iloc[:, 1:].values.astype(float)
pv_vals = df_pv.iloc[:, 1:].values.astype(float)

# 构建日期→数组的字典（避免mask错位）
load_dict = {}
pv_dict = {}
for i, d in enumerate(load_dates):
    key = d.strftime('%Y-%m-%d')
    load_dict[key] = load_vals[i]
    pv_dict[key] = pv_vals[i]
print(f"  负荷/光伏: {len(load_dict)} 天")

# 附件4：全年波动电价
df4 = pd.read_excel(os.path.join(BASE, "附件4.xlsx"))
fluc_dates = pd.to_datetime(df4.iloc[:, 0])
fluc_vals = df4.iloc[:, 1:].values.astype(float)
price_dict = {}
for i, d in enumerate(fluc_dates):
    key = d.strftime('%Y-%m-%d')
    price_dict[key] = fluc_vals[i]
print(f"  波动电价: {len(price_dict)} 天")

# 待求解日期：2025-02-01 ~ 2025-12-31
all_dates = sorted(load_dict.keys())
target_dates = [d for d in all_dates if '2025-02-01' <= d <= '2025-12-31']
print(f"  待求解天数: {len(target_dates)} (2025-02-01 ~ 2025-12-31)")

# 计算每个时段的价格偏差 δ_t = |波动-固定|（作为不确定集半径）
# 对每天用当天波动电价与固定电价的差
def get_delta(day_price_fluc):
    """返回每时段价格偏差幅度"""
    return np.abs(day_price_fluc - price_fixed)

# ============ 步骤2：预算鲁棒优化单日求解器 ============
def solve_budget_robust(day_load, day_pv, price_nominal, delta, Gamma, soc_prev=S0):
    """
    Bertsimas-Sim预算鲁棒优化单日模型

    变量:
      g[t]   : 计划购电功率 (kW)，here-and-now
      c[t]   : 充电功率 (kW)
      d[t]   : 放电功率 (kW)
      e[t]   : 紧急购电功率 (kW)
      s[t]   : 储能电量 (kWh)
      q      : 鲁棒对偶变量（全局）
      r[t]   : 鲁棒对偶变量（每时段）

    目标:
      min Σ_t [p̄_t * g_t * DT] + Γ*q + Σ_t r_t
            + 紧急购电成本（同样鲁棒化）
      s.t. q + r_t ≥ δ_t * g_t * DT
           紧急购电也鲁棒: q_em + r_em_t ≥ EM_MULT * δ_t * e_t * DT

    约束:
      功率平衡: g + pv + d + e >= load + c
      SOC递推: s[t+1] = s[t] + c*DT*ETA - d*DT/ETA
      SOC闭环: s[0]==soc_prev, s[N]==soc_prev
    """
    prob = pulp.LpProblem("BudgetRobust", pulp.LpMinimize)

    # 第一阶段：计划购电
    g = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(N)]
    # 第二阶段：电池+紧急购电
    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    e = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in range(N)]
    s = [pulp.LpVariable(f"s_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    # Bertsimas-Sim 对偶变量
    q = pulp.LpVariable("q", lowBound=0)       # 全局鲁棒变量
    r = [pulp.LpVariable(f"r_{t}", lowBound=0) for t in range(N)]
    q_em = pulp.LpVariable("q_em", lowBound=0)
    r_em = [pulp.LpVariable(f"re_{t}", lowBound=0) for t in range(N)]

    # 目标函数：名义成本 + 鲁棒惩罚项
    # 购电部分
    obj = pulp.lpSum(price_nominal[t] * g[t] * DT for t in range(N))
    # 鲁棒预算项（购电部分）
    obj += Gamma * q + pulp.lpSum(r[t] for t in range(N))
    # 紧急购电部分（名义=5*p̄，鲁棒惩罚=5*δ）
    obj += pulp.lpSum(EM_MULT * price_nominal[t] * e[t] * DT for t in range(N))
    obj += Gamma * q_em + pulp.lpSum(r_em[t] for t in range(N))
    prob += obj

    # 鲁棒对偶约束：q + r_t ≥ δ_t * g_t * DT
    for t in range(N):
        prob += q + r[t] >= delta[t] * g[t] * DT
        prob += q_em + r_em[t] >= EM_MULT * delta[t] * e[t] * DT

    # SOC闭环约束（修复bug①：首尾相等，禁止漂移）
    prob += s[0] == soc_prev
    prob += s[N] == soc_prev

    # 功率平衡 + SOC递推
    for t in range(N):
        prob += g[t] + day_pv[t] + d[t] + e[t] >= day_load[t] + c[t]
        prob += s[t+1] == s[t] + c[t]*DT*ETA - d[t]*DT/ETA

    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT))

    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"    [警告] 求解状态={pulp.LpStatus[prob.status]}")

    g_val = np.array([pulp.value(g[t]) for t in range(N)])
    c_val = np.array([pulp.value(c[t]) for t in range(N)])
    d_val = np.array([pulp.value(d[t]) for t in range(N)])
    e_val = np.array([pulp.value(e[t]) for t in range(N)])
    s_val = np.array([pulp.value(s[t]) for t in range(N+1)])
    cost = pulp.value(prob.objective)

    return g_val, c_val, d_val, e_val, s_val, cost


# ============ 步骤3：电价缩放灵敏度分析（独立模块，非鲁棒）============
def solve_deterministic(day_load, day_pv, price_arr, soc_prev=S0):
    """
    确定性单日优化（用于灵敏度分析）。
    注意：这不是鲁棒优化，只是单场景确定性LP。
    """
    prob = pulp.LpProblem("Det", pulp.LpMinimize)
    g = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(N)]
    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P_MAX) for t in range(N)]
    e = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in range(N)]
    s = [pulp.LpVariable(f"s_{t}", lowBound=S_MIN, upBound=S_MAX) for t in range(N+1)]

    prob += pulp.lpSum(price_arr[t]*g[t]*DT + EM_MULT*price_arr[t]*e[t]*DT for t in range(N))
    prob += s[0] == soc_prev
    prob += s[N] == soc_prev  # 闭环
    for t in range(N):
        prob += g[t] + day_pv[t] + d[t] + e[t] >= day_load[t] + c[t]
        prob += s[t+1] == s[t] + c[t]*DT*ETA - d[t]*DT/ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT))
    g_val = np.array([pulp.value(g[t]) for t in range(N)])
    c_val = np.array([pulp.value(c[t]) for t in range(N)])
    d_val = np.array([pulp.value(d[t]) for t in range(N)])
    e_val = np.array([pulp.value(e[t]) for t in range(N)])
    s_val = np.array([pulp.value(s[t]) for t in range(N+1)])
    return g_val, c_val, d_val, e_val, s_val, pulp.value(prob.objective)


def run_sensitivity(scale_list):
    """
    电价缩放灵敏度分析（纯确定性，非鲁棒）。
    对波动电价做缩放：scaled = mean + scale*(fluct - mean)
    scale=0 → 全天均值电价（无波动）
    scale=1 → 原始波动电价
    scale=1.5 → 放大波动
    """
    print("\n  --- 电价缩放灵敏度分析（确定性，非鲁棒）---")
    results = []
    for scale in scale_list:
        soc = S0
        total = 0.0
        t0 = time.time()
        for key in target_dates:
            fluc_p = price_dict[key]
            # 缩放：围绕当天均值
            day_mean = fluc_p.mean()
            scaled_p = day_mean + scale * (fluc_p - day_mean)
            _, _, _, _, _, cost = solve_deterministic(
                load_dict[key], pv_dict[key], scaled_p, soc)
            # SOC闭环，s[N]=s[0]，soc不变
            total += cost
        elapsed = time.time() - t0
        results.append((scale, total))
        print(f"    波动幅度×{scale:.2f}: {total:,.2f} 元 ({elapsed:.0f}s)")
    return results


# ============ 步骤4：批量求解预算鲁棒模型（不同Γ）============
print("\n" + "=" * 60)
print("步骤2：预算鲁棒优化（不同Γ值对比）")
print("=" * 60)

GAMMA_LIST = [0, 12, 36, 72, 144]  # Γ=0退化确定性；Γ=144最坏情况
gamma_results = {}

for Gamma in GAMMA_LIST:
    print(f"\n  Γ={Gamma}（保守度{'最低' if Gamma==0 else '最高' if Gamma==N else '中等'}）")
    soc = S0
    total_cost = 0.0
    t0 = time.time()
    daily_records = []

    for i, key in enumerate(target_dates):
        day_load = load_dict[key]
        day_pv = pv_dict[key]
        day_price = price_dict[key]      # 名义价格=当天实际波动电价
        delta = get_delta(day_price)     # 偏差幅度=|波动-固定|

        g_val, c_val, d_val, e_val, s_val, cost = solve_budget_robust(
            day_load, day_pv, day_price, delta, Gamma, soc)
        total_cost += cost
        # SOC闭环约束已保证s[N]==s[0]，所以soc不变（但仍记录以防万一）
        soc = s_val[-1]

        if (i+1) % 50 == 0 or i == 0:
            print(f"    [{i+1}/{len(target_dates)}] {key} 累计={total_cost:,.0f}")

    elapsed = time.time() - t0
    gamma_results[Gamma] = total_cost
    print(f"    全年总成本: {total_cost:,.2f} 元 ({elapsed:.0f}s)")

# ============ 步骤5：输出 result4_sensitivity.xlsx ============
print("\n" + "=" * 60)
print("步骤3：输出 result4_sensitivity.xlsx")
print("=" * 60)

# 先跑灵敏度分析
sens_results = run_sensitivity([0.0, 0.5, 0.75, 1.0, 1.25, 1.5])

wb = openpyxl.Workbook()

# --- Sheet1: 预算鲁棒Γ对比 ---
ws1 = wb.active
ws1.title = "预算鲁棒Γ对比"
ws1.append(["预算参数Γ", "全年成本(元)", "相对Γ=0增加(元)", "增加率", "说明"])
for Gamma in GAMMA_LIST:
    cost = gamma_results[Gamma]
    diff = cost - gamma_results[0]
    rate = f"{diff/gamma_results[0]*100:.2f}%" if gamma_results[0] > 0 else "0%"
    note = "确定性退化" if Gamma == 0 else ("最坏情况" if Gamma == N else f"同时{Gamma}个时段最坏")
    ws1.append([Gamma, round(cost, 2), round(diff, 2), rate, note])

# --- Sheet2: 电价缩放灵敏度 ---
ws2 = wb.create_sheet("电价缩放灵敏度")
ws2.append(["波动幅度缩放系数", "全年成本(元)", "说明"])
for scale, cost in sens_results:
    if scale == 0:
        note = "全天均值电价（无波动）"
    elif scale == 1.0:
        note = "原始波动电价"
    else:
        note = f"波动幅度×{scale}"
    ws2.append([f"×{scale}", round(cost, 2), note])

# --- Sheet3: 文字注释 ---
ws3 = wb.create_sheet("分析说明")
ws3.append(["问题", "解释"])
ws3.append([
    "为什么电价波动越大，总成本反而下降？",
    "这是因为储能的套利空间随波动幅度增大而扩大："
    "低谷电价时充电、高峰电价时放电的价差变大，"
    "电池在谷峰之间套利的收益超过了波动带来的额外购电成本。"
    "当波动幅度=0（全天均价）时，没有套利空间，电池只能平抑负荷，"
    "成本反而最高；波动越大，储能套利收益越大，总成本越低。"
])
ws3.append([
    "预算鲁棒Γ的含义",
    "Γ控制价格不确定集中允许同时偏离到最坏值的时段数量上限。"
    "Γ=0时不考虑不确定性（退化为确定性优化）；"
    "Γ=144时假设所有时段同时最坏（最保守）；"
    "中间值在保守性和经济性之间折中。"
])
ws3.append([
    "与两阶段随机规划的区别",
    "两阶段随机规划需要场景概率（k-means聚类5个场景），最小化期望成本；"
    "预算鲁棒优化不需要概率分布，仅知道价格偏差区间，"
    "通过Γ调节最坏情况覆盖范围。两者互补：SP量化平均成本，RO保护尾部风险。"
])

wb.save(os.path.join(CURRENT_DIR, "result4_sensitivity.xlsx"))
print(f"  已保存: {os.path.join(CURRENT_DIR, 'result4_sensitivity.xlsx')}")

# ============ 步骤6：汇总打印 ============
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print("【预算鲁棒优化 Γ对比】")
for Gamma in GAMMA_LIST:
    print(f"  Γ={Gamma:>3}: {gamma_results[Gamma]:>14,.2f} 元")
print("\n【电价缩放灵敏度（确定性）】")
for scale, cost in sens_results:
    print(f"  ×{scale:.2f}: {cost:>14,.2f} 元")
print("\n完成！")
