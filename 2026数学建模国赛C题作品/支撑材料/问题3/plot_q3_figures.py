# -*- coding: utf-8 -*-
"""
问题三 · 图片生成脚本（合并版，修正单位与模型硬伤）
=====================================================
本脚本一个文件覆盖「问题三合并版」文件夹内的全部5张图：
  图3_1_预报价值对比.png       —— 四策略 + Oracle完美预测下界（正确1300万量级）
  图3_2_偏差成本分解.png       —— 双饼图：总费用构成 + 偏差成本分解
  图3_3_典型日MPC滚动优化.png  —— 典型日(2025-06-21)光伏预报vs实际 + 计划vs调整购电(kWh)
  图3_3_费用分解.png           —— 全年费用横向堆叠条形（购电/调减/调增/紧急）
  图3_4_MPC机制示意图.png      —— Receding Horizon 滚动机制示意

同时输出与官方模板对齐的 result3.xlsx（时段单元格为kWh）。
模型内核：CAP_MAX=10800（附录1）、目标×DT、每日SOC闭环、调减罚0.5/调增溢价0.5、紧急购电5倍。

【运行前】确认 BASE_DIR 指向你的附件目录（含附件1/2/3.xlsx、附件5/result3.xlsx模板）。
"""
import os, numpy as np, pandas as pd, pulp, time, warnings
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings('ignore')

# ============ 路径配置 ============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))   # 附件目录（Andy\附件）
OUTPUT_DIR = CURRENT_DIR                                               # 输出到本脚本同目录
TEMPLATE_PATH = os.path.join(BASE_DIR, '附件5', 'result3.xlsx')       # 官方模板
RESULT_PATH = os.path.join(OUTPUT_DIR, 'result3.xlsx')
# ================================

# ============ 储能参数（附录1）============
CAP_MAX = 10800.0     # SOC上限（电量须保持1200-10800）
CAP_MIN = 1200.0
P_MAX = 5000.0
ETA = 0.9
SOC_INIT = 6000.0
DT = 10.0 / 60.0
N = 144

# ============ 费用参数（问题3）============
EMERGENCY_MULT = 5.0
DECREASE_PENALTY = 0.5   # 调减违约 = 交易电价50%
INCREASE_EXTRA = 0.5     # 调增超出部分 = 1.5倍（额外0.5倍溢价）

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11


# ============ 数据读取 ============
def load_price():
    df = pd.read_excel(os.path.join(BASE_DIR, '附件1.xlsx'))
    return df['电价'].values.astype(float)

def load_load_pv():
    df_l = pd.read_excel(os.path.join(BASE_DIR, '附件2.xlsx'), sheet_name='小区负载')
    df_p = pd.read_excel(os.path.join(BASE_DIR, '附件2.xlsx'), sheet_name='光伏发电实际功率')
    dates = pd.to_datetime(df_l.iloc[:, 0]).values
    d2i = {}
    for i, d in enumerate(dates):
        d2i[pd.Timestamp(d).strftime('%Y-%m-%d')] = i
    return d2i, df_l.iloc[:, 1:145].values.astype(float), df_p.iloc[:, 1:145].values.astype(float)

def load_forecast():
    df = pd.read_excel(os.path.join(BASE_DIR, '附件3.xlsx'))
    df['日期'] = df['日期'].ffill()
    fc = {}
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(str(row['日期']).strip())
            key = d.strftime('%Y-%m-%d')
        except Exception:
            continue
        hour = int(str(row['预报时刻']).strip().split(':')[0])
        fc[(key, hour)] = row.iloc[2:26].values.astype(float)
    return fc

def get_fc_10min(date_key, fc_hour, fc_data, current_power=None):
    key = (date_key, fc_hour)
    if key not in fc_data:
        return np.zeros(N)
    hourly = fc_data[key]
    full = np.zeros(25)
    for h in range(24):
        t = fc_hour + 1 + h
        if 0 <= t <= 24:
            full[t] = hourly[h]
    if current_power is not None:
        full[fc_hour] = current_power
    result = np.zeros(N)
    for i in range(N):
        t_min = (i + 1) * 10
        t_h = t_min / 60.0
        h0 = int(np.floor(t_h))
        h1 = min(h0 + 1, 24)
        result[i] = full[h0] + (t_h - h0) * (full[h1] - full[h0])
    return result


# ============ 统一LP求解（乘DT + SOC闭环 + 调整费拆分 + 状态检查）============
def solve_lp(price, load, pv, soc_init, planned=None, start_idx=0,
             end_idx=N, fixed_buy=None, pv_actual_past=None, time_limit=30, tag=''):
    n_opt = end_idx - start_idx
    prob = pulp.LpProblem("mpc", pulp.LpMinimize)
    chg = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    dis = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emg = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    buy = [pulp.LpVariable(f"b_{i}", lowBound=0) for i in range(n_opt)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N + 1)]
    prob += soc[0] == soc_init
    prob += soc[N] == soc_init   # ✅ SOC闭环

    obj = pulp.lpSum(price[start_idx + i] * buy[i] * DT for i in range(n_opt))
    obj += pulp.lpSum(EMERGENCY_MULT * price[i] * emg[i] * DT for i in range(N))
    if planned is not None:
        wd = [pulp.LpVariable(f"wd_{i}", lowBound=0) for i in range(n_opt)]
        wi = [pulp.LpVariable(f"wi_{i}", lowBound=0) for i in range(n_opt)]
        for i in range(n_opt):
            prob += wd[i] >= planned[start_idx + i] - buy[i]
            prob += wi[i] >= buy[i] - planned[start_idx + i]
        obj += pulp.lpSum(DECREASE_PENALTY * price[start_idx + i] * wd[i] * DT
                          + INCREASE_EXTRA * price[start_idx + i] * wi[i] * DT
                          for i in range(n_opt))
    prob += obj

    pv_use = pv_actual_past if pv_actual_past is not None else pv
    for i in range(N):
        if i < start_idx:
            b_val = fixed_buy[i] if fixed_buy is not None else 0
            prob += b_val + pv_use[i] + dis[i] + emg[i] >= load[i] + chg[i]
        else:
            prob += buy[i - start_idx] + pv[i] + dis[i] + emg[i] >= load[i] + chg[i]
        prob += soc[i + 1] == soc[i] + chg[i] * DT * ETA - dis[i] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] != 'Optimal':   # ✅ 状态检查
        print(f"  ⚠️ [{tag}] 求解非最优: {pulp.LpStatus[prob.status]}")

    buy_res = np.zeros(N)
    if fixed_buy is not None:
        buy_res[:start_idx] = fixed_buy[:start_idx]
    for i in range(n_opt):
        buy_res[start_idx + i] = pulp.value(buy[i])
    return (buy_res,
            np.array([pulp.value(chg[i]) for i in range(N)]),
            np.array([pulp.value(dis[i]) for i in range(N)]),
            np.array([pulp.value(emg[i]) for i in range(N)]),
            pulp.value(soc[N]), pulp.value(prob.objective))

def settle(price, load, pv_actual, buy_plan, soc_init, tag=''):
    prob = pulp.LpProblem("settle", pulp.LpMinimize)
    chg = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    dis = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emg = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N + 1)]
    prob += soc[0] == soc_init
    prob += soc[N] == soc_init
    prob += pulp.lpSum(EMERGENCY_MULT * price[i] * emg[i] * DT for i in range(N))
    for i in range(N):
        prob += buy_plan[i] + pv_actual[i] + dis[i] + emg[i] >= load[i] + chg[i]
        prob += soc[i + 1] == soc[i] + chg[i] * DT * ETA - dis[i] * DT / ETA
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"  ⚠️ [{tag}] 结算非最优: {pulp.LpStatus[prob.status]}")
    return (np.array([pulp.value(chg[i]) for i in range(N)]),
            np.array([pulp.value(dis[i]) for i in range(N)]),
            np.array([pulp.value(emg[i]) for i in range(N)]),
            pulp.value(soc[N]))


# ============ 单日MPC ============
def run_day_mpc(key, price, load, pv_act, fc_data, soc0, adj_hours=(6, 12, 18)):
    pv_fc0 = get_fc_10min(key, 0, fc_data)
    plan_buy, _, _, _, _, _ = solve_lp(price, load, pv_fc0, soc0, start_idx=0, end_idx=N, tag=f"{key}计划")

    adj_buy = plan_buy.copy()
    for ah in adj_hours:
        si = ah * 6
        cur_p = pv_act[si] if si < N else 0
        pv_fc = get_fc_10min(key, ah, fc_data, current_power=cur_p)
        new_buy, _, _, _, _, _ = solve_lp(
            price, load, pv_fc, soc0, planned=plan_buy,
            start_idx=si, end_idx=N, fixed_buy=adj_buy, pv_actual_past=pv_act, tag=f"{key}@{ah}点")
        next_ah = ah + 6 if ah < 18 else 24
        adj_buy[si:next_ah * 6] = new_buy[si:next_ah * 6]

    fc, fd, fe, soc_f = settle(price, load, pv_act, adj_buy, soc0, tag=key)

    buy_cost = np.sum(price * adj_buy * DT)
    dec_pen = DECREASE_PENALTY * np.sum(price[adj_buy < plan_buy] *
                                        (plan_buy[adj_buy < plan_buy] - adj_buy[adj_buy < plan_buy]) * DT)
    inc_pen = INCREASE_EXTRA * np.sum(price[adj_buy > plan_buy] *
                                      (adj_buy[adj_buy > plan_buy] - plan_buy[adj_buy > plan_buy]) * DT)
    emg_cost = np.sum(EMERGENCY_MULT * price * fe * DT)
    total = buy_cost + dec_pen + inc_pen + emg_cost
    return plan_buy, adj_buy, fc, fd, fe, soc_f, {
        'buy_cost': buy_cost, 'dec_pen': dec_pen, 'inc_pen': inc_pen,
        'emg_cost': emg_cost, 'total': total
    }

def oracle_day(price, load, pv_act, soc0, tag=''):
    buy, chg, dis, emg, soc_f, cost = solve_lp(price, load, pv_act, soc0,
                                               start_idx=0, end_idx=N, tag=tag)
    return cost


# ============ 写入 result3.xlsx（官方模板对齐，时段=kWh）============
def write_result3(dates, results, price):
    from openpyxl import load_workbook, Workbook
    if os.path.exists(TEMPLATE_PATH):
        wb = load_workbook(TEMPLATE_PATH)
        need = {'计划购电量', '调整购电量', '充放电量', '紧急购电量'}
        for s in need - set(wb.sheetnames):
            wb.create_sheet(s)
    else:
        wb = Workbook()
        wb.remove(wb.active)
        for s in ['计划购电量', '调整购电量', '充放电量', '紧急购电量']:
            wb.create_sheet(s)

    time_labels = []
    for i in range(N):
        sh, sm = divmod(i * 10, 60)
        eh, em = divmod((i + 1) * 10, 60)
        if eh == 24:
            time_labels.append(f"{sh}:{sm:02d}-0:00+1")
        else:
            time_labels.append(f"{sh}:{sm:02d}-{eh}:{em:02d}")

    ws = wb['计划购电量']
    if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
        ws.cell(1, 1, '日期\\时间')
        for i, t in enumerate(time_labels):
            ws.cell(1, i + 2, t)
        ws.cell(1, N + 2, '全天购电量')
        ws.cell(1, N + 3, '全天购电费')
    for day_idx, d in enumerate(dates):
        row = day_idx + 2
        planned_kwh = results['planned'][day_idx] * DT     # ✅ kW→kWh
        ws.cell(row, 1, d)
        for i in range(N):
            ws.cell(row, i + 2, round(float(planned_kwh[i]), 2))
        ws.cell(row, N + 2, round(float(np.sum(planned_kwh)), 2))
        ws.cell(row, N + 3, round(float(np.sum(price * planned_kwh)), 2))   # 计划购电费

    ws = wb['调整购电量']
    if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
        ws.cell(1, 1, '日期\\时间')
        for i, t in enumerate(time_labels):
            ws.cell(1, i + 2, t)
        ws.cell(1, N + 2, '全天购电量')
        ws.cell(1, N + 3, '全天购电费')
    for day_idx, d in enumerate(dates):
        row = day_idx + 2
        adjusted_kwh = results['adjusted'][day_idx] * DT   # ✅ kW→kWh
        ws.cell(row, 1, d)
        for i in range(N):
            ws.cell(row, i + 2, round(float(adjusted_kwh[i]), 2))
        ws.cell(row, N + 2, round(float(np.sum(adjusted_kwh)), 2))
        ws.cell(row, N + 3, round(float(results['costs'][day_idx]['total']), 2))  # 最终总费用

    ws = wb['充放电量']
    if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
        for c, h in enumerate(['日期', '时间段', '充电量', '放电量', '时刻', '储电量'], start=1):
            ws.cell(1, c, h)
    seg_labels = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
    row = 2
    for day_idx, d in enumerate(dates):
        for seg in range(6):
            if seg == 0:
                ws.cell(row, 1, d)
                ws.cell(row, 5, datetime(2025, d.month, d.day, 0, 0).time())
                ws.cell(row, 6, round(float(results['soc_0'][day_idx]), 2))
            ws.cell(row, 2, seg_labels[seg])
            ws.cell(row, 3, round(float(results['charge_4h'][day_idx][seg]), 2))
            ws.cell(row, 4, round(float(results['discharge_4h'][day_idx][seg]), 2))
            if seg == 1:
                ws.cell(row, 5, '24:00')
                ws.cell(row, 6, round(float(results['soc_24'][day_idx]), 2))
            row += 1

    ws = wb['紧急购电量']
    if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
        for c, h in enumerate(['日期', '购电时间段', '购电量'], start=1):
            ws.cell(1, c, h)
    row = 2
    current_date = None
    for em in results['emergency']:
        date_str, time_str, amount = em
        d = datetime.strptime(date_str, '%Y-%m-%d')
        if date_str != current_date:
            ws.cell(row, 1, d)
            current_date = date_str
        ws.cell(row, 2, time_str)
        ws.cell(row, 3, round(float(amount), 2))
        row += 1

    wb.save(RESULT_PATH)
    print(f"  result3.xlsx 已保存: {RESULT_PATH}")


# ============ 绘图 ============
def _draw_pie_external(ax, sizes, labels, colors, title, startangle=90):
    total = sum(sizes)
    wedges, _ = ax.pie(sizes, colors=colors, startangle=startangle,
                       wedgeprops=dict(edgecolor='white', linewidth=2), radius=0.85)
    entries = []
    for i, w in enumerate(wedges):
        ang_deg = (w.theta1 + w.theta2) / 2.0
        ang = np.deg2rad(ang_deg)
        entries.append({'x0': np.cos(ang), 'y0': np.sin(ang),
                        'pct': sizes[i] / total * 100, 'name': labels[i]})
    entries.sort(key=lambda e: e['y0'])
    min_gap = 0.20
    for i in range(1, len(entries)):
        if entries[i]['y0'] - entries[i - 1]['y0'] < min_gap:
            entries[i]['y0'] = entries[i - 1]['y0'] + min_gap
    for e in entries:
        side = 1.0 if e['x0'] >= 0 else -1.0
        ang_orig = np.arctan2(e['y0'], e['x0'])
        x_a = 0.87 * np.cos(ang_orig)
        y_a = 0.87 * np.sin(ang_orig)
        x_b = 1.12 * side
        y_b = e['y0']
        x_c = 1.48 * side
        ax.plot([x_a, x_b, x_c], [y_a, y_b, y_b], color='gray', linewidth=0.9, clip_on=False)
        ha = 'left' if side > 0 else 'right'
        ax.text(x_c + 0.03 * side, y_b, f"{e['name']}  {e['pct']:.1f}%",
                ha=ha, va='center', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.4, 1.4)


def plot_all(strat_cost, oracle_total, all_results, dates, pv_data, fc_data, d2i):
    names_short = ['仅0点', '0+6', '0+6+12', '全窗口']
    names = ['Oracle\n完美预测'] + names_short
    vals = [oracle_total] + [strat_cost[s] for s in names_short]

    # ---- 图3_1 预报价值对比（含Oracle下界）----
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50', '#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
    bars = ax.bar(names, [v / 1e4 for v in vals], color=colors, edgecolor='black', linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 1e4, f'{v / 1e4:.1f}万',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('全年成本（万元）')
    ax.set_title('MPC策略对比（含Oracle完美预测下界）', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_1_预报价值对比.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 图3_2 偏差成本分解（双饼图）----
    dec = sum(r[1][6]['dec_pen'] for r in all_results)
    inc = sum(r[1][6]['inc_pen'] for r in all_results)
    emg = sum(r[1][6]['emg_cost'] for r in all_results)
    buy = sum(r[1][6]['buy_cost'] for r in all_results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _draw_pie_external(axes[0], [buy, dec, inc, emg],
                       ['实际购电费', '调减违约金', '调增溢价', '紧急购电费'],
                       ['#4472C4', '#ED7D31', '#A5A5A5', '#E53935'], '总购电费用构成')
    _draw_pie_external(axes[1], [dec, inc, emg],
                       ['调减违约金', '调增溢价', '紧急购电费'],
                       ['#ED7D31', '#A5A5A5', '#E53935'], '偏差成本分解（调整+紧急）')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_2_偏差成本分解.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 图3_3 典型日MPC滚动优化（kWh单位）----
    target = datetime(2025, 6, 21)
    tk = target.strftime('%Y-%m-%d')
    day_idx = dates.index(target)
    planned = all_results[day_idx][1][0] * DT        # ✅ kW→kWh
    adjusted = all_results[day_idx][1][1] * DT       # ✅ kW→kWh
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    hours = np.arange(N) * DT
    if tk in d2i and pv_data is not None:
        pv_idx = d2i[tk]
        axes[0].plot(hours, pv_data[pv_idx], 'k-', linewidth=2, label='实际光伏')
        for h, c in [(0, '#4472C4'), (6, '#ED7D31'), (12, '#70AD47'), (18, '#FFC000')]:
            fcv = get_fc_10min(tk, h, fc_data, current_power=pv_data[pv_idx][h * 6])
            axes[0].plot(hours[h * 6:], fcv[h * 6:], '--', color=c, alpha=0.8, label=f'{h}:00预报')
    axes[0].set_ylabel('功率 (kW)')
    axes[0].set_title(f'{tk} 光伏发电预报与实际对比', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    axes[1].plot(hours, planned, '-', color='#4472C4', linewidth=1.5, label='计划购电量(0:00)')
    axes[1].plot(hours, adjusted, '-', color='#ED7D31', linewidth=1.5, label='最终调整购电量')
    axes[1].fill_between(hours, planned, adjusted, where=adjusted < planned,
                         alpha=0.3, color='green', label='调减区域')
    axes[1].fill_between(hours, planned, adjusted, where=adjusted > planned,
                         alpha=0.3, color='red', label='调增区域')
    axes[1].set_xlabel('时间 (小时)')
    axes[1].set_ylabel('购电量 (kWh)')
    axes[1].set_title(f'{tk} 计划购电量与调整购电量对比', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    for h in [6, 12, 18]:
        axes[0].axvline(x=h, color='gray', linestyle=':', alpha=0.5)
        axes[1].axvline(x=h, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_3_典型日MPC滚动优化.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 图3_3_费用分解（横向堆叠条形）----
    total_cost = buy + dec + inc + emg
    fig, ax = plt.subplots(figsize=(10, 4))
    cats = ['全年总费用构成']
    left = 0
    for label, val, color in [('购电费', buy, '#42A5F5'),
                              ('调减违约金', dec, '#FF9800'),
                              ('调增溢价', inc, '#AB47BC'),
                              ('紧急购电', emg, '#E53935')]:
        ax.barh(cats, val / 1e4, left=left / 1e4, color=color, label=label, edgecolor='black')
        if val / 1e4 > 30:
            ax.text(left / 1e4 + val / 2e4, 0, f'{val / 1e4:.0f}万\n{val / total_cost * 100:.1f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        left += val
    ax.set_xlabel('万元')
    ax.set_title('全窗口MPC全年费用分解', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_3_费用分解.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ---- 图3_4 MPC机制示意图 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (h, color) in enumerate([(0, '#4472C4'), (6, '#ED7D31'), (12, '#70AD47'), (18, '#FFC000')]):
        y = - i * 0.8
        ax.barh(y, h, height=0.4, left=0, color=color, alpha=0.3, edgecolor='black')
        ax.barh(y, 24 - h, height=0.4, left=h, color=color, alpha=0.7, edgecolor='black')
        ax.text(h / 2, y, '已交割\n(冻结)', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(h + (24 - h) / 2, y, f'优化窗口\n({h}:00预报)', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(-1, y, f'{h}:00', ha='right', va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(-2, 26)
    ax.set_ylim(-3, 0.5)
    ax.set_xlabel('时间 (小时)')
    ax.set_title('MPC滚动时域优化（Receding Horizon）机制示意图', fontweight='bold')
    ax.set_yticks([])
    ax.axvline(x=24, color='red', linestyle='--', alpha=0.5)
    ax.text(24, 0.3, '当日结束', ha='center', color='red', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_4_MPC机制示意图.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("  5张图已保存到:", OUTPUT_DIR)


# ============ 主流程 ============
def main():
    t0 = time.time()
    print("加载数据...")
    price = load_price()
    d2i, load_data, pv_data = load_load_pv()
    fc_data = load_forecast()

    dates = [datetime(2025, 2, 1) + timedelta(days=i) for i in range(334)]
    keys = [d.strftime('%Y-%m-%d') for d in dates]
    print(f"共{len(dates)}天")

    # 1月预热
    soc = SOC_INIT
    for d in [datetime(2025, 1, 1) + timedelta(days=i) for i in range(31)]:
        k = d.strftime('%Y-%m-%d')
        if k not in d2i:
            continue
        idx = d2i[k]
        pv_fc = get_fc_10min(k, 0, fc_data)
        buy, _, _, _, _, _ = solve_lp(price, load_data[idx], pv_fc, soc, tag=f"预热{k}")
        _, _, _, soc = settle(price, load_data[idx], pv_data[idx], buy, soc, tag=f"预热{k}")
    print(f"2/1初始SOC={soc:.0f}")

    strategies = {'仅0点': (), '0+6': (6,), '0+6+12': (6, 12), '全窗口': (6, 12, 18)}
    strat_cost = {k: 0.0 for k in strategies}
    strat_soc = {k: soc for k in strategies}
    oracle_total = 0.0

    all_results = []
    for di, key in enumerate(keys):
        if key not in d2i:
            continue
        idx = d2i[key]
        ld, pva = load_data[idx], pv_data[idx]

        oracle_total += oracle_day(price, ld, pva, soc, tag=f"{key}Oracle")

        day_res = {}
        for sn, ah in strategies.items():
            pb, ab, fc, fd, fe, sf, cb = run_day_mpc(
                key, price, ld, pva, fc_data, strat_soc[sn], adj_hours=ah)
            day_res[sn] = (pb, ab, fc, fd, fe, sf, cb)
            strat_cost[sn] += cb['total']
            strat_soc[sn] = sf
        all_results.append((key, day_res['全窗口']))
        if (di + 1) % 50 == 0:
            print(f"  [{di + 1}/{len(keys)}] {key} 用时{time.time() - t0:.0f}s")

    # ---- 控制台汇总 ----
    print("\n" + "=" * 60)
    print("策略对比（万元，正确口径：乘DT、SOC闭环、CAP_MAX=10800）")
    print("=" * 60)
    print(f"  Oracle完美预测:  {oracle_total / 1e4:.1f}万")
    for sn in strategies:
        c = strat_cost[sn]
        gap = (c - oracle_total) / oracle_total * 100
        print(f"  {sn:12s}: {c / 1e4:.1f}万  (距Oracle {gap:.1f}%)")
    print("\n预报边际价值:")
    prev = strat_cost['仅0点']
    for sn in ['0+6', '0+6+12', '全窗口']:
        print(f"  加{sn.split('+')[-1]}点: 省{(prev - strat_cost[sn]) / 1e4:.1f}万")
        prev = strat_cost[sn]

    # ---- 组装 result3 所需主结果（全窗口）----
    results = {'planned': [], 'adjusted': [], 'soc_0': [], 'soc_24': [],
               'charge_4h': [], 'discharge_4h': [], 'emergency': [], 'costs': []}
    for di, (key, day_res) in enumerate(all_results):
        planned, adjusted, fcharge, fdischarge, femergency, soc_final, cost_bd = day_res
        results['planned'].append(planned)
        results['adjusted'].append(adjusted)
        results['soc_0'].append(strat_soc['全窗口'])
        results['soc_24'].append(soc_final)
        results['costs'].append(cost_bd)
        c4 = np.zeros(6)
        d4 = np.zeros(6)
        for seg in range(6):
            c4[seg] = np.sum(fcharge[seg * 24:seg * 24 + 24] * DT)
            d4[seg] = np.sum(fdischarge[seg * 24:seg * 24 + 24] * DT)
        results['charge_4h'].append(c4)
        results['discharge_4h'].append(d4)
        em = femergency
        i = 0
        while i < N:
            if em[i] > 0.1:
                start_i = i
                while i < N and em[i] > 0.1:
                    i += 1
                end_i = i
                sm_, em_ = divmod(start_i * 10, 60)
                eh, em2 = divmod(end_i * 10, 60)
                results['emergency'].append(
                    (key, f"{sm_}:{em_:02d}-{eh}:{em2:02d}", np.sum(em[start_i:end_i] * DT)))
            else:
                i += 1

    print("\n写入 result3.xlsx ...")
    write_result3(dates, results, price)

    print("生成图片 ...")
    plot_all(strat_cost, oracle_total, all_results, dates, pv_data, fc_data, d2i)

    print(f"\n全部完成，总用时{time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
