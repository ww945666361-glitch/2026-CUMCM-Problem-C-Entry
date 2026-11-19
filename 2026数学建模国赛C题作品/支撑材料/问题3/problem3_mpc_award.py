# -*- coding: utf-8 -*-
"""
问题三：MPC滚动时域优化
=====================================
"""
import os, numpy as np, pandas as pd, pulp, time, warnings
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings('ignore')

# ============ 路径（基于脚本所在目录）============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))   # 附件目录（Andy\附件）
OUTPUT_DIR = CURRENT_DIR                                               # 输出到本脚本同目录
# ================================================

# ============ 储能参数 ============
CAP_MAX = 10800.0     # ✅ 修复：原题10800，不是12000
CAP_MIN = 1200.0
P_MAX = 5000.0
ETA = 0.9
SOC_INIT = 6000.0
DT = 10.0 / 60.0
N = 144

# ============ 费用参数 ============
EMERGENCY_MULT = 5.0
DECREASE_PENALTY = 0.5   # 调减罚0.5倍
INCREASE_EXTRA = 0.5     # 调增多付0.5倍（1.5-1.0）

# 字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
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
        except:
            continue
        hour = int(str(row['预报时刻']).strip().split(':')[0])
        fc[(key, hour)] = row.iloc[2:26].values.astype(float)
    return fc

def get_fc_10min(date_key, fc_hour, fc_data, current_power=None):
    """整点预报→10分钟插值"""
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


# ============ 单日LP（带DT、SOC闭环）============
def solve_lp(price, load, pv, soc_init, planned=None, start_idx=0,
             end_idx=N, fixed_buy=None, pv_actual_past=None, time_limit=30):
    """
    统一求解器。
    - planned: 计划购电（用于算调整费），None=首次计划
    - start_idx/end_idx: 可变购电时段范围
    - fixed_buy: 已交割购电（冻结段）
    - pv_actual_past: 已交割段实际光伏
    ✅ 修复：所有成本乘DT；SOC闭环 s[N]==s[0]
    """
    n_opt = end_idx - start_idx
    prob = pulp.LpProblem("mpc", pulp.LpMinimize)

    chg = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    dis = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emg = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    buy = [pulp.LpVariable(f"b_{i}", lowBound=0) for i in range(n_opt)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N + 1)]

    # ✅ SOC闭环：首尾相等
    prob += soc[0] == soc_init
    prob += soc[N] == soc_init

    # 目标：未来购电费（×DT）+ 全天紧急购电费（×DT）
    obj = pulp.lpSum(price[start_idx+i] * buy[i] * DT for i in range(n_opt))
    obj += pulp.lpSum(EMERGENCY_MULT * price[i] * emg[i] * DT for i in range(N))

    # 调整费：拆w_dec/w_inc，论文好写
    if planned is not None:
        wd = [pulp.LpVariable(f"wd_{i}", lowBound=0) for i in range(n_opt)]
        wi = [pulp.LpVariable(f"wi_{i}", lowBound=0) for i in range(n_opt)]
        for i in range(n_opt):
            prob += wd[i] >= planned[start_idx+i] - buy[i]
            prob += wi[i] >= buy[i] - planned[start_idx+i]
        obj += pulp.lpSum(DECREASE_PENALTY * price[start_idx+i] * wd[i] * DT
                          + INCREASE_EXTRA * price[start_idx+i] * wi[i] * DT
                          for i in range(n_opt))
    prob += obj

    # 功率平衡
    pv_use = pv_actual_past if pv_actual_past is not None else pv
    for i in range(N):
        if i < start_idx:
            b_val = fixed_buy[i] if fixed_buy is not None else 0
            prob += b_val + pv_use[i] + dis[i] + emg[i] >= load[i] + chg[i]
        else:
            prob += buy[i-start_idx] + pv[i] + dis[i] + emg[i] >= load[i] + chg[i]
        prob += soc[i+1] == soc[i] + chg[i]*DT*ETA - dis[i]*DT/ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    buy_res = np.zeros(N)
    if fixed_buy is not None:
        buy_res[:start_idx] = fixed_buy[:start_idx]
    for i in range(n_opt):
        buy_res[start_idx+i] = pulp.value(buy[i])

    return (buy_res,
            np.array([pulp.value(chg[i]) for i in range(N)]),
            np.array([pulp.value(dis[i]) for i in range(N)]),
            np.array([pulp.value(emg[i]) for i in range(N)]),
            pulp.value(soc[N]), pulp.value(prob.objective))


def settle(price, load, pv_actual, buy_plan, soc_init):
    """最终结算：固定购电，优化充放电最小化紧急购电。✅ SOC闭环"""
    prob = pulp.LpProblem("settle", pulp.LpMinimize)
    chg = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    dis = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emg = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N+1)]
    prob += soc[0] == soc_init
    prob += soc[N] == soc_init  # ✅ 闭环
    prob += pulp.lpSum(EMERGENCY_MULT * price[i] * emg[i] * DT for i in range(N))
    for i in range(N):
        prob += buy_plan[i] + pv_actual[i] + dis[i] + emg[i] >= load[i] + chg[i]
        prob += soc[i+1] == soc[i] + chg[i]*DT*ETA - dis[i]*DT/ETA
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    return (np.array([pulp.value(chg[i]) for i in range(N)]),
            np.array([pulp.value(dis[i]) for i in range(N)]),
            np.array([pulp.value(emg[i]) for i in range(N)]),
            pulp.value(soc[N]))


# ============ 单日MPC ============
def run_day_mpc(key, price, load, pv_act, fc_data, soc0, adj_hours=(6,12,18)):
    # 0:00计划
    pv_fc0 = get_fc_10min(key, 0, fc_data)
    plan_buy, _, _, _, _, _ = solve_lp(price, load, pv_fc0, soc0, start_idx=0, end_idx=N)

    adj_buy = plan_buy.copy()
    for ah in adj_hours:
        si = ah * 6
        cur_p = pv_act[si] if si < N else 0
        pv_fc = get_fc_10min(key, ah, fc_data, current_power=cur_p)
        new_buy, _, _, _, _, _ = solve_lp(
            price, load, pv_fc, soc0, planned=plan_buy,
            start_idx=si, end_idx=N, fixed_buy=adj_buy, pv_actual_past=pv_act)
        # 只执行到下一个调整点
        next_ah = ah + 6 if ah < 18 else 24
        adj_buy[si:next_ah*6] = new_buy[si:next_ah*6]

    # 最终结算
    fc, fd, fe, soc_f = settle(price, load, pv_act, adj_buy, soc0)

    # 费用分解（✅ 乘DT）
    buy_cost = np.sum(price * adj_buy * DT)
    dec_pen = INCREASE_EXTRA * np.sum(price[adj_buy < plan_buy] *
                                       (plan_buy[adj_buy < plan_buy] - adj_buy[adj_buy < plan_buy]) * DT)
    inc_pen = INCREASE_EXTRA * np.sum(price[adj_buy > plan_buy] *
                                       (adj_buy[adj_buy > plan_buy] - plan_buy[adj_buy > plan_buy]) * DT)
    emg_cost = np.sum(EMERGENCY_MULT * price * fe * DT)
    total = buy_cost + dec_pen + inc_pen + emg_cost

    return plan_buy, adj_buy, fc, fd, fe, soc_f, {
        'buy_cost': buy_cost, 'dec_pen': dec_pen, 'inc_pen': inc_pen,
        'emg_cost': emg_cost, 'total': total
    }


def oracle_day(price, load, pv_act, soc0):
    """Oracle完美预测下界：直接用实际光伏做单日LP"""
    buy, chg, dis, emg, soc_f, cost = solve_lp(price, load, pv_act, soc0,
                                                 start_idx=0, end_idx=N)
    return cost


# ============ 主流程 ============
def main():
    t0 = time.time()
    price = load_price()
    d2i, load_data, pv_data = load_load_pv()
    fc_data = load_forecast()

    dates = [datetime(2025,2,1)+timedelta(days=i) for i in range(334)]
    keys = [d.strftime('%Y-%m-%d') for d in dates]
    print(f"共{len(dates)}天")

    # 1月预热获取2/1初始SOC
    soc = SOC_INIT
    for d in [datetime(2025,1,1)+timedelta(days=i) for i in range(31)]:
        k = d.strftime('%Y-%m-%d')
        if k not in d2i: continue
        idx = d2i[k]
        pv_fc = get_fc_10min(k, 0, fc_data)
        buy, _, _, _, _, _ = solve_lp(price, load_data[idx], pv_fc, soc)
        _, _, _, soc = settle(price, load_data[idx], pv_data[idx], buy, soc)
    print(f"2/1初始SOC={soc:.0f}")

    # 4种策略 + Oracle
    strategies = {'仅0点': (), '0+6': (6,), '0+6+12': (6,12), '全窗口': (6,12,18)}
    strat_cost = {k: 0.0 for k in strategies}
    strat_soc = {k: soc for k in strategies}
    oracle_total = 0.0

    # 预报精度统计
    fc_mape = {0: [], 6: [], 12: [], 18: []}

    all_results = []
    for di, key in enumerate(keys):
        if key not in d2i: continue
        idx = d2i[key]
        ld, pva = load_data[idx], pv_data[idx]

        # Oracle
        oracle_total += oracle_day(price, ld, pva, soc)

        # 预报精度（只比较该预报时刻"之后"的白天光伏时段）
        for ah in [0,6,12,18]:
            fcv = get_fc_10min(key, ah, fc_data)
            # 只比较 ah点之后 的时段（预报有效范围）
            valid_start = ah * 6  # 该小时对应的时段索引
            mask = np.zeros(N, dtype=bool)
            mask[valid_start:] = True
            # 且实际光伏>50（白天）
            mask &= (pva > 50)
            if mask.sum() > 0:
                mape = np.mean(np.abs(fcv[mask] - pva[mask]) / np.maximum(pva[mask], 1))
                fc_mape[ah].append(mape)

        # 各策略
        day_res = {}
        for sn, ah in strategies.items():
            pb, ab, fc, fd, fe, sf, cb = run_day_mpc(
                key, price, ld, pva, fc_data, strat_soc[sn], adj_hours=ah)
            day_res[sn] = (pb, ab, fc, fd, fe, sf, cb)
            strat_cost[sn] += cb['total']
            strat_soc[sn] = sf

        all_results.append((key, day_res['全窗口']))
        if (di+1) % 50 == 0:
            print(f"  [{di+1}/{len(keys)}] {key} 用时{time.time()-t0:.0f}s")

    # 输出结果
    print("\n" + "="*60)
    print("策略对比（万元）")
    print("="*60)
    print(f"  Oracle完美预测:  {oracle_total/1e4:.1f}万")
    for sn in strategies:
        c = strat_cost[sn]
        gap = (c - oracle_total) / oracle_total * 100
        print(f"  {sn:12s}: {c/1e4:.1f}万  (距Oracle {gap:.1f}%)")

    print("\n预报精度MAPE:")
    for ah in [0,6,12,18]:
        print(f"  {ah}:00预报: MAPE={np.mean(fc_mape[ah])*100:.1f}%")

    # 边际价值
    print("\n预报边际价值:")
    prev = strat_cost['仅0点']
    for sn in ['0+6', '0+6+12', '全窗口']:
        print(f"  加{sn.split('+')[-1]}点: 省{(prev-strat_cost[sn])/1e4:.1f}万")
        prev = strat_cost[sn]

    # ===== 画图 =====
    plot_results(strat_cost, oracle_total, fc_mape, all_results, price, dates)

    # 保存结果
    print(f"\n总用时{time.time()-t0:.0f}s")


def plot_results(strat_cost, oracle_cost, fc_mape, all_results, price, dates):
    # 图1：策略对比+Oracle
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ['Oracle\n完美预测', '仅0点', '0+6', '0+6+12', '全窗口']
    vals = [oracle_cost, strat_cost['仅0点'], strat_cost['0+6'],
            strat_cost['0+6+12'], strat_cost['全窗口']]
    colors = ['#4CAF50', '#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
    bars = ax.bar(names, [v/1e4 for v in vals], color=colors, edgecolor='black', linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v/1e4, f'{v/1e4:.1f}万',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('全年成本（万元）')
    ax.set_title('MPC策略对比（含Oracle完美预测下界）', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_1_预报价值对比.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 图2：预报精度 vs 经济价值
    fig, ax1 = plt.subplots(figsize=(10, 6))
    hrs = [0, 6, 12, 18]
    mapes = [np.mean(fc_mape[h])*100 for h in hrs]
    ax1.bar([f'{h}:00' for h in hrs], mapes, color='#FF9800', alpha=0.7, label='预报MAPE(%)')
    ax1.set_ylabel('MAPE(%)', color='#FF9800')
    ax1.tick_params(axis='y', labelcolor='#FF9800')
    ax2 = ax1.twinx()
    costs = [strat_cost['仅0点'], strat_cost['0+6'], strat_cost['0+6+12'], strat_cost['全窗口']]
    ax2.plot([f'{h}:00' for h in hrs], [c/1e4 for c in costs], 'o-',
             color='#1565C0', lw=2, markersize=8, label='累计策略成本(万)')
    ax2.set_ylabel('累计成本（万元）', color='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#1565C0')
    ax1.set_title('预报精度(MAPE) vs 经济价值', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_2_预报精度与价值.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 图3：费用分解（横向堆叠条形图，非饼图）
    dec = sum(r[1][6]['dec_pen'] for r in all_results)
    inc = sum(r[1][6]['inc_pen'] for r in all_results)
    emg = sum(r[1][6]['emg_cost'] for r in all_results)
    buy = sum(r[1][6]['buy_cost'] for r in all_results)
    fig, ax = plt.subplots(figsize=(10, 4))
    cats = ['全年总费用构成']
    left = 0
    for label, val, color in [('购电费', buy, '#42A5F5'),
                               ('调减违约金', dec, '#FF9800'),
                               ('调增溢价', inc, '#AB47BC'),
                               ('紧急购电', emg, '#E53935')]:
        ax.barh(cats, val/1e4, left=left/1e4, color=color, label=label, edgecolor='black')
        if val/1e4 > 30:
            ax.text(left/1e4 + val/2e4, 0, f'{val/1e4:.0f}万\n{val/(buy+dec+inc+emg)*100:.1f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        left += val
    ax.set_xlabel('万元')
    ax.set_title('全窗口MPC全年费用分解', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '图3_3_费用分解.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print("图片已保存")


if __name__ == '__main__':
    main()
