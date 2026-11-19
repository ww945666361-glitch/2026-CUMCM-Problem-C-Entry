# -*- coding: utf-8 -*-
"""
问题4-3：波动电价下的问题3（MPC滚动优化 + 预报调整）
基于 problem3_mpc.py 修改，将附件1固定电价替换为附件4逐日波动电价
"""
import os
import numpy as np
import pandas as pd
import pulp
from datetime import datetime, timedelta
from openpyxl import load_workbook
import shutil
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "附件"))   # 附件目录（Andy\附件）
OUTPUT_DIR = CURRENT_DIR
TEMPLATE_PATH = os.path.join(BASE_DIR, '附件5', 'result4-3.xlsx')     # 官方模板
RESULT_PATH = os.path.join(OUTPUT_DIR, 'result4-3.xlsx')

# 复制模板
shutil.copy(TEMPLATE_PATH, RESULT_PATH)
print(f"模板已复制到: {RESULT_PATH}")

# 储能参数
CAP_MAX = 10800.0
CAP_MIN = 1200.0
P_MAX = 5000.0
ETA = 0.9
SOC_INIT = 6000.0
DT = 10.0 / 60.0
N = 144

# 费用参数
EMERGENCY_MULT = 5.0
DECREASE_PENALTY = 0.5
INCREASE_MULT = 1.5


# ============ 数据读取 ============
def load_price_dict():
    """读取附件4波动电价，返回 {date_str: array(144,)}"""
    df = pd.read_excel(os.path.join(BASE_DIR, '附件4.xlsx'))
    dates = pd.to_datetime(df.iloc[:, 0])
    prices = df.iloc[:, 1:].values.astype(float)
    price_dict = {}
    for i, d in enumerate(dates):
        price_dict[d.strftime('%Y-%m-%d')] = prices[i]
    print(f"波动电价: {len(price_dict)} 天")
    return price_dict


def load_load_and_pv():
    df_load = pd.read_excel(os.path.join(BASE_DIR, '附件2.xlsx'), sheet_name='小区负载')
    df_pv = pd.read_excel(os.path.join(BASE_DIR, '附件2.xlsx'), sheet_name='光伏发电实际功率')
    dates = pd.to_datetime(df_load.iloc[:, 0]).values
    load_data = df_load.iloc[:, 1:145].values.astype(float)
    pv_data = df_pv.iloc[:, 1:145].values.astype(float)
    date_to_idx = {}
    for i, d in enumerate(dates):
        key = pd.Timestamp(d).strftime('%Y-%m-%d')
        date_to_idx[key] = i
    return date_to_idx, load_data, pv_data


def load_forecast():
    df = pd.read_excel(os.path.join(BASE_DIR, '附件3.xlsx'))
    df['日期'] = df['日期'].ffill()
    forecast = {}
    for _, row in df.iterrows():
        date_str = str(row['日期']).strip()
        try:
            d = pd.to_datetime(date_str)
            date_key = d.strftime('%Y-%m-%d')
        except:
            continue
        hour_str = str(row['预报时刻']).strip()
        hour = int(hour_str.split(':')[0])
        vals = row.iloc[2:26].values.astype(float)
        forecast[(date_key, hour)] = vals
    return forecast


def get_forecast_for_day(date_key, forecast_hour, forecast_data, current_power=None):
    key = (date_key, forecast_hour)
    if key not in forecast_data:
        return np.zeros(N)
    hourly = forecast_data[key]
    full_hourly = np.zeros(25)
    for h in range(24):
        t_hour = forecast_hour + 1 + h
        if 0 <= t_hour <= 24:
            full_hourly[t_hour] = hourly[h]
    if current_power is not None:
        full_hourly[forecast_hour] = current_power
    result = np.zeros(N)
    for i in range(N):
        t_min = (i + 1) * 10
        t_hour = t_min / 60.0
        h0 = int(np.floor(t_hour))
        h1 = min(h0 + 1, 24)
        frac = t_hour - h0
        result[i] = full_hourly[h0] + frac * (full_hourly[h1] - full_hourly[h0])
    return result


# ============ LP 优化 ============
def solve_optimization(price, load, pv, soc_init, planned_purchase=None,
                       start_idx=0, end_idx=N, fixed_purchase=None, pv_actual=None):
    n_opt = end_idx - start_idx
    prob = pulp.LpProblem("microgrid", pulp.LpMinimize)

    charge = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    discharge = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emergency = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    purchase_var = [pulp.LpVariable(f"p_{i}", lowBound=0) for i in range(n_opt)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N + 1)]
    prob += soc[0] == soc_init

    obj = pulp.lpSum([price[start_idx + i] * purchase_var[i] * DT for i in range(n_opt)])
    obj += pulp.lpSum([EMERGENCY_MULT * price[i] * emergency[i] * DT for i in range(N)])

    if planned_purchase is not None:
        w = [pulp.LpVariable(f"w_{i}", lowBound=0) for i in range(n_opt)]
        for i in range(n_opt):
            p_i = planned_purchase[start_idx + i]
            prob += w[i] >= purchase_var[i] - p_i
            prob += w[i] >= p_i - purchase_var[i]
        obj += pulp.lpSum([DECREASE_PENALTY * price[start_idx + i] * w[i] * DT for i in range(n_opt)])

    prob += obj

    pv_use = pv_actual if pv_actual is not None else pv
    for i in range(N):
        if i < start_idx:
            p_val = fixed_purchase[i] if fixed_purchase is not None else 0
            prob += p_val + pv_use[i] + discharge[i] + emergency[i] >= load[i] + charge[i]
        else:
            idx = i - start_idx
            prob += purchase_var[idx] + pv[i] + discharge[i] + emergency[i] >= load[i] + charge[i]
        prob += soc[i + 1] == soc[i] + ETA * charge[i] * DT - discharge[i] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    purchase_result = np.zeros(N)
    if fixed_purchase is not None:
        purchase_result[:start_idx] = fixed_purchase[:start_idx]
    charge_arr = np.zeros(N)
    discharge_arr = np.zeros(N)
    emergency_arr = np.zeros(N)
    for i in range(N):
        charge_arr[i] = pulp.value(charge[i])
        discharge_arr[i] = pulp.value(discharge[i])
        emergency_arr[i] = pulp.value(emergency[i])
    for i in range(n_opt):
        purchase_result[start_idx + i] = pulp.value(purchase_var[i])

    soc_final = pulp.value(soc[N])
    return purchase_result, charge_arr, discharge_arr, emergency_arr, soc_final


def simulate_settlement(price, load, pv_actual, purchase_plan, soc_init):
    prob = pulp.LpProblem("settlement", pulp.LpMinimize)
    purchase = purchase_plan
    charge = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    discharge = [pulp.LpVariable(f"d_{i}", lowBound=0, upBound=P_MAX) for i in range(N)]
    emergency = [pulp.LpVariable(f"e_{i}", lowBound=0) for i in range(N)]
    soc = [pulp.LpVariable(f"s_{i}", lowBound=CAP_MIN, upBound=CAP_MAX) for i in range(N + 1)]
    prob += soc[0] == soc_init
    prob += pulp.lpSum([EMERGENCY_MULT * price[i] * emergency[i] * DT for i in range(N)])

    for i in range(N):
        prob += purchase[i] + pv_actual[i] + discharge[i] + emergency[i] >= load[i] + charge[i]
        prob += soc[i + 1] == soc[i] + ETA * charge[i] * DT - discharge[i] * DT / ETA

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    charge_arr = np.array([pulp.value(charge[i]) for i in range(N)])
    discharge_arr = np.array([pulp.value(discharge[i]) for i in range(N)])
    emergency_arr = np.array([pulp.value(emergency[i]) for i in range(N)])
    soc_final = pulp.value(soc[N])
    return charge_arr, discharge_arr, emergency_arr, soc_final


# ============ 单日MPC ============
def run_day_mpc(date_key, price, load, pv_actual, forecast_data, soc_init,
                adjust_hours=(6, 12, 18)):
    # 0:00 计划
    pv_forecast_0 = get_forecast_for_day(date_key, 0, forecast_data)
    planned_purchase, _, _, _, _ = solve_optimization(
        price, load, pv_forecast_0, soc_init, planned_purchase=None,
        start_idx=0, end_idx=N
    )

    adjusted_purchase = planned_purchase.copy()

    for adj_hour in [6, 12, 18]:
        if adj_hour not in adjust_hours:
            continue
        start_idx = adj_hour * 6
        current_power = pv_actual[start_idx] if start_idx < N else 0.0
        pv_forecast = get_forecast_for_day(date_key, adj_hour, forecast_data,
                                           current_power=current_power)
        adj_purchase, _, _, _, _ = solve_optimization(
            price, load, pv_forecast, soc_init,
            planned_purchase=planned_purchase,
            start_idx=start_idx, end_idx=N,
            fixed_purchase=adjusted_purchase,
            pv_actual=pv_actual
        )
        next_hour = adj_hour + 6 if adj_hour < 18 else 24
        next_idx = next_hour * 6
        adjusted_purchase[start_idx:next_idx] = adj_purchase[start_idx:next_idx]

    # 最终结算
    final_charge, final_discharge, final_emergency, soc_final = simulate_settlement(
        price, load, pv_actual, adjusted_purchase, soc_init
    )

    # 费用分解（均乘以DT转换为kWh对应的费用）
    plan_cost = np.sum(price * planned_purchase) * DT
    actual_purchase_cost = np.sum(price * adjusted_purchase) * DT
    decrease_penalty = 0.0
    increase_premium = 0.0
    for i in range(N):
        p = planned_purchase[i]
        a = adjusted_purchase[i]
        if a < p:
            decrease_penalty += DECREASE_PENALTY * price[i] * (p - a) * DT
        elif a > p:
            increase_premium += (INCREASE_MULT - 1.0) * price[i] * (a - p) * DT
    adjust_cost = decrease_penalty + increase_premium
    emergency_cost = np.sum(EMERGENCY_MULT * price * final_emergency) * DT
    total_cost = actual_purchase_cost + adjust_cost + emergency_cost

    cost_breakdown = {
        'plan_cost': plan_cost,
        'actual_purchase_cost': actual_purchase_cost,
        'decrease_penalty': decrease_penalty,
        'increase_premium': increase_premium,
        'adjust_cost': adjust_cost,
        'emergency_cost': emergency_cost,
        'total_cost': total_cost,
    }
    return (planned_purchase, adjusted_purchase, final_charge, final_discharge,
            final_emergency, soc_final, cost_breakdown)


# ============ 主流程 ============
def main():
    print("加载数据...")
    price_dict = load_price_dict()
    date_to_idx, load_data, pv_data = load_load_and_pv()
    forecast_data = load_forecast()

    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 12, 31)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    print(f"共 {len(dates)} 天需要处理")

    results = {
        'planned': [], 'adjusted': [], 'charge_4h': [], 'discharge_4h': [],
        'soc_0': [], 'soc_24': [], 'emergency': [], 'costs': [],
    }

    # 模拟1月份获取2-1初始SOC（用波动电价）
    print("模拟1月份以获取2月1日初始SOC...")
    soc_current = SOC_INIT
    jan_dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(31)]
    for d in jan_dates:
        key = d.strftime('%Y-%m-%d')
        if key not in date_to_idx or key not in price_dict:
            continue
        idx = date_to_idx[key]
        price = price_dict[key]
        load_d = load_data[idx]
        pv_d = pv_data[idx]
        pv_fc = get_forecast_for_day(key, 0, forecast_data)
        p, _, _, _, _ = solve_optimization(price, load_d, pv_fc, soc_current, planned_purchase=None)
        _, _, _, soc_final = simulate_settlement(price, load_d, pv_d, p, soc_current)
        soc_current = soc_final
    print(f"  2月1日初始SOC = {soc_current:.1f} kWh")

    # 策略对比
    strategies = {
        '仅0点预报': (),
        '0+6点': (6,),
        '0+6+12点': (6, 12),
        '0+6+12+18点': (6, 12, 18),
    }
    strategy_costs = {k: 0.0 for k in strategies}
    strategy_soc = {k: soc_current for k in strategies}

    import time
    t0 = time.time()
    for day_idx, d in enumerate(dates):
        key = d.strftime('%Y-%m-%d')
        if key not in date_to_idx or key not in price_dict:
            print(f"  跳过 {key}（数据缺失）")
            continue
        idx = date_to_idx[key]
        price = price_dict[key]  # 当天波动电价
        load_d = load_data[idx]
        pv_d = pv_data[idx]

        if day_idx % 30 == 0:
            elapsed = time.time() - t0
            print(f"处理 {key} ({day_idx+1}/{len(dates)}) 用时={elapsed:.0f}s...")

        soc0_main = strategy_soc['0+6+12+18点']
        day_results = {}
        for strat_name, adj_hours in strategies.items():
            (planned, adjusted, fcharge, fdischarge, femergency,
             soc_final, cost_bd) = run_day_mpc(
                key, price, load_d, pv_d, forecast_data, strategy_soc[strat_name],
                adjust_hours=adj_hours
            )
            day_results[strat_name] = (planned, adjusted, fcharge, fdischarge,
                                       femergency, soc_final, cost_bd)
            strategy_costs[strat_name] += cost_bd['total_cost']
            strategy_soc[strat_name] = soc_final

        planned, adjusted, fcharge, fdischarge, femergency, soc_final, cost_bd = \
            day_results['0+6+12+18点']

        results['planned'].append(planned)
        results['adjusted'].append(adjusted)
        results['soc_0'].append(soc0_main)
        results['soc_24'].append(soc_final)
        results['costs'].append(cost_bd)

        c4 = np.zeros(6)
        d4 = np.zeros(6)
        for seg in range(6):
            s = seg * 24
            e = s + 24
            c4[seg] = np.sum(fcharge[s:e] * DT)
            d4[seg] = np.sum(fdischarge[s:e] * DT)
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
                total_e = np.sum(em[start_i:end_i] * DT)
                start_min = start_i * 10
                end_min = end_i * 10
                sh, sm = divmod(start_min, 60)
                eh, em_ = divmod(end_min, 60)
                time_str = f"{sh}:{sm:02d}-{eh}:{em_:02d}"
                results['emergency'].append((key, time_str, total_e))
            else:
                i += 1

    print(f"\n所有日期处理完成！总用时={time.time()-t0:.0f}s")
    print(f"\n策略对比（总费用，元）：")
    for k, v in strategy_costs.items():
        print(f"  {k}: {v:,.2f}")

    # 写入 result4-3.xlsx
    print("\n写入 result4-3.xlsx ...")
    write_result4_3(dates, results, price_dict)

    # 汇总
    total_cost = sum(c['total_cost'] for c in results['costs'])
    total_purchase = sum(c['actual_purchase_cost'] for c in results['costs'])
    total_adjust = sum(c['adjust_cost'] for c in results['costs'])
    total_emerg = sum(c['emergency_cost'] for c in results['costs'])
    print(f"\n全年总费用: {total_cost:,.2f} 元")
    print(f"  实际购电费: {total_purchase:,.2f} 元")
    print(f"  调整费: {total_adjust:,.2f} 元")
    print(f"  紧急购电费: {total_emerg:,.2f} 元")
    print("完成！")


def write_result4_3(dates, results, price_dict):
    wb = load_workbook(RESULT_PATH)

    # ---- 计划购电量 ----
    ws = wb['计划购电量']
    for day_idx, d in enumerate(dates):
        row = day_idx + 2
        planned = results['planned'][day_idx] * DT  # kW -> kWh
        key = d.strftime('%Y-%m-%d')
        price = price_dict.get(key, np.zeros(N))
        for i in range(N):
            ws.cell(row=row, column=i + 2, value=round(float(planned[i]), 2))
        ws.cell(row=row, column=N + 2, value=round(float(np.sum(planned)), 2))
        ws.cell(row=row, column=N + 3, value=round(float(np.sum(price * planned)), 2))
    print(f"  计划购电量: {len(dates)} 行")

    # ---- 调整购电量 ----
    ws = wb['调整购电量']
    for day_idx, d in enumerate(dates):
        row = day_idx + 2
        adjusted = results['adjusted'][day_idx] * DT  # kW -> kWh
        for i in range(N):
            ws.cell(row=row, column=i + 2, value=round(float(adjusted[i]), 2))
        ws.cell(row=row, column=N + 2, value=round(float(np.sum(adjusted)), 2))
        ws.cell(row=row, column=N + 3, value=round(float(results['costs'][day_idx]['total_cost']), 2))
    print(f"  调整购电量: {len(dates)} 行")

    # ---- 充放电量 ----
    ws = wb['充放电量']
    for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row_cells:
            cell.value = None
    seg_labels = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
    row = 2
    for day_idx, d in enumerate(dates):
        for seg in range(6):
            if seg == 0:
                ws.cell(row=row, column=1, value=d)
                ws.cell(row=row, column=5, value=datetime(2025, d.month, d.day, 0, 0).time())
                ws.cell(row=row, column=6, value=round(float(results['soc_0'][day_idx]), 2))
            ws.cell(row=row, column=2, value=seg_labels[seg])
            ws.cell(row=row, column=3, value=round(float(results['charge_4h'][day_idx][seg]), 2))
            ws.cell(row=row, column=4, value=round(float(results['discharge_4h'][day_idx][seg]), 2))
            if seg == 1:
                ws.cell(row=row, column=5, value='24:00')
                ws.cell(row=row, column=6, value=round(float(results['soc_24'][day_idx]), 2))
            row += 1
    print(f"  充放电量: {len(dates) * 6} 行")

    # ---- 紧急购电量 ----
    ws = wb['紧急购电量']
    for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row_cells:
            cell.value = None
    row = 2
    current_date = None
    for em in results['emergency']:
        date_str, time_str, amount = em
        d = datetime.strptime(date_str, '%Y-%m-%d')
        if date_str != current_date:
            ws.cell(row=row, column=1, value=d)
            current_date = date_str
        ws.cell(row=row, column=2, value=time_str)
        ws.cell(row=row, column=3, value=round(float(amount), 2))
        row += 1
    print(f"  紧急购电量: {row - 2} 条")

    wb.save(RESULT_PATH)
    print(f"  已保存到 {RESULT_PATH}")


if __name__ == '__main__':
    main()
