# -*- coding: utf-8 -*-
"""
问题1：微网日前经济调度 —— MILP 模型 + LP/MILP 对比 + 影子价格 + 参数灵敏度
============================================================================
覆盖大纲要求：
  基础   : MILP 模型（0-1 变量 z(t) 保证充放电互斥）
  推荐   : LP 与 MILP 结果对比（目标值 / 购电量 / 充放电量 / 同时充放电时段 / 求解时间）
  加分   : 影子价格（LP 对偶值，MILP 整数模型无对偶，故从 LP 松弛提取）
  必做   : 参数扰动灵敏度（充放电效率±5%、SOC上/下限±10%、最大充放电功率±20%）

模型口径（与 data_loader / preprocess 完全一致）：
  slot t=0..143 对应 [t*10min,(t+1)*10min)，DT=1/6 h；
  g_t 计划购电功率(kW)，c_t/d_t 储能充放电功率(kW)，E_t 储电量(kWh)；
  目标：min Σ price_t·g_t·DT；
  约束：功率平衡 pv+g+d = load+c；储能递推 E_{t+1}=E_t+η·c·DT-d·DT/η；
        SOC 上下限；充放电功率上限；初末储电量 E_0=E_144=6000；非负。

运行：python solve_q1.py
输出：result1_分析.xlsx（计划购电量 / 充放电量 / LP_MILP对比 / 影子价格 / 灵敏度）
     
"""
from __future__ import annotations

import time
import sys
import os

import numpy as np
import pandas as pd
import pulp
from openpyxl import Workbook

from data_loader import load_annex1

# ----------------------------- 物理常量（与 preprocess 一致） -----------------------------
DT = 1.0 / 6.0          # 时段长度（小时）
PERIODS = 144           # 每天时段数
ETA = 0.9               # 充放电效率
E_MIN = 1200.0          # 储电量下限 kWh
E_MAX = 10800.0         # 储电量上限 kWh
E_INIT = 6000.0         # 初末储电量 kWh
P_MAX = 5000.0          # 最大充/放电功率 kW
EPS = 1e-6              # 判定"同时充放电"的数值阈值


# ======================================================================
# 模型构建：use_milp=True 加 0-1 互斥变量；False 为纯 LP 松弛
# 说明：储能 SOC 上下限、功率上限均写成显式约束（而非变量上下界），
#       以便从 LP 对偶中提取这些约束的影子价格。
# ======================================================================
def build_model(use_milp: bool = True,
                eta: float = ETA, e_min: float = E_MIN, e_max: float = E_MAX,
                p_max: float = P_MAX, s0: float = E_INIT,
                price=None, load=None, pv=None) -> pulp.LpProblem:
    price = price if price is not None else PRICE
    load = load if load is not None else LOAD
    pv = pv if pv is not None else PV

    prob = pulp.LpProblem("Q1" + ("_MILP" if use_milp else "_LP"), pulp.LpMinimize)

    g = [pulp.LpVariable(f"g_{t}", lowBound=0) for t in range(PERIODS)]
    c = [pulp.LpVariable(f"c_{t}", lowBound=0) for t in range(PERIODS)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0) for t in range(PERIODS)]
    E = [pulp.LpVariable(f"E_{t}", lowBound=0) for t in range(PERIODS + 1)]
    z = None
    if use_milp:
        z = [pulp.LpVariable(f"z_{t}", cat="Binary") for t in range(PERIODS)]

    # 目标：全天购电费用最小
    prob += pulp.lpSum(price[t] * g[t] * DT for t in range(PERIODS))

    # 初末储电量约束
    prob += (E[0] == s0, "soc_init")
    prob += (E[PERIODS] == s0, "soc_term")

    for t in range(PERIODS):
        # 功率平衡：光伏 + 计划购电 + 储能放电 = 负载 + 储能充电
        prob += (pv[t] + g[t] + d[t] == load[t] + c[t], f"bal_{t}")
        # 储能电量递推
        prob += (E[t + 1] == E[t] + eta * c[t] * DT - d[t] * DT / eta, f"soc_{t}")
        # 显式约束：储电量上下限（用于提取影子价格）
        prob += (E[t + 1] <= e_max, f"e_ub_{t + 1}")
        prob += (E[t + 1] >= e_min, f"e_lb_{t + 1}")
        # 显式约束：充放电功率上限（用于提取影子价格）
        prob += (c[t] <= p_max, f"pch_{t}")
        prob += (d[t] <= p_max, f"pdis_{t}")
        # 充放电互斥（MILP）：z=1 充电、z=0 放电
        if use_milp:
            prob += (c[t] <= p_max * z[t], f"cm_{t}")
            prob += (d[t] <= p_max * (1 - z[t]), f"dm_{t}")

    return prob


# ======================================================================
# 求解并汇总指标
# ======================================================================
def solve_and_summarize(prob: pulp.LpProblem, use_milp: bool, price=None,
                        load=None, pv=None) -> dict:
    price = price if price is not None else PRICE
    load = load if load is not None else LOAD
    pv = pv if pv is not None else PV

    t0 = time.perf_counter()
    if use_milp:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
    elapsed = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"求解失败: status={status}")

    g = np.array([pulp.value(prob.variablesDict()[f"g_{t}"]) for t in range(PERIODS)])
    c = np.array([pulp.value(prob.variablesDict()[f"c_{t}"]) for t in range(PERIODS)])
    d = np.array([pulp.value(prob.variablesDict()[f"d_{t}"]) for t in range(PERIODS)])
    E = np.array([pulp.value(prob.variablesDict()[f"E_{t}"]) for t in range(PERIODS + 1)])

    cost = float(pulp.value(prob.objective))
    purchase_kwh = float(np.sum(g * DT))
    charge_kwh = float(np.sum(c * DT))
    discharge_kwh = float(np.sum(d * DT))
    both = int(np.sum((c > EPS) & (d > EPS)))  # 同时充放电时段数

    return {
        "prob": prob, "status": status, "time": elapsed,
        "g": g, "c": c, "d": d, "E": E,
        "cost": cost, "purchase_kwh": purchase_kwh,
        "charge_kwh": charge_kwh, "discharge_kwh": discharge_kwh,
        "both": both,
    }


# ======================================================================
# 影子价格：仅 LP 松弛模型有对偶（MILP 整数模型无对偶）
# ======================================================================
def shadow_prices(lp_res: dict) -> pd.DataFrame:
    prob = lp_res["prob"]
    rows = []
    for name, constr in prob.constraints.items():
        pi = constr.pi  # CBC 对 LP 返回对偶值
        rows.append({"约束": name, "类型": _constraint_type(name),
                     "影子价格": float(pi) if pi is not None else np.nan})
    df = pd.DataFrame(rows)
    return df


def _constraint_type(name: str) -> str:
    if name.startswith("bal_"):
        return "功率平衡(单位负载的边际成本)"
    if name.startswith("e_ub_"):
        return "储电量上限"
    if name.startswith("e_lb_"):
        return "储电量下限"
    if name.startswith("pch_"):
        return "充电功率上限"
    if name.startswith("pdis_"):
        return "放电功率上限"
    if name.startswith("soc_init"):
        return "初始储电量"
    if name.startswith("soc_term"):
        return "终端储电量"
    if name.startswith("soc_"):
        return "储能递推(内部)"
    return "其他"


# ======================================================================
# 灵敏度分析：关键参数 ± 扰动
# ======================================================================
def sensitivity(use_milp: bool = True) -> pd.DataFrame:
    rows = []
    base = solve_and_summarize(build_model(use_milp=use_milp), use_milp)["cost"]
    rows.append({"扰动参数": "基准", "扰动幅度": "—", "参数取值": "—",
                 "总购电费(元)": base, "变化率": "—"})

    cases = [
        ("充放电效率", "−5%（0.855）", ETA * 0.95, "eta"),
        ("充放电效率", "+5%（0.945）", ETA * 1.05, "eta"),
        ("SOC 上限", "−10%（9720）", E_MAX * 0.90, "e_max"),
        ("SOC 上限", "+10%（11880）", E_MAX * 1.10, "e_max"),
        ("SOC 下限", "−10%（1080）", E_MIN * 0.90, "e_min"),
        ("SOC 下限", "+10%（1320）", E_MIN * 1.10, "e_min"),
        ("最大充放电功率", "−20%（4000kW）", P_MAX * 0.80, "p_max"),
        ("最大充放电功率", "+20%（6000kW）", P_MAX * 1.20, "p_max"),
    ]
    for label, amp, val, kind in cases:
        kw = {"eta": dict(eta=val), "e_max": dict(e_max=val),
              "e_min": dict(e_min=val), "p_max": dict(p_max=val)}[kind]
        res = solve_and_summarize(build_model(use_milp=use_milp, **kw), use_milp)
        chg = (res["cost"] - base) / base * 100.0
        rows.append({"扰动参数": label, "扰动幅度": amp, "参数取值": val,
                     "总购电费(元)": res["cost"],
                     "变化率": f"{chg:+.2f}%"})
    return pd.DataFrame(rows)


# ======================================================================
# 输出：result1_分析.xlsx + 控制台三线表
# ======================================================================
def write_excel(lp_res: dict, milp_res: dict, sens: pd.DataFrame,
                shadow: pd.DataFrame) -> str:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(out_dir, "result1_分析.xlsx")
    wb = Workbook()

    # ---- Sheet1 计划购电量（严格对齐附件5模板：144数据行，无合计行） ----
    ws = wb.active
    ws.title = "计划购电量"
    time_labels = _time_labels()
    g_kwh = lp_res["g"] * DT
    ws.append(["时间段", "购电量"])
    for t in range(PERIODS):
        ws.append([time_labels[t], round(float(g_kwh[t]), 4)])

    # ---- Sheet2 充放电量（严格对齐附件5模板：时刻列 0:00/24:00 在第1、2行） ----
    ws = wb.create_sheet("充放电量")
    ws.append(["时间段", "充电量", "放电量", "时刻", "储电量"])
    blocks = [(0, 24, "0:00-4:00"), (24, 48, "4:00-8:00"), (48, 72, "8:00-12:00"),
              (72, 96, "12:00-16:00"), (96, 120, "16:00-20:00"), (120, 144, "20:00-24:00")]
    for bi, (s, e, nm) in enumerate(blocks):
        ws.append([nm,
                   round(float(lp_res["c"][s:e].sum() * DT), 4),
                   round(float(lp_res["d"][s:e].sum() * DT), 4),
                   "0:00" if bi == 0 else ("24:00" if bi == 1 else None),
                   round(lp_res["E"][0], 4) if bi == 0
                   else (round(lp_res["E"][-1], 4) if bi == 1 else None)])

    # ---- Sheet3 LP 与 MILP 对比 ----
    ws = wb.create_sheet("LP_MILP对比")
    ws.append(["指标", "LP 模型", "MILP 模型", "差异"])
    metrics = [
        ("总购电费（元）", lp_res["cost"], milp_res["cost"]),
        ("全天购电量（kWh）", lp_res["purchase_kwh"], milp_res["purchase_kwh"]),
        ("总充电量（kWh）", lp_res["charge_kwh"], milp_res["charge_kwh"]),
        ("总放电量（kWh）", lp_res["discharge_kwh"], milp_res["discharge_kwh"]),
        ("同时充放电时段数", float(lp_res["both"]), float(milp_res["both"])),
        ("求解时间（s）", lp_res["time"], milp_res["time"]),
    ]
    for name, lp_v, milp_v in metrics:
        diff = (milp_v - lp_v) if isinstance(lp_v, (int, float)) and isinstance(milp_v, (int, float)) else "—"
        ws.append([name, round(float(lp_v), 4) if isinstance(lp_v, (int, float)) else lp_v,
                   round(float(milp_v), 4) if isinstance(milp_v, (int, float)) else milp_v,
                   "—" if diff == "—" else (round(float(diff), 4))])

    # ---- Sheet4 影子价格 ----
    ws = wb.create_sheet("影子价格")
    ws.append(["约束", "类型", "影子价格(元/单位RHS)"])
    for _, r in shadow.iterrows():
        ws.append([r["约束"], r["类型"], None if np.isnan(r["影子价格"]) else round(r["影子价格"], 6)])

    # ---- Sheet5 灵敏度 ----
    ws = wb.create_sheet("灵敏度分析")
    ws.append(list(sens.columns))
    for _, r in sens.iterrows():
        val = round(float(r["参数取值"]), 4) if isinstance(r["参数取值"], (int, float)) else r["参数取值"]
        ws.append([r["扰动参数"], r["扰动幅度"], val,
                   round(r["总购电费(元)"], 2), r["变化率"]])

    wb.save(out)

    # ---- 提交版：result1_MILP_final.xlsx（严格匹配参考 result1 模板：时间段+购电量 144行） ----
    sub = os.path.join(out_dir, "result1_MILP_final.xlsx")
    wb2 = Workbook()
    ws2 = wb2.active
    ws2.title = "Sheet1"
    ws2.append(["时间段", "购电量"])
    for t in range(PERIODS):
        ws2.append([time_labels[t], round(float(g_kwh[t]), 4)])
    wb2.save(sub)

    return out


def _time_labels() -> list[str]:
    labels = []
    for t in range(PERIODS):
        sm = (t + 1) * 10
        em = (t + 2) * 10
        def fmt(m):
            if m >= 1440:
                m -= 1440
                return f"{m // 60}:{m % 60:02d}+1"
            return f"{m // 60}:{m % 60:02d}"
        labels.append(f"{fmt(sm)}-{fmt(em)}")
    return labels


def print_tables(lp_res, milp_res, shadow, sens):
    print("=" * 72)
    print("表1  LP 与 MILP 模型求解结果对比（问题1，单日代表日）")
    print("=" * 72)
    print(f"{'指标':<18}{'LP 模型':>16}{'MILP 模型':>16}{'差异':>12}")
    print("-" * 62)
    for name, lp_v, milp_v in [
        ("总购电费（元）", lp_res["cost"], milp_res["cost"]),
        ("全天购电量（kWh）", lp_res["purchase_kwh"], milp_res["purchase_kwh"]),
        ("总充电量（kWh）", lp_res["charge_kwh"], milp_res["charge_kwh"]),
        ("总放电量（kWh）", lp_res["discharge_kwh"], milp_res["discharge_kwh"]),
        ("同时充放电时段数", lp_res["both"], milp_res["both"]),
        ("求解时间（s）", lp_res["time"], milp_res["time"]),
    ]:
        print(f"{name:<18}{lp_v:>16.4f}{milp_v:>16.4f}{(milp_v - lp_v):>12.4f}")

    print()
    print("=" * 72)
    print("表2  关键约束影子价格（LP 对偶，元/单位RHS）")
    print("=" * 72)
    grp = shadow.groupby("类型")["影子价格"].agg(["count", "mean", "min", "max"])
    grp = grp.round(4)
    print(f"{'约束类型':<26}{'数量':>6}{'平均':>10}{'最小':>10}{'最大':>10}")
    print("-" * 62)
    for typ, row in grp.iterrows():
        print(f"{typ:<26}{int(row['count']):>6}{row['mean']:>10.4f}"
              f"{row['min']:>10.4f}{row['max']:>10.4f}")
    # 最贵约束
    non_internal = shadow[~shadow["类型"].str.contains("内部")].copy()
    non_internal = non_internal.replace({np.nan: 0.0})
    top = non_internal.reindex(non_internal["影子价格"].abs().sort_values(ascending=False).index).head(5)
    print("\n影子价格绝对值最大的5条约束（最'贵'约束）：")
    for _, r in top.iterrows():
        print(f"  {r['约束']:<12}{r['类型']:<24}{r['影子价格']:.6f}")

    print()
    print("=" * 72)
    print("表3  关键参数灵敏度分析（基准费用 = %.2f 元）" % sens.iloc[0]["总购电费(元)"])
    print("=" * 72)
    print(f"{'扰动参数':<12}{'扰动幅度':<20}{'参数取值':<14}{'总购电费(元)':>16}{'变化率':>10}")
    print("-" * 72)
    for _, r in sens.iterrows():
        val = round(float(r["参数取值"]), 4) if isinstance(r["参数取值"], (int, float)) else r["参数取值"]
        print(f"{r['扰动参数']:<12}{str(r['扰动幅度']):<20}{str(val):<14}"
              f"{r['总购电费(元)']:>16.2f}{str(r['变化率']):>10}")


# ======================================================================
# 主流程
# ======================================================================
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    a1 = load_annex1()
    PRICE = a1["price"].to_numpy(float)
    LOAD = a1["load_kw"].to_numpy(float)
    PV = a1["pv_pred_kw"].to_numpy(float)

    print("=" * 72)
    print("问题1 求解：MILP 模型 + LP/MILP 对比 + 影子价格 + 参数灵敏度")
    print("=" * 72)

    print("\n[1/4] 求解 LP 模型 ...")
    lp_res = solve_and_summarize(build_model(use_milp=False), use_milp=False)
    print(f"      LP 目标 = {lp_res['cost']:.4f} 元 (求解 {lp_res['time']:.3f}s)")

    print("[2/4] 求解 MILP 模型（0-1 互斥）...")
    milp_res = solve_and_summarize(build_model(use_milp=True), use_milp=True)
    print(f"      MILP 目标 = {milp_res['cost']:.4f} 元 (求解 {milp_res['time']:.3f}s)")

    print("[3/4] 提取影子价格（LP 对偶）...")
    shadow = shadow_prices(lp_res)

    print("[4/4] 参数扰动灵敏度 ...")
    sens = sensitivity(use_milp=True)

    print_tables(lp_res, milp_res, shadow, sens)

    out_file = write_excel(lp_res, milp_res, sens, shadow)
    sub_file = os.path.join(os.path.dirname(out_file), "result1_MILP_final.xlsx")
    print(f"\n已保存: {out_file}")
    print(f"提交版: {sub_file}")
    print("完成！")
