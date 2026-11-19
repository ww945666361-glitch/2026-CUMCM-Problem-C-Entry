# -*- coding: utf-8 -*-
"""
preprocess.py —— 单位转换、预报对齐、数据校验模块（阶段0）

核心物理口径（严禁改动）：
    * 10分钟一个时段，dt = 1/6 h；电量(kWh) = 功率(kW) * dt
    * 电价单位 元/kWh，不是功率，绝对不能乘 dt
    * 储能递推：E_t = E_{t-1} + 0.9*c_t - d_t/0.9（在 model.py 实现）

时序口径与 data_loader 一致：slot t=0..143 对应 [t*10min, (t+1)*10min)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------- 全局物理常量 -----------------------------
DT = 1.0 / 6.0          # 时段长度（小时）
PERIODS = 144           # 每天时段数

ETA = 0.9               # 充放电效率
E_CAPACITY = 12000.0    # 最大容量 kWh
E_MIN = 1200.0          # 运行电量下限 kWh
E_MAX = 10800.0         # 运行电量上限 kWh
E_INIT = 6000.0         # 2025-01-01 0:00 初始电量 kWh
P_MAX_KW = 5000.0       # 最大充/放电功率 kW
STEP_MAX_KWH = P_MAX_KW * DT        # 单时段最大充/放电量 = 833.333... kWh

EMERGENCY_MULT = 5.0    # 紧急购电电价倍数
ADJUST_DOWN_MULT = 0.5  # 计划调减违约结算倍数
ADJUST_UP_MULT = 1.5    # 计划调增结算倍数

FORECAST_ISSUE_HOURS = (0, 6, 12, 18)  # 附件3预报发布时刻


# ----------------------------- 单位转换 -----------------------------
def kw_to_kwh(power_kw):
    """功率(kW) -> 单时段电量(kWh)，标量/数组/DataFrame 通用。"""
    return power_kw * DT


def annex1_to_arrays(a1: pd.DataFrame) -> dict:
    """
    将附件1单日数据拆成 LP 可直接使用的数组。

    返回 dict：
        price      : np.ndarray[144] 电价 元/kWh（不乘 dt）
        load_kwh   : np.ndarray[144] 负载电量 kWh = 负载功率 * dt
        pv_pred_kwh: np.ndarray[144] 预报光伏电量 kWh = 预测功率 * dt
    """
    _check_len(a1, "附件1")
    return {
        "price": a1["price"].to_numpy(dtype=float),
        "load_kwh": kw_to_kwh(a1["load_kw"].to_numpy(dtype=float)),
        "pv_pred_kwh": kw_to_kwh(a1["pv_pred_kw"].to_numpy(dtype=float)),
    }


# ----------------------------- 持续性预报（问题2口径） -----------------------------
def persistence_forecast(actual: pd.DataFrame, date) -> pd.Series:
    """
    0:00 持续性预报：把前一日的实际曲线（负载或光伏）当作当日预报。

    actual : 365×144 日矩阵（data_loader.load_annex2 的输出，单位 kW）
    date   : 目标日期（当日）；必须存在前一日数据（调度期 2025-02-01 起满足）
    返回   : pd.Series，index=slot 0..143，单位仍为 kW（未转电量）
    """
    date = pd.Timestamp(date).normalize()
    prev = date - pd.Timedelta(days=1)
    if prev not in actual.index:
        raise ValueError(f"[preprocess] {date.date()} 的前一日 {prev.date()} "
                         f"不在实际数据中，无法构造持续性预报")
    return actual.loc[prev].astype(float).copy()


# ----------------------------- 附件3预报对齐 -----------------------------
def forecast_hourly_to_slots(hourly: pd.Series, issue_hour: int) -> np.ndarray:
    """
    把某次发布的24小时整点光伏预报展开为144个10分钟时段的“时段平均功率”(kW)。

    口径（经附件2实际数据验证）：预报N小时 = (issue_hour + N):00 整点的瞬时功率。
    slot t 的功率 = 区间两端点瞬时功率线性插值后的均值 (v_s + v_e)/2，
    其中端点时刻由相邻整点线性插值得到（端点取均值等价于对线性功率曲线积分，
    保证功率×dt 后能量口径正确）。

    边界处理：
      * 发布整点 issue_hour:00 本身没有预报值，用 h=1 的值同值外推
        （物理上相邻一小时功率在该点连续；夜间该值本就为0）；
      * 超过当日24:00的预报步长（如6:00发布的h19~h24指向次日）直接丢弃；
      * slot 起点早于 issue_hour 的位置返回 NaN，由调用方沿用上一版预报
        （已过去/未更新时段不用新预报覆盖）。

    hourly     : pd.Series, index=1..24，附件3某一行（kW）
    issue_hour : 0/6/12/18
    返回       : np.ndarray[144]，未覆盖位置为 np.nan，单位 kW
    """
    if issue_hour not in FORECAST_ISSUE_HOURS:
        raise ValueError(f"[preprocess] 非法预报时刻: {issue_hour}")

    # 发布时刻起的“整点 -> 功率”锚点
    anchors: dict[int, float] = {issue_hour: float(hourly.loc[1])}
    for h in range(1, 25):
        hr = issue_hour + h
        if hr <= 24:  # 超出当日24:00的步长不用
            anchors[hr] = float(hourly.loc[h])
    hours = np.array(sorted(anchors), dtype=float)
    values = np.array([anchors[int(h)] for h in hours], dtype=float)

    slots = np.full(PERIODS, np.nan)
    for t in range(PERIODS):
        start_hour = t / 6.0          # slot起点（小时）
        if start_hour < issue_hour:
            continue                  # 发布时刻之前的slot不覆盖
        end_hour = start_hour + DT
        v_start = float(np.interp(start_hour, hours, values))
        v_end = float(np.interp(end_hour, hours, values))
        slots[t] = 0.5 * (v_start + v_end)
    return slots


def merge_forecast(base: np.ndarray, new: np.ndarray) -> np.ndarray:
    """
    用新预报更新当日slot序列：new中非NaN的位置覆盖base，NaN位置保留base。
    保证“未获得新预报的时段沿用上一版”的滚动口径。
    """
    _check_len(base, "base预报")
    _check_len(new, "new预报")
    return np.where(np.isnan(new), base, new)


def build_day_pv_forecast(annex3: pd.DataFrame, date,
                          issue_hours: tuple[int, ...]) -> np.ndarray:
    """
    按给定预报发布时刻序列，合成当日最终使用的光伏“时段平均功率”序列(kW)。
    按时间顺序依次 merge，后发布的预报覆盖未冻结时段。

    例：issue_hours=(0,) 仅0点预报；(0,6) 0点+6点；(0,6,12,18) 全部4次。
    """
    date = pd.Timestamp(date).normalize()
    if not issue_hours:
        raise ValueError("[preprocess] issue_hours 不能为空")
    merged = None
    for ih in issue_hours:
        row = annex3.loc[(date, ih)]
        slots = forecast_hourly_to_slots(row, ih)
        merged = slots if merged is None else merge_forecast(merged, slots)
    if np.isnan(merged).any():
        raise ValueError(f"[preprocess] {date.date()} 预报合成后仍存在未覆盖slot")
    return merged


# ----------------------------- 数据校验 -----------------------------
def _check_len(arr, name: str) -> None:
    if len(arr) != PERIODS:
        raise ValueError(f"[preprocess] {name} 长度应为{PERIODS}，实际{len(arr)}")


def validate_nonneg_finite(arr, name: str = "数据") -> None:
    """校验全日序列/矩阵：有限且非负。发现问题直接抛错，不做静默填充。"""
    a = np.asarray(arr, dtype=float)
    if not np.isfinite(a).all():
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"[preprocess] {name} 存在 {bad} 个 NaN/Inf")
    if (a < 0).any():
        bad = int((a < 0).sum())
        raise ValueError(f"[preprocess] {name} 存在 {bad} 个负值")


def validate_price(price, name: str = "电价") -> None:
    """电价校验：有限且严格为正。"""
    p = np.asarray(price, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"[preprocess] {name} 存在 NaN/Inf")
    if (p <= 0).any():
        raise ValueError(f"[preprocess] {name} 存在非正电价")


# ======================================================================
# 自测：python preprocess.py
# ======================================================================
if __name__ == "__main__":
    import data_loader as dl

    print("=" * 70)
    print("preprocess 自测")
    print("=" * 70)

    # ---------- 1. 单位转换抽查 ----------
    a1 = dl.load_annex1()
    arr = annex1_to_arrays(a1)
    assert abs(arr["load_kwh"][0] - a1["load_kw"].iloc[0] / 6.0) < 1e-9
    assert abs(arr["pv_pred_kwh"][10] - a1["pv_pred_kw"].iloc[10] / 6.0) < 1e-9
    assert abs(STEP_MAX_KWH - 5000.0 / 6.0) < 1e-9
    print(f"[1] kW->kWh 抽查通过: 首时段负载 "
          f"{a1['load_kw'].iloc[0]:.4f} kW -> {arr['load_kwh'][0]:.4f} kWh")
    print(f"    单时段充放电上限 = {STEP_MAX_KWH:.4f} kWh")
    validate_nonneg_finite(arr["load_kwh"], "附件1负载电量")
    validate_nonneg_finite(arr["pv_pred_kwh"], "附件1光伏电量")
    validate_price(arr["price"], "附件1电价")

    # ---------- 2. 持续性预报：2025-02-01 必须取到 2025-01-31 实际 ----------
    load_kw, pv_kw = dl.load_annex2()
    pf = persistence_forecast(load_kw, "2025-02-01")
    actual_0131 = load_kw.loc[pd.Timestamp("2025-01-31")]
    assert np.allclose(pf.to_numpy(), actual_0131.to_numpy())
    print(f"[2] 持续性预报通过: 2025-02-01 0点预报 == 2025-01-31 实际, "
          f"t=0 负载 {pf.iloc[0]:.2f} kW")

    # ---------- 3. 附件3展开：覆盖范围、昼夜形状 ----------
    a3 = dl.load_annex3()
    d = pd.Timestamp("2025-06-21")
    s0 = forecast_hourly_to_slots(a3.loc[(d, 0)], 0)
    s6 = forecast_hourly_to_slots(a3.loc[(d, 6)], 6)
    assert np.isnan(s6[:36]).all() and not np.isnan(s6[36:]).any(), "6点预报覆盖范围错误"
    print(f"[3] 6:00预报覆盖slot 36..143，slot0..35为NaN: 通过")
    print(f"    6/21 0点预报夜间t=0={s0[0]:.3f} kW, "
          f"正午t=72(12:00-12:10)={s0[72]:.1f} kW, t=143={s0[143]:.3f} kW")
    merged = build_day_pv_forecast(a3, d, (0, 6, 12, 18))
    assert not np.isnan(merged).any()
    print(f"    4次预报合成后无NaN: 通过, 全天平均 {merged.mean():.1f} kW")

    # ---------- 4. 预报对齐口径实证：线性区间均值 vs 小时块均值 ----------
    def expand_blockmean(row):
        """对照口径：fc[h] 当作 [h-1:00,h:00) 小时块均值，块内6个slot同值。"""
        out = np.zeros(144)
        for h in range(1, 25):
            out[(h - 1) * 6:h * 6] = float(row.loc[h])
        return out

    a_err, c_err, a_pred, c_pred = [], [], [], []
    for day in dl.SCHEDULE_DATES:
        row = a3.loc[(day, 0)]
        actual = pv_kw.loc[day].to_numpy()
        a_slots = forecast_hourly_to_slots(row, 0)
        c_slots = expand_blockmean(row)
        a_err.append(np.abs(a_slots - actual))
        c_err.append(np.abs(c_slots - actual))
        a_pred.append(a_slots)
        c_pred.append(c_slots)
    all_actual = pv_kw.loc[dl.SCHEDULE_DATES].to_numpy().ravel()
    print("[4] 0点光伏预报 vs 当日实际（全年334天）:")
    for name, err_lst, pred_lst in [
        ("A_线性区间均值(本模块)", a_err, a_pred),
        ("C_小时块均值(对照)", c_err, c_pred),
    ]:
        mae = np.concatenate(err_lst).mean()
        corr = np.corrcoef(np.concatenate(pred_lst).ravel(), all_actual)[0, 1]
        print(f"    {name:22s} MAE={mae:8.2f} kW  相关系数={corr:.4f}")

    # ---------- 5. 校验函数必须对脏数据抛错 ----------
    for bad, fn, tag in [
        (np.array([1.0, np.nan]), validate_nonneg_finite, "NaN"),
        (np.array([1.0, -2.0]), validate_nonneg_finite, "负值"),
        (np.array([0.5, 0.0]), validate_price, "零电价"),
    ]:
        try:
            fn(bad, tag)
        except ValueError:
            pass
        else:
            raise AssertionError(f"校验未拦截 {tag}")
    print("[5] 脏数据(NaN/负值/零电价)拦截: 通过")

    print("preprocess 自测全部通过 ✓")
