# -*- coding: utf-8 -*-
"""
data_loader.py —— 附件1~4 原始数据读取模块（阶段0）

职责边界：
    只负责读取与结构解析，返回的功率数据保持原始单位 kW、电价保持 元/kWh，
    不做 kW->kWh 转换、不做任何缺失值填充。单位转换与数据校验在 preprocess.py。

统一时序约定（全项目唯一口径）：
    slot t = 0..143 表示物理区间 [t*10min, (t+1)*10min)，
    t=0   -> 0:00-0:10，t=143 -> 23:50-24:00。
    附件中的列/行标签是“区间结束时刻”（0:10, 0:20, ..., 23:50, '0:00+1'），
    其出现顺序与 slot 0..143 一一对应，本模块按位置解析、不依赖标签字符串。

数据契约：
    日矩阵（附件2/4）: DataFrame, index=DatetimeIndex(365天, 归一化到0点),
                       columns=RangeIndex(0..143), 值=float
    附件3           : DataFrame, MultiIndex (date, issue_hour),
                       columns=Int64Index 1..24（预报N小时）, 值=float(kW整点值)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 附件目录：优先同级“附件”文件夹，兼容旧布局“附件1”文件夹（两者取存在者）
_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = _BASE_DIR / "附件"
if not DATA_DIR.exists():
    DATA_DIR = _BASE_DIR / "附件1"

PERIODS = 144  # 每天10分钟时段数

# 原始中文表头 -> 内部字段名（用映射隔离原始表头，后续模块只认内部名）
_ANNEX1_RENAME = {
    "时间": "time_label",
    "电价": "price",
    "小区负载": "load_kw",
    "光伏发电预测功率": "pv_pred_kw",
}
_ANNEX3_RENAME = {"日期": "date", "预报时刻": "issue_time"}


def _assert(cond: bool, msg: str) -> None:
    """结构断言：数据文件与预期不符时直接报错，不做静默兜底。"""
    if not cond:
        raise ValueError(f"[data_loader] 数据结构异常: {msg}")


def load_annex1(path: Path | str | None = None) -> pd.DataFrame:
    """
    读取附件1（单日代表日：电价 + 负载 + 光伏预测功率）。

    返回 DataFrame，144行，index=slot(0..143)，列：
        time_label : 原始时间标签（datetime.time 或 '0:00+1' 字符串），仅留档/调试
        price      : 电价 元/kWh
        load_kw    : 小区负载功率 kW
        pv_pred_kw : 光伏发电预测功率 kW
    """
    path = Path(path) if path is not None else DATA_DIR / "附件1.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns=_ANNEX1_RENAME)

    _assert(len(df) == PERIODS, f"附件1应为{PERIODS}行，实际{len(df)}行")
    for col in ("price", "load_kw", "pv_pred_kw"):
        _assert(col in df.columns, f"附件1缺少列: {col}")

    df = df[["time_label", "price", "load_kw", "pv_pred_kw"]].copy()
    df.insert(0, "slot", np.arange(PERIODS))
    df[["price", "load_kw", "pv_pred_kw"]] = df[
        ["price", "load_kw", "pv_pred_kw"]
    ].astype(float)
    return df.set_index("slot")


def _load_daily_matrix(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """
    读取“日期行 × 144时段列”布局的矩阵（附件2的两个sheet、附件4）。

    返回 DataFrame，365行×144列：
        index   = DatetimeIndex（2025-01-01 .. 2025-12-31）
        columns = RangeIndex 0..143（slot）
        值      = float（原始单位，kW 或 元/kWh，由调用方解释）
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    _assert(df.shape[1] == PERIODS + 1,
            f"{path.name}[{sheet_name}] 应为{PERIODS + 1}列，实际{df.shape[1]}列")
    _assert(len(df) == 365, f"{path.name}[{sheet_name}] 应为365天，实际{len(df)}行")

    dates = pd.to_datetime(df.iloc[:, 0]).dt.normalize()
    _assert(dates.is_unique, f"{path.name}[{sheet_name}] 日期列存在重复")
    expected = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    _assert(dates.reset_index(drop=True).equals(pd.Series(expected)),
            f"{path.name}[{sheet_name}] 日期不是连续的2025全年")

    mat = df.iloc[:, 1:].astype(float)
    mat.index = pd.DatetimeIndex(dates)
    mat.columns = pd.RangeIndex(PERIODS)
    _assert(not mat.isna().any().any(), f"{path.name}[{sheet_name}] 存在空值")
    return mat.sort_index()


def load_annex2(path: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取附件2：全年实际小区负载、全年实际光伏发电功率。

    返回 (load_kw, pv_kw)，均为 365×144 的 DataFrame，单位 kW。
    """
    path = Path(path) if path is not None else DATA_DIR / "附件2.xlsx"
    load_kw = _load_daily_matrix(path, sheet_name="小区负载")
    pv_kw = _load_daily_matrix(path, sheet_name="光伏发电实际功率")
    return load_kw, pv_kw


def load_annex4(path: Path | str | None = None) -> pd.DataFrame:
    """
    读取附件4：全年波动电价。

    返回 365×144 的 DataFrame，单位 元/kWh（电价不是功率，禁止乘 dt）。
    """
    path = Path(path) if path is not None else DATA_DIR / "附件4.xlsx"
    return _load_daily_matrix(path, sheet_name=0)


def load_annex3(path: Path | str | None = None) -> pd.DataFrame:
    """
    读取附件3：0:00/6:00/12:00/18:00 发布的未来24小时整点光伏功率预报。

    原始布局：每天4行，日期列在4行中只填第1行（需向下填充）。
    “预报N小时”解读为 发布时刻后第 N 个整点（T+N:00）的瞬时功率 kW
    （该口径已用附件2实际数据验证，见 preprocess 自测）。

    返回 DataFrame：
        index   = MultiIndex (date[Timestamp], issue_hour[0/6/12/18])
        columns = Int64Index 1..24（预报步长，小时）
        值      = float，预报功率 kW
    """
    path = Path(path) if path is not None else DATA_DIR / "附件3.xlsx"
    df = pd.read_excel(path).rename(columns=_ANNEX3_RENAME)

    _assert(len(df) == 365 * 4, f"附件3应为{365 * 4}行，实际{len(df)}行")
    _assert(df.shape[1] == 26, f"附件3应为26列，实际{df.shape[1]}列")

    # 日期列向下填充（同一天4行只在首行写日期）
    df["date"] = pd.to_datetime(df["date"].ffill()).dt.normalize()
    # 预报时刻 '0:00'/'6:00' ... -> 小时整数
    df["issue_hour"] = df["issue_time"].astype(str).str.split(":").str[0].astype(int)

    _assert(set(df["issue_hour"].unique()) == {0, 6, 12, 18},
            "附件3预报时刻不是 {0,6,12,18}")
    _assert(df.groupby("date").size().eq(4).all(), "附件3存在每天不足4次预报")
    _assert(df["date"].nunique() == 365, "附件3日期不是365天")

    horizons = list(range(1, 25))
    val_cols = [f"预报{h}小时" for h in horizons]
    out = df[["date", "issue_hour"] + val_cols].copy()
    out[val_cols] = out[val_cols].astype(float)
    _assert(not out[val_cols].isna().any().any(), "附件3预报值存在空值")

    out = out.set_index(["date", "issue_hour"]).sort_index()
    out.columns = pd.Index(horizons, dtype="int64")
    return out


# ---- 日期集合常量（全项目共用，避免各处手写日期范围） ----
YEAR_DATES = pd.date_range("2025-01-01", "2025-12-31", freq="D")
SCHEDULE_DATES = pd.date_range("2025-02-01", "2025-12-31", freq="D")  # 334天


# ======================================================================
# 自测：python data_loader.py
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("data_loader 自测")
    print("=" * 70)

    a1 = load_annex1()
    print(f"[附件1] shape={a1.shape}, 列={list(a1.columns)}")
    print(f"        首行标签={a1['time_label'].iloc[0]!r}, "
          f"末行标签={a1['time_label'].iloc[-1]!r}")
    print(f"        电价范围 [{a1['price'].min():.4f}, {a1['price'].max():.4f}], "
          f"负载范围 [{a1['load_kw'].min():.1f}, {a1['load_kw'].max():.1f}] kW")

    load_kw, pv_kw = load_annex2()
    print(f"[附件2] load {load_kw.shape}, pv {pv_kw.shape}, "
          f"日期 {load_kw.index[0].date()}..{load_kw.index[-1].date()}, "
          f"slot列 {load_kw.columns[0]}..{load_kw.columns[-1]}")

    a3 = load_annex3()
    print(f"[附件3] shape={a3.shape}, index层级={a3.index.names}, "
          f"步长列={a3.columns[0]}..{a3.columns[-1]}")
    row0 = a3.loc[(pd.Timestamp("2025-01-01"), 0)]
    print(f"        2025-01-01 0:00预报: h1={row0[1]:.2f}(应≈0夜), "
          f"h7={row0[7]:.2f}(日出≈0), h12={row0[12]:.1f}(中午高), "
          f"h24={row0[24]:.2f}(次日0点≈0)")

    a4 = load_annex4()
    print(f"[附件4] shape={a4.shape}, "
          f"电价范围 [{a4.min().min():.4f}, {a4.max().max():.4f}] 元/kWh")
    print(f"[常量] 全年{len(YEAR_DATES)}天, 调度期{len(SCHEDULE_DATES)}天 "
          f"({SCHEDULE_DATES[0].date()}..{SCHEDULE_DATES[-1].date()})")
    print("data_loader 自测通过 ✓")
