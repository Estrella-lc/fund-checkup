"""
step4_indicators.py —— 清洗 + 计算指标 + 合并冗余项 + 等权打分
================================================================
运行:  python src/step4_indicators.py

输入:
    data/raw/fund_basic.csv        成立时间/规模/经理
    data/raw/fund_manager_all.csv  经理累计从业时间（代理口径）
    data/raw/fund_rating_all.csv   三方评级
    data/raw/fee_all.csv           费率明细（长表）
    data/raw/nav/nav_<code>.csv    净值历史（累计净值）
输出:
    data/processed/master.csv      一只基金一行，含原始指标 + 分项得分 + 总分

口径说明（写报告时要一致）:
    - 收益/风险指标基于「累计净值」日收益序列，取最近 3 年
    - 年化收益 = (末值/首值)^(244/交易日数) - 1
    - 最大回撤 = min(净值/历史峰值 - 1) 的绝对值
    - 卡玛 = 年化收益 / 最大回撤（收益为负时无经济意义，置空）
    - 综合费率 = 管理费 + 托管费 + 销售服务费（%）
    - 费后年化 = 年化收益 - 综合费率 - 持有1年赎回费率（%）
    - 经理从业年限 = 累计从业天数 / 365（代理口径，非本基金任职年限）

打分口径（v3，2026-10-29 修订）:
    相关性检验发现「年化收益/超额收益/费后年化」三者 Spearman = 1.00，
    「最大回撤/波动率/下行波动率」两两 0.93–0.98，即两组内部在测同一件事。
    直接 11 项等权，等于把「涨得多」和「颠不颠」各算三遍。
    因此先合并成 7 个独立信号，再等权：
        综合收益 = mean(年化收益, 超额收益, 费后年化) 的分位分
        综合风险 = mean(最大回撤, 波动率, 下行波动率) 的分位分
        其余 5 项单列：卡玛、综合费率、赎回费、经理从业年限、评级均值
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
NAV_DIR = RAW / "nav"

START = pd.Timestamp("2023-10-29")   # 近 3 年
TRADING_DAYS = 244

# 最终参与等权的 7 个独立信号
FINAL_COLS = [
    "分_综合收益", "分_综合风险", "分_卡玛",
    "分_综合费率", "分_赎回费1年", "分_经理从业年限", "分_评级均值",
]


# ---------------------------------------------------------------- 净值指标
def nav_metrics(code: str) -> dict | None:
    path = NAV_DIR / f"nav_{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ["日期", "累计净值"]
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce", format="mixed")
    df = df.dropna().sort_values("日期")
    # 数据源里有把缺失值填成 0 的行（例：005627 在 2024-09-28 累计净值 = 0），
    # 一个 0 会让最大回撤算出 100%、波动率变成 NaN。净值不可能为 0，直接剔除。
    df = df[df["累计净值"] > 0]
    df = df[df["日期"] >= START]
    if len(df) < 400:          # 数据太短算不准，至少两年
        return None

    nav = df["累计净值"].astype(float).values
    ret = np.diff(nav) / nav[:-1]
    n = len(ret)

    ann_ret = (nav[-1] / nav[0]) ** (TRADING_DAYS / n) - 1
    vol = np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS)
    downside = ret[ret < 0]
    dvol = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 5 else np.nan

    peak = np.maximum.accumulate(nav)
    mdd = abs(np.min(nav / peak - 1))

    # 卡玛 = 年化收益 / 最大回撤。亏损基金的卡玛为负，排序上自然垫底，
    # 符合「性价比」的经济含义；不能因为负值就置空——那会把亏损基金整批踢出样本。
    calmar = round(ann_ret / mdd, 3) if mdd > 0 else np.nan

    return {
        "基金代码": code,
        "年化收益": round(ann_ret * 100, 2),
        "波动率": round(vol * 100, 2),
        "下行波动率": round(dvol * 100, 2) if not np.isnan(dvol) else np.nan,
        "最大回撤": round(mdd * 100, 2),
        "卡玛": calmar,
        "区间交易日": n,
    }


# ---------------------------------------------------------------- 费率
def parse_fee(code: str, rows: pd.DataFrame) -> dict:
    """从长表里取管理费/托管费/销售服务费，以及持有 365 天的赎回费率。"""
    out = {"基金代码": code, "管理费": np.nan, "托管费": np.nan,
           "销售服务费": np.nan, "赎回费1年": np.nan}
    other = rows[rows["费用类型"] == "其他费用"]
    for _, r in other.iterrows():
        name = str(r["条件或名称"])
        val = pd.to_numeric(r["费用"], errors="coerce")
        if "管理费" in name:
            out["管理费"] = val
        elif "托管费" in name:
            out["托管费"] = val
        elif "销售服务费" in name:
            out["销售服务费"] = val

    sell = rows[rows["费用类型"] == "卖出规则"]
    best = np.nan
    for _, r in sell.iterrows():
        cond = str(r["条件或名称"])
        val = pd.to_numeric(r["费用"], errors="coerce")
        nums = [float(x) for x in __import__("re").findall(r"[\d.]+", cond)]
        if len(nums) >= 2 and nums[0] <= 365 <= nums[1]:
            best = val
            break
    if np.isnan(best) and len(sell):
        best = pd.to_numeric(sell["费用"], errors="coerce").min()
    out["赎回费1年"] = best
    return out


def build_fee_table() -> pd.DataFrame:
    fee = pd.read_csv(RAW / "fee_all.csv", dtype={"基金代码": str})
    recs = [parse_fee(c, g) for c, g in fee.groupby("基金代码")]
    t = pd.DataFrame(recs)
    t["综合费率"] = t[["管理费", "托管费", "销售服务费"]].fillna(0).sum(axis=1)
    return t


# ---------------------------------------------------------------- 打分
def score(df: pd.DataFrame) -> pd.DataFrame:
    """组内分位打分（0–100，越大越好）→ 合并冗余项 → 7 项等权。"""
    df = df.copy()
    df["组别"] = np.where(df["基金类型"].astype(str).str.contains("指数"), "被动指数", "主动权益")

    good_high = {"年化收益": 1, "超额收益": 1, "卡玛": 1, "评级均值": 1,
                 "经理从业年限": 1, "费后年化": 1}
    good_low = {"最大回撤": -1, "波动率": -1, "下行波动率": -1,
                "综合费率": -1, "赎回费1年": -1}

    for col, direction in {**good_high, **good_low}.items():
        if col not in df.columns:
            continue
        pct = df.groupby("组别")[col].rank(pct=True)
        df[f"分_{col}"] = (pct if direction == 1 else 1 - pct) * 100

    # 合并冗余项：三项收益取均值当一项，三项风险取均值当一项
    df["分_综合收益"] = df[["分_年化收益", "分_超额收益", "分_费后年化"]].mean(axis=1)
    df["分_综合风险"] = df[["分_最大回撤", "分_波动率", "分_下行波动率"]].mean(axis=1)

    df["参与项数"] = df[FINAL_COLS].notna().sum(axis=1)
    df["总分"] = df[FINAL_COLS].mean(axis=1).round(1)
    return df


# ---------------------------------------------------------------- 检验
def worst_codes(frame: pd.DataFrame, col: str, n: int = 100) -> set:
    picks = []
    for _, g in frame.groupby("组别"):
        picks += g.nsmallest(n // 2, col)["基金代码"].tolist()
    return set(picks)


def missing_sensitivity(df: pd.DataFrame) -> None:
    """口径敏感性：指标不全的基金被剔除 vs 用组内中位数填补后全放进来，最差名单变多少。"""
    filled = df.copy()
    for c in FINAL_COLS:
        filled[c] = filled[c].fillna(filled.groupby("组别")[c].transform("median"))
    filled["总分_填补"] = filled[FINAL_COLS].mean(axis=1).round(1)

    full = df[df["参与项数"] == len(FINAL_COLS)]
    a, b = worst_codes(full, "总分"), worst_codes(filled, "总分_填补")
    overlap = len(a & b) / max(len(a), 1)
    print(f"\n【口径敏感性】严格 {len(full)} 只（7 项齐全）vs 放宽 {len(filled)} 只（中位数填补）")
    print(f"   最差 100 只名单重合度: {overlap*100:.0f}%")
    print(f"   {'结论稳健：缺失处理不改变更差名单' if overlap > 0.9 else '结论敏感：放宽样本后名单变化明显，需在报告中说明'}")


def weight_sensitivity(df: pd.DataFrame) -> None:
    """权重敏感性：换两套非等权权重，看最差 100 只是否换人。

    「最差名单」是这个项目的结论载体，它稳，结论才稳。
    """
    schemes = {
        "等权（基准）": {c: 1 for c in FINAL_COLS},
        "收益优先": {"分_综合收益": .30, "分_卡玛": .20, "分_综合风险": .15,
                 "分_综合费率": .15, "分_评级均值": .10, "分_经理从业年限": .05, "分_赎回费1年": .05},
        "风险优先": {"分_综合风险": .35, "分_卡玛": .25, "分_综合收益": .15,
                 "分_综合费率": .10, "分_评级均值": .05, "分_经理从业年限": .05, "分_赎回费1年": .05},
    }
    base = worst_codes(df, "总分")
    print("\n【权重敏感性】以「最差 100 只」为结论载体，换权重后换多少人")
    for name, w in schemes.items():
        parts = sum(df[c].fillna(df[c].median()) * wt for c, wt in w.items()) / sum(w.values())
        df[f"总分_{name}"] = parts.round(1)
        other = worst_codes(df, f"总分_{name}")
        print(f"   {name}: 重合 {len(base & other)}/100 只（{len(base & other)}%）")


if __name__ == "__main__":
    print("① 读取基本信息与评级")
    basic = pd.read_csv(RAW / "fund_basic.csv", dtype={"基金代码": str})
    basic["成立时间"] = pd.to_datetime(basic["成立时间"], errors="coerce", format="mixed")
    master_list = pd.read_csv(RAW / "fund_master.csv", dtype={"基金代码": str})
    # fund_basic 里雪球已返回一列「基金类型」，与 master 表同名列会撞车，显式改名再合并
    master_list = master_list.rename(columns={"基金类型": "基金类型_分类"})
    basic = basic.merge(master_list[["基金代码", "基金类型_分类"]], on="基金代码", how="left")
    basic["基金类型"] = basic["基金类型_分类"].fillna(basic.get("基金类型"))

    rating = pd.read_csv(RAW / "fund_rating_all.csv", dtype={"代码": str})
    rating = rating.rename(columns={"代码": "基金代码"})
    rcols = ["上海证券", "招商证券", "济安金信", "晨星评级"]
    rating["评级均值"] = rating[rcols].mean(axis=1)
    basic = basic.merge(rating[["基金代码", "评级均值"]], on="基金代码", how="left")

    print("② 计算净值指标")
    metrics = [m for c in basic["基金代码"] if (m := nav_metrics(c)) is not None]
    nav_df = pd.DataFrame(metrics)
    print(f"   可算指标的基金: {len(nav_df)} 只")

    print("③ 汇总费率")
    fee_df = build_fee_table()

    print("④ 合并 + 打分")
    df = basic.merge(nav_df, on="基金代码").merge(fee_df, on="基金代码", how="left")
    df["费后年化"] = df["年化收益"] - df["综合费率"] - df["赎回费1年"].fillna(0)

    # 经理指标：雪球基本信息接口不含「任职时间」，改用天天基金经理全量表的「累计从业时间」做代理。
    # 口径差异必须写进报告：从业年限 = 这个人当基金经理多久，不是「管这只基金多久」。
    mgr = pd.read_csv(RAW / "fund_manager_all.csv", dtype={"现任基金代码": str})
    mgr = (mgr.drop_duplicates("现任基金代码")
              .rename(columns={"现任基金代码": "基金代码", "累计从业时间": "从业天数"}))
    df = df.merge(mgr[["基金代码", "从业天数"]], on="基金代码", how="left")
    df["经理从业年限"] = (df["从业天数"].astype(float) / 365).round(1)
    print(f"   经理从业年限覆盖: {df['经理从业年限'].notna().sum()}/{len(df)}"
          f" ({round(df['经理从业年限'].notna().mean()*100,1)}%)")

    df = df.dropna(subset=["最大回撤"])
    df["组别"] = np.where(df["基金类型"].astype(str).str.contains("指数"), "被动指数", "主动权益")
    df["超额收益"] = df["年化收益"] - df.groupby("组别")["年化收益"].transform("median")

    df = score(df)

    # 决策（2026-10-29）：只保留 7 项全齐的样本，保证每只基金同一套标准
    missing_sensitivity(df)
    n_all = len(df)
    df = df[df["参与项数"] == len(FINAL_COLS)].copy()
    print(f"\n样本收敛：{n_all} → {len(df)} 只（7 项齐全），剔除 {n_all - len(df)} 只")
    weight_sensitivity(df)

    out = PROC / "master.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已输出 {len(df)} 只 → {out}")
    print(f"   主动权益 {(df['组别']=='主动权益').sum()} 只 | 被动指数 {(df['组别']=='被动指数').sum()} 只")
    print("\n总分最高 5 只:")
    print(df.nlargest(5, "总分")[["基金代码", "基金名称", "组别", "年化收益", "最大回撤", "综合费率", "总分"]].to_string(index=False))
    print("\n总分最低 5 只:")
    print(df.nsmallest(5, "总分")[["基金代码", "基金名称", "组别", "年化收益", "最大回撤", "综合费率", "总分"]].to_string(index=False))
