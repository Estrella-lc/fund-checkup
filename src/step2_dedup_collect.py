"""
step2_dedup_collect.py —— 去重 + 采集基金基本信息
================================================
运行（先小样本验证）:
    python src/step2_dedup_collect.py --limit 20
确认无误后跑全量（约 1-2 小时，建议挂着）:
    python src/step2_dedup_collect.py --limit 0

这一步做两件事:
    1. 把 17875 个代码收敛成「一只基金一行」的代表名单（优先保留 A 类份额）
    2. 逐只调用雪球接口拿成立时间/规模/经理/评级，做断点续传
"""

import argparse
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
RANDOM_SAMPLE = True


# ------------------------------------------------------------------ 去重
def strip_share_class(name: str) -> str:
    """把「华夏红利价值混合A」→「华夏红利价值混合」，用于识别同一只基金的不同份额。"""
    s = str(name)
    s = re.sub(r"[（(].*?[)）]", "", s)          # 去掉 (后端) (LOF) (人民币) 这类括号标记
    s = re.sub(r"(联接)?[ABCDEHIORY]$", "", s.strip())  # 去掉末尾份额字母（含 Y 类养老金份额）
    s = re.sub(r"[ABCDEHIORY]类$", "", s.strip())
    return s.strip()


def share_class(name: str) -> str:
    """提取份额类别，留作 C 类专题分析用。"""
    s = str(name)
    m = re.search(r"[ABCE]$", s.strip())
    if m:
        return m.group(0)
    if "后端" in s:
        return "后端"
    return "无后缀"


def build_master() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fund_list_focused.csv", dtype={"基金代码": str})
    df["去后缀名称"] = df["基金简称"].map(strip_share_class)
    df["份额类别"] = df["基金简称"].map(share_class)

    # 场内基金（ETF/LOF）代码以 15/16/50/51/52/56/58 开头。
    # 排除理由：场内靠券商交易，成本是佣金而非申购费，与场外 A/C 份额的费率口径不可比；
    # 且雪球接口不覆盖。这个取舍要写进报告的方法说明。
    onsite_mask = df["基金代码"].str.match(r"^(15|16|50|51|52|56|58)")
    onsite = df[onsite_mask].copy()
    onsite.to_csv(RAW / "fund_onsite_excluded.csv", index=False, encoding="utf-8-sig")
    print(f"排除场内基金 {len(onsite)} 只（已单独存盘，供说明用）")
    df = df[~onsite_mask].copy()

    # 每个名称下挑一个代表：A 类优先 → 无后缀 → C 类 → 代码最小
    priority = {"A": 0, "无后缀": 1, "C": 2, "E": 3, "B": 4, "后端": 5}
    df["优先级"] = df["份额类别"].map(priority).fillna(9)
    df = df.sort_values(["去后缀名称", "优先级", "基金代码"])
    master = df.drop_duplicates(subset="去后缀名称", keep="first").copy()

    # 同时把全部份额留一份，供专题分析
    df.drop(columns=["优先级"]).to_csv(RAW / "fund_all_shares.csv", index=False, encoding="utf-8-sig")
    master.drop(columns=["优先级"]).to_csv(RAW / "fund_master.csv", index=False, encoding="utf-8-sig")

    print(f"代码数 {len(df)} → 独立基金 {len(master)} 只")
    print("代表份额构成:", master["份额类别"].value_counts().to_dict())
    return master


# ------------------------------------------------------------------ 采集
def parse_one(code: str) -> dict:
    """单只基金基本信息：接口返回 item/value 长表，转成一行宽表。"""
    raw = ak.fund_individual_basic_info_xq(symbol=code)
    kv = dict(zip(raw["item"], raw["value"]))
    kv["基金代码"] = code

    # "75.79亿" → 75.79（亿元），后面算规模因子要用数字
    size = str(kv.get("最新规模", "")).replace("亿", "")
    try:
        kv["规模_亿元"] = float(size)
    except ValueError:
        kv["规模_亿元"] = None

    # 成立时间统一成日期，后面算"成立年限"
    try:
        kv["成立时间"] = pd.to_datetime(kv.get("成立时间"), errors="coerce")
    except Exception:
        kv["成立时间"] = pd.NaT
    return kv


def collect(master: pd.DataFrame, limit: int, sleep: float = 0.4) -> None:
    out_path = RAW / "fund_basic.csv"
    # 断点续传：已有的不再拉
    if out_path.exists():
        done = pd.read_csv(out_path, dtype={"基金代码": str})["基金代码"].tolist()
        print(f"已完成 {len(done)} 只，继续拉剩下的")
    else:
        done = []

    todo = master[~master["基金代码"].isin(done)]
    if RANDOM_SAMPLE:
        # 随机抽样：避免按代码顺序取导致"同一家公司/同一类产品扎堆"，缺失率被放大
        todo = todo.sample(frac=1.0, random_state=42)
    if limit:
        todo = todo.head(limit)
    print(f"本次待采集 {len(todo)} 只")

    rows, failed, t0 = [], [], time.time()
    for i, code in enumerate(todo["基金代码"], 1):
        try:
            rows.append(parse_one(code))
        except Exception as e:
            failed.append((code, type(e).__name__))
        time.sleep(sleep)
        if i % 50 == 0:
            used = time.time() - t0
            print(f"  {i}/{len(todo)}  已用 {used/60:.1f} 分钟  预计剩余 {used/i*(len(todo)-i)/60:.1f} 分钟")

    if rows:
        new_df = pd.DataFrame(rows)
        old = pd.read_csv(out_path, dtype={"基金代码": str}) if out_path.exists() else pd.DataFrame()
        pd.concat([old, new_df], ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"写入 {len(new_df)} 条 → {out_path.name}")
    if failed:
        pd.DataFrame(failed, columns=["基金代码", "异常"]).to_csv(RAW / "failed.log", index=False)
        print(f"失败 {len(failed)} 只，已记入 failed.log")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20, help="0 表示全量")
    p.add_argument("--sleep", type=float, default=0.4, help="每次请求间隔秒数")
    p.add_argument("--ordered", action="store_true", help="按代码顺序取（默认随机抽样）")
    args = p.parse_args()

    if args.ordered:
        RANDOM_SAMPLE = False
    master = build_master()
    collect(master, limit=args.limit, sleep=args.sleep)
