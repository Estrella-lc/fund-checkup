"""
step3_collect_nav_fee.py —— 采集净值历史与费率明细（只采成立满 1 年的样本）
==========================================================================
先小样本验证:  python src/step3_collect_nav_fee.py --limit 5
全量(约 3-4 小时, 建议睡前挂):  python src/step3_collect_nav_fee.py --limit 0

产出:
    data/raw/nav_<代码>.csv     每只基金一个净值文件（累计净值口径）
    data/raw/fee_all.csv        所有基金的费率明细（追加式）
说明:
    - 净值用「累计净值走势」: 含历史分红的累积口径。未做分红再投资的精确复权,
      与支付宝等平台展示的收益率会有小幅差异 —— 这个口径差异要写进报告局限。
    - 费率来自 fund_individual_detail_info_xq, 长表转宽表后追加存储。
"""

import argparse
import json
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
NAV_DIR = RAW / "nav"
NAV_DIR.mkdir(parents=True, exist_ok=True)
FEE_PATH = RAW / "fee_all.csv"

CUTOFF = pd.Timestamp("2025-10-29")  # 成立满 1 年


def load_targets() -> pd.DataFrame:
    basic = pd.read_csv(RAW / "fund_basic.csv", dtype={"基金代码": str})
    basic["成立时间"] = pd.to_datetime(basic["成立时间"], errors="coerce", format="mixed")
    ok = basic[basic["成立时间"] <= CUTOFF].copy()
    print(f"目标样本: {len(ok)} 只（成立满 1 年）")
    return ok


def fetch_nav(code: str) -> pd.DataFrame:
    return ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")


def fetch_fee(code: str) -> pd.DataFrame | None:
    try:
        fee = ak.fund_individual_detail_info_xq(symbol=code)
        fee.insert(0, "基金代码", code)
        return fee
    except Exception:
        return None


def done_navs() -> set:
    return {p.stem.split("_")[-1] for p in NAV_DIR.glob("nav_*.csv")}


def run(targets: pd.DataFrame, limit: int, sleep: float) -> None:
    done = done_navs()
    todo = targets[~targets["基金代码"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"本次待采集 {len(todo)} 只（净值文件已有 {len(done)} 个）")

    fee_frames, fee_done = [], set()
    if FEE_PATH.exists():
        old_fee = pd.read_csv(FEE_PATH, dtype={"基金代码": str})
        fee_done = set(old_fee["基金代码"].unique())
        print(f"费率表已有 {old_fee['基金代码'].nunique()} 只")

    t0, failed = time.time(), []
    for i, code in enumerate(todo["基金代码"], 1):
        try:
            nav = fetch_nav(code)
            nav.to_csv(NAV_DIR / f"nav_{code}.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            failed.append((code, "nav", type(e).__name__))
        if code not in fee_done:
            fee = fetch_fee(code)
            if fee is not None:
                fee_frames.append(fee)
            else:
                failed.append((code, "fee", "-"))
        time.sleep(sleep)
        if i % 100 == 0:
            used = time.time() - t0
            eta = used / i * (len(todo) - i) / 60
            print(f"  {i}/{len(todo)}  已用 {used/60:.0f} 分钟  预计剩余 {eta:.0f} 分钟")
        if fee_frames and i % 200 == 0:
            pd.concat(fee_frames, ignore_index=True).to_csv(
                FEE_PATH, mode="a", header=not FEE_PATH.exists(), index=False, encoding="utf-8-sig")
            fee_frames = []

    if fee_frames:
        pd.concat(fee_frames, ignore_index=True).to_csv(
            FEE_PATH, mode="a", header=not FEE_PATH.exists(), index=False, encoding="utf-8-sig")
    if failed:
        pd.DataFrame(failed, columns=["基金代码", "项目", "异常"]).to_csv(
            RAW / "failed_step3.log", index=False)
    print(f"完成。净值文件 {len(done_navs())} 个, 失败 {len(failed)} 条")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args()
    run(load_targets(), args.limit, args.sleep)
