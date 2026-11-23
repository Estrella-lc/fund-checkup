"""
step1_fetch.py —— 阶段0/第1周：基金基础数据采集
=============================================
运行方式:
    python src/step1_fetch.py

你要做的事（TODO 标出）:
    1. 逐段读懂代码，跑一遍；
    2. 扩展 fetch_all_funds()：把「费率」「经理」「评级」三张明细表拉下来；
    3. 处理限流：每批之间 sleep，失败重试，raw 数据落盘后不重复拉。

AkShare 接口速查（写代码前先读文档参数说明）:
    - fund_name_em()            全量基金列表（代码/简称/类型）
    - fund_purchase_em()        申购赎回费率表
    - fund_open_fund_info_em()  单只基金净值（indicator 参数选曲线类型）
    - fund_manager_em()         基金经理
    - fund_rating_all()         三方评级
"""

import time
from pathlib import Path

import akshare as ak
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 本项目聚焦的三类基金（组合型/货币型/ QDII 暂不纳入，原因写进指标草案）
FOCUS_TYPES = ["混合型", "股票型", "指数型"]


def fetch_fund_list() -> pd.DataFrame:
    """全量基金列表，按类型过滤后落盘。"""
    df = ak.fund_name_em()
    df.to_csv(RAW_DIR / "fund_list_all.csv", index=False, encoding="utf-8-sig")

    mask = df["基金类型"].str.contains("|".join(FOCUS_TYPES), na=False)
    focused = df[mask].copy()
    focused.to_csv(RAW_DIR / "fund_list_focused.csv", index=False, encoding="utf-8-sig")
    print(f"全量 {len(df)} 只 → 聚焦 {len(focused)} 只（{', '.join(FOCUS_TYPES)}）")
    return focused


def fetch_one_nav(symbol: str) -> pd.DataFrame:
    """单只基金累计净值（示例：华夏回报 002001）。"""
    nav = ak.fund_open_fund_info_em(symbol=symbol, indicator="累计净值走势")
    return nav


if __name__ == "__main__":
    # TODO 1: 跑通列表拉取，观察「基金类型」字段的取值分布
    focused = fetch_fund_list()
    print(focused["基金类型"].value_counts())

    # TODO 2: 挑 3 只不同类型的基金，拉净值，看数据长什么样
    for code in ["002001", "110011", "510300"]:
        nav = fetch_one_nav(code)
        print(code, "→", len(nav), "条净值，最新日期:", nav["净值日期"].iloc[-1])

    # TODO 3（第1周 D3-D4）: 批量拉取明细表
    #   提示：两万多只逐只拉取要几个小时，务必：
    #   - 先过滤（剔除成立<1年，用 fund_purchase_em 里的字段判断）
    #   - 每 100 只 sleep 几秒，raw 落盘后跳过已完成的
    #   - 写进 try/except，失败记入 failed.log，别让一只基金卡死全程
    pass
