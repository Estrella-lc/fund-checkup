"""
step1b_check.py —— 数据体检：在正式分析前先查三个问题
=====================================================
运行:  python src/step1b_check.py

这个脚本不产出分析结论，只回答三个"如果不查清楚，后面全白做"的问题：
    Q1  17875 只里，有多少其实是同一只基金的 A/C 类份额？
    Q2  基金简称里的后缀命名有多混乱？（决定能不能用规则去重）
    Q3  能不能拿到"成立日期"？（决定怎么剔除成立不满 1 年的基金）
"""

import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# ---------------------------------------------------------------- Q1 + Q2
df = pd.read_csv(RAW / "fund_list_focused.csv", dtype={"基金代码": str})
print(f"聚焦样本: {len(df)} 只\n")

# 去掉名称末尾的份额后缀：A / B / C / E / A1 / C类 / (QDII) 等
def strip_share_class(name: str) -> str:
    s = str(name)
    s = re.sub(r"[（(].*?[)）]", "", s)                  # 去掉括号内容
    s = re.sub(r"(联接)?[ABCE]$", "", s.strip())         # 去掉末尾份额字母
    s = re.sub(r"[ABCE]类$", "", s.strip())
    return s.strip()

df["去后缀名称"] = df["基金简称"].map(strip_share_class)
dup = df.groupby("去后缀名称").size().sort_values(ascending=False)

print("Q1 同一名称下有多少个代码（>1 说明存在多份额）")
print(f"  · 去重后的独立名称数: {dup.size}")
print(f"  · 有多份额的名称数:   {(dup > 1).sum()}")
print(f"  · 最多份额的那几个:")
for name, cnt in dup.head(5).items():
    codes = df.loc[df["去后缀名称"] == name, "基金代码"].tolist()
    print(f"      {name}: {cnt} 个代码 {codes[:4]}")

print("\nQ2 名称后缀的实际形态（抽样看规则能不能覆盖）")
suffixes = (
    df["基金简称"].astype(str)
      .str.extract(r"([ABCE]|A类|C类|[（(][^)）]*[)）])$")[0]
      .value_counts(dropna=False)
      .head(10)
)
print(suffixes.to_string())

# ---------------------------------------------------------------- Q3
print("\nQ3 尝试获取成立日期（接口可能改版，逐个试）")
import akshare as ak  # noqa: E402

candidates = {
    "fund_individual_basic_info_xq": lambda c: ak.fund_individual_basic_info_xq(symbol=c),
    "fund_individual_basic_info_hk_xq": lambda c: ak.fund_individual_basic_info_hk_xq(symbol=c),
}
for name, fn in candidates.items():
    try:
        res = fn("002001")
        print(f"  [成功] {name}")
        print("         返回字段:", list(res.columns)[:8])
        print(res.head(3).to_string())
        break
    except Exception as e:
        print(f"  [失败] {name}: {type(e).__name__}")
