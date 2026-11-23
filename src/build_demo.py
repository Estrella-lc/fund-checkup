# -*- coding: utf-8 -*-
"""
生成作品集演示页的数据文件：docs/data.js

为什么单独一步：GitHub Pages 是静态托管，页面本身不能跑 Python。
所以把「代码算好的数据卡」预先固化成一份 JS 常量，页面只做展示——
既保证页面上每个数字都来自真实计算，又不需要后端。

用法：
  python src/build_demo.py        # 生成 docs/data.js
"""

import json
from pathlib import Path

from card import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data.js"

# 展示样本：覆盖最高分/最低分/明星产品/行业指数/债券指数/中等主动，形成对照
CODES = [
    ("011803", "分数最高：小规模、低回撤"),
    ("012523", "分数最低：高波动＋高费率"),
    ("005827", "名气与体检背离：明星产品"),
    ("040025", "中等主动：科技主题"),
    ("012414", "行业指数：单一赛道"),
    ("013169", "债券指数：低波动对照组"),
]

RISK_LINES = [15, 30]   # 稳健型 / 进取型承受线


def slim(card: dict) -> dict:
    """裁掉演示页用不到的长文本，控制体积。"""
    if not card:
        return {}
    keep = ["基金代码", "基金名称", "基金类型", "组别", "同类样本量", "成立时间",
            "统计区间", "画像标签", "体检总分", "七维度", "结论字段", "数据缺口"]
    out = {k: card.get(k) for k in keep}
    strat = out["七维度"].get("③策略一致性", {})
    if strat.get("投资策略"):
        strat["投资策略"] = strat["投资策略"][:80] + "…"
    return out


def main():
    bundle = []
    for code, tag in CODES:
        variants = {}
        for r in RISK_LINES:
            variants[str(r)] = slim(build(code, risk_line=r))
        if not variants[str(RISK_LINES[0])]:
            continue
        bundle.append({"tag": tag, "variants": variants})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// 由 src/build_demo.py 自动生成，请勿手改。\n"
        f"window.CARDS = {payload};\n"
        "window.RISK_LINES = [15, 30];\n",
        encoding="utf-8")
    print(f"✅ 已生成 {OUT}（{len(bundle)} 只基金 × {len(RISK_LINES)} 档承受线，"
          f"{len(payload)/1024:.0f} KB）")


if __name__ == "__main__":
    main()
