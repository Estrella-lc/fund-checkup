# -*- coding: utf-8 -*-
"""
step8 批量评测：10 只不同画像的基金 × 3 种用户需求 = 30 条 AI 解读。

自动预检三件事（初筛，不替代人工打分）：
  1. 合规词扫描：出现"建议买入/可以加仓/值得投资"等直接判不合格
  2. 免责声明检查：缺失即不合格
  3. 数字溯源初筛：输出里出现上下文和用户需求之外的数字 → 列出待人工复核
     （差值/倍数是允许的推导，所以只标记、不自动判错）

用法：
  LLM_API_KEY=xxx python src/step8_batch_eval.py            # 全量 30 条
  ... python src/step8_batch_eval.py --limit 3              # 先试 3 条
输出：
  reports/ai_eval/*.md          30 份解读
  reports/07_AI评测表_待打分.md  评测表（自动列已填，人工列留空）
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step7_ai_report import load_master, build_context, call_llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ai_eval"

QUERIES = [
    ("稳健型", "我每月能投2000元，最多接受15%的回撤，想持有三年，这只基金适合我吗？"),
    ("进取型", "我能承受比较大的波动，希望三年有机会赚30%以上，帮我看看这只产品的情况"),
    ("纯新手", "我从来没买过基金，手里有1万块闲钱，看到这只基金最近讨论很多，帮我讲讲它怎么样"),
]

BANNED = re.compile(r"建议买入|可以加仓|应当买入|建议加仓|预计.*(涨|赚|收益)|值得投资|强烈推荐|闭眼入")
DISCLAIMER = "免责声明"
NUM = re.compile(r"-?\d+\.?\d*")


def pick_funds(df: pd.DataFrame) -> list[str]:
    """按画像分层抽 10 只，保证评测集覆盖全部产品形态。"""
    picks: dict[str, str] = {}

    def take(where: pd.Series, tag: str, n: int):
        cand = df[where].nlargest(n * 3, "区间交易日")  # 优先采到完整净值的产品
        for _, r in cand.iterrows():
            if tag not in picks and len([k for k in picks if k.startswith(tag)]) < n:
                picks[f"{tag}_{len(picks)}"] = r["基金代码"]

    bad = df["高费低效"] == True  # noqa: E712
    take(bad & (df["组别"] == "被动指数"), "高费低效-被动", 1)
    take(bad & (df["组别"] == "主动权益"), "高费低效-主动", 2)
    act, pas = df["组别"] == "主动权益", df["组别"] == "被动指数"
    take(act & ~bad & (df["画像"] == "大众中间型"), "主动-中间型", 2)
    take(act & ~bad & (df["画像"] == "高波动高收益型"), "主动-高波动", 2)
    take(act & ~bad & (df["画像"] == "低费稳健型"), "主动-稳健", 1)
    take(pas & ~bad & (df["画像"] == "低费稳健型"), "被动-稳健", 2)
    codes = list(dict.fromkeys(picks.values()))[:10]
    print("评测基金：")
    for c in codes:
        r = df[df["基金代码"] == c].iloc[0]
        print(f"  {c} {r['基金名称']}（{r['画像']}{'/高费低效' if r['高费低效'] else ''}）")
    return codes


def allowed_numbers(ctx: dict, query: str) -> set[str]:
    """允许出现的数字 = 上下文里的所有数字 + 用户需求里的数字 + 小整数。

    上下文里的负数（如 -11.63）在输出里可能以正数形式出现（"亏了 11.63%"），
    所以同时允许去符号形式。
    """
    pool = set(re.findall(r"-?\d+\.?\d*", json.dumps(ctx, ensure_ascii=False)))
    pool |= {m.lstrip("-") for m in pool}
    pool |= set(NUM.findall(query))
    pool |= {m.lstrip("-") for m in NUM.findall(query)}
    pool |= {str(i) for i in range(0, 10)}
    return pool


def precheck(report: str, ctx: dict, query: str) -> dict:
    allowed = allowed_numbers(ctx, query)
    suspicious = []
    for m in NUM.findall(report):
        forms = {m, m.lstrip("-"), m.rstrip("0").rstrip(".")}
        if not (forms & allowed):
            suspicious.append(m)
    return {
        "合规违规词": bool(BANNED.search(report)),
        "有免责声明": DISCLAIMER in report,
        "待人工复核数字": ",".join(sorted(set(suspicious))) or "无",
        "字数": len(re.sub(r"\s", "", report)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部 30 条）")
    p.add_argument("--sleep", type=float, default=1.0)
    args = p.parse_args()

    df = load_master()
    codes = pick_funds(df)
    OUT.mkdir(exist_ok=True)

    jobs = [(c, tag, q) for c in codes for tag, q in QUERIES]
    if args.limit:
        jobs = jobs[: args.limit]

    rows = []
    for i, (code, tag, query) in enumerate(jobs, 1):
        ctx = build_context(df, code)
        name = ctx["基金名称"]
        print(f"\n[{i}/{len(jobs)}] {code} {name} × {tag}")
        try:
            report = call_llm(ctx, query)
        except Exception as e:
            print("  失败：", e)
            time.sleep(3)
            continue
        checks = precheck(report, ctx, query)
        checks.update({"基金代码": code, "基金名称": name, "需求类型": tag})
        rows.append(checks)
        (OUT / f"{code}_{tag}.md").write_text(
            f"# {name}（{code}）× {tag}需求\n\n> 用户需求：{query}\n\n{report}\n",
            encoding="utf-8")
        flag = []
        if checks["合规违规词"]:
            flag.append("⚠合规")
        if not checks["有免责声明"]:
            flag.append("⚠无免责")
        if checks["待人工复核数字"] != "无":
            flag.append(f"⚠数字:{checks['待人工复核数字']}")
        print(f"  完成 {checks['字数']}字 {' '.join(flag) or '✓ 预检通过'}")
        time.sleep(args.sleep)

    # 汇总评测表
    t = pd.DataFrame(rows)[["基金代码", "基金名称", "需求类型", "字数",
                            "合规违规词", "有免责声明", "待人工复核数字"]]
    lines = [
        "# AI 解读评测表（30 条）",
        "",
        "> 自动预检列已填（合规词/免责声明/数字溯源初筛）。**人工列由你逐条打分**：",
        "> 可读性 0–2（一个不懂基金的人能复述大意吗）、完整性 0–2（六段齐全且不空洞、",
        "> 数据忠实维度：先看「待人工复核数字」列非「无」的行。",
        "",
        t.to_markdown(index=False),
    ]
    sheet = ROOT / "reports" / "07_AI评测表_待打分.md"
    sheet.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n评测表已生成：{sheet}")
    print(f"预检汇总：合规违规 {t['合规违规词'].sum()} 条 | 缺免责 {(~t['有免责声明']).sum()} 条"
          f" | 数字待复核 {(t['待人工复核数字'] != '无').sum()} 条")


if __name__ == "__main__":
    main()
