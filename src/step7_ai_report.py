# -*- coding: utf-8 -*-
"""
step7 AI 解读层：把体检指标翻译成用户能看懂的话。

分工原则（也是面试答案）：
  · 代码负责精确计算——筛选、打分、分位排名（本文件的数据层函数）
  · LLM 负责语言生成——把算好的数字翻译成人话（call_llm）
  · 模型禁止使用上下文之外的数字（见 SYSTEM_PROMPT 规则 1）

用法：
  # 单只基金体检报告（需要先设置环境变量 LLM_API_KEY / LLM_API_BASE / LLM_MODEL）
  python src/step7_ai_report.py --code 012414

  # 带用户需求画像的个性化解读
  python src/step7_ai_report.py --code 012414 --query "每月能投2000元，最多接受15%回撤，想持有三年"

环境变量（OpenAI 兼容格式，智谱/DeepSeek/通义等均可）：
  LLM_API_KEY   必填
  LLM_API_BASE  如 https://open.bigmodel.cn/api/paas/v4
  LLM_MODEL     如 glm-4-flash
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "ai_reports"

# ---------------------------------------------------------------- 数据层

def load_master() -> pd.DataFrame:
    df = pd.read_csv(PROC / "clustered.csv", dtype={"基金代码": str})
    # 优先读 step5 落库的列（单一真相源）；老数据无该列时才回退复算
    if "高费低效" in df.columns:
        return df
    STD = {"主动权益": 1.40, "被动指数": 0.60}
    bad = pd.Series(False, index=df.index)
    for g, sub in df.groupby("组别"):
        thr = STD[g]
        bad.loc[sub.index] = (sub["综合费率"] >= thr) & (
            sub["卡玛"] <= sub["卡玛"].quantile(.25))
    df["高费低效"] = bad
    return df


def pct_rank_in_group(df: pd.DataFrame, code: str, col: str, higher_better: bool) -> float | None:
    """该基金某指标在同类中的分位（0–100，越大越好）。

    为什么给分位而不是原始值：模型直接说"年化 4.4%"用户没法判断好坏，
    说"高于同类 62% 的产品"才是可决策的信息。分位由代码算，模型只负责引用。
    """
    row = df[df["基金代码"] == code]
    if row.empty or col not in df.columns:
        return None
    g = df[df["组别"] == row.iloc[0]["组别"]][col].dropna()
    if g.empty or pd.isna(row.iloc[0][col]):
        return None
    r = (g < row.iloc[0][col]).mean() * 100
    return round(r if higher_better else 100 - r, 1)


def build_context(df: pd.DataFrame, code: str) -> dict | None:
    """组装喂给模型的全部事实。**模型只许用这个 dict 里的数字。**"""
    row = df[df["基金代码"] == code]
    if row.empty:
        print(f"找不到基金 {code}")
        return None
    r = row.iloc[0]

    def f(col, nd=2):
        v = r.get(col)
        return None if pd.isna(v) else round(float(v), nd)

    metrics = {}
    for col, higher, key in [
        ("年化收益", True, "年化收益_pct"),
        ("最大回撤", False, "最大回撤_pct"),   # 回撤越小越好
        ("综合费率", False, "综合费率_pct"),   # 费率越低越好
        ("费后年化", True, "费后年化_pct"),
        ("卡玛", True, "卡玛_收益回撤比"),
        ("经理从业年限", True, "经理从业年限_年"),
    ]:
        metrics[key] = {
            "本基金": f(col),
            "同类分位": pct_rank_in_group(df, code, col, higher),
            "分位说明": "同类分位=优于多少比例的同类基金。0.1 表示只优于 0.1%（几乎垫底），50 表示中游，90 表示前 10%。",
        }

    return {
        "基金代码": code,
        "基金名称": str(r.get("基金名称", "")),
        "基金类型": str(r.get("基金类型", r.get("组别", ""))),
        "基金公司": str(r.get("基金公司", "未知")),
        "统计区间": "2023-10 至 2026-10（近三年）",
        "同类样本量": int((df["组别"] == r["组别"]).sum()),
        "体检总分": f("总分", 1),
        "画像标签": str(r.get("画像", "")),
        "高费低效标记": bool(r["高费低效"]),
        "指标": metrics,
        "数据说明": "以上数字均由代码从公开净值与费率数据计算得出，是本回答唯一可引用的数据来源。",
    }

# ---------------------------------------------------------------- 解读层

SYSTEM_PROMPT = """你是一个基金信息工具"基金体检"，面向没有金融背景的普通用户，把体检指标翻译成通俗易懂的诊断报告。

【硬性规则，违反任何一条即为不合格输出】
1. 数字封闭：只能使用用户上下文 JSON 中出现的数字，以及由这些数字直接计算得出的对比（如差值、倍数，需注明算法）。上下文里没有的信息一律回答"暂无数据"，禁止编造、估算、引用任何外部知识或数字。
2. 术语翻译：禁止让"年化""卡玛""分位"等术语裸奔出现。必须翻译成人话，例如"年化收益 4.4%"→"平均一年赚 4.4%"。表述分位时必须精确："同类分位 0.1"→"几乎排在同类最后"，禁止把 0.1 说成"倒数第一"这种绝对化说法。
3. 需求对照：如果用户给出了风险承受能力（如"最多接受15%回撤"），必须把它和该基金的最大回撤做显式对比（例如"你的承受线是 15%，它历史上最惨时亏了 49.45%，超出承受线 34 个百分点"）。同理，用户提到金额或期限时，要结合费用和回撤给出对应解释。
4. 合规红线：你提供的是信息，不是投资建议。全文禁止出现"建议买入""可以加仓""预计会涨""值得投资"等表述。结尾必须保留免责声明。
5. 固定结构（六段，用 markdown 标题）：
   ## 一句话结论
   ## 赚钱能力（对照同类说）
   ## 风险体验（回答"最惨的时候亏多少"；如有用户承受线，必须对比）
   ## 费用（回答"每年交多少、值不值"；必须使用费后年化——即扣完所有费用后实际到手多少）
   ## 适合谁、不适合谁（结合用户需求；无用户需求则按一般稳健型用户假设，并注明这是假设）
   ## 免责声明（固定：本报告由算法生成，数据截至统计区间末，历史业绩不代表未来表现，不构成投资建议。）
6. 篇幅：全文 300–500 字，每段不超过 3 句话，禁止堆砌数字。
"""

# 注意：智谱 GLM 不接受只有 system 一条消息的请求（报错 1214），
# 因此拆成 system（规则）+ user（数据与需求）两条。
RULES_PROMPT = SYSTEM_PROMPT.rstrip()


def call_llm(context: dict, query: str) -> str:
    base = os.environ.get("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "glm-4-flash")
    if not key:
        raise SystemExit("未设置 LLM_API_KEY。智谱 Key 申请：https://open.bigmodel.cn （glm-4-flash 免费）")

    resp = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0.3,   # 低温度：解读任务要稳定，不要发散
            "messages": [
                {"role": "system", "content": RULES_PROMPT},
                {"role": "user", "content": (
                    "用户上下文（唯一数据来源）：\n"
                    + json.dumps(context, ensure_ascii=False, indent=1)
                    + "\n\n用户需求（自然语言，可能为空）：\n"
                    + (query or "（用户未提供具体需求）"))},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    print(f"[tokens] prompt {usage.get('prompt_tokens')} / completion {usage.get('completion_tokens')}")
    return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="基金代码，如 012414")
    p.add_argument("--query", default="", help="用户需求原文，如 '最多接受15%回撤'")
    args = p.parse_args()

    df = load_master()
    ctx = build_context(df, args.code)
    if ctx is None:
        return
    print("=== 喂给模型的上下文（评测时逐数字对照它） ===")
    print(json.dumps(ctx, ensure_ascii=False, indent=1))
    print()

    t0 = time.time()
    report = call_llm(ctx, args.query)
    print(f"=== 生成耗时 {time.time()-t0:.1f}s ===\n")
    print(report)

    OUT.mkdir(exist_ok=True)
    tag = args.query[:12].replace(" ", "") or "通用"
    path = OUT / f"{args.code}_{tag}.md"
    path.write_text(
        f"# AI 解读样例：{ctx['基金名称']}（{args.code}）\n\n"
        f"> 用户需求：{args.query or '（无，按一般稳健型用户）'}\n"
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n\n{report}\n",
        encoding="utf-8")
    print(f"\n已存：{path}")


if __name__ == "__main__":
    main()
