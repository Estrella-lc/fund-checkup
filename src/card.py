# -*- coding: utf-8 -*-
"""
数据卡（Data Card）：把"算账"和"说话"彻底分开。

为什么要有这一层（面试可讲）：
  v2 的 30 条评测里出了 7 处真错误（23%），**没有一处是编造外部数据**，
  全是模型在给定数字上算错或理解反：把 30% 承受线写成 15%、回撤 10.75%<15%
  却说"超出 34 个百分点"、"34"这个数上下文里根本不存在。

  → 结论：光在 Prompt 里写"不要瞎编"没用。凡是涉及比较、差值、方向的判断，
     必须由代码算成一个**结论字段**（如 "回撤是否超出承受线": "否"），
     模型只负责引用原话。模型彻底不碰算术，这一类错误就没有发生空间。

用法：
  python src/card.py --code 012414
  python src/card.py --code 012414 --risk 15      # 给风险承受线，自动算出对照结论
  python src/card.py --code 012414 --json         # 输出 JSON，直接喂给 LLM

只依赖标准库（本机无 pandas 也能跑）。
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "clustered.csv"

# 综合费率"行业标准档"：主动权益 1.40%、被动指数 0.60%
STD_FEE = {"主动权益": 1.40, "被动指数": 0.60}


def _f(v, nd=2):
    if v is None or str(v).strip() == "":
        return None
    try:
        return round(float(v), nd)
    except ValueError:
        return None


def load() -> list[dict]:
    with open(MASTER, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct_rank(rows: list[dict], col: str, value, higher_better: bool):
    """同类分位 0–100（越大越好）。0.1 表示只优于 0.1% 的同类。"""
    vals = [x for x in (_f(r.get(col)) for r in rows) if x is not None]
    if value is None or not vals:
        return None
    below = sum(1 for v in vals if v < value) / len(vals) * 100
    return round(below if higher_better else 100 - below, 1)


def build(code: str, risk_line: float | None = None) -> dict | None:
    allrows = load()
    row = next((r for r in allrows if r["基金代码"] == code), None)
    if row is None:
        print(f"找不到基金 {code}")
        return None

    grp = row["组别"]
    peers = [r for r in allrows if r["组别"] == grp]
    n = len(peers)

    g = lambda c, nd=2: _f(row.get(c), nd)

    # ---- 七个维度：原始值 + 同类分位 + 同类排名（自然语言化的"位置"）
    def dim(col, higher, label, unit=""):
        val = g(col) if unit != "年" else _f(row.get(col), 1)
        p = pct_rank(peers, col, val, higher)
        # 排名分母用「该字段有值的同类家数」，避免缺失值把名次虚高
        cnt = sum(1 for x in (_f(r.get(col)) for r in peers) if x is not None)
        return {
            "指标": label,
            "本基金": val,
            "单位": unit,
            "同类分位": p,
            "同类排名": (f"{max(1, round(cnt * (100 - p) / 100))}/{cnt}" if p is not None else None),
        }

    dims = {
        "①风险调整后收益": {
            "卡玛比率": dim("卡玛", True, "卡玛（年化收益/最大回撤）"),
            "年化收益_pct": dim("年化收益", True, "年化收益", "%"),
            "费后年化_pct": dim("费后年化", True, "费后年化", "%"),
            "超额收益_pct_vs同类中位数": dim("超额收益", True, "相对同类中位数超额", "%"),
        },
        "②回撤与波动": {
            "最大回撤_pct": dim("最大回撤", False, "最大回撤", "%"),
            "波动率_pct": dim("波动率", False, "年化波动率", "%"),
            "下行波动_pct": dim("下行波动率", False, "下行波动率", "%"),
        },
        "③策略一致性": {
            "基金类型": grp,
            "业绩比较基准": (row.get("业绩比较基准") or "").strip() or None,
            "投资策略": ((row.get("投资策略") or "").strip()[:120] or None),
            "判断能力": "仅有基准与超额，无持仓/风格序列，只能做方向性判断，不能验证漂移",
        },
        "④持仓集中度": {
            "数据状态": "缺失（无重仓股/行业分布/换手率）",
            "处理方式": "输出时必须整项标注【信息不足，暂不评估】",
        },
        "⑤经理_or_跟踪": {
            "基金经理": (row.get("基金经理") or "").strip() or None,
            "经理从业年限_年": dim("经理从业年限", True, "经理从业年限"),
            "基金公司": (row.get("基金公司") or "").strip() or None,
            "数据状态": ("指数基金：跟踪误差/跟踪偏离度缺失" if grp == "被动指数"
                         else "主动基金：任职起始日、任职回报、在管规模缺失"),
        },
        "⑥规模与流动性": {
            "最新规模_亿元": dim("规模_亿元", True, "最新规模（亿元）"),
            "数据状态": "无规模时序/持有人结构/场内成交额与折溢价",
        },
        "⑦费率": {
            "综合费率_pct": dim("综合费率", False, "综合费率", "%"),
            "管理费_pct": g("管理费"),
            "托管费_pct": g("托管费"),
            "销售服务费_pct": g("销售服务费"),
            "赎回费1年_pct": g("赎回费1年"),
        },
    }

    # ---- 结论字段：比较类判断全部在这里算完，模型只引用（核心设计）
    conclusions = {}

    ret, net = g("年化收益"), g("费后年化")
    if ret is not None and net is not None:
        conclusions["费用侵蚀_pp"] = round(ret - net, 2)
        conclusions["费用侵蚀说明"] = (
            f"年化收益 {ret}% 扣费后为 {net}%，费用吃掉 {round(ret-net,2)} 个百分点"
        )

    mdd = g("最大回撤")
    if risk_line is not None and mdd is not None:
        diff = round(risk_line - mdd, 2)
        exceeded = mdd > risk_line
        conclusions["承受线对照"] = {
            "用户承受线_pct": risk_line,
            "本基金最大回撤_pct": mdd,
            "是否超出承受线": "是" if exceeded else "否",
            "差值_pp": abs(diff),
            "允许的表述": (f"最大回撤 {mdd}% 超出你 {risk_line}% 的承受线，超出 {abs(diff)} 个百分点"
                        if exceeded else
                        f"最大回撤 {mdd}% 低于你 {risk_line}% 的承受线，在其承受范围内（余量 {abs(diff)} 个百分点）"),
            "禁止的表述": [
                ("禁止说'超出'" if not exceeded else "禁止说'在承受范围内'"),
                "禁止自行重算差值，只能引用上面这句",
            ],
        }

    # 高费低效：优先读 step5 落库的列（单一真相源）；列缺失时回退到本地复算，
    # 但必须与 step5 口径完全一致——用 pandas quantile，不要用排序取位（差 1 只的坑）。
    fee, calmar = g("综合费率"), g("卡玛")
    thr = STD_FEE.get(grp)
    flag = None
    if "高费低效" in row and str(row.get("高费低效")).strip() != "":
        flag = str(row["高费低效"]).strip().lower() in ("true", "1", "1.0")
        conclusions["高费低效标记"] = flag
        conclusions["判定口径"] = ("读取 clustered.csv 落库列；口径＝综合费率 ≥ 行业标准档"
                             f"（{grp} {thr}%）且卡玛处于组内后 25%，由 step5 统一计算")
    elif fee is not None and calmar is not None and thr is not None:
        c25 = sorted(x for x in (_f(r.get("卡玛")) for r in peers) if x is not None)
        q25 = c25[int(len(c25) * 0.25)] if c25 else None
        flag = bool(fee >= thr and q25 is not None and calmar <= q25)
        conclusions["高费低效标记"] = flag
        conclusions["判定口径"] = (
            f"（回退复算，clustered.csv 缺列）综合费率 {fee}% {'≥' if fee >= thr else '<'} 行业标准档 {thr}%，"
            f"卡玛 {calmar} {'处于' if flag else '未处于'}同类后 25%（阈值 {round(q25,3) if q25 else 'NA'}）"
        )

    # 收益可解释性：注意口径——本项目的「超额收益」= 年化收益 − 同组中位数（同类相对），
    # 不是相对业绩比较基准的超额。两者含义不同，措辞必须写清，否则会误导读者。
    ex = g("超额收益")
    if ex is not None:
        conclusions["超额收益方向"] = ("跑赢同类中位数" if ex > 0 else "跑输同类中位数")
        conclusions["超额收益口径"] = "年化收益 − 同组（主动权益/被动指数）所有样本的中位数；非相对业绩比较基准"
        conclusions["收益可解释性"] = (
            f"高于同类中位数 {abs(ex)} 个百分点" if ex > 0 else
            f"低于同类中位数 {abs(ex)} 个百分点"
        ) + ("；但无持仓与分年度数据，无法判断这来自选股能力还是行业行情，"
             "归因部分必须标注【信息不足】")

    return {
        "数据卡版本": "v1",
        "基金代码": code,
        "基金名称": row["基金名称"],
        "基金类型": row["基金类型"],
        "组别": grp,
        "同类样本量": n,
        "成立时间": row.get("成立时间"),
        "统计区间": "2023-10 至 2026-10（近三年）",
        "画像标签": row.get("画像"),
        "体检总分": g("总分", 1),
        "七维度": dims,
        "结论字段": conclusions,
        "数据缺口": ["持仓集中度（重仓股/行业分布/换手率）",
                 "夏普比率（未纳入无风险利率，用卡玛替代）",
                 "分年度收益与分年度基准",
                 ("跟踪误差/跟踪偏离度" if grp == "被动指数" else "经理任职起始日与任职回报"),
                 "规模时序与持有人结构"],
        "使用说明": ("本数据卡由代码计算生成，是评价唯一可引用的数据来源。"
                 "结论字段中的'允许的表述'必须原样引用，禁止改写数字或方向。"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--risk", type=float, default=None, help="用户风险承受线（最大回撤 %%）")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    card = build(a.code, a.risk)
    if card is None:
        return
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
