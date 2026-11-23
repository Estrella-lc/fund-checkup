"""
step5_cluster.py —— K-Means 聚类出产品画像 + 圈定"高费低效"群体
=================================================================
运行:  python src/step5_cluster.py

输入:  data/processed/master.csv（step4 输出，含 7 项打分）
输出:  data/processed/clustered.csv
        reports/02_聚类与高费低效群体.md

两步走，刻意分开：
  1) 聚类（无监督）：不预设好坏，让算法按"收益/风险/费率/经理"把产品自动归堆，看市场真实的产品形态
  2) 高费低效（规则）：不看聚类结果，直接用"费率高位 + 性价比低位"的硬规则圈人，避免循环论证

为什么分开：如果先聚类再指着某一堆说"这就是高费低效"，等于用自己定的标签证明自己，
面试官一问就塌。聚类负责"描述市场长什么样"，规则负责"谁该被点名"，两者互相印证。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

# 聚类用的原始指标（不用分位分，用真实业务口径，方便解释）
FEATURES = ["年化收益", "最大回撤", "波动率", "综合费率", "经理从业年限"]
LABEL_MAP = {
    "年化收益": "年化收益%", "最大回撤": "最大回撤%", "波动率": "波动率%",
    "综合费率": "综合费率%", "经理从业年限": "经理从业年限",
}


def pick_k(X: np.ndarray, ks=range(3, 8), seed: int = 42) -> tuple[int, dict]:
    """用轮廓系数选 k：轮廓系数衡量"堆内够紧、堆间够远"，范围 -1~1，越大越好。"""
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best = max(scores, key=scores.get)
    return best, scores


def name_cluster(row: pd.Series, df: pd.DataFrame) -> str:
    """按相对全样本的高低给簇起名，规则写死，不靠感觉。"""
    def z(col):
        s = df[col]
        return (row[col] - s.mean()) / s.std() if s.std() else 0

    fee, ret, mdd, mgr = z("综合费率"), z("年化收益"), z("最大回撤"), z("经理从业年限")

    if fee > 0.5 and ret < -0.3:
        return "高费低效型"
    if fee < -0.5 and mdd < -0.3:
        return "低费稳健型"
    if ret > 0.5 and mdd > 0.5:
        return "高波动高收益型"
    if ret > 0.3 and fee < 0.2:
        return "性价比均衡型"
    if mgr < -0.3:
        return "新手经理型"
    if fee > 0.5:
        return "高费平庸型"
    return "大众中间型"


def cluster_group(df: pd.DataFrame, group: str, ks) -> pd.DataFrame:
    sub = df[df["组别"] == group].copy().reset_index(drop=True)
    X = StandardScaler().fit_transform(sub[FEATURES].astype(float))

    k, scores = pick_k(X, ks)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sub["簇"] = km.labels_

    # 用簇均值代表这个簇，再按相对位置命名
    centers = sub.groupby("簇")[FEATURES].mean()
    names = {}
    for cid, row in centers.iterrows():
        names[cid] = name_cluster(row, sub)
    sub["画像"] = sub["簇"].map(names)

    print(f"\n=== {group}（{len(sub)} 只）k={k} ===")
    print("   轮廓系数：" + " ".join(f"k={k2}:{v:.3f}" for k2, v in scores.items()))

    prof = (sub.groupby("画像")
               .agg(产品数=("基金代码", "size"), 年化收益=("年化收益", "mean"),
                    最大回撤=("最大回撤", "mean"), 波动率=("波动率", "mean"),
                    综合费率=("综合费率", "mean"), 卡玛=("卡玛", "mean"),
                    经理从业年限=("经理从业年限", "mean"), 总分=("总分", "mean"))
               .sort_values("产品数", ascending=False))
    prof["占比"] = (prof["产品数"] / len(sub) * 100).round(1)
    prof = prof.round(2)
    print(prof.to_string())
    return sub


def high_fee_low_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """规则圈人：仍收「行业标准档」费率 且 风险调整后收益（卡玛）在组内后 25%。

    【为什么不用分位数切费率 —— 这里改过一次，是最重要的一处修正】
    最初写的是「组内费率前 25%」，跑通后去查分布才发现：主动权益基金中
    71% 的综合费率完全相同，都是 1.40%（1.2% 管理费 + 0.2% 托管费的标准档），
    于是 P50 和 P75 都等于 1.40，「费率 >= P75」实际选中了 75% 的样本，
    这条条件几乎没筛掉任何人——729 只入选产品中 87% 仅靠「卡玛后 25%」就能选出。

    费率是**离散档位**（76 个取值），不是连续变量。对离散变量用分位数切分，
    等于假设每个分位点都对应不同的产品，这个假设在大量并列值下直接失效。

    所以改成按**绝对档位**切：主动权益 1.40%、被动指数 0.60% 是两类的标准档，
    取「>= 标准档」= 没有享受到降费红利的产品。
    业务含义也随之更清楚了：不是"收得特别贵"，而是"收着行业标准价，却没交出标准业绩"。
    """
    STD_FEE = {"主动权益": 1.40, "被动指数": 0.60}   # 两类基金的标准费率档（%/年）
    flags = []
    for g, sub in df.groupby("组别"):
        thr = STD_FEE.get(g, sub["综合费率"].median())
        fee_hi = sub["综合费率"] >= thr
        calmar_lo = sub["卡玛"] <= sub["卡玛"].quantile(0.25)
        flags.append(sub[fee_hi & calmar_lo])
    return pd.concat(flags)


if __name__ == "__main__":
    df = pd.read_csv(PROC / "master.csv", dtype={"基金代码": str})
    print(f"样本 {len(df)} 只（主动 {(df['组别']=='主动权益').sum()} / 被动 {(df['组别']=='被动指数').sum()}）")

    parts = [cluster_group(df, "主动权益", range(3, 8)),
             cluster_group(df, "被动指数", range(2, 6))]
    out = pd.concat(parts, ignore_index=True)

    # ---------- 高费低效群体 ----------
    bad = high_fee_low_efficiency(out)
    # 判定结果必须落库：否则 step7 / card.py 各自重算，口径一改就悄悄漂移。
    # （2026-11-10 复核发现三处实现，pandas 分位 vs 排序取位导致 1097/1098 只差异）
    out["高费低效"] = out["基金代码"].isin(bad["基金代码"])
    out.to_csv(PROC / "clustered.csv", index=False, encoding="utf-8-sig")

    good = out[~out["高费低效"]]
    print(f"\n=== 高费低效群体（费率 >= 行业标准档 且 卡玛组内后 25%）===")
    print(f"规模 {len(bad)} 只，占样本 {len(bad)/len(out)*100:.1f}%"
          f"（主动 {(bad['组别']=='主动权益').sum()} / 被动 {(bad['组别']=='被动指数').sum()}）")
    cmp = pd.DataFrame({
        "高费低效组": [bad["综合费率"].mean(), bad["年化收益"].mean(), bad["最大回撤"].mean(),
                   bad["卡玛"].mean(), bad["费后年化"].mean()],
        "其余样本": [good["综合费率"].mean(), good["年化收益"].mean(), good["最大回撤"].mean(),
                 good["卡玛"].mean(), good["费后年化"].mean()],
    }, index=["综合费率%", "年化收益%", "最大回撤%", "卡玛", "费后年化%"]).round(2)
    cmp["差值"] = (cmp["高费低效组"] - cmp["其余样本"]).round(2)
    print(cmp.to_string())

    # 这些产品集中在哪些公司
    print("\n高费低效产品最集中的 8 家基金公司:")
    print(bad["基金公司"].value_counts().head(8).to_string())

    print("\n典型样本（按规模取前 5，规模数据可得者）:")
    cols = ["基金代码", "基金名称", "组别", "综合费率", "年化收益", "最大回撤", "卡玛", "总分"]
    show = bad.dropna(subset=["规模_亿元"]).nlargest(5, "规模_亿元") if "规模_亿元" in bad else bad.head(5)
    print(show[cols].to_string(index=False))

    # ---------- 写报告 ----------
    md = ["# 聚类画像与高费低效群体\n",
          f"> 样本 {len(out)} 只（全量采集完成），统计区间 2023-10 至 2026-10-28，数据源 AkShare / 天天基金。\n",
          "## 一、产品画像分布\n"]
    for g in ["主动权益", "被动指数"]:
        sub = out[out["组别"] == g]
        prof = (sub.groupby("画像")
                   .agg(产品数=("基金代码", "size"), 年化收益=("年化收益", "mean"),
                        最大回撤=("最大回撤", "mean"), 综合费率=("综合费率", "mean"),
                        卡玛=("卡玛", "mean"), 总分=("总分", "mean"))
                   .sort_values("产品数", ascending=False).round(2))
        prof["占比%"] = (prof["产品数"] / len(sub) * 100).round(1)
        md.append(f"\n### {g}（{len(sub)} 只）\n")
        md.append(prof.to_markdown())
    md.append("\n## 二、高费低效群体\n")
    md.append(f"定义：**综合费率 ≥ 行业标准档**（主动权益 1.40%、被动指数 0.60%）**且** 卡玛处于组内后 25%。\n"
              f"\n> 为什么不用「费率前 25%」：费率只有 76 个取值，主动权益中 71% 的基金费率完全相同（1.40%），\n"
              f"> P50 与 P75 并列，按分位数切会让「前 25%」实际选中 75% 的样本，条件形同虚设。详见 step5 源码注释。\n")
    md.append(cmp.to_markdown())
    md.append("\n### 集中在哪些公司\n")
    md.append(bad["基金公司"].value_counts().head(10).to_frame("产品数").to_markdown())
    (REP / "02_聚类与高费低效群体.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✅ 已输出 {PROC/'clustered.csv'} 与 {REP/'02_聚类与高费低效群体.md'}")
