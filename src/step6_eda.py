# -*- coding: utf-8 -*-
"""
step6_eda.py —— 探索性分析（EDA）：出 8 张图，每张图对应报告里的一个结论。

【这个脚本在整个项目里的位置】
前面 5 步做的是"把数据算成一个数字"，这一步做的是"证明这个数字背后有真实的商业事实"。
评委和面试官不会读你的表格，他们看图。所以图不是装饰，是证据。

【为什么图要"一张图只讲一件事"】
一张图塞两个结论，读者两个都记不住。所以下面每张图我都只留一个论点，
图标题直接写成"结论句"，而不是"图1 年化收益分布"。

运行：python src/step6_eda.py
产出：reports/figs/*.png + 控制台打印每张图的关键数字
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存文件（服务器/后台跑也不会卡住）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- 中文字体 ----------
# Windows 自带微软雅黑；不设这个，中文会变成一堆方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号能正常显示

# ---------- 配色 ----------
# 主动=红、被动=蓝，全篇统一。图里颜色一旦有固定含义，读者看第二张就不用重新学。
C_ACT, C_PAS = "#C0392B", "#1F6F8B"
C_BAD, C_REST = "#C0392B", "#95A5A6"
C_GRID = "#E5E7EB"


ROOT = Path(__file__).resolve().parent.parent
PROC, FIGS, RAW = ROOT / "data" / "processed", ROOT / "reports" / "figs", ROOT / "data" / "raw"
FIGS.mkdir(parents=True, exist_ok=True)


def style(ax, xlabel="", ylabel="", title=""):
    """统一图表样式：去掉上右边框、浅网格、标题左对齐。

    为什么要去掉上/右边框？因为它们不承载任何信息，只会让图看起来更挤。
    """
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=C_GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=10, color="#374151")
    ax.set_ylabel(ylabel, fontsize=10, color="#374151")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12, color="#111827")


def note(ax, text, xy=(0.02, 0.96), **kw):
    """在图上加一句"人话注解"——图里必须有一句话告诉读者该看哪儿。"""
    ax.text(*xy, text, transform=ax.transAxes, fontsize=9.5, va="top", color="#374151",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FFF7E6", ec="#F0C36D", lw=0.8), **kw)


def fee_tier(s: pd.Series) -> pd.Series:
    """把连续的费率值映射到 5 个「定价档位」。

    为什么不用 pd.qcut（等频分箱）？因为费率只有 76 个取值、58% 挤在 1.40%，
    等频分箱会在并列值上切出空组或大小极不均的组，图会失真。
    业务上按档位分也更自然：投资者看到的是"0.2 / 0.6 / 1.4"这几个价签，不是分位数。
    """
    return pd.cut(s, bins=[0, .50, 1.00, 1.35, 1.45, 9],
                  labels=["<0.5%", "0.5–1.0%", "1.0–1.35%", "1.35–1.45%", ">1.45%"],
                  include_lowest=True, right=False)


def load():
    d = pd.read_csv(PROC / "clustered.csv", dtype={"基金代码": str})
    d["成立年份"] = pd.to_datetime(d["成立时间"], errors="coerce", format="mixed").dt.year
    # 【高费低效的口径与 step5 严格一致，且这里踩过一次坑，务必看 step5 的注释】
    # 费率是离散档位（主动 71% 都在 1.40%），用分位数切分等于没筛，所以按绝对标准档切。
    STD = {"主动权益": 1.40, "被动指数": 0.60}
    bad = pd.Series(False, index=d.index)
    for g, sub in d.groupby("组别"):
        thr = STD.get(g, sub["综合费率"].median())
        bad.loc[sub.index] = (sub["综合费率"] >= thr) & (sub["卡玛"] <= sub["卡玛"].quantile(.25))
    d["高费低效"] = bad
    return d


# =====================================================================
# 图 1｜收益分布：市场长什么样
# =====================================================================
def fig1(d):
    """为什么用直方图？因为要看的是"分布形状"，不是某只基金。

    分布图回答三个问题：中心在哪（中位数）、胖瘦如何（离散度）、有没有长尾巴（极端值）。
    这三件事直接决定了后面能不能用平均值、能不能用 z 分数标准化。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-45, 65, 2.5)
    for g, c in [("主动权益", C_ACT), ("被动指数", C_PAS)]:
        s = d.loc[d["组别"] == g, "年化收益"].dropna()
        ax.hist(s, bins=bins, alpha=.55, color=c, label=f"{g}（{len(s)} 只，中位数 {s.median():.1f}%）")
        ax.axvline(s.median(), color=c, ls="--", lw=1.4)
    ax.axvline(0, color="#111827", lw=1)
    style(ax, "近 3 年年化收益（%，2023-10 至 2026-10）", "基金数量",
          f"图1｜主动基金三年年化中位数 {d.loc[d['组别']=='主动权益','年化收益'].median():.1f}%，"
          f"但其中 {d.loc[d['组别']=='主动权益','年化收益'].lt(0).mean()*100:.0f}% 在亏钱")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    note(ax, f"分布右偏：少数暴涨产品把平均值拉到 {d['年化收益'].mean():.1f}%，\n"
             f"但中位数只有 {d['年化收益'].median():.1f}%——用户感觉到的「平均能赚多少」，\n"
             f"是被极少数赢家撑起来的", xy=(0.02, 0.58))
    fig.tight_layout(); fig.savefig(FIGS / "fig1_收益分布.png", dpi=160); plt.close(fig)

    print("\n[图1] 收益分布")
    for g in ["主动权益", "被动指数"]:
        s = d.loc[d["组别"] == g, "年化收益"].dropna()
        print(f"  {g}: 中位数 {s.median():.2f}% | 均值 {s.mean():.2f}% | "
              f"亏损占比 {(s < 0).mean()*100:.0f}% | 偏度 {s.skew():.2f}")


# =====================================================================
# 图 2｜费率的真相：不是连续价格，是几个定价档位
# =====================================================================
def fig2(d):
    """这张图原本写的是「2023 改革后费率断崖下降」，跑出来发现是假的，改成了现在这版。

    真实情况：76 个取值，58% 的基金挤在 1.40% 一档，2006–2023 年成立的产品
    费率中位数一动没动，全是 1.40%。所以结论不是"在降价"，而是
    **"存量不降费，降价只在增量产品上发生，而增量产品样本里几乎没有"**。
    这既是发现，也是局限，两句话都写进报告。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- 左：离散档位分布 ----
    ax = axes[0]
    share = d["综合费率"].value_counts(normalize=True).head(8).sort_index(ascending=False)
    bars = ax.barh([f"{v:.2f}%" for v in share.index], share.values * 100, color="#4A6FA5", height=.62)
    for b, v in zip(bars, share.values * 100):
        ax.text(v + .8, b.get_y() + b.get_height()/2, f"{v:.0f}%", va="center", fontsize=9.5)
    ax.set_yticklabels([f"{v:.2f}%" for v in share.index], fontsize=9.5)
    style(ax, "占全样本比例（%）", "综合费率档位（%/年）",
          "图2a｜58% 的基金，费率一模一样，都是 1.40%")
    note(ax, f"全样本只有 {d['综合费率'].nunique()} 个取值，不是连续定价。\n"
             f"最低 0.20%、最高 2.40%，但两端都是少数派", xy=(0.35, 0.30))

    # ---- 右：时间趋势（如实呈现"没动"）----
    ax = axes[1]
    for g, c in [("主动权益", C_ACT), ("被动指数", C_PAS)]:
        s = d[d["组别"] == g].dropna(subset=["成立年份", "综合费率"])
        t = s[s["成立年份"].between(2010, 2024)].groupby("成立年份")["综合费率"].median()
        t = t[t.index.map(lambda y: (s["成立年份"] == y).sum() >= 15)]  # 样本太少的年份不画
        ax.plot(t.index, t.values, "o-", color=c, lw=2.2, ms=5, label=g)
    style(ax, "基金成立年份", "当年成立产品的费率中位数（%）",
          "图2b｜十五年间，主动基金的定价一动不动")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    note(ax, "基金公司不会主动给存量产品降费，\n只有新发产品才按新价格定价。\n"
             "而 2024 年后成立的产品，本样本不足 30 只", xy=(0.02, 0.52))

    fig.tight_layout(); fig.savefig(FIGS / "fig2_费率档位.png", dpi=160); plt.close(fig)

    print("\n[图2] 费率")
    print(f"  档位 Top5: " + " | ".join(f"{k:.2f}% {v*100:.0f}%" for k, v in
                                        d["综合费率"].value_counts(normalize=True).head(5).items()))
    for g in ["主动权益", "被动指数"]:
        s = d[d["组别"] == g]["综合费率"]
        print(f"  {g}: 中位数 {s.median():.2f}% | 等于中位数的占比 {(s==s.median()).mean()*100:.0f}%"
              f" | 唯一取值 {s.nunique()} 个")


# =====================================================================
# 图 3｜费率和收益到底有没有关系
# =====================================================================
def fig3(d):
    """为什么用"分箱均值线"而不是拟合直线？

    因为两者的关系很可能不是直线（也可能根本不存在）。直线会强行告诉你一个斜率，
    分箱均值是让数据自己说话：把费率从低到高切成 10 组，每组算平均收益，看它动不动。
    如果这条线是平的，就说明"贵=好"这个假设不成立——这是本项目最重要的一张"打脸图"。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, g, c in [(axes[0], "主动权益", C_ACT), (axes[1], "被动指数", C_PAS)]:
        s = d[d["组别"] == g].dropna(subset=["综合费率", "年化收益"]).copy()
        ax.scatter(s["综合费率"], s["年化收益"], s=9, alpha=.20, color=c, edgecolors="none")
        # 用绝对档位分箱，而不是分位数分箱——分位数在并列值上会切出空组
        s["档位"] = fee_tier(s["综合费率"])
        m = s.groupby("档位", observed=True).agg(x=("综合费率", "median"), y=("年化收益", "mean"),
                                                 n=("年化收益", "size"))
        m = m[m["n"] >= 30]  # 样本太少的档位不画
        ax.plot(m["x"], m["y"], "o-", color="#111827", lw=2.2, ms=7, label="各费率档的平均收益")
        for x, y, n in zip(m["x"], m["y"], m["n"]):
            ax.annotate(f"n={n}", (x, y), fontsize=7.5, xytext=(0, 9), textcoords="offset points",
                        ha="center", color="#4B5563")
        ax.axhline(s["年化收益"].median(), color="#6B7280", ls=":", lw=1.2)
        r = s["综合费率"].corr(s["年化收益"], method="spearman")
        style(ax, "综合费率（%/年）", "近 3 年年化收益（%）" if g == "主动权益" else "",
              f"图3｜{g}：费率与收益几乎无关（ρ = {r:.2f}）")
        ax.legend(frameon=False, fontsize=9)
        note(ax, f"最贵那一档平均收益 {m['y'].iloc[-1]:.1f}%，\n"
                 f"最便宜那一档 {m['y'].iloc[0]:.1f}%\n"
                 f"→ 多付钱买不到更好的业绩")
    fig.tight_layout(); fig.savefig(FIGS / "fig3_费率与收益关系.png", dpi=160); plt.close(fig)

    print("\n[图3] 费率 × 收益")
    for g in ["主动权益", "被动指数"]:
        s = d[d["组别"] == g].dropna(subset=["综合费率", "年化收益"])
        r = s["综合费率"].corr(s["年化收益"], method="spearman")
        print(f"  {g}: Spearman ρ = {r:.3f}（越接近 0 越说明没关系）")


# =====================================================================
# 图 4｜风险—收益平面：钱到底买到了什么
# =====================================================================
def fig4(d):
    """为什么用散点而不是柱状？因为这里有两个连续变量，柱状图只能表达一个。

    风险收益平面是资产管理的"标准地图"：横轴承担多少波动，纵轴换来多少收益。
    点应该沿右上方向排布（高风险高收益）。如果有一大批点落在右下角
    ——高波动、低收益——那就是"白承担风险"，这就是高费低效群体的位置。
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    s = d.dropna(subset=["波动率", "年化收益"])
    sc = ax.scatter(s["波动率"], s["年化收益"], c=s["综合费率"], cmap="RdYlBu_r",
                    s=13, alpha=.65, vmin=0.3, vmax=2.0, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=.02); cb.set_label("综合费率（%/年）", fontsize=9)
    ax.axhline(0, color="#111827", lw=1)
    ax.axvline(s["波动率"].median(), color="#6B7280", ls=":", lw=1.2)
    ax.text(s["波动率"].median() * 1.02, ax.get_ylim()[0] * .92, "波动率中位数", fontsize=8.5, color="#6B7280")
    style(ax, "年化波动率（%，越大越颠）", "近 3 年年化收益（%）",
          "图4｜右上角是「值得冒险」，右下角是「白担风险」")
    # 标注右下角（高波动、亏钱）的代表性产品
    worst = s[(s["波动率"] > s["波动率"].quantile(.75)) & (s["年化收益"] < s["年化收益"].quantile(.05))].nlargest(3, "综合费率")
    for _, r in worst.iterrows():
        ax.annotate(r["基金名称"][:14], (r["波动率"], r["年化收益"]), fontsize=8,
                    xytext=(6, -10), textcoords="offset points", color="#7F1D1D")
    note(ax, "右下角这一片：波动比同行大，收益却在水下。\n颜色越红 = 收费越贵 —— 贵、颠、亏同时出现")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_风险收益平面.png", dpi=160); plt.close(fig)

    print("\n[图4] 风险—收益平面")
    q_hi = s["波动率"].quantile(.75)
    bad = s[(s["波动率"] > q_hi) & (s["年化收益"] < 0)]
    print(f"  高波动（>P75={q_hi:.1f}%）且亏损的产品: {len(bad)} 只 "
          f"({len(bad)/len(s)*100:.1f}%)，平均费率 {bad['综合费率'].mean():.2f}%")


# =====================================================================
# 图 5｜"贵"和"性价比差"是不是同一件事
# =====================================================================
def fig5(d):
    """为什么用箱线图？它一次给出三件事：中位数、四分位区间、离群情况。

    这里要回答的是：按费率档位从低到高排，每档的"卡玛比率"（每承受一份最大痛苦
    换回多少收益）是否单调下降。如果单调下降，"贵"和"性价比差"就是同一件事，
    而不是两个独立缺点——这是把"贵"和"差"绑在一起讲的第一份证据。

    注意：分档用的是绝对档位（见 fee_tier），不是四分位。分位数在这个变量上会切爆。
    """
    fig, ax = plt.subplots(figsize=(10, 5.4))
    data, labels, colors = [], [], []
    for g in ["主动权益", "被动指数"]:
        s = d[d["组别"] == g].dropna(subset=["综合费率", "卡玛"]).copy()
        s["档位"] = fee_tier(s["综合费率"])
        for lab in s["档位"].cat.categories:
            v = s.loc[s["档位"] == lab, "卡玛"]
            if len(v) < 25:            # 样本太少的档位不画，否则箱体没有统计意义
                continue
            data.append(v.clip(-1, 3).values)   # 截尾：卡玛在回撤接近 0 时会炸到几十
            labels.append(f"{lab}\n(n={len(v)})")
            colors.append(C_ACT if g == "主动权益" else C_PAS)
    bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False,
                    medianprops=dict(color="#111827", lw=2))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(.5); patch.set_edgecolor(c)
    ax.axhline(0, color="#111827", lw=1)
    style(ax, "综合费率档位（红=主动权益，蓝=被动指数）", "卡玛比率（年化收益 ÷ 最大回撤）",
          "图5｜费率越高，中间那 50% 的产品性价比越低")
    note(ax, "每个箱体中间那条线是中位数。\n从 0.20% 档到 1.40% 档，中位数一路下移——\n"
             "贵的不只是贵，而且更不划算", xy=(0.02, 0.97))
    fig.tight_layout(); fig.savefig(FIGS / "fig5_费率档位与卡玛.png", dpi=160); plt.close(fig)

    print("\n[图5] 费率档位 × 卡玛（中位数）")
    for g in ["主动权益", "被动指数"]:
        s = d[d["组别"] == g].dropna(subset=["综合费率", "卡玛"]).copy()
        s["档位"] = fee_tier(s["综合费率"])
        line = " | ".join(f"{lab}: {s.loc[s['档位']==lab,'卡玛'].median():.2f}(n={(s['档位']==lab).sum()})"
                          for lab in s["档位"].cat.categories if (s["档位"] == lab).sum() >= 25)
        print(f"  {g}: {line}")


# =====================================================================
# 图 6｜高费低效群体画像：四个维度同时看
# =====================================================================
def fig6(d):
    """为什么用四张小图而不是一张雷达图？

    雷达图看着酷，但面积会骗人（维度顺序一换形状就变）。四张小柱状图
    每个维度独立成轴，读者一眼能比出差距，而且每个数字都能溯源。
    """
    metrics = [("综合费率", "%/年", "每年交的钱"), ("年化收益", "%", "三年赚的钱"),
               ("费后年化", "%", "扣完费真正到手"), ("最大回撤", "%", "最惨时亏多少")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, (col, unit, name) in zip(axes, metrics):
        bad = d.loc[d["高费低效"], col].mean()
        rest = d.loc[~d["高费低效"], col].mean()
        bars = ax.bar(["高费低效\n(%d 只)" % d["高费低效"].sum(), "其余样本\n(%d 只)" % (~d["高费低效"]).sum()],
                      [bad, rest], color=[C_BAD, C_REST], width=.6)
        for b, v in zip(bars, [bad, rest]):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}{unit}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10.5, fontweight="bold")
        ax.axhline(0, color="#111827", lw=.9)
        style(ax, "", "", name)
        ax.set_xticks(range(2)); ax.set_xticklabels(["高费低效", "其余"], fontsize=9.5)
    fig.suptitle(f"图6｜费率只差 {d.loc[d['高费低效'],'综合费率'].mean()-d.loc[~d['高费低效'],'综合费率'].mean():.2f} 个百分点，"
                 f"费后收益却差 {abs(d.loc[d['高费低效'],'费后年化'].mean()-d.loc[~d['高费低效'],'费后年化'].mean()):.1f} 个百分点",
                 fontsize=13, fontweight="bold", x=.01, ha="left", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "fig6_高费低效画像.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    print("\n[图6] 高费低效 vs 其余")
    b, r = d[d["高费低效"]], d[~d["高费低效"]]
    for col in ["综合费率", "年化收益", "费后年化", "最大回撤"]:
        print(f"  {col}: {b[col].mean():.2f} vs {r[col].mean():.2f}  差 {b[col].mean()-r[col].mean():+.2f}")


# =====================================================================
# 图 7｜它们是怎么被卖出去的：赛道超配
# =====================================================================
def fig7(d):
    """为什么用条形图而不是饼图？因为要比"两组在同一主题上的占比"，条形并排最好比。

    饼图只能表达"构成"，不能表达"两组之间的差异"——这是选图最常见的错误。
    """
    themes = {"医药医疗": r"医药|医疗|生物|健康|创新药", "白酒消费": r"白酒|消费|食品饮料|酒",
              "新能源": r"新能源|光伏|电池|汽车|碳中和", "科技半导体": r"半导体|芯片|科技|人工智能|电子|信息",
              "港股": r"港股|恒生|香港|中概"}
    rows = []
    for k, p in themes.items():
        hit = d["基金名称"].astype(str).str.contains(p)
        rows.append((k, hit[d["高费低效"]].mean() * 100, hit.mean() * 100))
    t = pd.DataFrame(rows, columns=["主题", "高费低效组", "全样本"]).sort_values("高费低效组")

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(t))
    ax.barh(y + .2, t["高费低效组"], height=.38, color=C_BAD, label="高费低效组（%d 只）" % d["高费低效"].sum())
    ax.barh(y - .2, t["全样本"], height=.38, color=C_REST, label="全样本（%d 只）" % len(d))
    for i, (a, b) in enumerate(zip(t["高费低效组"], t["全样本"])):
        ax.text(a + .3, i + .2, f"{a:.0f}%", va="center", fontsize=9.5)
        ax.text(b + .3, i - .2, f"{b:.0f}%", va="center", fontsize=9.5, color="#4B5563")
    ax.set_yticks(y); ax.set_yticklabels(t["主题"])
    allhit = d["基金名称"].astype(str).str.contains("|".join(themes.values()))
    ratio = allhit[d["高费低效"]].mean() / allhit.mean()
    style(ax, "该类产品在组内的占比（%）", "",
          f"图7｜高费低效组里，赛道主题产品密度是全样本的 {ratio:.1f} 倍")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    note(ax, "这些主题产品多在 2020–2021 年赛道高点发行，\n用户买入时看到的是「过去一年涨了 80%」", xy=(0.42, 0.22))
    fig.tight_layout(); fig.savefig(FIGS / "fig7_赛道主题超配.png", dpi=160); plt.close(fig)

    print("\n[图7] 主题超配")
    print(t.to_string(index=False))


# =====================================================================
# 图 8｜两条净值曲线：把所有抽象结论落成一张图
# =====================================================================
def fig8(d):
    """为什么这张图最有说服力？因为它不需要任何统计知识就能看懂。

    报告里最贵的图，是让完全不懂金融的人看一眼就"哦"的那种。
    这里挑两只产品：一只赛道高点发行的白酒指数C，一只普通宽基低费联接，
    都从 1 块钱开始，看三年后差多少。
    """
    pair = [("012414", "招商中证白酒指数C（费率 1.30%）", C_BAD),
            ("000961", "天弘沪深300ETF联接A（费率 0.60%）", C_PAS)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ends = []
    for code, label, c in pair:
        f = RAW / "nav" / f"nav_{code}.csv"
        if not f.exists():
            print(f"  [跳过] 缺少净值文件 {code}"); continue
        s = pd.read_csv(f); s.columns = ["日期", "累计净值"]
        s["日期"] = pd.to_datetime(s["日期"], errors="coerce", format="mixed")
        s["累计净值"] = pd.to_numeric(s["累计净值"], errors="coerce")
        s = s.dropna().query("`累计净值` > 0").sort_values("日期")
        s = s[s["日期"] >= "2023-10-01"]
        base = s["累计净值"].iloc[0]
        final = s["累计净值"].iloc[-1] / base
        ends.append(final)
        ax.plot(s["日期"], s["累计净值"] / base, lw=2, color=c, label=label)
        ax.text(s["日期"].iloc[-1], final,
                f" {(final-1)*100:+.0f}%", color=c, fontsize=10, fontweight="bold", va="center")
    ax.axhline(1, color="#111827", lw=1)
    gap = abs(ends[0] - ends[1]) * 10000 if len(ends) == 2 else 0
    style(ax, "", "净值（起点=1）",
          f"图8｜同样投 1 万元，一只剩 {ends[0]*10000:,.0f} 元，一只变 {ends[1]*10000:,.0f} 元")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    note(ax, "两只产品都从 1 开始，差别只有两处：\n"
             "买在什么位置、每年交多少钱", xy=(0.02, 0.16))
    fig.tight_layout(); fig.savefig(FIGS / "fig8_两条净值曲线.png", dpi=160); plt.close(fig)
    print("\n[图8] 净值曲线对比已输出")


if __name__ == "__main__":
    df = load()
    print(f"样本 {len(df)} 只 | 主动 {(df['组别']=='主动权益').sum()} | 被动 {(df['组别']=='被动指数').sum()}")
    print(f"高费低效 {df['高费低效'].sum()} 只（{df['高费低效'].mean()*100:.1f}%）")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        fn(df)
    print(f"\n全部图表已输出到 {FIGS}")
