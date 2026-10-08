import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/claude/data"
OUT = "/mnt/user-data/outputs/llm_pricing_analysis"
TAB = f"{OUT}/tables"
CH = f"{OUT}/charts"
for d in (OUT, TAB, CH):
    os.makedirs(d, exist_ok=True)

p = pd.read_csv(f"{DATA}/price_snapshots.csv")
s = pd.read_csv(f"{DATA}/scenario_costs.csv")
LATEST = p.snapshot_date.max()
PREV = sorted(p.snapshot_date.unique())[0]
K = ["provider", "model_id", "price_tier"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})
PCOL = {"OpenAI": "#10a37f", "Anthropic": "#d97757", "Google": "#4285f4", "xAI": "#222222",
        "Mistral": "#fa520f", "Groq": "#f55036", "DeepSeek": "#4d6bfe", "Kimi": "#8e44ad", "TypeSafe": "#7f8c8d"}


def save(df, name):
    df.to_csv(f"{TAB}/{name}.csv", index=False)
    return df


#1. Data quality 
dq = pd.DataFrame({
    "column": p.columns,
    "dtype": p.dtypes.astype(str).values,
    "n_missing": p.isna().sum().values,
    "pct_missing": (p.isna().mean() * 100).round(1).values,
    "n_unique": p.nunique().values,
})
save(dq, "01_data_quality")
checks = pd.DataFrame([
    ["rows price_snapshots", len(p)], ["rows scenario_costs", len(s)],
    ["snapshot dates", p.snapshot_date.nunique()],
    ["duplicate key rows (date,provider,model,tier)", int(p.duplicated(["snapshot_date"] + K).sum())],
    ["rows with $0 input price", int((p.input_usd_per_1m == 0).sum())],
    ["rows missing output price (embedding/moderation)", int(p.output_usd_per_1m.isna().sum())],
    ["rows where output < input price", int(((p.output_usd_per_1m < p.input_usd_per_1m)).sum())],
    ["promotional rows", int(p.is_promotional.sum())],
], columns=["check", "value"])
save(checks, "01b_integrity_checks")

#2. Snapshot diff
a = p[p.snapshot_date == PREV].set_index(K)
b = p[p.snapshot_date == LATEST].set_index(K)
added = b.loc[b.index.difference(a.index)].reset_index()
removed = a.loc[a.index.difference(b.index)].reset_index()
common = a.index.intersection(b.index)
price_cols = ["input_usd_per_1m", "output_usd_per_1m", "cached_input_usd_per_1m"]
changed = [c for c in price_cols if (b.loc[common, c].fillna(-1) != a.loc[common, c].fillna(-1)).any()]
diff = pd.concat([
    added.assign(change="added")[["change"] + K + ["input_usd_per_1m", "output_usd_per_1m"]],
    removed.assign(change="removed")[["change"] + K + ["input_usd_per_1m", "output_usd_per_1m"]],
])
save(diff, "02_snapshot_diff")
diff_summary = pd.DataFrame([
    ["rows in " + PREV, len(a)], ["rows in " + LATEST, len(b)],
    ["rows added", len(added)], ["rows removed", len(removed)],
    ["shared rows", len(common)], ["shared rows with a price change", len(changed) and 1 or 0],
    ["new model_ids", ", ".join(sorted(set(added.model_id) - set(removed.model_id)))],
], columns=["metric", "value"])
diff_summary.loc[diff_summary.metric == "shared rows with a price change", "value"] = int(
    sum((b.loc[common, c].fillna(-1) != a.loc[common, c].fillna(-1)).sum() for c in price_cols))
save(diff_summary, "02b_diff_summary")

#3. Provider summary (latest) 
cur = p[p.snapshot_date == LATEST].copy()
tok = cur[(cur.modality == "text") & cur.model_family.isin(["chat", "reasoning"]) & (~cur.is_alias)]
std = tok[(tok.price_tier == "standard") | ((tok.provider == "DeepSeek") & (tok.price_tier == "off_peak"))].copy()  # DeepSeek has no standard tier: off_peak used as the base rate
std["blended_3to1"] = (3 * std.input_usd_per_1m + std.output_usd_per_1m) / 4
std["out_in_ratio"] = std.output_usd_per_1m / std.input_usd_per_1m
prov = cur.groupby("provider").agg(rows=("model_id", "size"), models=("model_id", "nunique"),
                                    tiers=("price_tier", "nunique")).reset_index()
ps = std.groupby("provider").agg(
    std_models=("model_id", "nunique"),
    min_input=("input_usd_per_1m", "min"), median_input=("input_usd_per_1m", "median"),
    max_input=("input_usd_per_1m", "max"),
    min_output=("output_usd_per_1m", "min"), median_output=("output_usd_per_1m", "median"),
    max_output=("output_usd_per_1m", "max"),
    median_blended_3to1=("blended_3to1", "median"),
    median_out_in_ratio=("out_in_ratio", "median")).reset_index()
prov = prov.merge(ps, on="provider", how="left").round(3)
save(prov, "03_provider_summary")

#4. Leaderboards
lb = std.sort_values("blended_3to1")[["provider", "model_id", "input_usd_per_1m", "output_usd_per_1m",
                                       "cached_input_usd_per_1m", "blended_3to1", "out_in_ratio"]].round(4)
lb.insert(0, "rank_cheapest", range(1, len(lb) + 1))
save(lb, "04_standard_price_leaderboard")
top_exp = lb.sort_values("blended_3to1", ascending=False).head(15)
save(top_exp, "04b_most_expensive_15")

#5. Tier multipliers vs standard 
base = tok[tok.price_tier == "standard"].set_index(["provider", "model_id"])
rows = []
for t in sorted(tok.price_tier.unique()):
    if t in ("standard", "peak", "off_peak"):
        continue
    x = tok[tok.price_tier == t].set_index(["provider", "model_id"])
    j = x.join(base[["input_usd_per_1m", "output_usd_per_1m"]], rsuffix="_std", how="inner")
    if j.empty:
        continue
    j["in_mult"] = j.input_usd_per_1m / j.input_usd_per_1m_std
    j["out_mult"] = j.output_usd_per_1m / j.output_usd_per_1m_std
    j = j.reset_index()
    rows.append(j.assign(tier=t)[["provider", "model_id", "tier", "in_mult", "out_mult"]])
tm = pd.concat(rows)
save(tm.round(3), "05a_tier_multipliers_detail")
tms = tm.groupby(["provider", "tier"]).agg(n=("model_id", "nunique"), in_mult_median=("in_mult", "median"),
                                           out_mult_median=("out_mult", "median")).round(3).reset_index()
save(tms, "05_tier_multipliers_vs_standard")

#6. Cached-input discount 
cd = std.dropna(subset=["cached_input_usd_per_1m"]).copy()
cd["cache_pct_of_input"] = cd.cached_input_usd_per_1m / cd.input_usd_per_1m * 100
cds = cd.groupby("provider").agg(n=("model_id", "nunique"), median_cache_pct=("cache_pct_of_input", "median"),
                                 min_pct=("cache_pct_of_input", "min"), max_pct=("cache_pct_of_input", "max")).round(1).reset_index()
cds["median_saving_pct"] = (100 - cds.median_cache_pct).round(1)
save(cds, "06_cached_input_discount")

#7. Long-context premium
lc_pairs = [("standard_long_context", "standard"), ("long_context", "standard"),
            ("batch_long_context", "batch"), ("flex_long_context", "flex"),
            ("fast_long_context", "fast"), ("priority_long_context", "priority")]
lr = []
for lt, bt in lc_pairs:
    x = cur[cur.price_tier == lt].set_index(["provider", "model_id"])
    y = cur[cur.price_tier == bt].set_index(["provider", "model_id"])
    j = x.join(y[["input_usd_per_1m", "output_usd_per_1m"]], rsuffix="_base", how="inner").reset_index()
    if j.empty:
        continue
    j["in_prem"] = j.input_usd_per_1m / j.input_usd_per_1m_base
    j["out_prem"] = j.output_usd_per_1m / j.output_usd_per_1m_base
    lr.append(j.assign(long_tier=lt, base_tier=bt)[["provider", "model_id", "long_tier", "base_tier", "in_prem", "out_prem",
                                                  "context_threshold_tokens"]])
lc = pd.concat(lr).round(3)
save(lc, "07a_long_context_detail")
lcs = lc.groupby(["provider", "long_tier"]).agg(n=("model_id", "nunique"), in_prem_median=("in_prem", "median"),
                                                out_prem_median=("out_prem", "median"),
                                                threshold=("context_threshold_tokens", "median")).round(3).reset_index()
save(lcs, "07_long_context_premium")

#8. Scenario costs 
sc = s[s.snapshot_date == LATEST]
sc_sum = sc.groupby(["scenario_id", "provider"]).total_usd.agg(["count", "min", "median", "max"]).round(3).reset_index()
save(sc_sum, "08_scenario_cost_by_provider")
cheap = (sc.sort_values("total_usd").groupby("scenario_id").head(5)
         .sort_values(["scenario_id", "total_usd"])[["scenario_id", "provider", "model_id", "price_tier_used", "total_usd"]])
save(cheap, "08b_cheapest5_per_scenario")
dear = (sc.sort_values("total_usd", ascending=False).groupby("scenario_id").head(5)
        .sort_values(["scenario_id", "total_usd"], ascending=[True, False])[["scenario_id", "provider", "model_id", "price_tier_used", "total_usd"]])
save(dear, "08c_priciest5_per_scenario")
spread = sc.groupby("scenario_id").total_usd.agg(cheapest="min", priciest="max", median="median").round(3)
spread["max_to_min_x"] = (spread.priciest / spread.cheapest.replace(0, np.nan)).round(0)
save(spread.reset_index(), "08d_scenario_spread")

#9. Generation / family trends (Anthropic, OpenAI) 
fam = std[std.provider.isin(["Anthropic", "OpenAI"])][["provider", "model_id", "input_usd_per_1m", "output_usd_per_1m"]]
save(fam.sort_values(["provider", "output_usd_per_1m"]), "09_anthropic_openai_standard_prices")

#10. Peak vs off-peak (DeepSeek) 
ds = cur[cur.provider == "DeepSeek"].pivot_table(index="model_id", columns="price_tier",
                                                  values=["input_usd_per_1m", "output_usd_per_1m"])
ds.columns = [f"{a}_{b}" for a, b in ds.columns]
ds = ds.reset_index()
ds["peak_premium_x"] = (ds.input_usd_per_1m_peak / ds.input_usd_per_1m_off_peak).round(2)
save(ds, "10_deepseek_peak_vs_offpeak")

#11. Promotions 
promo = cur[cur.is_promotional & (cur.price_tier == "standard")][["provider", "model_id", "input_usd_per_1m",
                                                                 "output_usd_per_1m", "notes"]]
save(promo, "11_promotional_pricing")

#12. Embeddings
emb = cur[cur.model_family == "embedding"][["provider", "model_id", "price_tier", "input_usd_per_1m"]].sort_values("input_usd_per_1m")
save(emb, "12_embedding_prices")

# CHARTS 
# 1 provider median blended
fig, ax = plt.subplots(figsize=(7, 4))
d = prov.dropna(subset=["median_blended_3to1"]).sort_values("median_blended_3to1")
ax.barh(d.provider, d.median_blended_3to1, color=[PCOL[x] for x in d.provider])
ax.set_xscale("log"); ax.set_xlabel("Median blended price, USD / 1M tokens (3:1 in:out, log scale)")
ax.set_title(f"Median standard-tier price by provider ({LATEST})")
for i, v in enumerate(d.median_blended_3to1): ax.text(v * 1.05, i, f"${v:.2f}", va="center")
plt.tight_layout(); plt.savefig(f"{CH}/01_provider_median_price.png"); plt.close()

# 2 input vs output scatter
fig, ax = plt.subplots(figsize=(7, 5))
for pr, g in std.groupby("provider"):
    ax.scatter(g.input_usd_per_1m, g.output_usd_per_1m, s=28, alpha=.8, label=pr, color=PCOL[pr])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Input USD / 1M"); ax.set_ylabel("Output USD / 1M")
ax.set_title("Input vs output price, standard tier (log-log)")
lim = [std.input_usd_per_1m.min(), std.output_usd_per_1m.max()]
ax.plot([lim[0], lim[1] / 5], [lim[0] * 5, lim[1]], "k--", lw=.7, alpha=.5)
ax.text(lim[0] * 1.2, lim[0] * 7, "output = 5x input", fontsize=8, alpha=.7)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(f"{CH}/02_input_vs_output_scatter.png"); plt.close()

# 3 cheapest / priciest 15
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
c15 = lb.head(15).iloc[::-1]; e15 = top_exp.iloc[::-1]
axs[0].barh(c15.model_id, c15.blended_3to1, color=[PCOL[x] for x in c15.provider]); axs[0].set_title("15 cheapest (blended USD/1M)")
axs[1].barh(e15.model_id, e15.blended_3to1, color=[PCOL[x] for x in e15.provider]); axs[1].set_title("15 most expensive (blended USD/1M)")
plt.tight_layout(); plt.savefig(f"{CH}/03_cheapest_vs_priciest.png"); plt.close()

# 4 tier multipliers
t4 = tms[tms.tier.isin(["batch", "flex", "fast", "priority"])]
piv = t4.pivot(index="provider", columns="tier", values="out_mult_median")
ax = piv.plot(kind="bar", figsize=(8, 4), width=.8)
ax.axhline(1, color="k", lw=.8); ax.set_ylabel("Median output price multiple of standard")
ax.set_title("Service-tier price multiples vs standard"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{CH}/04_tier_multipliers.png"); plt.close()

# 5 cache discount
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(cds.provider, cds.median_saving_pct, color=[PCOL[x] for x in cds.provider])
ax.set_ylabel("Median saving on cached input (%)"); ax.set_title("Prompt-cache discount vs fresh input")
for i, v in enumerate(cds.median_saving_pct): ax.text(i, v + 1, f"{v:.0f}%", ha="center")
ax.set_ylim(0, 105)
plt.tight_layout(); plt.savefig(f"{CH}/05_cache_discount.png"); plt.close()

# 6 long-context premium
l6 = lcs[lcs.long_tier.isin(["standard_long_context", "long_context"])]
fig, ax = plt.subplots(figsize=(7, 3.8))
lab = l6.provider + "\n(" + l6.long_tier.str.replace("_", " ") + ")"
x = np.arange(len(l6))
ax.bar(x - .2, l6.in_prem_median, .4, label="Input"); ax.bar(x + .2, l6.out_prem_median, .4, label="Output")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.axhline(1, color="k", lw=.8)
ax.set_ylabel("Multiple of base price"); ax.set_title("Long-context price premium"); ax.legend()
plt.tight_layout(); plt.savefig(f"{CH}/06_long_context_premium.png"); plt.close()

# 7 scenario boxplot (log)
order = ["chat", "coding_agent", "cached_agent", "long_context_analysis", "batch_extraction"]
fig, ax = plt.subplots(figsize=(8, 4.5))
data = [sc[sc.scenario_id == o].total_usd.clip(lower=1e-4) for o in order]
ax.boxplot(data, tick_labels=order, showfliers=True); ax.set_yscale("log")
ax.set_ylabel("Cost per run, USD (log)"); ax.set_title("Scenario cost distribution across models")
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(f"{CH}/07_scenario_cost_boxplot.png"); plt.close()

# 8 Anthropic vs OpenAI standard
fig, ax = plt.subplots(figsize=(8, 5))
for pr in ["Anthropic", "OpenAI"]:
    g = fam[fam.provider == pr]
    ax.scatter(g.input_usd_per_1m, g.output_usd_per_1m, label=pr, color=PCOL[pr], s=35)
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend()
ax.set_xlabel("Input USD/1M"); ax.set_ylabel("Output USD/1M"); ax.set_title("Anthropic vs OpenAI price positions (standard)")
plt.tight_layout(); plt.savefig(f"{CH}/08_anthropic_vs_openai.png"); plt.close()

# 9 snapshot change
fig, ax = plt.subplots(figsize=(7, 3.8))
cnt = p.groupby(["provider", "snapshot_date"]).size().unstack()
cnt.plot(kind="bar", ax=ax); ax.set_ylabel("Price rows"); ax.set_title("Rows per provider by snapshot"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{CH}/09_rows_per_snapshot.png"); plt.close()

#  EXCEL 
with pd.ExcelWriter(f"{OUT}/llm_pricing_analysis.xlsx", engine="openpyxl") as xw:
    for f in sorted(os.listdir(TAB)):
        df = pd.read_csv(f"{TAB}/{f}")
        df.to_excel(xw, sheet_name=f[:-4][:31], index=False)
        ws = xw.sheets[f[:-4][:31]]
        ws.freeze_panes = "A2"
        for i, col in enumerate(df.columns, 1):
            w = min(max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) + 2, 45)
            ws.column_dimensions[ws.cell(1, i).column_letter].width = w

# stash key numbers for report
import json
key = {
    "latest": LATEST, "prev": PREV,
    "n_models_latest": int(cur.model_id.nunique()),
    "n_std": int(len(std)),
}
json.dump(key, open("/home/claude/key.json", "w"))
print("done")
for n in ["01b_integrity_checks", "02b_diff_summary", "03_provider_summary", "05_tier_multipliers_vs_standard",
          "06_cached_input_discount", "07_long_context_premium", "08d_scenario_spread", "10_deepseek_peak_vs_offpeak"]:
    print("\n==", n); print(pd.read_csv(f"{TAB}/{n}.csv").to_string())
print("\n== cheapest 10"); print(lb.head(10).to_string())
print("\n== priciest 10"); print(top_exp.head(10).to_string())
print("\n== cheapest5 per scenario"); print(cheap.to_string())

