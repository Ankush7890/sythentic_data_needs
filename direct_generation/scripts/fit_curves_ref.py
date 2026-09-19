"""Fitted learning curves and variance decomposition (2026-09-17).

Every learning curve {n: [AUROC per draw]} is fit with a four-parameter log-logistic
    A(n) = L + (U - L) / (1 + (n / m) ** (-k))
L = floor (AUROC at n -> 0), U = ceiling (level), m = half-gain size (the n at which half of U - L is in hand),
k = steepness in log n.  n_90 = m * 9 ** (1/k) is the size at which 90% of the gain is in hand.
Fit: grid over (m, k), exact least squares for (L, U) on every individual draw, constraints 0.5 <= L <= U <= 1.
Bootstrap: draws resampled within each size, refit, percentile intervals.
Variance decomposition: sequential (type I) sums of squares of log10 m and of U on the factors of each design.

Called from make_tables.py with the curves it has already loaded; writes
  tables/fit_decomp.tex   variance decomposition table
  tables/fit_splits.tex   per-split summary of U and m (appendix)
  figures/fit_scatter.tex the level-vs-bend scatter (main text)
  data/fits.csv           every fitted arm with bootstrap intervals
"""
import math, csv, statistics as st
from collections import defaultdict
import numpy as np

M_GRID = np.exp(np.linspace(math.log(3.0), math.log(5000.0), 90))
K_GRID = np.exp(np.linspace(math.log(0.25), math.log(8.0), 36))
MM, KK = np.meshgrid(M_GRID, K_GRID, indexing="ij")
MM = MM.ravel(); KK = KK.ravel()          # (G,)
L_MIN = 0.5                               # floor: a random direction scores 0.5 in expectation
B = 200                                   # bootstrap replicates
M_MIN = 10.0                              # smallest measured size: a half-gain size below it is reported as "<10" and clamped for the decomposition
RNG = np.random.default_rng(20260917)


def _points(curve):
    ns, ys = [], []
    for n, v in curve.items():
        for y in v: ns.append(n); ys.append(y)
    return np.array(ns, float), np.array(ys, float)


def fit(ns, ys):
    """-> dict(L, U, m, k, rmse). Exact LS for (L, U) at every (m, k) grid point, constrained 0.5 <= L <= U <= 1."""
    F = 1.0 / (1.0 + (ns[None, :] / MM[:, None]) ** (-KK[:, None]))   # (G, N) logistic feature
    N = len(ys); Sy = ys.sum(); Sf = F.sum(1); Sff = (F * F).sum(1); Sfy = F @ ys; Syy = ys @ ys
    det = N * Sff - Sf * Sf
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (N * Sfy - Sf * Sy) / det; a = (Sy - c * Sf) / N        # y = a + c f  (a = L, c = U - L)
    a = np.where(np.isfinite(a), a, ys.mean()); c = np.where(np.isfinite(c), c, 0.0)
    # constraints: c >= 0 ; a >= L_MIN ; a + c <= 1
    c = np.maximum(c, 0.0)
    # if a + c > 1: refit with U = 1 -> y - f = a (1 - f)
    over = a + c > 1.0
    if over.any():
        g = 1.0 - F[over]; a_o = (g * (ys[None, :] - F[over])).sum(1) / (g * g).sum(1)
        a_o = np.clip(a_o, L_MIN, 1.0); a[over] = a_o; c[over] = 1.0 - a_o
    low = a < L_MIN
    if low.any():
        a[low] = L_MIN; c[low] = np.clip((F[low] @ (ys - L_MIN)) / Sff[low], 0.0, 1.0 - L_MIN)
    pred = a[:, None] + c[:, None] * F
    sse = ((pred - ys[None, :]) ** 2).sum(1)
    i = int(np.argmin(sse))
    return dict(L=float(a[i]), U=float(a[i] + c[i]), m=float(MM[i]), k=float(KK[i]), rmse=math.sqrt(sse[i] / N))


def n90(f): return f["m"] * 9.0 ** (1.0 / f["k"])


def fit_curve(curve, boot=True):
    ns, ys = _points(curve)
    f = fit(ns, ys); f["gain"] = f["U"] - f["L"]
    A = lambda n: f["L"] + (f["U"] - f["L"]) / (1.0 + (n / f["m"]) ** (-f["k"]))
    f["gain_obs"] = A(ns.max()) - A(ns.min())   # fitted gain inside the measured range (flat curves: < 0.02)
    f["n90"] = n90(f)
    if boot:
        Us, ms, n90s = [], [], []
        for _ in range(B):
            c = {n: [v[i] for i in RNG.integers(0, len(v), len(v))] for n, v in curve.items()}
            bn, by = _points(c); g = fit(bn, by)
            Us.append(g["U"]); ms.append(g["m"]); n90s.append(n90(g))
        f["U_lo"], f["U_hi"] = np.percentile(Us, [2.5, 97.5]); f["U_bvar"] = float(np.var(Us)); f["lm_bvar"] = float(np.var(np.log10(np.maximum(ms, M_MIN))))
        f["m_lo"], f["m_hi"] = np.percentile(ms, [2.5, 97.5])
        f["n90_lo"], f["n90_hi"] = np.percentile(n90s, [2.5, 97.5])
    return f


def anova(rows, y, order):
    """Sequential (type I) fraction of the total sum of squares of rows[i][y] explained by each factor in `order`."""
    yv = np.array([r[y] for r in rows], float); yv = yv - yv.mean(); sst = float(yv @ yv)
    cols = [np.ones((len(rows), 1))]; out = {}; prev = 0.0
    for f in order:
        lv = sorted(set(r[f] for r in rows))
        cols.append(np.array([[1.0 if r[f] == l else 0.0 for l in lv[1:]] for r in rows]))
        X = np.concatenate(cols, 1); coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        ssr = sst - float(((yv - X @ coef) ** 2).sum()); out[f] = (ssr - prev) / sst; prev = ssr
    out["resid"] = 1.0 - prev / sst
    return out


def eta_range(rows, y, factors):
    """For each factor: the fraction of SS it explains when entered first and when entered last."""
    res = {}
    for f in factors:
        first = anova(rows, y, [f])[f]
        last = anova(rows, y, [g for g in factors if g != f] + [f])[f]
        res[f] = (first, last)
    return res


CSV_COLS = ["model", "concept", "split", "recipe", "variant", "gen", "sizes", "L", "U", "U_lo", "U_hi", "m", "m_lo", "m_hi", "k", "n90", "n90_lo", "n90_hi", "gain_obs", "rmse", "U_bvar", "lm_bvar"]

def fit_all(arms):
    """arms: list of dict(model, concept, split, recipe, gen, variant, curve). Fits every curve with >= 5 sizes and writes data/fits.csv."""
    rows = []
    for d in arms:
        if len(d["curve"]) < 5: continue
        f = fit_curve(d["curve"]); rows.append({**d, **f, "sizes": " ".join(str(n) for n in sorted(d["curve"]))})
    with open("data/fits.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(CSV_COLS)
        for r in rows:
            w.writerow([r["model"], r["concept"], r["split"], r["recipe"], r["variant"], r["gen"], r["sizes"]] +
                       [f"{r[c]:.4f}" for c in ("L", "U", "U_lo", "U_hi")] + [f"{r[c]:.1f}" for c in ("m", "m_lo", "m_hi")] + [f"{r['k']:.3f}"] + [f"{r[c]:.1f}" for c in ("n90", "n90_lo", "n90_hi")] + [f"{r['gain_obs']:.4f}", f"{r['rmse']:.4f}", f"{r['U_bvar']:.6f}", f"{r['lm_bvar']:.6f}"])
    return rows

def load_rows():
    rows = []
    for r in csv.DictReader(open("data/fits.csv")):
        rows.append({**{k: r[k] for k in ("model", "concept", "split", "recipe", "variant", "gen", "sizes")}, **{k: float(r[k]) for k in CSV_COLS[7:]}})
    return rows

def run(rows, SPLN, CONCEPTS, OUT, FIG):
    """Decomposes the fitted rows and writes the tables and the figure."""
    for r in rows:
        r["nsizes"] = len(r["sizes"].split()); r["lm"] = math.log10(max(r["m"], M_MIN)); r["ln90"] = math.log10(min(r["n90"], 1e6))
    # ---- variance decomposition
    flat = lambda r: r["gain_obs"] < 0.02
    fac = [r for r in rows if r["recipe"] in ("general", "detailed") and not flat(r)]
    for r in fac: r["prompt"] = r["recipe"]
    pvr = [r for r in rows if r["recipe"] == "llm" and r["nsizes"] == 7 and not flat(r)]
    gm = [r for r in rows if r["model"] == "gemma27b" and not flat(r)]
    for r in gm: r["rv"] = r["recipe"] + ":" + r["variant"]
    def noise(rs, y):  # fraction of the total variance that is bootstrap (draw) noise
        tot = float(np.var([r[y] for r in rs])); return float(np.mean([r[y + "_bvar"] for r in rs])) / tot if tot > 0 else 0.0
    D = {"n": {"fac": len(fac), "pv": len(pvr), "gm": len(gm), "all": len(rows), "flat": sum(flat(r) for r in rows)}}
    D["fac"] = {y: eta_range(fac, y, ["split", "model", "gen", "prompt"]) for y in ("lm", "U")}
    D["gm"] = {y: eta_range(gm, y, ["split", "rv", "gen"]) for y in ("lm", "U")}
    D["pv"] = {y: {"split": (anova(pvr, y, ["split"])["split"],) * 2} for y in ("lm", "U")}
    D["resid"] = {k: {y: anova(rs, y, fs)["resid"] for y in ("lm", "U")} for k, rs, fs in (("fac", fac, ["split", "model", "gen", "prompt"]), ("gm", gm, ["split", "rv", "gen"]), ("pv", pvr, ["split"]))}
    D["noise"] = {k: {y: noise(rs, y) for y in ("lm", "U")} for k, rs in (("fac", fac), ("gm", gm), ("pv", pvr))}
    D["concept"] = {y: (anova(fac, y, ["concept"])["concept"], anova(fac, y, ["concept", "split"])["split"]) for y in ("lm", "U")}
    for r in fac: r["sp"] = r["split"] + "|" + r["prompt"]
    D["interact"] = {y: anova(fac, y, ["split", "model", "gen", "prompt", "sp"])["sp"] for y in ("lm", "U")}  # split x prompt interaction
    lines = []
    FN = {"split": "split", "model": "probe model", "gen": "generator", "prompt": "prompt (general / detailed)", "rv": "recipe (general, detailed, LLM-written, targeted)"}
    def block(title, key, factors):
        lines.append(f"\\emph{{{title}}} ({D['n'][key]} curves) & & & & \\\\")
        for f in factors:
            a1, a2 = D[key]["lm"][f]; b1, b2 = D[key]["U"][f]
            lines.append(f"\\quad {FN[f]} & {a1:.2f} & {a2:.2f} & {b1:.2f} & {b2:.2f} \\\\")
        lines.append(f"\\quad residual (interactions and noise) & \\multicolumn{{2}}{{c}}{{{D['resid'][key]['lm']:.2f}}} & \\multicolumn{{2}}{{c}}{{{D['resid'][key]['U']:.2f}}} \\\\")
        lines.append(f"\\quad \\quad of which draw noise (bootstrap) & \\multicolumn{{2}}{{c}}{{{D['noise'][key]['lm']:.2f}}} & \\multicolumn{{2}}{{c}}{{{D['noise'][key]['U']:.2f}}} \\\\")
    block("Four probe models $\\times$ 14 splits $\\times$ four generators $\\times$ two prompts", "fac", ["split", "model", "gen", "prompt"])
    block("Gemma-3-27B-IT, every recipe", "gm", ["split", "rv", "gen"])
    block(f"Five LLM-written prompts $\\times$ {['four', 'five', 'six'][len(set(r['split'] for r in pvr)) - 4]} splits (Gemma-3-27B-IT)", "pv", ["split"])
    (OUT / "fit_decomp.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    # ---- per-split appendix table: Gemma, general / detailed (range over generators), LLM prompts (range over prompts), targeted
    lines = []
    def rng_(vals, fmt): return (fmt(min(vals)) + "--" + fmt(max(vals))) if len(vals) > 1 else fmt(vals[0])
    fU = lambda v: f"{v:.2f}"; fm = lambda v: ("$<$10" if v < M_MIN else f"{v:.0f}")
    for key, cname, _, _ in CONCEPTS:
        lines.append(f"\\emph{{{cname}}}" + " &" * 8 + " \\\\")
        splits = [s for s in dict.fromkeys(r["split"] for r in rows if r["concept"] == key and r["model"] == "gemma27b")]
        for sp in splits:
            cells = []
            for rec in ("general", "detailed", "llm", "targeted"):
                rs = [r for r in rows if r["model"] == "gemma27b" and r["split"] == sp and r["recipe"] == rec]
                if not rs: cells += ["--", "--"]; continue
                ms_ = sorted(max(r["m"], M_MIN) for r in rs if not flat(r))
                cells += [rng_([r["U"] for r in rs], fU), (rng_(ms_, fm) if ms_ else "flat")]
            lines.append(f"\\quad {SPLN.get(sp, sp)} & " + " & ".join(cells) + " \\\\")
    (OUT / "fit_splits.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    # ---- figure: level (U) vs half-gain size (m), Gemma, three panels, colour = split, marker = recipe
    SPLIT_STYLE = ["blue", "red", "green!60!black", "orange", "violet", "teal"]
    MARK = {"general": "o", "detailed": "*", "llm": "triangle*", "targeted": "square*"}
    fig = ["\\begin{tikzpicture}",
           "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.9cm}, width=0.36\\linewidth, height=3.7cm,",
           "  xmode=log, log basis x=10, xmin=7, xmax=6000, xtick={10,30,100,300,1000,3000}, xticklabels={$\\le$10,30,100,300,1000,3000}, ymin=0.5, ymax=1.0, grid=major,",
           "  xlabel={half-gain size $m$ (samples)}, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, title style={font=\\small},",
           "  legend style={font=\\tiny, draw=none, row sep=-2pt}, legend columns=2, error bars/x dir=both, error bars/x explicit, error bars/error bar style={opacity=0.22, very thin}]"]
    for i, (key, cname, _, _) in enumerate(CONCEPTS):
        fig.append(f"\\nextgroupplot[title={{{cname}}}, legend to name=fitlegend{i}" + (", ylabel={level $U$ (fitted ceiling)}" if i == 0 else "") + "]")
        splits = [s for s in dict.fromkeys(r["split"] for r in rows if r["concept"] == key and r["model"] == "gemma27b")]
        for j, sp in enumerate(splits):
            col = SPLIT_STYLE[j % len(SPLIT_STYLE)]
            for rec in ("general", "detailed", "llm", "targeted"):
                rs = [r for r in rows if r["model"] == "gemma27b" and r["split"] == sp and r["recipe"] == rec and not flat(r)]
                if not rs: continue
                def pt(r):  # bars only where the interval is informative (spans less than 100x)
                    m, lo, hi = (max(r[c], M_MIN) for c in ("m", "m_lo", "m_hi"))
                    return f"({m:.1f},{r['U']:.4f})" + (f" -= ({m-lo:.1f},0) += ({hi-m:.1f},0)" if hi / lo < 100 else " -= (0,0) += (0,0)")
                pts = " ".join(pt(r) for r in rs)
                fill = "fill=" + col if rec != "general" else "fill=white"
                fig.append(f"\\addplot[only marks, color={col}, mark={MARK[rec]}, mark size=1.7pt, mark options={{{fill}, draw={col}}}, forget plot] coordinates {{{pts}}};")
            # one legend entry per split (detailed marker)
            fig.append(f"\\addlegendimage{{only marks, color={col}, mark=*, mark size=1.7pt}}\\addlegendentry{{{{{SPLN.get(sp, sp)}}}}}")
    fig += ["\\end{groupplot}"]
    fig += [f"\\node[anchor=north] at ($(group c{i+1}r1.south)+(0,-0.75cm)$) {{\\pgfplotslegendfromname{{fitlegend{i}}}}};" for i in range(3)]
    fig += ["\\end{tikzpicture}"]
    (FIG / "fit_scatter.tex").write_text("\n".join(fig) + "\n")
    return rows, D


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    J = json.load(open("data/curves.json"))
    arms = [{**a, "curve": {int(n): v for n, v in a["curve"].items()}} for a in J["arms"]]
    if len(sys.argv) > 1 and sys.argv[1].isdigit(): B = int(sys.argv[1])
    rows = load_rows() if "--from-csv" in sys.argv else fit_all(arms)   # --from-csv: redraw tables and figure from data/fits.csv without refitting
    rows, D = run(rows, J["SPLN"], J["CONCEPTS"], Path("tables"), Path("figures"))
    print(json.dumps({k: v for k, v in D.items()}, indent=1, default=float))
