"""Generate the paper's LaTeX tables from cached results."""
import json
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "tables"
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")
NAMES = {"noop": "No-op", "retrain2": "Retrain-2", "finetune": "Fine-tune",
         "neggrad": "NegGrad", "scrub": "SCRUB", "ssd": "SSD"}


def load(path):
    return json.loads((RESULTS / path).read_text())


def num(x):
    return f"{x:.4g}"


def pval(x):
    return r"$<0.001$" if x < .001 else f"{x:.3g}"


def esc(text):
    return str(text).replace("_", r"\_")


def table(caption, label, columns, header, rows, size=r"\small"):
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
{size}
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{{columns}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def settings():
    rows = [
        ("Synthetic suite", "MLP", "10", "Benchmark-specific structural subset", "All class 0 (500)"),
        ("MNIST random", "SmallCNN", "10", r"5\% random (3,000)", "600"),
        ("MNIST class", "SmallCNN", "10", "Class 9 (5,949)", "600"),
        ("MNIST targeted", "SmallCNN", "10", "Cycle-support targeted (3,000)", "600"),
        ("MNIST matched", "SmallCNN", "10", "Class-matched random (3,000)", "600"),
        ("FashionMNIST random", "SmallCNN", "10", r"5\% random (3,000)", "600"),
        ("FashionMNIST class", "SmallCNN", "10", "Class 9 (6,000)", "600"),
        ("SVHN random", "ResNet-18", "10", r"5\% random (3,662)", "600"),
        ("SVHN class", "ResNet-18", "10", "Class 9 (4,659)", "600"),
        ("CIFAR-10 random 1", "ResNet-18", "10", r"1\% random (500)", "600"),
        ("CIFAR-10 random 5", "ResNet-18", "10", r"5\% random (2,500)", "600"),
        ("CIFAR-10 random 10", "ResNet-18", "10", r"10\% random (5,000)", "600"),
        ("CIFAR-10 class", "ResNet-18", "10", "Class 9 (5,000)", "600"),
        ("CIFAR-10 targeted", "ResNet-18", "10", "Cycle-support targeted (2,500)", "600"),
        ("CIFAR-10 matched", "ResNet-18", "10", "Class-matched random (2,500)", "600"),
        ("CIFAR-10 pretrained", "ResNet-18 (ImageNet init.)", "10", r"5\% random (2,500)", "600"),
        ("CIFAR-100 class", "ResNet-18", "10", "Fine class 30 (500)", "600"),
    ]
    return table("Experimental settings.", "settings", "lllll",
                 r"Run & Model & \# seeds & Forget set & Probe size",
                 [" & ".join(row) + r" \\" for row in rows], r"\scriptsize")


def main_results():
    metrics, mia = load("m4_random/metrics.json"), load("m4_random/mia.json")
    # TRR/alpha/eta from the frozen-grid analysis (penultimate H1 cell)
    frozen = load("m4_random/analysis.json")["penultimate_H1"]["methods"]
    rows = []
    for method in METHODS:
        a = metrics[method]["acc"]
        m = frozen.get(method, {"TRR": 1.0, "alpha": 0.0, "eta": 0.0})
        auc = fmean(mia["original" if method == "noop" else method])
        rows.append(" & ".join([NAMES[method], *(num(a[k]) for k in ("retain", "forget", "test")),
                                num(auc), *(num(m[k]) for k in ("TRR", "alpha", "eta"))]) + r" \\")
    return table(r"CIFAR-10 random 5\% deletion results (means over seeds).", "main", "lrrrrrrr",
                 r"Method & Retain acc. & Forget acc. & Test acc. & MIA AUC & TRR & $\alpha$ & $\eta$", rows)


def equivalence():
    stats = load("m4_random/formal_stats.json")
    eq, bh = stats["equivalence"], stats["bh_fdr"]["cells"]
    cells = (("penultimate_H0", r"Penultimate $H_0$"),
             ("penultimate_H1", r"Penultimate $H_1$"),
             ("logits_H0", r"Logits $H_0$"), ("logits_H1", r"Logits $H_1$"))
    rows = []
    for method in METHODS:
        for i, (key, cell) in enumerate(cells):
            value, test = eq[key]["methods"][method], bh[key][method]
            q = pval(test["q"]) + (r"$^*$" if test["reject"] else "")
            ci = f"[{num(value['CI'][0])}, {num(value['CI'][1])}]"
            rows.append(" & ".join((NAMES[method] if i == 0 else "", cell, num(value["D_UR"]),
                                    ci, num(eq[key]["delta"]), q, esc(value["decision"]))) + r" \\")
        if method != METHODS[-1]:
            rows.append(r"\addlinespace")
    return table("Formal equivalence tests; stars mark BH-significant entries.", "equivalence", "llrrrrl",
                 r"Method & Cell & $D_{UR}$ & CI & $\delta$ & BH $q$ & Decision", rows, r"\scriptsize")


def ablation():
    data = load("m4_random/ablations.json")
    titles = {"distance": "Distance", "vectorization": "Vectorization",
              "probe_subset": "Probe subset", "point_count": "Point count"}
    rows = []
    for family in titles:
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{titles[family]}}}}} \\")
        for variant, value in data[family].items():
            rows.append(" & ".join((esc(variant), num(value["I_topo"]), pval(value["p"]),
                                    num(value["TRR"]["retrain2"]), num(value["TRR"]["scrub"]))) + r" \\")
        rows.append(r"\addlinespace")
    oracle = data["oracle"]["single_retrain"]
    rows.append(r"\multicolumn{5}{l}{\emph{Single-oracle sensitivity}} \\")
    for bound in ("min", "max"):
        rows.append(" & ".join((bound.capitalize(), "--", "--", num(oracle["retrain2"][bound]),
                                num(oracle["scrub"][bound]))) + r" \\")
    return table("Ablations and single-oracle sensitivity.", "ablation", "lrrrr",
                 r"Variant & $I_{\mathrm{topo}}$ & $p$ & TRR (Retrain-2) & TRR (SCRUB)", rows)


def operational():
    distinguish = load("m4_random/distinguisher.json")
    relearn = load("m4_class/relearn.json")
    correlations = load("m4_random/formal_stats.json")["correlations"]
    features = (("pen_H1", r"Pen. $H_1$"), ("pen_H0", r"Pen. $H_0$"),
                ("logits_H1", r"Logits $H_1$"), ("concat", "Concatenated"))
    rows = [r"\multicolumn{5}{l}{\emph{Distinguisher AUC}} \\"]
    for task, values in distinguish.items():
        rows.append(" & ".join((esc(task), *(num(values[key]["auc"]) for key, _ in features))) + r" \\")
    rows += [r"\addlinespace", r"\multicolumn{5}{l}{\emph{CIFAR-10 class relearning epoch-AUC}} \\"]
    for condition, values in relearn.items():
        auc = values["auc"]
        rows.append(f"{esc(condition)} & {num(fmean(auc))} $\\pm$ {num(stdev(auc))} & & & \\\\")
    rows += [r"\addlinespace", r"\multicolumn{5}{l}{\emph{Rank correlations}} \\"]
    for key, label in (("mia_auc", "TRR vs. MIA"), ("relearn_auc", "TRR vs. relearning")):
        value = correlations[key]
        rows.append(f"{label} & $\\rho={num(value['rho'])}$ & $p={pval(value['p'])}$ & & \\\\")
    header = "Task / condition & " + " & ".join(label for _, label in features)
    return table("Operational distinguishability, relearning, and rank correlations.", "operational",
                 "lrrrr", header, rows, r"\scriptsize")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = (("table1_settings.tex", settings()), ("table2_main.tex", main_results()),
               ("table3_equivalence.tex", equivalence()), ("table4_ablation.tex", ablation()),
               ("table5_operational.tex", operational()))
    for name, content in outputs:
        path = OUT / name
        path.write_text(content)
        assert path.exists() and r"\bottomrule" in path.read_text()
        print(f"\n--- {name} ---\n" + "\n".join(content.splitlines()[:9]))


if __name__ == "__main__":
    main()
