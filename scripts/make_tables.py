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
SCENARIOS = ("m4_random1", "m4_random", "m4_random10", "m4_class",
             "m4_targeted", "m4_matched", "m4_random_pretrained",
             "exp_cifar100_class", "exp_fashionmnist_random",
             "exp_fashionmnist_class", "exp_svhn_random", "exp_svhn_class",
             "m2_class")
APPROXIMATE = ("finetune", "neggrad", "scrub", "ssd")


def load(path):
    return json.loads((RESULTS / path).read_text())


def num(x):
    return f"{x:.4g}"


def pval(x):
    return r"$<0.001$" if x < .001 else f"{x:.3g}"


def esc(text):
    return str(text).replace("_", r"\_")


def bh(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    qvalues, running = [0.] * len(pvalues), 1.
    for rank in range(len(pvalues), 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * len(pvalues) / rank)
        qvalues[index] = running
    return qvalues


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
    return table(r"CIFAR-10 random 5\% deletion results (means over seeds). TRR, $\alpha$, and $\eta$ come from the frozen-grid penultimate-$H_1$ cell.", "main", "lrrrrrrr",
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
            dci = value.get("dur_minus_delta_CI", value["CI"])
            ci = f"[{num(dci[0])}, {num(dci[1])}]"
            decision = value.get("joint_decision", value["decision"])
            rows.append(" & ".join((NAMES[method] if i == 0 else "", cell, num(value["D_UR"]),
                                    num(eq[key]["delta"]), ci, q, esc(decision))) + r" \\")
        if method != METHODS[-1]:
            rows.append(r"\addlinespace")
    return table(r"Proximity-based oracle-equivalence decisions with jointly bootstrapped margin (CI is for $D_{UR}-\delta$; oracle and method seeds resampled together). Stars mark BH-significant DIFFERENCE tests (method vs oracle).", "equivalence", "llrrrrl",
                 r"Method & Cell & $D_{UR}$ & $\delta$ & CI($D_{UR}-\delta$) & BH $q$ & Decision", rows, r"\scriptsize")


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
    return table(r"Ablations and single-oracle sensitivity; all variants are penultimate-$H_1$.", "ablation", "lrrrr",
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
    caption = ("Operational distinguishability, relearning, and rank correlations. "
               "Relearning epoch-AUC has zero variance for conditions whose forget "
               "accuracy saturates after the first epoch (curves are identically "
               r"$(0,1,1,\ldots)$). Rank correlations pool methods per seed and "
               "exclude NegGrad, whose utility collapse (retain accuracy $0.02$) "
               "places it outside the regime the correlation is meant to probe; "
               "the no-op is excluded for lacking a relearning outcome.")
    return table(caption, "operational", "lrrrr", header, rows, r"\scriptsize")


def matrix():
    rows = []
    for scenario in SCENARIOS:
        analysis = load(f"{scenario}/analysis.json")
        labels = [(cell, method) for cell, values in analysis.items()
                  for method in values.get("methods", {})]
        qvalues = bh([analysis[cell]["methods"][method]["p"]
                      for cell, method in labels])
        adjusted = dict(zip(labels, qvalues))
        for cell, values in analysis.items():
            gate_open = values["gate_open"]
            anchor = pval(adjusted[(cell, "retrain2")]) if gate_open else ""
            worst = pval(max(adjusted[(cell, method)] for method in APPROXIMATE)) \
                if gate_open else ""
            rows.append(" & ".join((esc(scenario), esc(cell), num(values["I_topo"]),
                                    pval(values["p"]), "y" if gate_open else "n",
                                    anchor, worst)) + r" \\")
    header = (r"Scenario & Cell & $I_{\mathrm{topo}}$ & Gate $p$ & Gate open & "
              r"Anchor BH $q$ & Worst approximate BH $q$")
    halves = (rows[:28], rows[28:])
    return "\n".join(table(f"Complete scenario-by-cell evidence matrix (part {i}/2).",
                           f"matrix{i}", "llrrlrr", header, half, r"\scriptsize")
                     for i, half in enumerate(halves, 1))


VERDICT_NAMES = {"retrain": "Exact retrain (oracle)", "retrain2": "Exact retrain (held out)",
                 "original": "No-op", "finetune": "Fine-tune", "neggrad": "NegGrad",
                 "scrub": "SCRUB", "ssd": "SSD", "neggrad_plus": "NegGrad+",
                 "scrub_tuned": "SCRUB (tuned)", "ssd_tuned": "SSD (tuned)"}


def verdict():
    data = load("m4_random/joint_verdict.json")
    ranges, conditions = data["oracle_ranges"], data["conditions"]
    rows = []
    for key, label in VERDICT_NAMES.items():
        row = conditions[key]
        mean, q = row["mean"], row["topology_q"]
        rows.append(" & ".join((
            label, num(mean["retain"]), num(mean["forget"]), num(mean["test"]),
            pval(q) if q is not None else "--",
            esc(row["mode"] or "---"))) + r" \\")
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{6}{l}{\emph{Destructive noise control}} \\")
    for key, row in conditions.items():
        if not key.startswith("noise_"):
            continue
        mean = row["mean"]
        rows.append(" & ".join((
            f"$\\sigma={key.split('_')[1]}$", num(mean["retain"]), num(mean["forget"]),
            num(mean["test"]), "--", esc(row["mode"] or "---"))) + r" \\")
    caption = (r"Joint verdict on CIFAR-10 random 5\%. A condition is audit-consistent "
               r"(mode ``---'') only if retain/test accuracy, forget accuracy, and the "
               r"topological method test all agree with the oracle. Oracle reference "
               f"ranges: retain $[{num(ranges['retain'][0])}, {num(ranges['retain'][1])}]$, "
               f"forget $[{num(ranges['forget'][0])}, {num(ranges['forget'][1])}]$, "
               f"test $[{num(ranges['test'][0])}, {num(ranges['test'][1])}]$. "
               r"Only the two exact-retrain cohorts pass.")
    return table(caption, "verdict", "lrrrrl",
                 r"Condition & Retain & Forget & Test & Topology $q$ & Failure mode",
                 rows, r"\scriptsize")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = (("table1_settings.tex", settings()), ("table2_main.tex", main_results()),
               ("table3_equivalence.tex", equivalence()), ("table4_ablation.tex", ablation()),
               ("table5_operational.tex", operational()),
               ("table6_matrix.tex", matrix()), ("table7_verdict.tex", verdict()))
    for name, content in outputs:
        path = OUT / name
        path.write_text(content)
        assert path.exists() and r"\bottomrule" in path.read_text()
        lines = content.splitlines()
        print(f"\n--- {name} ---\n" + "\n".join(
            lines if name == "table6_matrix.tex" else lines[:9]))


if __name__ == "__main__":
    main()
