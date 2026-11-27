"""Export complete experiment tables, plots, paired drops and error examples."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_retrieval_index import sha256
from scripts.evaluate_models import CONDITIONS, data_paths
from scripts import generate_noise
from scripts.run_experiments import SELECTION_RULE, completed_run, fingerprint
from src.evaluation.runs import METRICS, read_jsonl, write_csv, write_json
from src.retrieval.evaluation import aggregate, degradation


def noise_coverage(query_sets, config):
    """Audit existing files in memory; never regenerate canonical noise files."""
    clean = query_sets["clean"]
    analyzed = generate_noise.analyze_queries(clean, config)
    result = []
    for condition in CONDITIONS[1:]:
        if condition.startswith("mixed_"):
            probability = config["mixed_probabilities"][condition.removeprefix("mixed_")]
        else:
            probability = config["single_probability"]
        generated = [generate_noise.generate_query(row, tokens, condition, probability, config["seed"])
                     for row, tokens in analyzed]
        expected, audits = [x[0] for x in generated], [x[1] for x in generated]
        observed = query_sets[condition]
        if [(r["id"], r["text"]) for r in expected] != [(r["id"], r["text"]) for r in observed]:
            raise ValueError(f"Noise does not match the fixed generator: {condition}")
        stats = generate_noise.summarize(clean, observed, audits, probability)
        result.append(dict(condition=condition, target_token_probability=probability,
            actual_token_noise_rate=stats["actual_token_noise_rate"],
            changed_queries=stats["changed_queries"], unchanged_queries=stats["unchanged_queries"],
            collision_groups=stats["collision_group_count"], new_collision_groups=stats["new_collision_group_count"]))
    return result


def select_configuration(rows):
    candidates = []
    pairs = sorted({(row["branch"], row["model"]) for row in rows})
    for branch, model in pairs:
        group = [row for row in rows if (row["branch"], row["model"]) == (branch, model)]
        if len(group) != 8 or {row["condition"] for row in group} != set(CONDITIONS):
            raise ValueError("Selection requires all eight conditions per candidate")
        clean = next(row for row in group if row["condition"] == "clean")
        candidates.append(dict(branch=branch, model=model,
            mean_noisy_ndcg=mean(row["ndcg@10"] for row in group if row["condition"] != "clean"),
            clean_ndcg=clean["ndcg@10"]))
    candidates.sort(key=lambda row: (-row["mean_noisy_ndcg"], -row["clean_ndcg"], row["branch"], row["model"]))
    return candidates


def make_plots(rows, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/lam/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    pairs = sorted({(row["branch"], row["model"]) for row in rows})
    lookup = {(row["branch"], row["model"], row["condition"]): row for row in rows}
    labels = [f"{branch} / {model.upper()}" for branch, model in pairs]
    for metric in METRICS:
        matrix = np.array([[lookup[b, m, c][metric] for c in CONDITIONS] for b, m in pairs])
        fig, ax = plt.subplots(figsize=(12, 5.4), layout="constrained")
        picture = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(8), CONDITIONS, rotation=25, ha="right")
        ax.set_yticks(range(len(pairs)), labels)
        ax.set_title(f"{rows[0]['split'].upper()} — {metric} (full corpus)")
        for i in range(len(pairs)):
            for j in range(8):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i, j] > .55 else "black", fontsize=9)
        fig.colorbar(picture, ax=ax, label=metric)
        filename = metric.replace("@", "_at_")
        fig.savefig(output_dir / f"{filename}.png", dpi=180)
        fig.savefig(output_dir / f"{filename}.svg")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    levels = ("clean", "mixed_low", "mixed_medium", "mixed_high")
    for ax, metric in zip(axes, METRICS):
        for branch, model in pairs:
            ax.plot(range(4), [lookup[branch, model, c][metric] for c in levels],
                    marker="o", label=f"{branch}/{model}")
        ax.set_xticks(range(4), ("clean", "low", "medium", "high"))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mixed noise condition (actual rates differ)")
        ax.set_title(metric)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.savefig(output_dir / "mixed_noise.png", dpi=180)
    fig.savefig(output_dir / "mixed_noise.svg")
    plt.close(fig)


def report(experiment_dir, output_dir, data_dir, split_dir, noise_config):
    experiment_dir, output_dir = Path(experiment_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Report already exists; choose a new output directory")
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    complete = json.loads((experiment_dir / "complete.json").read_text(encoding="utf-8"))
    if (complete["completed_runs"] != 48 or manifest["expected_runs"] != 48
            or complete["summary_sha256"] != sha256(experiment_dir / "summary.csv")):
        raise ValueError("Report requires a complete, intact 48-run experiment")
    paths, qrels = data_paths(Path(data_dir), Path(split_dir), config["split"])
    if ({name: sha256(path) for name, path in paths.items()} != config["queries_sha256"]
            or sha256(qrels) != config["qrels_sha256"]):
        raise ValueError("Report inputs differ from experiment inputs")
    query_sets = {name: read_jsonl(path) for name, path in paths.items()}
    texts = {name: {row["id"]: row["text"] for row in data} for name, data in query_sets.items()}
    coverage = noise_coverage(query_sets, json.loads(Path(noise_config).read_text(encoding="utf-8")))
    rows, paired, worst, run_hashes = [], [], [], {}
    for branch in config["branches"]:
        for model in config["models"]:
            clean_metrics = None
            clean_top = None
            for condition in CONDITIONS:
                directory = experiment_dir / branch / model / condition
                run_config = dict(experiment=fingerprint(config), branch=branch, model=model, condition=condition)
                summary = completed_run(directory, run_config)
                run_hashes[f"{branch}/{model}/{condition}"] = sha256(directory / "complete.json")
                with (directory / "per_query.csv").open(encoding="utf-8", newline="") as stream:
                    per_query = {row["qid"]: {metric: float(row[metric]) for metric in METRICS}
                                 for row in csv.DictReader(stream)}
                if set(per_query) != set(texts["clean"]):
                    raise ValueError("Per-query metrics have missing/extra IDs")
                measured = aggregate(list(per_query.values()))
                if any(abs(summary[metric] - measured[metric]) > 1e-12 for metric in METRICS):
                    raise ValueError("Per-query metrics disagree with summary")
                top = {qid: [] for qid in per_query}
                for ranking in read_jsonl(directory / "rankings.jsonl"):
                    if ranking["rank"] <= 3:
                        top[ranking["qid"]].append(ranking["doc_id"])
                if condition == "clean":
                    clean_metrics, clean_top = per_query, top
                changed = [qid for qid in per_query if texts["clean"][qid] != texts[condition][qid]]
                changed_scores = aggregate([per_query[qid] for qid in changed]) if changed else None
                rows.append({**summary, **{metric + "_changed_only": changed_scores[metric]
                                          if changed_scores else None for metric in METRICS}})
                if condition == "clean":
                    continue
                errors = []
                for qid, noisy in per_query.items():
                    row = dict(branch=branch, model=model, condition=condition, qid=qid,
                        changed=texts["clean"][qid] != texts[condition][qid],
                        **{metric + "_clean": clean_metrics[qid][metric] for metric in METRICS},
                        **{metric + "_noisy": noisy[metric] for metric in METRICS},
                        **degradation(clean_metrics[qid], noisy))
                    paired.append(row)
                    if row["changed"] and row["ndcg@10_drop_abs"] > 0:
                        errors.append(dict(**row, clean_text=texts["clean"][qid], noisy_text=texts[condition][qid],
                            clean_top3=" | ".join(clean_top[qid]), noisy_top3=" | ".join(top[qid])))
                errors.sort(key=lambda row: (-row["ndcg@10_drop_abs"], row["qid"]))
                worst.extend(errors[:5])
    candidates = select_configuration(rows)
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "paired_drops.csv", paired)
    write_csv(output_dir / "worst_queries.csv", worst)
    write_csv(output_dir / "noise_coverage.csv", coverage)
    write_csv(output_dir / "configuration_comparison.csv", candidates)
    if config["split"] == "val":
        write_json(output_dir / "selection.json", dict(split="val", completed_runs=48,
            selection_rule=SELECTION_RULE, selected=candidates[0], protocol=config["protocol"],
            indexes=config["indexes"], raw_corpus_sha256=config["raw_corpus_sha256"],
            manifest_sha256=sha256(experiment_dir / "manifest.json"),
            summary_sha256=sha256(experiment_dir / "summary.csv")))
    make_plots(rows, output_dir)
    pairs = [(row["branch"], row["model"]) for row in candidates]
    lookup = {(row["branch"], row["model"], row["condition"]): row for row in rows}
    lines = [f"# Kết quả metrics — {config['split']}", "",
        f"Đủ **48/48 lượt**, **{rows[0]['queries']} queries/lượt**, full corpus; tham số baseline giữ nguyên.", "",
        "## Chất lượng clean và trung bình bảy điều kiện nhiễu", "",
        "| Term | Model | Clean MRR@10 | Clean nDCG@10 | Clean Recall@100 | Mean noisy nDCG@10 |",
        "|---|---|---:|---:|---:|---:|"]
    for candidate in candidates:
        clean = lookup[candidate["branch"], candidate["model"], "clean"]
        lines.append(f"| {candidate['branch']} | {candidate['model']} | {clean['mrr@10']:.4f} | "
                     f"{clean['ndcg@10']:.4f} | {clean['recall@100']:.4f} | {candidate['mean_noisy_ndcg']:.4f} |")
    for metric in METRICS:
        lines += ["", f"## {metric}", "", "| Điều kiện | " + " | ".join(f"{b}/{m}" for b, m in pairs) + " |",
                  "|---|" + "---:|" * len(pairs)]
        for condition in CONDITIONS:
            lines.append(f"| {condition} | " + " | ".join(f"{lookup[b, m, condition][metric]:.4f}"
                                                          for b, m in pairs) + " |")
        lines += ["", f"![{metric}]({metric.replace('@', '_at_')}.png)"]
    lines += ["", "## Độ phủ nhiễu thực tế", "",
        "| Điều kiện | Token thực sự đổi | Query đổi | Query không đổi | Collision mới |",
        "|---|---:|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['condition']} | {row['actual_token_noise_rate']:.2%} | {row['changed_queries']} | "
                     f"{row['unchanged_queries']} | {row['new_collision_groups']} |")
    if config["split"] == "val":
        best = candidates[0]
        lines += ["", "## Chọn cấu hình bằng validation", "",
            "Quy tắc ghi trước trong manifest: tối đa hóa trung bình nDCG@10 trên bảy điều kiện nhiễu; "
            "nếu bằng nhau, ưu tiên clean nDCG@10 rồi thứ tự tên term/model. Không tuning hay sửa model.", "",
            f"Cấu hình được chọn: **{best['branch']} + {best['model']}**; mean noisy nDCG@10 = "
            f"**{best['mean_noisy_ndcg']:.4f}**. `selection.json` khóa cấu hình trước khi chạy test."]
    else:
        lines += ["", "## Diễn giải test", "",
            "Các cấu hình và tham số đã khóa trước test bằng selection validation. Bảng test phục vụ "
            "so sánh sáu baseline, không được dùng chọn lại cấu hình. Tham chiếu selection qua SHA-256 trong manifest."]
    lines += ["", "## Suy giảm và phân tích lỗi", "",
        "`summary.csv`: điểm toàn tập, drop tuyệt đối/phần trăm và điểm chỉ trên query thực sự đổi. "
        "`paired_drops.csv`: ghép sạch–nhiễu theo cùng qid, model, term. Khi điểm clean bằng 0, drop % để trống (N/A). "
        "Điểm toàn tập vẫn tính cả query không đổi và query có ranking rỗng.", "",
        "`worst_queries.csv`: tối đa năm query có nDCG@10 giảm mạnh nhất cho mỗi tổ hợp term/model/noise, "
        "kèm văn bản sạch/nhiễu và top 3 document IDs để đối chiếu. Đây là mẫu chọn theo lỗi, không phải mẫu ngẫu nhiên.", "",
        "![Mixed noise](mixed_noise.png)", "", "## Giới hạn", "",
        "Nhiễu tổng hợp có thể làm đổi nghĩa; độ phủ phonetic/abbreviation thấp nên điểm toàn tập bị chi phối "
        "bởi query không đổi. Cần xem thêm cột changed_only. Qrels có thể chưa liệt kê hết tài liệu liên quan. "
        "Không suy ra ý nghĩa thống kê từ chênh lệch điểm nhỏ. Test đã từng được xem xét trong giai đoạn trước.", "",
        "Rà soát ngữ nghĩa và nhận xét cụ thể được ghi riêng trong tài liệu bàn giao của Lam.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output_dir / "provenance.json", dict(manifest_sha256=sha256(experiment_dir / "manifest.json"),
        complete_sha256=sha256(experiment_dir / "complete.json"), run_complete_sha256=run_hashes,
        noise_config_sha256=sha256(noise_config), report_code_sha256=sha256(Path(__file__))))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/zalo_legal")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/lam/splits/seed42")
    parser.add_argument("--noise-config", type=Path, default=ROOT / "configs/noise.json")
    args = parser.parse_args()
    candidates = report(args.experiment_dir, args.output_dir, args.data_dir, args.split_dir, args.noise_config)
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
