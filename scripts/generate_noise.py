"""Sinh bảy tập truy vấn nhiễu; chỉ dùng thư viện chuẩn Python.

Các quy tắc là mô phỏng lỗi tổng hợp, cần rà soát mẫu trước khi báo cáo.
Không thay đổi file truy vấn sạch, corpus hoặc qrels.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "1.1.0"
KINDS = ("no_diacritics", "telex_typo", "phonetic", "abbrev")
TONE_KEYS = {
    "\u0301": ("s", "1"),
    "\u0300": ("f", "2"),
    "\u0309": ("r", "3"),
    "\u0303": ("x", "4"),
    "\u0323": ("j", "5"),
}
SHAPE_MARKS = {"\u0302", "\u0306", "\u031b"}
WORD = re.compile(r"^([^\w]*)([^\W\d_]+)([^\w]*)$", re.UNICODE)
VIETNAMESE_LETTERS = set("abcdefghijklmnopqrstuvwxyzđ")


def strip_diacritics(text):
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    ).translate(str.maketrans({"đ": "d", "Đ": "D"}))


def make_rng(seed, *parts):
    # Không dùng hash(): giá trị hash của Python có thể đổi giữa các process.
    payload = json.dumps([seed, *parts], ensure_ascii=False).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_queries(path):
    rows, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: dòng trống trong JSONL")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: bản ghi phải là object")
            qid, text = row.get("id"), row.get("text")
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"{path}:{line_number}: id phải là chuỗi không rỗng")
            if qid in seen:
                raise ValueError(f"{path}:{line_number}: ID trùng: {qid}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: text không hợp lệ")
            if text != unicodedata.normalize("NFC", text):
                raise ValueError(f"{path}:{line_number}: text chưa ở dạng Unicode NFC")
            seen.add(qid)
            rows.append(row)
    if not rows:
        raise ValueError("File queries không có dữ liệu")
    return rows


def validate_qrels(path, rows):
    positive_ids, all_ids = set(), set()
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("Qrels thiếu cột query-id/corpus-id/score")
        for row in reader:
            all_ids.add(row["query-id"])
            if float(row["score"]) > 0:
                positive_ids.add(row["query-id"])
    query_ids = {row["id"] for row in rows}
    if all_ids != query_ids or not query_ids.issubset(positive_ids):
        raise ValueError("Qrels phải khớp tập query ID và mỗi query phải có nhãn dương")


def validate_config(config):
    for probability in [config["single_probability"], *config["mixed_probabilities"].values()]:
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("Xác suất nhiễu phải nằm trong [0, 1]")
    levels = config["mixed_probabilities"]
    if set(levels) != {"low", "medium", "high"}:
        raise ValueError("Cần đủ ba mức low, medium, high")
    if not levels["low"] <= levels["medium"] <= levels["high"]:
        raise ValueError("Xác suất cần thỏa low <= medium <= high")
    for name in ("abbreviations", "phonetic"):
        for word, alternatives in config[name].items():
            if not word.isalpha() or word != word.lower() or not alternatives:
                raise ValueError(f"Mục từ không hợp lệ trong {name}: {word}")
            for alternative in alternatives:
                if (
                    not isinstance(alternative, str)
                    or not alternative.isalpha()
                    or alternative == word
                    or alternative != unicodedata.normalize("NFC", alternative)
                ):
                    raise ValueError(f"Biến thể không hợp lệ trong {name}: {word}")


def preserve_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.capitalize()
    return replacement


def telex_candidates(word):
    """Mô phỏng lỗi phím dấu, không xóa/đảo ký tự tùy ý.

    Ví dụ: phạt -> phat (thiếu j/5), phatj hoặc phat5 (phím dấu
    còn dạng thô); đầu -> đàu (thiếu một phím a tạo â); được -> dược
    (thiếu phím d thứ hai). Đây là mô phỏng, không phải bộ gõ đầy đủ.
    """
    candidates = []
    decomposed = unicodedata.normalize("NFD", word)
    tone = next((c for c in decomposed if c in TONE_KEYS), None)
    if tone:
        untoned = unicodedata.normalize(
            "NFC", "".join(c for c in decomposed if c not in TONE_KEYS)
        )
        telex_key, vni_key = TONE_KEYS[tone]
        candidates.extend([
            (untoned, "missing_tone_key"),
            (untoned + telex_key, "uncomposed_telex_tone_key"),
            (untoned + vni_key, "uncomposed_vni_tone_key"),
        ])
    for index, char in enumerate(word):
        parts = unicodedata.normalize("NFD", char)
        if any(mark in parts for mark in SHAPE_MARKS):
            plain = unicodedata.normalize(
                "NFC", "".join(c for c in parts if c not in SHAPE_MARKS)
            )
            candidates.append((word[:index] + plain + word[index + 1:], "missing_shape_key"))
        if char in "đĐ":
            plain = "D" if char == "Đ" else "d"
            candidates.append((word[:index] + plain + word[index + 1:], "missing_d_key"))
    unique, seen = [], {word}
    for replacement, rule in candidates:
        if replacement not in seen:
            seen.add(replacement)
            unique.append((replacement, rule))
    return unique


def token_options(raw, token_index, config):
    match = WORD.fullmatch(raw)
    if not match:
        return {}, "number_identifier_or_nonword"
    prefix, word, suffix = match.groups()
    lowered = word.lower()
    if not set(strip_diacritics(lowered)).issubset(VIETNAMESE_LETTERS):
        return {}, "non_vietnamese_letters"
    if lowered in config["protected_words"]:
        return {}, "protected_word"
    if (len(word) > 1 and word.isupper()) or (
        config["protect_capitalized_noninitial"] and token_index > 0 and word[0].isupper()
    ):
        return {}, "capitalized_name_or_acronym"
    # Bảo vệ cả kiểu viết hoa pha trộn như iPhone.
    if any(c.isupper() for c in word[1:]):
        return {}, "mixed_case_identifier"

    options = {}
    plain = strip_diacritics(word)
    if plain != word:
        options["no_diacritics"] = [(plain, "strip_diacritics")]
    keyboard = telex_candidates(word)
    if keyboard:
        options["telex_typo"] = keyboard
    for kind, dictionary in (("phonetic", "phonetic"), ("abbrev", "abbreviations")):
        alternatives = config[dictionary].get(lowered, [])
        if alternatives:
            options[kind] = [
                (preserve_case(word, replacement), f"dictionary:{lowered}")
                for replacement in alternatives
            ]
    return {
        kind: [(prefix + replacement + suffix, rule) for replacement, rule in values]
        for kind, values in options.items()
    }, None


def analyze_queries(rows, config):
    analyzed = []
    for row in rows:
        tokens = []
        matches = list(re.finditer(r"\S+", row["text"]))
        for index, match in enumerate(matches):
            raw = match.group()
            counted = any(char.isalnum() for char in raw)
            options, protected = token_options(raw, index, config) if counted else ({}, None)
            word_match = WORD.fullmatch(raw)
            neighbors = matches[max(0, index - 1):index] + matches[index + 1:index + 2]
            if (
                word_match and word_match.group(2).lower() in config["numeric_units"]
                and any(any(char.isdigit() for char in neighbor.group()) for neighbor in neighbors)
            ):
                options, protected = {}, "unit_next_to_number"
            tokens.append({
                "index": index, "start": match.start(), "end": match.end(),
                "raw": raw, "counted": counted, "options": options,
                "protected": protected,
            })
        analyzed.append((row, tokens))
    return analyzed


def generate_query(row, tokens, condition, probability, seed):
    mixed = condition.startswith("mixed_")
    namespace = "mixed" if mixed else condition
    changes, selected, protected, eligible, unavailable = [], 0, 0, 0, 0
    chunks, cursor = [], 0
    for token in tokens:
        raw, replacement = token["raw"], token["raw"]
        if token["counted"]:
            options = token["options"] if mixed else {
                condition: token["options"][condition]
            } if condition in token["options"] else {}
            protected += int(token["protected"] is not None)
            eligible += bool(options)
            # Hai lượt RNG riêng: khi tăng mức mixed, các vị trí cũ giữ
            # nguyên biến thể; chỉ bổ sung các vị trí được chọn thêm.
            picked = make_rng(seed, namespace, row["id"], token["index"], "select").random() < probability
            if picked:
                selected += 1
                if options:
                    variant_rng = make_rng(seed, namespace, row["id"], token["index"], "variant")
                    kinds = [kind for kind in KINDS if kind in options]
                    kind = variant_rng.choice(kinds)
                    replacement, rule = variant_rng.choice(options[kind])
                    changes.append({
                        "token_index": token["index"], "start": token["start"],
                        "end": token["end"], "before": raw, "after": replacement,
                        "kind": kind, "rule": rule,
                    })
                else:
                    unavailable += 1
        chunks.extend([row["text"][cursor:token["start"]], replacement])
        cursor = token["end"]
    chunks.append(row["text"][cursor:])
    noisy = {**row, "text": "".join(chunks)}
    counted = sum(token["counted"] for token in tokens)
    audit = {
        "qid": row["id"], "total_tokens": counted,
        "eligible_tokens": eligible, "protected_tokens": protected,
        "selected_tokens": selected, "selected_without_change": unavailable,
        "changed_tokens": len(changes), "changes": changes,
    }
    return noisy, audit


def collision_groups(clean, noisy):
    groups = defaultdict(list)
    for original, variant in zip(clean, noisy):
        groups[variant["text"]].append(original)
    return [
        {
            "text": text, "qids": [row["id"] for row in members],
            "distinct_clean_texts": len({row["text"] for row in members}),
        }
        for text, members in groups.items() if len(members) > 1
    ]


def summarize(clean, noisy, audits, probability):
    total = sum(audit["total_tokens"] for audit in audits)
    changed = sum(audit["changed_tokens"] for audit in audits)
    queries_changed = sum(a["text"] != b["text"] for a, b in zip(clean, noisy))
    kinds = Counter(change["kind"] for audit in audits for change in audit["changes"])
    collisions = collision_groups(clean, noisy)
    return {
        "query_count": len(clean), "target_token_probability": probability,
        "total_tokens": total,
        "eligible_tokens": sum(audit["eligible_tokens"] for audit in audits),
        "protected_tokens": sum(audit["protected_tokens"] for audit in audits),
        "selected_tokens": sum(audit["selected_tokens"] for audit in audits),
        "selected_without_change": sum(audit["selected_without_change"] for audit in audits),
        "changed_tokens": changed, "actual_token_noise_rate": changed / total if total else 0,
        "changed_queries": queries_changed, "unchanged_queries": len(clean) - queries_changed,
        "changed_query_rate": queries_changed / len(clean),
        "type_counts": dict(sorted(kinds.items())),
        "type_proportions": {kind: count / changed for kind, count in sorted(kinds.items())},
        "collision_group_count": len(collisions),
        "queries_in_collisions": sum(len(group["qids"]) for group in collisions),
        "new_collision_group_count": sum(group["distinct_clean_texts"] > 1 for group in collisions),
        "collisions": collisions,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("Seed phải là số nguyên")
    if args.sample_size < 0:
        raise ValueError("sample-size phải >= 0")
    rows = read_queries(args.input)
    qrels = args.qrels
    if qrels is None and args.input.with_name("qrels_test.tsv").is_file():
        qrels = args.input.with_name("qrels_test.tsv")
    if qrels is not None:
        validate_qrels(qrels, rows)
    output_dir = args.output_dir if args.output_dir is not None else args.input.parent / "noise"
    # Kiểm tra mọi đường dẫn trước khi ghi để không ghi đè đầu vào.
    conditions = [(kind, config["single_probability"]) for kind in KINDS]
    conditions.extend((f"mixed_{level}", config["mixed_probabilities"][level]) for level in ("low", "medium", "high"))
    inputs = {args.input.resolve(), args.config.resolve()}
    if qrels is not None:
        inputs.add(qrels.resolve())
    output_paths = [output_dir / f"queries_{condition}.jsonl" for condition, _ in conditions]
    output_paths += [args.report_dir / f"{condition}.changes.jsonl" for condition, _ in conditions]
    output_paths += [args.report_dir / name for name in ("stats.json", "config_used.json", "summary.md", "samples.md")]
    if len({path.resolve() for path in output_paths}) != len(output_paths) or any(path.resolve() in inputs for path in output_paths):
        raise ValueError("Đường dẫn đầu ra trùng nhau hoặc ghi đè đầu vào")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    analyzed = analyze_queries(rows, config)
    stats = {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": sha256(Path(__file__)),
        "config_source_sha256": sha256(args.config),
        "python_version": sys.version.split()[0],
        "input_file": args.input.name, "input_sha256": sha256(args.input),
        "qrels_sha256": sha256(qrels) if qrels is not None else None,
        "seed": config["seed"], "query_count": len(rows),
        "token_definition": "whitespace-delimited spans containing a letter or digit; indices are zero-based",
        "review_status": "synthetic rules; human review pending",
        "conditions": {},
    }
    summary = [
        "# Báo cáo sinh nhiễu", "",
        f"Seed: {config['seed']}. Số truy vấn mỗi điều kiện: {len(rows)}.", "",
        "Quy tắc tổng hợp phiên bản đầu; cần rà soát mẫu trước khi dùng kết quả chính.", "",
        "| Điều kiện | Xác suất chọn token | Token thực sự đổi | Câu thay đổi | Nhóm trùng văn bản |",
        "|---|---:|---:|---:|---:|",
    ]
    samples = [
        "# Mẫu truy vấn nhiễu cần rà soát", "",
        "Lấy mẫu tái lập từ các câu thực sự thay đổi, không đại diện cho tỷ lệ câu thay đổi toàn tập.",
        "Xem stats.json để biết các câu không thay đổi và tỷ lệ nhiễu thực tế.", "",
    ]
    for condition, probability in conditions:
        generated = [generate_query(row, tokens, condition, probability, config["seed"]) for row, tokens in analyzed]
        noisy, audits = [item[0] for item in generated], [item[1] for item in generated]
        if [row["id"] for row in noisy] != [row["id"] for row in rows]:
            raise ValueError("ID hoặc thứ tự queries bị thay đổi")
        output_path = output_dir / f"queries_{condition}.jsonl"
        audit_path = args.report_dir / f"{condition}.changes.jsonl"
        write_jsonl(output_path, noisy)
        write_jsonl(audit_path, audits)
        condition_stats = summarize(rows, noisy, audits, probability)
        condition_stats.update({"output_file": output_path.name, "output_sha256": sha256(output_path), "audit_file": audit_path.name, "audit_sha256": sha256(audit_path)})
        stats["conditions"][condition] = condition_stats
        summary.append(
            f"| {condition} | {probability:.0%} | {condition_stats['actual_token_noise_rate']:.2%} "
            f"| {condition_stats['changed_queries']}/{len(rows)} | {condition_stats['collision_group_count']} |"
        )
        changed_indices = [i for i, audit in enumerate(audits) if audit["changed_tokens"]]
        review_indices = sorted(make_rng(config["seed"], "review", condition).sample(changed_indices, min(args.sample_size, len(changed_indices))))
        samples.extend([f"## {condition}", ""])
        for index in review_indices:
            example = {
                "qid": rows[index]["id"], "clean": rows[index]["text"],
                "noisy": noisy[index]["text"], "changes": audits[index]["changes"],
            }
            samples.extend(["```json", json.dumps(example, ensure_ascii=False, indent=2), "```", ""])
        print(f"{condition}: {len(noisy)} queries; changed_tokens={condition_stats['actual_token_noise_rate']:.2%}; changed_queries={condition_stats['changed_queries']}")
    summary.extend([
        "", "Tỷ lệ token tính trên token gốc chứa chữ hoặc số; không ép mỗi câu phải có lỗi.",
        "Mức mixed dùng cùng lượt chọn và cùng biến thể ở các vị trí chung; low là tập con của medium và high.",
        "Telex/VNI chỉ mô phỏng thiếu phím dấu/phím tạo chữ và phím dấu còn dạng thô.",
        "Bỏ dấu chỉ áp dụng trên token được chọn với single_probability; giữ nguyên token không được chọn. Tất cả điều kiện bảo vệ token có số, mã/ký hiệu, viết hoa và đơn vị tiền đã cấu hình.",
        "Bảo vệ tên riêng là heuristic, không bảo đảm nhận diện hết thực thể. Các cặp phát âm có thể mơ hồ về nghĩa, cần đọc mẫu trong ngữ cảnh.",
        "Xem stats.json để biết token không áp dụng được, phân bố loại lỗi và nhóm trùng văn bản.", "",
    ])
    write_json(args.report_dir / "config_used.json", config)
    write_json(args.report_dir / "stats.json", stats)
    (args.report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (args.report_dir / "samples.md").write_text("\n".join(samples), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/zalo_legal/queries_clean.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/noise.json")
    parser.add_argument("--qrels", type=Path, help="Mặc định kiểm tra qrels_test.tsv cạnh file input nếu có")
    parser.add_argument("--output-dir", type=Path, help="Mặc định ghi vào thư mục noise cạnh file input")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "outputs/noise")
    parser.add_argument("--seed", type=int, help="Ghi đè seed trong config")
    parser.add_argument("--sample-size", type=int, default=10, help="Số câu đã thay đổi lấy mẫu mỗi điều kiện")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
