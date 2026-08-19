import json
from pathlib import Path

print("SCRIPT_VERSION=three_way_v2")
print("RUNNING_FILE=", __file__)
STANDARD_PATH = Path(
    "/root/headinfer_ref/results/qwen7b_9216/standard_sync/"
    "headinfer_reference_standard_9216_report.json"
)

HEADINFER_PATH = Path(
    "/root/headinfer_ref/results/qwen7b_9216/headinfer_sync/"
    "headinfer_reference_headinfer_9216_report.json"
)

RESIDENT_PATH = Path(
    "/root/headinfer_ref/results/qwen7b_9216/resident_r14_sync_v2/"
    "headinfer_reference_resident_9216_report.json"
)


def parse_token_ids(value):
    if isinstance(value, list):
        return [int(token) for token in value]

    if isinstance(value, str):
        parsed = json.loads(value)
        return [int(token) for token in parsed]

    raise TypeError(f"Unsupported generated_token_ids type: {type(value)}")


def load_sequences(path):
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("rows")

    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No non-empty 'rows' list in: {path}")

    sequences = []
    for index, row in enumerate(rows):
        if "generated_token_ids" not in row:
            raise KeyError(
                f"rows[{index}] has no generated_token_ids in: {path}"
            )
        sequences.append(parse_token_ids(row["generated_token_ids"]))

    return sequences


def is_stable(sequences):
    reference = sequences[0]
    return all(sequence == reference for sequence in sequences)


def first_mismatch(left, right):
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {
                "index": index,
                "left_token": left_token,
                "right_token": right_token,
            }

    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left_length": len(left),
            "right_length": len(right),
        }

    return None


def main():
    standard = load_sequences(STANDARD_PATH)
    headinfer = load_sequences(HEADINFER_PATH)
    resident = load_sequences(RESIDENT_PATH)

    standard_ids = standard[0]
    headinfer_ids = headinfer[0]
    resident_ids = resident[0]

    standard_stable = is_stable(standard)
    headinfer_stable = is_stable(headinfer)
    resident_stable = is_stable(resident)

    headinfer_matches_standard = headinfer_ids == standard_ids
    resident_matches_standard = resident_ids == standard_ids
    resident_matches_headinfer = resident_ids == headinfer_ids

    print("standard_stable=", standard_stable)
    print("headinfer_stable=", headinfer_stable)
    print("resident_stable=", resident_stable)
    print("headinfer_matches_standard=", headinfer_matches_standard)
    print("resident_matches_standard=", resident_matches_standard)
    print("resident_matches_headinfer=", resident_matches_headinfer)

    print("standard_ids=", standard_ids)
    print("headinfer_ids=", headinfer_ids)
    print("resident_ids=", resident_ids)

    if not resident_matches_standard:
        print(
            "resident_vs_standard_first_mismatch=",
            first_mismatch(resident_ids, standard_ids),
        )

    passed = (
        standard_stable
        and headinfer_stable
        and resident_stable
        and headinfer_matches_standard
        and resident_matches_standard
        and resident_matches_headinfer
    )

    print("three_way_correctness=", "PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()