import argparse
import json
from collections import defaultdict


def collect_distinct_question_prefixes_by_tag(dataset_path, prefix_length):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    tag_to_prefixes = defaultdict(set)

    for scene_id, scene_obj in data.items():
        key_frames = scene_obj.get("key_frames", {})
        for frame_id, frame_obj in key_frames.items():
            qa_sections = frame_obj.get("QA", {})
            for qa_type, qa_list in qa_sections.items():
                for qa in qa_list:
                    question = qa.get("Q", "")
                    tag = qa.get("tag", "NO_TAG")
                    prefix = question[:prefix_length].strip().lower()
                    if prefix:
                        if isinstance(tag, list):
                            tags = tag if tag else ["NO_TAG"]
                        else:
                            tags = [tag if tag else "NO_TAG"]
                        for t in tags:
                            tag_to_prefixes[t].add(prefix)

    for tag, prefixes in sorted(tag_to_prefixes.items(), key=lambda x: x[0]):
        print(f"\nTag: {tag} ({len(prefixes)} distinct prefixes)")
        for prefix in sorted(prefixes):
            print(prefix)
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect all distinct question prefixes of a given length, grouped by tag."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file (e.g., data/drivelm/v1_1_train_nus.json)",
    )
    parser.add_argument(
        "--prefix_length",
        type=int,
        required=True,
        help="Number of characters to use for the question prefix.",
    )

    args = parser.parse_args()
    collect_distinct_question_prefixes_by_tag(args.dataset_path, args.prefix_length)
