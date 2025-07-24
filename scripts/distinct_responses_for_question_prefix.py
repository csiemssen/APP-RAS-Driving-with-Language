import argparse
import json


def collect_distinct_responses(dataset_path, question_prefix):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    prefix = question_prefix.strip().lower()
    responses = set()
    matched_questions = set()

    for scene_id, scene_obj in data.items():
        key_frames = scene_obj.get("key_frames", {})
        for frame_id, frame_obj in key_frames.items():
            qa_sections = frame_obj.get("QA", {})
            for qa_type, qa_list in qa_sections.items():
                for qa in qa_list:
                    question = qa.get("Q", "")
                    answer = qa.get("A", "")
                    if question.lower().startswith(prefix):
                        responses.add(answer)
                        matched_questions.add(question)

    print(
        f"Found {len(responses)} distinct responses for questions starting with '{question_prefix}':\n"
    )
    for resp in sorted(responses):
        print(resp)
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect all distinct responses for questions starting with a given prefix in the dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file (e.g., data/drivelm/v1_1_train_nus.json)",
    )
    parser.add_argument(
        "--question_prefix",
        type=str,
        required=True,
        help="Prefix string to match at the start of questions (case-insensitive)",
    )

    args = parser.parse_args()
    collect_distinct_responses(args.dataset_path, args.question_prefix)
