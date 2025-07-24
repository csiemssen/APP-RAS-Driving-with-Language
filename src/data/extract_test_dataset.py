# Adopted from https://github.com/OpenDriveLab/DriveLM. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import json
import random

from src.utils.logger import get_logger

logger = get_logger(__name__)

PERCEPTION_MOVING_STATUS_OPTIONS = [
    "Back up",
    "Backward",
    "Bending over",
    "Reverse parking",
    "Turn left",
    "Turn right",
    "Going ahead",
]

BEHAVIOR_STEERING_OPTIONS = [
    "going straight",
    "slightly steering to the left",
    "slightly steering to the right",
    "steering to the left",
    "steering to the right",
]

BEHAVIOR_SPEED_OPTIONS = [
    "not moving",
    "driving slowly",
    "driving with normal speed",
    "driving fast",
    "driving very fast",
]


def extract_data(root_path, save_path, patch_accuracy_options=True, exclude_tags=[]):
    with open(root_path, "r") as f:  # , \
        train_file = json.load(f)

    test_data = dict()

    # TODO: convert the data into test data, containing the importance, multiple choice questions, graph questions
    for scene_id in train_file.keys():
        scene_data = train_file[scene_id]["key_frames"]

        # for test file
        test_data[scene_id] = dict()
        test_data[scene_id]["key_frames"] = dict()

        for frame_id in scene_data.keys():
            frame_data_infos = scene_data[frame_id]["key_object_infos"]
            frame_data_qa = scene_data[frame_id]["QA"]
            image_paths = scene_data[frame_id]["image_paths"]

            # for test file
            test_data[scene_id]["key_frames"][frame_id] = dict()
            # test_data[scene_id]['key_frames'][frame_id]['key_object_infos'] = frame_data_infos
            test_data[scene_id]["key_frames"][frame_id]["QA"] = dict()
            test_data[scene_id]["key_frames"][frame_id]["image_paths"] = image_paths
            test_data[scene_id]["key_frames"][frame_id]["QA"]["perception"] = []
            test_data[scene_id]["key_frames"][frame_id]["QA"]["prediction"] = []
            test_data[scene_id]["key_frames"][frame_id]["QA"]["planning"] = []
            test_data[scene_id]["key_frames"][frame_id]["QA"]["behavior"] = []

            # get the classes of the important objects
            classes = []
            for obj_id in frame_data_infos.keys():
                obj_data = frame_data_infos[obj_id]
                classes.append(obj_data["Visual_description"].split(".")[0])
                logger.debug(classes)

            # get the location of the important objects
            locations = []
            for obj_id in frame_data_infos.keys():
                locations.append(obj_id)
                logger.debug(locations)

            # get the questions and answers of the perception
            perception = frame_data_qa["perception"]
            prediction = frame_data_qa["prediction"]
            planning = frame_data_qa["planning"]
            behavior = frame_data_qa["behavior"]

            for qa in perception:
                question = qa["Q"]
                answer = qa["A"]

                # according to the classes to select the corresponding question
                flag = 1
                for cl in classes:
                    if cl.lower() not in answer.lower():
                        flag = 0
                if flag == 1:
                    qa["tag"] = [2]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "perception"
                    ].append(qa)
                    break

            # get the multiple choice questions and answers
            for qa in perception:
                question = qa["Q"]
                answer = qa["A"]
                if "What is the moving status of object".lower() in question.lower():
                    qa["tag"] = [0]
                    if patch_accuracy_options and not any(
                        opt in question for opt in ["A.", "B.", "C.", "D."]
                    ):
                        option_strs, correct_letter = generate_perception_options(
                            answer
                        )
                        qa["Q"] = (
                            f"{question} Please select the correct answer from the following options: "
                            + " ".join(option_strs)
                        )
                        qa["A"] = correct_letter
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "perception"
                    ].append(qa)
                    break

            # get the graph questions and answers
            for qa in prediction:
                question = qa["Q"]
                answer = qa["A"]

                # according to the location to select the corresponding question
                flag = 1
                for loc in locations:
                    if loc.lower() not in answer.lower():
                        flag = 0
                if flag == 1:
                    qa["tag"] = [3]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "prediction"
                    ].append(qa)
                    break

            # get the yes or no questions and answers
            for qa in prediction:
                question = qa["Q"]
                answer = qa["A"]
                if "yes" in answer.lower() or "no" in answer.lower():
                    qa["tag"] = [0]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "prediction"
                    ].append(qa)
                    break

            # get the three questions from the planning "safe actions", "collision", ""
            actions_question_added = False
            collision_question_added = False
            safe_actions_question_added = False
            for qa in planning:
                question = qa["Q"]
                answer = qa["A"]
                if (
                    "What actions could the ego vehicle take".lower()
                    in question.lower()
                    and not actions_question_added
                ):
                    qa["tag"] = [1]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "planning"
                    ].append(qa)
                    actions_question_added = True
                if (
                    "lead to a collision" in question.lower()
                    and not collision_question_added
                ):
                    qa["tag"] = [1]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "planning"
                    ].append(qa)
                    collision_question_added = True
                if (
                    "safe actions" in question.lower()
                    and not safe_actions_question_added
                ):
                    qa["tag"] = [1]
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "planning"
                    ].append(qa)
                    safe_actions_question_added = True

                # Check if all question types have been added and exit the loop
                if (
                    actions_question_added
                    and collision_question_added
                    and safe_actions_question_added
                ):
                    break

            for qa in behavior:
                question = qa["Q"]
                answer = qa["A"]
                qa["tag"] = [0]
                if patch_accuracy_options and not any(
                    opt in question for opt in ["A.", "B.", "C."]
                ):
                    option_strs, correct_letter = generate_behavior_options(answer)
                    qa["Q"] = (
                        f"{question} Please select the correct answer from the following options: "
                        + " ".join(option_strs)
                    )
                    qa["A"] = correct_letter
                test_data[scene_id]["key_frames"][frame_id]["QA"]["behavior"].append(qa)

    if exclude_tags is not None:
        for scene_id in test_data.keys():
            for frame_id in test_data[scene_id]["key_frames"].keys():
                qa_types = test_data[scene_id]["key_frames"][frame_id]["QA"]
                for qa_type in qa_types.keys():
                    qa_list = qa_types[qa_type]
                    qa_types[qa_type] = [
                        qa
                        for qa in qa_list
                        if not any(tag in exclude_tags for tag in qa.get("tag", []))
                    ]

    with open(save_path, "w") as f:
        json.dump(test_data, f, indent=4)


def generate_perception_options(answer):
    options = set(PERCEPTION_MOVING_STATUS_OPTIONS)
    answer_clean = answer.strip().rstrip(".")
    answer_option = None

    for opt in options:
        if answer_clean.lower() in opt.lower():
            answer_option = opt
            break

    if not answer_option:
        answer_option = answer.strip().rstrip(".")

    options.discard(answer_option)

    other_options = random.sample(list(options), 3)
    all_options = [answer_option] + other_options
    random.shuffle(all_options)

    option_letters = ["A", "B", "C", "D"]
    option_strs = [
        f"{letter}. {opt}" for letter, opt in zip(option_letters, all_options)
    ]
    correct_letter = option_letters[all_options.index(answer_option)]

    return option_strs, correct_letter


def generate_behavior_options(answer):
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if len(sentences) < 2:
        answer_option = answer.strip()
        options = [answer_option]
        while len(options) < 3:
            candidate = f"The ego vehicle is {random.choice(BEHAVIOR_STEERING_OPTIONS)}. The ego vehicle is {random.choice(BEHAVIOR_SPEED_OPTIONS)}."
            if candidate not in options:
                options.append(candidate)
        random.shuffle(options)
    else:
        answer_option = f"{sentences[0]}. {sentences[1]}."
        options = [answer_option]
        while len(options) < 3:
            candidate = f"The ego vehicle is {random.choice(BEHAVIOR_STEERING_OPTIONS)}. The ego vehicle is {random.choice(BEHAVIOR_SPEED_OPTIONS)}."
            if candidate not in options:
                options.append(candidate)
        random.shuffle(options)

    option_letters = ["A", "B", "C"]
    option_strs = [f"{letter}. {opt}" for letter, opt in zip(option_letters, options)]
    correct_letter = option_letters[options.index(answer_option)]
    return option_strs, correct_letter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Test Data")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/drivelm/v1_1_train_nus.json",
        help="path to the root dataset file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/drivelm/v1_1_test_nus.json",
        help="path to save test dataset to",
    )
    parser.add_argument(
        "--exclude_tags",
        type=str,
        default="[]",
        help="question tags to exclude from the test dataset",
    )
    parser.add_argument(
        "--patch_accuracy_options",
        action="store_true",
        help="If set, patch accuracy questions (tag 0) to have multiple-choice options.",
    )
    args = parser.parse_args()
    exclude_tags = json.loads(args.exclude_tags) if args.exclude_tags else []

    extract_data(
        args.root_path,
        args.save_path,
        args.patch_accuracy_options,
        exclude_tags=[int(tag) for tag in exclude_tags],
    )
