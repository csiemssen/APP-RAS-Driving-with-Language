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
from dataclasses import dataclass
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionTag:
    tag: List[int]
    subtype: Optional[str] = None


def get_question_tag_test(
    question, answer, qa_type, classes=None, locations=None
) -> Optional[QuestionTag]:
    if qa_type == "perception" and classes is not None:
        if all(cl.lower() in answer.lower() for cl in classes):
            return QuestionTag(tag=[2], subtype="importance")

    if (
        qa_type == "perception"
        and "what is the moving status of object" in question.lower()
    ):
        return QuestionTag(tag=[0], subtype="moving_status")

    if qa_type == "prediction" and locations is not None:
        if all(loc.lower() in answer.lower() for loc in locations):
            return QuestionTag(tag=[3], subtype="graph")

    if qa_type == "prediction":
        if "yes" in answer.lower() or "no" in answer.lower():
            return QuestionTag(tag=[0], subtype="yes_no")

    if qa_type == "planning":
        if "What actions could the ego vehicle take".lower() in question.lower():
            return QuestionTag(tag=[1], subtype="actions")
        if "lead to a collision" in question.lower():
            return QuestionTag(tag=[1], subtype="collision")
        if "safe actions" in question.lower():
            return QuestionTag(tag=[1], subtype="safe_actions")

    if qa_type == "behavior":
        return QuestionTag(tag=[0])

    return None


def get_question_tag_eval(question, qa_type):
    q_lower = question.lower()
    match qa_type:
        case "perception":
            if "what are the important objects in the current scene" in q_lower:
                return QuestionTag(tag=[2], subtype="importance")
            if "what is the moving status of object" in q_lower:
                return QuestionTag(tag=[0], subtype="moving_status")

        case "prediction":
            if (
                "what object should the ego vehicle notice first when the ego vehicle is getting to the next possible location"
                in q_lower
            ):
                return QuestionTag(tag=[3], subtype="graph")
                if q_lower.strip().startswith(("are there", "is", "will", "would")):
                    return QuestionTag(tag=[0], subtype="yes_no")

        case "planning":
            if "what actions could the ego vehicle take" in q_lower:
                return QuestionTag(tag=[1], subtype="actions")
            if "lead to a collision" in q_lower:
                return QuestionTag(tag=[1], subtype="collision")
            if "safe actions" in q_lower:
                return QuestionTag(tag=[1], subtype="safe_actions")

        case "behavior":
            return QuestionTag(tag=[0])

    return None


def get_key_objects_classes(key_object_infos):
    classes = []
    for obj_id in key_object_infos.keys():
        obj_data = key_object_infos[obj_id]
        classes.append(obj_data["Visual_description"].split(".")[0])
    return classes


def get_key_objects_locations(key_object_infos):
    locations = []
    for obj_id in key_object_infos.keys():
        locations.append(obj_id)
    return locations


def extract_data(root_path, save_path, exclude_tags=[]):
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
            classes = get_key_objects_classes(frame_data_infos)

            # get the location of the important objects
            locations = get_key_objects_locations(frame_data_infos)

            # get the questions and answers of the perception
            perception = frame_data_qa["perception"]
            prediction = frame_data_qa["prediction"]
            planning = frame_data_qa["planning"]
            behavior = frame_data_qa["behavior"]

            # Perception: get the importance questions
            # perception_tagged = set()
            for qa in perception:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="perception", classes=classes
                )
                if (
                    question_tag and 2 in question_tag.tag
                    # and 2 not in perception_tagged
                ):
                    qa["tag"] = question_tag.tag
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "perception"
                    ].append(qa)
                    # perception_tagged.add(2)
                    # break

            # Perception: get the moving status questions
            for qa in perception:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="perception", classes=classes
                )
                if (
                    question_tag and 0 in question_tag.tag
                    # and 0 not in perception_tagged
                ):
                    qa["tag"] = question_tag.tag
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "perception"
                    ].append(qa)
                    # perception_tagged.add(0)
                    # break

            # Prediction: graph
            prediction_tagged = set()
            for qa in prediction:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="prediction", locations=locations
                )
                if (
                    question_tag
                    and 3 in question_tag.tag
                    and 3 not in prediction_tagged
                ):
                    qa["tag"] = question_tag.tag
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "prediction"
                    ].append(qa)
                    prediction_tagged.add(3)
                    break

            # Prediction: yes/no
            for qa in prediction:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="prediction", locations=locations
                )
                if (
                    question_tag
                    and 0 in question_tag.tag
                    and 0 not in prediction_tagged
                ):
                    qa["tag"] = question_tag.tag
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "prediction"
                    ].append(qa)
                    prediction_tagged.add(0)
                    break

            # Planning: safe actions, collision, actions
            planning_subtypes_added = set()
            for qa in planning:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="planning"
                )
                if question_tag and question_tag.subtype not in planning_subtypes_added:
                    qa["tag"] = question_tag.tag
                    test_data[scene_id]["key_frames"][frame_id]["QA"][
                        "planning"
                    ].append(qa)
                    planning_subtypes_added.add(question_tag.subtype)

                # Check if all question types have been added and exit the loop
                if planning_subtypes_added == {
                    "actions",
                    "collision",
                    "safe_actions",
                }:
                    break

            for qa in behavior:
                question = qa["Q"]
                answer = qa["A"]
                question_tag = get_question_tag_test(
                    question, answer, qa_type="behavior"
                )
                if question_tag is None:
                    continue
                qa["tag"] = question_tag.tag
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
    args = parser.parse_args()
    exclude_tags = json.loads(args.exclude_tags) if args.exclude_tags else []

    extract_data(
        args.root_path,
        args.save_path,
        exclude_tags=[int(tag) for tag in exclude_tags],
    )
