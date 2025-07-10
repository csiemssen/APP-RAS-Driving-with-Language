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
import os
import re
import sys

import language_evaluation
import numpy as np

sys.path.append(".")
from evaluation.chat_evaluator_factory import create_evaluator


class evaluation_suit:
    def __init__(self, chat_provider=None, chat_model=None):
        self.language_eval = language_evaluation.CocoEvaluator(
            coco_types=["BLEU", "ROUGE_L", "CIDEr"]
        )
        self.GPT = []

        if chat_provider is None:
            self.chat_eval = None
            print("Chat provider is not set. Skipping chat evaluation.")
        else:
            self.chat_eval = create_evaluator(provider=chat_provider, model=chat_model)

        self.accuracy = {"answer": [], "GT": [], "idx": []}
        self.language = {"answer": [], "GT": [], "idx": []}
        self.match = {"match": {"answer": [], "GT": [], "idx": []}, "GPT": []}
        self.per_question_scores = {}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            idx = self.accuracy["idx"][i]

            if answer == GT:
                score = 1.0
            else:
                score = 0.0
            scores.append(score)
            self.per_question_scores.setdefault(idx, {})["accuracy"] = score

        if len(scores) == 0:
            return 0.0

        scores = sum(scores) / len(scores)
        return scores

    def eval_chat(self, data):
        if not self.chat_eval:
            print("Chat evaluation is not set up. Skipping evaluation.")
            return 0.0

        results = [self.chat_eval.forward(item) for item in data]

        scores = [float(result[0]) for result in results]
        tokens_used = sum(result[1] for result in results)

        print(f"Chat evaluation used {tokens_used} tokens.")

        if len(scores) == 0:
            return 0.0

        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        idx = self.language["idx"]

        if not idx:
            return None

        results_gen = self.language_eval.run_evaluation(answer, GT)

        for idx, answer, GT in zip(idx, answer, GT):
            results_gen = self.language_eval.run_evaluation([answer], [GT])
            self.per_question_scores.setdefault(idx, {}).update(results_gen)

        results_gen_dict = {f"val/{k}": v for k, v in results_gen.items()}
        return results_gen_dict

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            idx = self.match["match"]["idx"][i]
            _, F1_score = self.match_result(answer, GT)
            score = F1_score * 100
            outs1.append(score)
            self.per_question_scores.setdefault(idx, {})["match"] = score

        if len(outs1) == 0:
            return 0.0

        outs1 = sum(outs1) / len(outs1)

        if not self.chat_provider:
            print("Chat evaluation is not set up. Skipping evaluation.")
            return outs1
        else:
            outs2 = self.eval_chat(self.match["GPT"])
            return (outs1 + outs2) / 2.0

    def eval_graph(self, question):
        # check if answer in self.graph
        question_nums = re.findall(r"\d+\.\d+", question)
        question_nums = np.array(
            [list(map(float, x.split()))[0] for x in question_nums]
        ).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """

        answer_nums = re.findall(r"\d+\.\d+", answer)
        GT_nums = re.findall(r"\d+\.\d+", GT)
        # transform string into float
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        answer_nums = np.array(
            [list(map(float, x.split()))[0] for x in answer_nums]
        ).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(
            -1, 2
        )
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float("inf")
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)
                GT_nums = np.delete(GT_nums, closest_id, axis=0)
            else:
                false_positives += 1

        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT, idx, question):
        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
            self.accuracy["idx"].append(idx)
        if 1 in tag:
            self.GPT.append((answer, GT))
        if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
            self.language["idx"].append(idx)
        if 3 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
            self.match["match"]["idx"].append(idx)
            self.match["GPT"].append((answer, GT))

        self.per_question_scores.setdefault(idx, {}).update(
            {"tag": tag, "question": question, "GT": GT, "answer": answer}
        )

    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["chatgpt"] = self.eval_chat(self.GPT)
        scores["language"] = self.eval_language()
        scores["match"] = self.eval_match()
        scores["per_question_scores"] = self.per_question_scores

        return scores


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--prediction_file",
        type=str,
        default="./data/output/output.json",
        help="path to prediction file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/drivelm/v1_1_test_nus.json",
        help="path to test file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/local_eval/result.json",
        help="path to output file",
    )

    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="Whether to skip missing predictions completely or use worst case response for missing prediction.",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to override existing output file if it exists.",
    )

    parser.add_argument(
        "--chat_provider",
        type=str,
        default=None,
        choices=["openai", "google"],
        help="Chat evaluation provider to use (default: google)",
    )

    parser.add_argument(
        "--chat_model",
        type=str,
        default=None,
        help="Model to use for chat evaluation (e.g., gpt-3.5-turbo, gpt-4, gemini-1.5-flash, claude-3-sonnet-20240229)",
    )

    args = parser.parse_args()

    with open(args.prediction_file, "r") as f:  # , \
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}

    with open(args.test_file, "r") as f:
        test_file = json.load(f)

    evaluation = evaluation_suit(
        chat_provider=args.chat_provider,
        chat_model=args.chat_model,
    )

    missing_predictions_counter = 0
    available_predictions_counter = 0
    questions_counter = 0
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]["key_frames"]

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]["QA"]
            first_flag = True

            for i, qa in enumerate(
                frame_data_qa["perception"]
                + frame_data_qa["prediction"]
                + frame_data_qa["planning"]
                + frame_data_qa["behavior"]
            ):
                question = qa["Q"]
                GT = qa["A"]
                tag = qa["tag"]
                idx = scene_id + "_" + frame_id + "_" + str(i)

                questions_counter += 1
                if idx in pred_file:
                    if question != pred_file[idx]["question"]:
                        raise Exception(
                            f"Question mismatch for {idx}. Expected: {question}, Found: {pred_file[idx]['question']}"
                        )

                    predict = pred_file[idx]["answer"]
                    available = True
                    available_predictions_counter += 1
                else:
                    predict = ""
                    available = False
                    missing_predictions_counter += 1
                    print(f"Warning: No prediction found for {idx}")

                if args.ignore_missing and not available:
                    print(
                        f"Skipping missing prediction for {idx} as ignore_missing is set to True."
                    )
                    continue

                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    evaluation.forward(tag, predict, GT, idx, question)
                else:
                    if evaluation.eval_graph(question):
                        res = evaluation.forward(tag, predict, GT, idx, question)

    output = evaluation.evaluation()
    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("match score: ", output["match"])
    print("language score: ", output["language"])

    # Normalize to 0-1 and combine the scores: chatgpt, language, match, accuracy
    scores = []
    weights = [0.4, 0.2, 0.2, 0.2]

    # chatGPT
    score = output["chatgpt"] / 100.0
    scores.append(score)

    # language
    if output["language"]:
        score = 0
        for idx, key in enumerate(output["language"].keys()):
            if idx < 4:
                score += output["language"][key] / 4.0 / 3.0
            elif idx == 4:
                score += output["language"][key] / 3.0
            else:
                score += output["language"][key] / 10.0 / 3.0
        scores.append(score)

    # match
    score = output["match"] / 100.0
    scores.append(score)

    # accuracy
    score = output["accuracy"]
    scores.append(score)

    final_score = sum([x * y for x, y in zip(scores, weights)])
    print("final score: ", final_score)
    print(
        f"Evaluated {questions_counter} questions, "
        f"available predictions: {available_predictions_counter}, "
        f"missing predictions: {missing_predictions_counter}"
    )

    pred_dir = os.path.dirname(args.prediction_file)
    pred_base = os.path.basename(args.prediction_file)

    out_base = re.sub(r"(output)?\.json$", "", pred_base)
    output_file = os.path.join(args.output_path, f"{out_base}results.json")

    counter = 1
    while os.path.exists(output_file) and not args.override:
        print(f"Warning: {output_file} already exists.")
        output_file = os.path.join(
            args.output_path, f"{out_base}results_{counter}.json"
        )
        counter += 1

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)
        print(f"Results saved to {output_file}")
