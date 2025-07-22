import os

import yaml

from src.utils.utils import has_options


class SystemPromptProvider:
    def __init__(self, config_path=None):
        self.prompts = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.prompts = yaml.safe_load(f)

    def get_approach_prompt(
        self, use_grid: bool = False, use_reasoning: bool = False
    ) -> str:
        approach = self.prompts.get("approach_prompt", {})
        prompt = approach.get("base", "You are an autonomous driving assistant. ")

        grid_prompts = approach.get("use_grid", {})
        if use_grid:
            prompt += grid_prompts.get(
                "enabled",
                "You are provided with a 2×3 image grid representing the current driving scene from different camera angles.\n"
                "The **top row**, from left to right, contains images from the following cameras:\n"
                "- FRONT_LEFT\n"
                "- FRONT\n"
                "- FRONT_RIGHT\n"
                "The **bottom row**, from left to right, contains images from:\n"
                "- BACK_LEFT\n"
                "- BACK\n"
                "- BACK_RIGHT\n",
            )
        else:
            prompt += grid_prompts.get(
                "disabled",
                "You receive a single image from the front camera. ",
            )

        if use_reasoning:
            prompt += approach.get(
                "use_reasoning",
                "In addition to the image input, you receive context from a previously answered question that should be used as information for the following question. ",
            )

        return prompt

    def get_general_prompt(self):
        return self.prompts.get(
            "general_prompt",
            "Objects are labeled as <c, CAM, x, y>, where c is the ID, CAM is the camera name, and x, y are the 2D coordinates of the object center. Provide accurate, concise, and context-aware response for the given question",
        )

    def get_question_type_prompt(self, question_type: str):
        # Try to get from config first
        config_prompts = self.prompts.get("question_type_prompts", {})
        if question_type in config_prompts:
            return config_prompts[question_type]

        return ""

    def get_question_specific_prompt(self, question_type: str, question: str) -> str:
        q_lower = question.lower()
        specific = self.prompts.get("question_specific_prompts", {}).get(
            question_type, {}
        )
        match question_type:
            case "perception":
                if (
                    "what are the important objects in the current scene" in q_lower
                ):  # metric: language
                    return specific.get(
                        "importance_objects",
                        (
                            "Your response must follow this exact structure:\n\n"
                            "1. Start with a single sentence listing the important objects using their **type**, **color**, and **relative position** with respect to the ego vehicle.\n"
                            "- Join multiple objects in a single sentence using commas, and end the sentence with a period.\n\n"
                            "2. On a new sentence, list the **IDs of those objects**, using their full **object descriptors in angle brackets** exactly as given in the input. Use this format:\n"
                            "'The IDs of these objects are <...>, <...>, and <...>.'\n"
                            "- You must list the descriptors in the **same order** as the objects were mentioned.\n"
                            "- Do not change, shorten, or reformat the descriptors in any way.\n\n"
                            "3. Do not include more or fewer objects than those intended.\n"
                            "- Do not provide any explanation, reasoning, or formatting beyond the two sentences.\n\n"
                            "Example (structure only):\n"
                            "There is a gray SUV to the back of the ego vehicle, a blue sedan to the front, and a white truck to the back. The IDs of these objects are <c1,CAM_BACK,960.0,495.8>, <c2,CAM_FRONT,912.5,486.7>, and <c3,CAM_BACK,870.8,479.2>.\n"
                        ),
                    )
                if "what is the moving status of object" in q_lower and has_options(
                    question, 2
                ):  # metric: accuracy
                    return specific.get(
                        "moving_status",
                        (
                            "Select the most likely choice from the given options\n"
                            "The question includes a list of answer options labeled A, B, C, D etc.\n"
                            "You must select the **correct answer by returning only the letter** corresponding to the best option.\n"
                            "Do not output anything except a single uppercase letter (e.g., 'A', 'B', 'C', 'D').\n"
                        ),
                    )
            case "prediction":
                if (
                    "what object should the ego vehicle notice first when the ego vehicle is getting to the next possible location"
                    in q_lower
                ):  # metric: match
                    return specific.get(
                        "graph",
                        (
                            "Your response must follow these requirements strictly:\n\n"
                            "1. **Always describe exactly three objects** in the order of importance: first, second, and third.\n"
                            "2. **Always refer to each object using its full descriptor** in angle bracket.\n"
                            "- Incorrect: the ego vehicle should first notice the traffic sign.\n"
                            "- Correct: The ego vehicle should first notice `<c4,CAM_FRONT,729.1,801.9>`.\n"
                            "3. For each object, follow this full structure:\n"
                            "- 'The ego vehicle should [first/then/finally,...] notice <OBJECT_DESCRIPTOR>, which is in the state of [OBJECT_STATE], so the ego vehicle should [ACTION].'\n"
                            "4. Use clear transition phrases to separate the three observations, such as: 'first', 'then', 'lastly', 'secondly', 'finally', etc.\n"
                            "5. Do not provide extra commentary or reasoning beyond what is described above."
                            "- Always use the **exact full object descriptor**.\n"
                        ),
                    )
                if q_lower.strip().startswith(
                    ("are there", "is", "will", "would")
                ):  # metric: accuracy
                    return specific.get(
                        "yes_no",
                        "Respond only with ‘Yes.’ or ‘No.’ (including the period). Do not provide any additional text, explanation, or variation.\n",
                    )
            case "planning":  # metric: gpt
                if "what actions could the ego vehicle take" in q_lower:
                    return specific.get(
                        "actions",
                        (
                            "Your response should include three parts:\n"
                            "1. **The action** the ego vehicle could take (e.g., turn right, stay stationary, slow down, brake, accelerate, ...).\n"
                            "2. **The reason** for this action (e.g., to avoid a collision, to follow traffic rules, no safety issue,...).\n"
                            "3. **The probability or likelihood** of this action (e.g., high probability, low probability, ...).\n"
                            "4. You can use variations in phrasing, but all three parts — action, reason, and probability — must be clearly expressed.\n\n"
                            "Example:\n"
                            "- The action is to turn right. The reason is to avoid a collision. The probability is high.\n"
                        ),
                    )
                if "lead to a collision" in q_lower:
                    return specific.get(
                        "collision",
                        (
                            "Your response must include:\n"
                            "1. A description of one or more ego vehicle actions that could plausibly lead to a collision (e.g., turning, accelerating, reversing, etc.).\n"
                            "2. A clear indication that those actions could lead to a collision with the specified object.\n"
                            "3. The object must be referred to using its **full descriptor**, exactly as provided in the question.\n"
                            "4. You can use variations in phrasing,, but ensure the object descriptor is present and clearly linked to the action.\n\n"
                            "Example:\n"
                            "Turning right and accelerating through the intersection can lead to a collision with <c2,CAM_FRONT_RIGHT,985.8,610.8>.\n"
                        ),
                    )
                if "safe actions" in q_lower:
                    return specific.get(
                        "safe_actions",
                        (
                            "Your response must include:\n"
                            "1. A list of one or more safe driving actions the ego vehicle can reasonably take (e.g., keep going at the same speed, decelerate gradually without braking, slightly offset to the left, etc.).\n"
                            "2. The actions should be written in natural language, joined with commas, 'and', or 'or'.\n"
                            "3. Do not include any explanation, reasoning, or commentary—only the actions.\n"
                            "4. Use complete, grammatically correct phrases.\n"
                            "5. You may vary the phrasing and number of actions, but all should be context-appropriate and safe.\n\n"
                            "Example:\n"
                            "Keep going at the same speed, decelerate gradually without braking, or slightly offset to the right."
                        ),
                    )
            case "behavior":  # metric: accuracy
                if has_options(question, 2):
                    return specific.get(
                        "default",
                        (
                            "Select the most likely choice from the given options\n"
                            "The question includes a list of answer options labeled A, B, C, etc.\n"
                            "You must select the **correct answer by returning only the letter** corresponding to the best option.\n"
                            "Do not output anything except a single uppercase letter (e.g., 'A', 'B', 'C').\n"
                        ),
                    )
        return None

    def get_system_prompt(self, question_type: str, question: str, **approach_kwargs):
        approach_prompt = self.get_approach_prompt(**approach_kwargs)
        general_prompt = self.get_general_prompt()
        question_type_prompt = self.get_question_type_prompt(question_type)
        question_specific_prompt = self.get_question_specific_prompt(
            question_type, question
        )

        sections = [
            approach_prompt,
            general_prompt,
            question_type_prompt,
            question_specific_prompt,
        ]

        return "\n".join([s for s in sections if s and s.strip()])
