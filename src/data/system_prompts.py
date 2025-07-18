import os

import yaml


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
                "You are provided with a grid of images of the current situation. Starting from the upper left, the upper row shows images from the 'FRONT_LEFT', 'FRONT' and 'FRONT_RIGHT' cameras respectively. Starting from the bottom left, the lower row shows images from the 'BACK_LEFT', 'BACK' and 'BACK_RIGHT' cameras respectively. ",
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

        default_prompts = {
            "perception": (
                "Answer questions about what objects are present and where they are located "
                "relative to the ego vehicle. Include object types and spatial positions (e.g., front left, back right)."
            ),
            "prediction": (
                "Predict the likely future state, movement, or role of a given object in the scene. "
                "Consider direction, intent, or whether the object will interact with the ego vehicle."
            ),
            "planning": (
                "Decide how the ego vehicle should act in the current scene based on relevant objects and traffic context. "
                "Recommend safe or unsafe actions, estimate collision risk, or determine object priority if asked."
            ),
            "behavior": (
                "Describe the current motion or behavior of the ego vehicle (e.g., going straight, slowing down). "
                "Focus only on the ego vehicle’s observed driving behavior."
            ),
        }
        return default_prompts.get(question_type, "")

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
                if "what is the moving status of object" in q_lower:  # metric: accuracy
                    return specific.get(
                        "moving_status",
                        (
                            "Your response must consist of **only one phrase** selected from the list below including the period.\n"
                            "**Do not add any explanation or extra text.**\n\n"
                            "**Allowed phrases:**\n"
                            "- Back up.\n"
                            "- Backward.\n"
                            "- Bending over.\n"
                            "- Reverse parking.\n"
                            "- Turn left.\n"
                            "- Turn right.\n"
                            "- Going ahead.\n"
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
                        "safe_actions", "placeholder safe actions prompt"
                    )
            case "behavior":  # metric: accuracy
                return specific.get(
                    "default",
                    (
                        "Predict the behavior of the ego vehicle. Your response must consist of two full sentences:\n"
                        "1. The first sentence describes the vehicle's **steering or directional behavior**.\n"
                        "2. The second sentence describes the vehicle's **speed or motion status**.\n\n"
                        "Use only one phrase from each list below. Insert each into the template:\n"
                        '"The ego vehicle is [PHRASE]."\n\n'
                        "**Allowed steering/direction phrases:**\n"
                        "- going straight\n"
                        "- slightly steering to the left\n"
                        "- slightly steering to the right\n"
                        "- steering to the left\n"
                        "- steering to the right\n\n"
                        "**Allowed speed/motion phrases:**\n"
                        "- not moving\n"
                        "- driving slowly\n"
                        "- driving with normal speed\n"
                        "- driving fast\n"
                        "- driving very fast\n\n"
                        "Do not include any other text or variation. Combine each selected phrase into the sentence format exactly as shown.\n",
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
