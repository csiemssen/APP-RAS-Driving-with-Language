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
                if "what are the important objects in the current scene" in q_lower:
                    return specific.get(
                        "importance_objects",
                        "placeholder importance objects prompt",
                    )
                if "what is the moving status of object" in q_lower:
                    return specific.get(
                        "moving_status", "placeholder moving status prompt"
                    )
            case "prediction":
                if (
                    "what object should the ego vehicle notice first when the ego vehicle is getting to the next possible location"
                    in q_lower
                ):
                    return specific.get("graph", "placeholder graph prompt")
                if q_lower.strip().startswith(("are there", "is", "will", "would")):
                    return specific.get("yes_no", "placeholder yes/no prompt")
            case "planning":
                if "what actions could the ego vehicle take" in q_lower:
                    return specific.get("actions", "placeholder actions prompt")
                if "lead to a collision" in q_lower:
                    return specific.get("collision", "placeholder collision prompt")
                if "safe actions" in q_lower:
                    return specific.get(
                        "safe_actions", "placeholder safe actions prompt"
                    )
            case "behavior":
                return specific.get("default", "placeholder behavior prompt")
        return None

    def get_system_prompt(self, question_type: str, question: str, **approach_kwargs):
        approach_prompt = self.get_approach_prompt(**approach_kwargs)
        general_prompt = self.get_general_prompt()
        question_type_prompt = self.get_question_type_prompt(question_type)
        question_specific_prompt = self.get_question_specific_prompt(
            question_type, question
        )
        return f"{approach_prompt}\n{general_prompt}\n\n{question_type_prompt}\n\n{question_specific_prompt}"
