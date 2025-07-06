def get_approach_prompt(
    use_grid: bool = False,
    use_reasoning: bool = False,
) -> str:
    prompt = "You are an autonomous driving assistant. "
    if use_grid:
        prompt += (
            "You are provided with a grid of images of the current situation. "
            "Starting from the upper left, the upper row shows images from the 'FRONT_LEFT', 'FRONT' and 'FRONT_RIGHT' cameras respectively. "
            "Starting from the bottom left, the lower row shows images from the 'BACK_LEFT', 'BACK' and 'BACK_RIGHT' cameras respectively. "
        )
    else:
        prompt += "You receive a single image from the front camera. "
    if use_reasoning:
        prompt += "In addition to the image input, you receive context from a previously answered question that should be used as information for the following question. "
    return prompt


def get_general_prompt():
    return (
        "Objects are labeled as <c, CAM, x, y>, where c is the ID, CAM is the camera name, and x, y are the 2D coordinates of the object center."
        "Provide accurate, concise, and context-aware response for the given question"
    )


def get_question_type_prompt(question_type: str):
    prompts = {
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
    return prompts.get(question_type, "")


def get_system_prompt(question_type: str, **approach_kwargs):
    approach_prompt = get_approach_prompt(**approach_kwargs)
    general_prompt = get_general_prompt()
    specific_prompt = get_question_type_prompt(question_type)
    return f"{approach_prompt}\n{general_prompt}\n\n{specific_prompt}"
