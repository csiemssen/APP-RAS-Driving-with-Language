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


def get_question_specific_prompt(question_type: str, question: str) -> str:
    q_lower = question.lower()
    match question_type:
        case "perception":
            if (
                "what are the important objects in the current scene" in q_lower
            ):  # metric: language
                return "placeholder importance objects prompt"
            if "what is the moving status of object" in q_lower:  # metric: accuracy
                return "placeholder moving status prompt"

        case "prediction":
            if (
                "what object should the ego vehicle notice first when the ego vehicle is getting to the next possible location"
                in q_lower
            ):  # metric: match
                return "placeholder graph prompt"

            if q_lower.strip().startswith(("are there", "is", "will", "would")):
                return "placeholder yes/no prompt"

        case "planning":
            if "what actions could the ego vehicle take" in q_lower:  # metric: gpt
                return "placeholder actions prompt"
            if "lead to a collision" in q_lower:  # metric: gpt
                return "placeholder collision prompt"
            if "safe actions" in q_lower:  # metric: gpt
                return "placeholder safe actions prompt"

        case "behavior":  # metric: accuracy
            return "placeholder behavior prompt"

    return None


def get_system_prompt(question_type: str, question: str, **approach_kwargs):
    approach_prompt = get_approach_prompt(**approach_kwargs)
    general_prompt = get_general_prompt()
    question_type_prompt = get_question_type_prompt(question_type)
    question_specific_prompt = get_question_specific_prompt(question_type, question)

    return f"{approach_prompt}\n{general_prompt}\n\n{question_type_prompt}\n\n{question_specific_prompt}"
