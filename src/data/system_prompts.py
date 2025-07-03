def get_general_prompt():
    return (
        "You are an autonomous driving AI assistant. You receive a single image from the front camera. "
        "Objects are labeled as <c, CAM, x, y>, where c is the ID, CAM is the camera name, and x, y are the 2D coordinates of the object center."
        "Provide accurate, concise, and context-aware response in the appropriate format for the question type that complete the request"
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


def get_system_prompt(question_type: str):
    general_prompt = get_general_prompt()
    specific_prompt = get_question_type_prompt(question_type)
    return f"{general_prompt}\n\n{specific_prompt}"
