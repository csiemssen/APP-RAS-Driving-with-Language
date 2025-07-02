from typing import Any, Dict, List, Tuple

from src.data.generate_descriptor_qas import (
    generate_descriptor_answer,
    generate_descriptor_core_question,
)
from src.data.query_item import QueryItem
from src.utils.logger import get_logger
from src.utils.utils import parse_key_objects

logger = get_logger(__name__)


def generate_reasoning_context(item: QueryItem) -> List[Tuple[str, str]]:
    if item.qa_type == "augmented" or item.qa_type == "perception":
        return []

    key_objects = parse_key_objects(item.question)
    if not key_objects:
        logger.debug(
            f"No key objects found in question: {item.question}. Skipping reasoning context generation."
        )
        return []

    descriptor_question = generate_descriptor_core_question(key_objects)

    if not descriptor_question:
        logger.info(
            f"Descriptor question could not be generated for key objects: {key_objects}. Skipping reasoning context generation."
        )
        return []

    descriptor_answer = generate_descriptor_question_ground_truth(
        key_objects, item.key_object_info
    )

    if not descriptor_answer:
        logger.info(
            f"No ground truth answer generated for descriptor question: {descriptor_question}. Skipping reasoning context generation."
        )

    return [(descriptor_question, descriptor_answer)]


def generate_descriptor_question_ground_truth(
    key_objects: List[str], key_object_info: Dict[str, Any]
) -> str:
    answers = []
    for obj_id in key_objects:
        # Add back the angle brackets for matching with key_object_info keys
        obj_id_with_brackets = f"<{obj_id}>"

        if obj_id_with_brackets in key_object_info:
            answer = generate_descriptor_answer(
                obj_id_with_brackets, key_object_info[obj_id_with_brackets]
            )
            answers.append(answer)
        else:
            logger.warning(f"No information available for object {obj_id}.")
            answers.append(f"No information available for object {obj_id}.")

    return "; ".join(answers)
