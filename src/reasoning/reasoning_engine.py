import re
from typing import Dict, List, Tuple

from src.data.query_item import QueryItem
from src.models.base_inference import BaseInferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine:
    def __init__(self, engine: BaseInferenceEngine):
        self.engine = engine

    def process_batch(self, batch_items: List[QueryItem]) -> List[QueryItem]:
        item_map: Dict[str, QueryItem] = {item.qa_id: item for item in batch_items}

        descriptor_items: List[Tuple[str, QueryItem]] = []

        # Generate descriptor questions for each item
        for item in batch_items:
            if item.qa_type == "augmented" or item.qa_type == "perception":
                continue

            key_objects = self._parse_key_objects(item.question)

            descriptor_question = self._generate_descriptor_question(key_objects)

            if descriptor_question:
                desc_item = QueryItem(
                    question=descriptor_question,
                    image_path=item.image_path,
                    qa_id=f"{item.qa_id}_reasioning",
                    qa_type=item.qa_type,
                    key_object_info=item.key_object_info,
                    system_prompt=item.system_prompt,
                )

                desc_item.formatted_message = desc_item.format_message(
                    self.engine.message_formatter
                )

                descriptor_items.append((item.qa_id, desc_item))

        # Get Answers for descriptor questions
        if descriptor_items:
            descriptor_messages = [
                desc_item.formatted_message for _, desc_item in descriptor_items
            ]

            descriptor_answers = self.engine.predict_batch([descriptor_messages])

            for i, (parent_id, desc_item) in enumerate(descriptor_items):
                answer = descriptor_answers[i]
                parent_item = item_map[parent_id]

                parent_item.context_pairs.append((desc_item.question, answer))

            for item in batch_items:
                if item.context_pairs:
                    item.formatted_message = item.format_message(
                        self.engine.message_formatter
                    )

        # return batch with updated context pairs
        logger.info(
            f"Processed {len(batch_items)} items with reasoning engine and added {len(descriptor_items)} descriptor qas."
        )
        return batch_items

    def _parse_key_objects(self, question: str) -> List[str]:
        """Extract key objects enclosed in angle brackets from the question."""
        pattern = r"<(.*?)>"
        return re.findall(pattern, question)

    def _generate_descriptor_question(self, key_objects: List[str]) -> str:
        """Generate descriptor questions for the key objects."""
        if not key_objects:
            return []

        object_list = ", ".join([f"<{obj}>" for obj in key_objects])
        if len(key_objects) == 1:
            return f"What is the object {object_list}? What is its state?"
        return f"What are the objects {object_list}? What are their states?"
