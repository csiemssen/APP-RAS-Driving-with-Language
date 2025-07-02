from typing import Dict, List, Tuple

from src.data.generate_descriptor_qas import (
    generate_descriptor_core_question,
)
from src.data.query_item import QueryItem
from src.models.base_inference import BaseInferenceEngine
from src.utils.logger import get_logger
from src.utils.utils import parse_key_objects

logger = get_logger(__name__)


class ReasoningEngine:
    def __init__(self, engine: BaseInferenceEngine):
        self.engine = engine

    def process_batch(self, batch_items: List[QueryItem]) -> List[QueryItem]:
        if not self.engine:
            raise ValueError("Inference engine is required for evaluation mode.")

        item_map: Dict[str, QueryItem] = {item.qa_id: item for item in batch_items}
        descriptor_items: List[Tuple[str, QueryItem]] = []

        for item in batch_items:
            if item.qa_type == "augmented" or item.qa_type == "perception":
                continue

            key_objects = parse_key_objects(item.question)
            if not key_objects:
                logger.debug(
                    f"No key objects found in question: {item.question}. Skipping reasoning context generation."
                )
                continue
            descriptor_question = generate_descriptor_core_question(key_objects)

            if descriptor_question:
                desc_item = QueryItem(
                    question=descriptor_question,
                    image_path=item.image_path,
                    qa_id=f"{item.qa_id}_reasoning",
                    qa_type=item.qa_type,
                    key_object_info=item.key_object_info,  # note not available in eval mode
                    system_prompt=item.system_prompt,
                )
                desc_item.formatted_message = desc_item.format_message(
                    self.engine.message_formatter
                )
                descriptor_items.append((item.qa_id, desc_item))

        if descriptor_items:
            descriptor_messages = [
                [desc_item.formatted_message] for _, desc_item in descriptor_items
            ]
            descriptor_answers = self.engine.predict_batch(descriptor_messages)

            for i, (parent_id, desc_item) in enumerate(descriptor_items):
                answer = descriptor_answers[i]
                parent_item = item_map[parent_id]
                parent_item.context_pairs.append((desc_item.question, answer))

            for item in batch_items:
                if item.context_pairs:
                    item.formatted_message = item.format_message(
                        self.engine.message_formatter
                    )

        logger.info(
            f"Processed {len(batch_items)} items with reasoning engine and added {len(descriptor_items)} descriptor qas."
        )
        return batch_items
