from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.data.message_formats import MessageFormat


@dataclass
class QueryItem:
    question: str
    image_path: str
    qa_id: str
    qa_type: str
    tags: List[str]
    key_object_info: Optional[Dict[str, Any]] = None
    system_prompt: str = None
    ground_truth_answer: Optional[str] = None

    formatted_message: Optional[List[Dict[str, Any]]] = None

    context_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def format_message(self, formatter: MessageFormat) -> List[Dict[str, Any]]:
        self.formatted_message = formatter.format(
            question=self.question,
            image_path=self.image_path,
            system_prompt=self.system_prompt,
            answer=self.ground_truth_answer,
            key_object_info=self.key_object_info,
            context=self.context_pairs if self.context_pairs else None,
        )
        return self.formatted_message
