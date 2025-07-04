from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        content = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": "Question: " + question},
            {
                "type": "image",
                "image": f"file://{image_path}",
            },
        ]

        if key_object_info:
            content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }
            )

        if context:
            for context_q, context_a in context:
                content.append(
                    {"type": "text", "text": f"Context Question: {context_q}"}
                )
                content.append({"type": "text", "text": f"Context Answer: {context_a}"})

        return {
            "role": "user",
            "content": content,
        }


class QwenTrainingMessageFormat(MessageFormat):
    """
    Message format adhering to OAI implementation
    """

    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        content = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": "Question: " + question},
            {
                "type": "image",
                "image": f"file://{image_path}",
            },
        ]

        if key_object_info:
            content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }
            )

        if context:
            for context_q, context_a in context:
                content.append(
                    {"type": "text", "text": f"Context Question: {context_q}"}
                )
                content.append({"type": "text", "text": f"Context Answer: {context_a}"})

        return {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer or ""},
                    ],
                },
            ]
        }


class InternVLMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        full_prompt = system_prompt + "\n\nQuestion: " + question

        if key_object_info:
            full_prompt += "\n\nKey object infos:\n" + str(key_object_info)

        if context:
            for context_q, context_a in context:
                full_prompt += (
                    f"\n\nContext Question: {context_q}\nContext Answer: {context_a}"
                )

        return {
            "text": full_prompt,
            "image_path": image_path,
        }


class GemmaMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        content = [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": "Question: " + question},
        ]

        if key_object_info:
            content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }
            )

        if context:
            for context_q, context_a in context:
                content.append(
                    {"type": "text", "text": f"Context Question: {context_q}"}
                )
                content.append({"type": "text", "text": f"Context Answer: {context_a}"})

        return {
            "role": "user",
            "content": content,
        }
