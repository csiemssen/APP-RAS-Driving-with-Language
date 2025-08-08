from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str = None,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        image_path: Optional[str],
        system_prompt: str = None,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})

        if key_object_info:
            content.append(
                {
                    "type": "text",
                    "text": "List of objects in the scene:\n"
                    + key_object_info.__str__(),
                }
            )

        if context:
            for context_q, context_a in context:
                content.append(
                    {"type": "text", "text": f"Context Question: {context_q}"}
                )
                content.append({"type": "text", "text": f"Context Answer: {context_a}"})

        content.append({"type": "text", "text": "Question: " + question})
        if image_path:
            content.append(
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                }
            )
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
        system_prompt: str = None,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        user_content = []
        if system_prompt:
            user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text", "text": "Question: " + question})
        user_content.append(
            {
                "type": "image",
                "image": f"file://{image_path}",
            }
        )
        if key_object_info:
            user_content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }
            )

        if context:
            for context_q, context_a in context:
                user_content.append(
                    {"type": "text", "text": f"Context Question: {context_q}"}
                )
                user_content.append(
                    {"type": "text", "text": f"Context Answer: {context_a}"}
                )

        return {
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ]
        }


class InternVLMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        image_path: str,
        system_prompt: str = None,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        full_prompt = ""
        if system_prompt:
            full_prompt += system_prompt + "\n\n"
        full_prompt += "Question: " + question

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
        system_prompt: str = None,
        answer: Optional[str] = None,
        key_object_info: Optional[dict] = None,
        context: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        content = [
            {
                "type": "image",
                "image": image_path,
            }
        ]
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})
        content.append({"type": "text", "text": "Question: " + question})
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
