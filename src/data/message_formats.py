from abc import ABC, abstractmethod
from typing import Optional


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> dict[str, str | list[dict[str, str]]]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> dict[str, str | list[dict[str, str]]]:
        image_dict = {
            "type": "image",
            "image": f"file://{image_path}",
        }
        if min_pixels is not None:
            image_dict["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_dict["max_pixels"] = max_pixels

        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})
        content.append({"type": "text", "text": "Question: " + question})
        content.append(image_dict)
        if key_object_info:
            content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
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
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> dict[str, str | list[dict[str, str]]]:
        image_dict = {
            "type": "image",
            "image": f"file://{image_path}",
        }
        if min_pixels is not None:
            image_dict["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_dict["max_pixels"] = max_pixels

        user_content = []
        if system_prompt:
            user_content.append({"type": "text", "text": system_prompt})
        user_content.append({"type": "text", "text": "Question: " + question})
        user_content.append(image_dict)
        if key_object_info:
            user_content.append(
                {
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }
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
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> dict[str, str | list[dict[str, str]]]:
        full_prompt = system_prompt + "\n\nQuestion: " + question
        if key_object_info:
            full_prompt += "\n\nKey object infos:\n" + str(key_object_info)

        return {
            "text": full_prompt,
            "image_path": image_path,
        }


class GemmaMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
        answer: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> dict[str, str | list[dict[str, str]]]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": "Question: " + question},
                *(
                    [
                        {
                            "type": "text",
                            "text": "Key object infos:\n" + key_object_info.__str__(),
                        }
                    ]
                    if key_object_info
                    else []
                ),
            ],
        }
