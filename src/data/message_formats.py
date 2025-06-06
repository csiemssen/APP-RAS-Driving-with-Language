from abc import ABC, abstractmethod


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self, question: str, key_object_info: dict, image_path: str
    ) -> dict[str, str | list[dict[str, str]]]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self, question: str, key_object_info: dict, image_path: str
    ) -> dict[str, str | list[dict[str, str]]]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {"type": "text", "text": question},
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


class InternVLMessageFormat(MessageFormat):
    def format(
        self, question: str, key_object_info: dict, image_path: str
    ) -> dict[str, str | list[dict[str, str]]]:
        full_prompt = question
        if key_object_info:
            full_prompt += "\n\nKey object infos:\n" + str(key_object_info)

        return {
            "text": full_prompt,
            "image_path": image_path,
        }


class GemmaMessageFormat(MessageFormat):
    def format(
        self, question: str, key_object_info: dict, image_path: str
    ) -> dict[str, str | list[dict[str, str]]]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
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
