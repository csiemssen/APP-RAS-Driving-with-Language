from abc import ABC, abstractmethod


class MessageFormat(ABC):

    @abstractmethod
    def format(self, question: str, key_object_info:dict, image_path: str) -> dict[str, str | list[dict[str, str]]]:
        pass

class QwenMessageFormat(MessageFormat):

    def format(self, question: str, key_object_info: dict, image_path: str) -> dict[str, str | list[dict[str, str]]]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {
                    "type": "text",
                    "text": question
                },
                *([{
                    "type": "text",
                    "text": "Key object infos:\n" + key_object_info.__str__(),
                }] if key_object_info else []),
            ],
        }
