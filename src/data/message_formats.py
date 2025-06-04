from abc import ABC, abstractmethod


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
    ) -> dict[str, str | list[dict[str, str]]]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
    ) -> dict[str, str | list[dict[str, str]]]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": "Question: " + question},
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                *(
                    [
                        {
                            "type": "text",
                            "text": "Key object infos:\n"
                            + key_object_info.__str__(),
                        }
                    ]
                    if key_object_info
                    else []
                ),
            ],
        }


class InternVLMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str,
    ) -> dict[str, str | list[dict[str, str]]]:
        full_prompt = system_prompt + "\n\nQuestion: " + question
        if key_object_info:
            full_prompt += "\n\nKey object infos:\n" + str(key_object_info)

        return {
            "text": full_prompt,
            "image_path": image_path,
        }
