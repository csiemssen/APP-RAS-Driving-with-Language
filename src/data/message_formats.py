from abc import ABC, abstractmethod


class MessageFormat(ABC):
    @abstractmethod
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str = None,
        answer: str = "",
    ) -> dict[str, str | list[dict[str, str]]]:
        pass


class QwenMessageFormat(MessageFormat):
    def format(
        self,
        question: str,
        key_object_info: dict,
        image_path: str,
        system_prompt: str = None,
        answer: str = "",
    ) -> dict[str, str | list[dict[str, str]]]:
        content = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})
        content.append({"type": "text", "text": "Question: " + question})
        content.append(
            {
                "type": "image",
                "image": f"file://{image_path}",
            }
        )
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
        system_prompt: str = None,
        answer: str = "",
    ) -> dict[str, str | list[dict[str, str]]]:
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
        system_prompt: str = None,
        answer: str = "",
    ) -> dict[str, str | list[dict[str, str]]]:
        full_prompt = ""
        if system_prompt:
            full_prompt += system_prompt + "\n\n"
        full_prompt += "Question: " + question
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
        system_prompt: str = None,
        answer: str = "",
    ) -> dict[str, str | list[dict[str, str]]]:
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
        return {
            "role": "user",
            "content": content,
        }
