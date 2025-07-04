import unittest

from src.data.message_formats import (
    GemmaMessageFormat,
    InternVLMessageFormat,
    QwenMessageFormat,
)


class TestMessageFormat(unittest.TestCase):
    def test_format_of_qwen_message(self):
        # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
        message_format = QwenMessageFormat()
        question = "Describe this image."
        key_object_info = {"object": "car", "color": "red"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"

        formatted_message = message_format.format(
            question=question,
            key_object_info=key_object_info,
            image_path=image_path,
            system_prompt=system_prompt,
            max_pixels=1,
            min_pixels=0,
        )

        expected_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": "Question: " + question},
                {
                    "type": "image",
                    "image": "file:///path/to/your/image.jpg",
                    "min_pixels": 0,
                    "max_pixels": 1,
                },
                {
                    "type": "text",
                    "text": "Key object infos:\n{'object': 'car', 'color': 'red'}",
                },
            ],
        }
        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message does not match the expected qwen message format",
        )

    def test_format_of_internvl_message(self):
        # https://huggingface.co/OpenGVLab/InternVL3-2B
        message_format = InternVLMessageFormat()
        question = "What is the color of the car?"
        key_object_info = {"object": "car", "color": "blue"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"

        formatted_message = message_format.format(
            question, key_object_info, image_path, system_prompt
        )

        expected_message = {
            "text": "This is the system prompt\n\nQuestion: What is the color of the car?\n\nKey object infos:\n{'object': 'car', 'color': 'blue'}",
            "image_path": image_path,
        }
        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message does not match the expected InternVL message format",
        )

    def test_format_of_gemma_message(self):
        message_format = GemmaMessageFormat()
        question = "What is the capital of France?"
        key_object_info = {"country": "France", "capital": "Paris"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"
        formatted_message = message_format.format(
            question, key_object_info, image_path, system_prompt
        )

        expected_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": "Question: " + question},
                {
                    "type": "text",
                    "text": "Key object infos:\n{'country': 'France', 'capital': 'Paris'}",
                },
            ],
        }

        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message does not match the expected Gemma message format",
        )


if __name__ == "__main__":
    unittest.main()
