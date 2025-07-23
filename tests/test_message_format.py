import unittest

from src.data.message_formats import (
    GemmaMessageFormat,
    InternVLMessageFormat,
    QwenMessageFormat,
)


class TestMessageFormat(unittest.TestCase):
    def test_format_of_qwen_message(self):
        message_format = QwenMessageFormat()
        question = "Describe this image."
        key_object_info = {"object": "car", "color": "red"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"
        context = [("What is this?", "This is a car.")]

        formatted_message = message_format.format(
            question,
            image_path,
            system_prompt,
            key_object_info=key_object_info,
            context=context,
        )

        expected_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": "Question: " + question},
                    {
                        "type": "image",
                        "image": "file:///path/to/your/image.jpg",
                    },
                    {
                        "type": "text",
                        "text": "Key object infos:\n{'object': 'car', 'color': 'red'}",
                    },
                    {"type": "text", "text": "Context Question: What is this?"},
                    {"type": "text", "text": "Context Answer: This is a car."},
                ],
            }
        ]
        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message with context does not match the expected qwen message format",
        )

    def test_format_of_internvl_message(self):
        message_format = InternVLMessageFormat()
        question = "What is the color of the car?"
        key_object_info = {"object": "car", "color": "blue"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"
        context = [("What is this?", "This is a car.")]

        formatted_message = message_format.format(
            question,
            image_path,
            system_prompt,
            key_object_info=key_object_info,
            context=context,
        )

        expected_message = [
            {
                "text": "This is the system prompt\n\nQuestion: What is the color of the car?\n\nKey object infos:\n{'object': 'car', 'color': 'blue'}\n\nContext Question: What is this?\nContext Answer: This is a car.",
                "image_path": image_path,
            }
        ]
        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message with context does not match the expected InternVL message format",
        )

    def test_format_of_gemma_message(self):
        message_format = GemmaMessageFormat()
        question = "What is the capital of France?"
        key_object_info = {"country": "France", "capital": "Paris"}
        image_path = "/path/to/your/image.jpg"
        system_prompt = "This is the system prompt"
        context = [("What is this?", "This is a map of France.")]

        formatted_message = message_format.format(
            question,
            image_path,
            system_prompt,
            key_object_info=key_object_info,
            context=context,
        )

        expected_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": "Question: " + question},
                    {
                        "type": "text",
                        "text": "Key object infos:\n{'country': 'France', 'capital': 'Paris'}",
                    },
                    {"type": "text", "text": "Context Question: What is this?"},
                    {
                        "type": "text",
                        "text": "Context Answer: This is a map of France.",
                    },
                ],
            }
        ]

        self.assertEqual(
            formatted_message,
            expected_message,
            "Formatted message with context does not match the expected Gemma message format",
        )


if __name__ == "__main__":
    unittest.main()
