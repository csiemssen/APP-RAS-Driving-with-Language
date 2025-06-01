import unittest
import pytest
from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat, InternVLMessageFormat

class TestMessageFormat(unittest.TestCase):
    def test_format_of_qwen_message(self):
      # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
      message_format = QwenMessageFormat()
      question = "Describe this image."
      key_object_info = {"object": "car", "color": "red"}
      image_path = "/path/to/your/image.jpg"

      formatted_message = message_format.format(question, key_object_info, image_path)

      expected_message = {
          "role": "user",
          "content": [
              {"type": "image", "image": "file:///path/to/your/image.jpg"},
              {"type": "text", "text": "Describe this image."},
              {"type": "text", "text": "Key object infos:\n{'object': 'car', 'color': 'red'}"},
          ],
      }
      self.assertEqual(formatted_message, expected_message, "Formatted message does not match the expected qwen message format")

    def test_format_of_internvl_message(self):
      # https://huggingface.co/OpenGVLab/InternVL3-2B
      message_format = InternVLMessageFormat()
      question = "What is the color of the car?"
      key_object_info = {"object": "car", "color": "blue"}
      image_path = "/path/to/your/image.jpg"

      formatted_message = message_format.format(question, key_object_info, image_path)

      expected_message = {
          "text": "What is the color of the car?\n\nKey object infos:\n{'object': 'car', 'color': 'blue'}",
          "image_path": image_path,
      }
      self.assertEqual(formatted_message, expected_message, "Formatted message does not match the expected InternVL message format")

if __name__ == "__main__":
    unittest.main()
