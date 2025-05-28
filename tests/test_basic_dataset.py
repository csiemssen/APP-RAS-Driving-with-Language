import unittest
import pytest
from src.data.basic_dataset import DriveLMImageDataset
from src.data.message_formats import QwenMessageFormat

@pytest.mark.dataset
class TestDriveLMImageDataset(unittest.TestCase):
    def test_train_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="train")
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

    def test_val_dataset_is_not_empty(self):
        dataset = DriveLMImageDataset(message_format=QwenMessageFormat(), split="val")
        self.assertGreater(len(dataset), 0, "Validation dataset should not be empty")

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

if __name__ == "__main__":
    unittest.main()
