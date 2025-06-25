import re
import unittest

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from src.data.basic_dataset import DriveLMImageDataset, simple_dict_collate
from src.data.message_formats import InternVLMessageFormat
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.reasoning.reasoning_engine import ReasoningEngine
from src.utils.utils import is_cuda


@pytest.mark.reasoning
class TestReasoningInternVL(unittest.TestCase):
    def test_reasoning_engine_intern_vl(self):
        if is_cuda():
            engine = InternVLInferenceEngine(
                "OpenGVLab/InternVL3-2B",
                use_4bit=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            engine = InternVLInferenceEngine("OpenGVLab/InternVL3-2B")

        engine.load_model()

        reasoning_engine = ReasoningEngine(engine)

        # Load the dataset
        dataset = DriveLMImageDataset(
            InternVLMessageFormat(), split="val", use_grid=True
        )

        filtered_dataset = [
            item
            for item in dataset
            if item.qa_type in ["prediction", "planning"]
            and re.search(r"<(.*?)>", item.question)
        ]

        test_set = Subset(filtered_dataset, np.arange(2))
        dataloader = DataLoader(test_set, batch_size=2, collate_fn=simple_dict_collate)
        for batch in dataloader:
            processed_batch = reasoning_engine.process_batch(batch)

            # Verify that context pairs were added
            for original_item, processed_item in zip(batch, processed_batch):
                self.assertEqual(
                    original_item.qa_id,
                    processed_item.qa_id,
                    "QA IDs should match.",
                )
                self.assertGreater(
                    len(processed_item.context_pairs),
                    0,
                    "Processed item should have context pairs added.",
                )


if __name__ == "__main__":
    unittest.main()
