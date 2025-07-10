# Adopted from https://github.com/OpenDriveLab/DriveLM. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse

from .chat_evaluator import GeminiEvaluator, OpenAIEvaluator


def create_evaluator(provider, model):
    providers = {
        "openai": OpenAIEvaluator,
        "google": GeminiEvaluator,
    }

    provider = provider.lower()
    if provider not in providers:
        raise ValueError(
            f"Unsupported provider: {provider}. Valid options: {', '.join(providers.keys())}"
        )

    return providers[provider](model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test chat evaluators")
    parser.add_argument(
        "--provider",
        default="google",
        choices=["openai", "google"],
        help="Chat provider to test",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use for the selected provider",
    )

    args = parser.parse_args()

    evaluator = create_evaluator(provider=args.provider, model=args.model)

    data = [
        (
            "The ego vehicle should notice the bus next, as it is the third object in the image. The bus is stopped at the intersection, and the ego vehicle should be cautious when approaching the intersection to ensure it does not collide with the bus.",
            "Firstly, notice <c3,CAM_FRONT_LEFT,1075.5,382.8>. The object is a traffic sign, so the ego vehicle should continue at the same speed. Secondly, notice <c2,CAM_FRONT,836.3,398.3>. The object is a traffic sign, so the ego vehicle should accelerate and continue ahead. Thirdly, notice <c1,CAM_BACK,991.7,603.0>. The object is stationary, so the ego vehicle should continue ahead at the same speed.",
        ),
        # Add more data here
    ]

    scores = [evaluator.evaluate_batch(item) for item in data]

    print(scores)
