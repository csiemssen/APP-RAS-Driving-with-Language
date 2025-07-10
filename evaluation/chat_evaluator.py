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

import os
import time
from abc import ABC, abstractmethod
from multiprocessing import Pool

import dotenv


class ChatEvaluator(ABC):
    def __init__(self, model=None):
        self.model = model
        self.client = None
        self.system_message = (
            "You are an evaluator who rates answers based on correctness."
        )
        # Load environment variables from .env file before setting up the client
        dotenv.load_dotenv()
        self._setup_client()

    @abstractmethod
    def _setup_client(self):
        pass

    def _call_api(self, messages, **kwargs):
        pass

    def prepare_messages(self, prompt):
        messages = [{"role": "system", "content": self.system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})

        return messages

    def create_prompt(self, answer, GT):
        return (
            "Rate my answer based on the correct answer out of 100, with higher scores "
            "indicating that the answer is closer to the correct answer, and you should "
            "be accurate to single digits like 62, 78, 41,etc. Output the number only. "
            f"This is the correct answer: {GT}. "
            f"This is my answer: {answer}"
        )

    def forward(self, data):
        answer, GT = data
        prompt = self.create_prompt(answer, GT)
        messages = self.prepare_messages(prompt)

        reply, total_tokens = self._call_api(messages, max_tokens=3000)
        return reply.strip(), total_tokens

    def evaluate_batch(self, data_list, num_workers=32):
        with Pool(num_workers) as p:
            scores = p.map(self.forward, data_list)
        return scores


class OpenAIEvaluator(ChatEvaluator):
    def _setup_client(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def _call_api(self, messages, **kwargs):
        model = self.model or "gpt-3.5-turbo"
        max_tokens = kwargs.get("max_tokens", 40)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens,
        )

        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return reply, total_tokens


class GeminiEvaluator(ChatEvaluator):
    def _setup_client(self):
        from google import genai

        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()

    def _call_api(self, messages, **kwargs):
        from google.genai import types

        model = self.model or "gemini-2.0-flash"
        max_tokens = kwargs.get("max_tokens", 40)

        response = self.client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=self.system_message,
                temperature=0.6,
                max_output_tokens=max_tokens,
            ),
        )

        reply = response.text
        total_tokens = response.usage_metadata.total_token_count
        return reply, total_tokens

    def _call_api_batch(self, data_list, **kwargs):
        model = self.model or "gemini-2.0-flash"

        inline_requests = [
            {
                "contents": [
                    {
                        "parts": [{"text": self.create_prompt(answer, GT)}],
                        "role": "user",
                    }
                ]
            }
            for answer, GT in data_list
        ]

        batch_job = self.client.batches.create(
            model=model or "gemini-2.0-flash",
            src=inline_requests,
        )

        job_name = self.client.batches.get(name=batch_job.name)

        print(f"Created batch job: {job_name}")

        while batch_job.state.name not in set(
            [
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ]
        ):
            print(f"Current state: {batch_job.state.name}")
            time.sleep(30)  # Wait for 30 seconds before polling again
            batch_job = self.client.batches.get(name=job_name)

        print(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == "JOB_STATE_FAILED":
            print(f"Error: {batch_job.error}")

        if batch_job.state.name == "JOB_STATE_SUCCEEDED":
            print(f"Batch job {batch_job.name} completed successfully.")
            token = self.client.count_tokens(
                [
                    response.text
                    for response in batch_job.dest.inlined_responses.usage_metadata.total_token_count
                ]
            )
            return batch_job.dest.inlined_responses, token

    def prepare_messages(self, prompt):
        return [prompt]

    def evaluate_batch(self, data_list, max_batch_size=5):
        chunks = [
            data_list[i : i + max_batch_size]
            for i in range(0, len(data_list), max_batch_size)
        ]

        all_scores = []
        total_tokens_used = 0

        for chunk in chunks:
            scores, tokens_used = self._call_api_batch(chunk, model=self.model)
            all_scores.extend(scores)
            total_tokens_used += tokens_used

        return all_scores, total_tokens_used
