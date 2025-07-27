import openai
import json
import time
from pydantic import ValidationError
from langchain.output_parsers import PydanticOutputParser
from schemas.output_schema import AnswerOutput
from llm.prompt_template import get_prompt_template
from utils import remove_think_content 


class OnlineLLM:
    def __init__(self, api_key: str):
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = api_key
        self.model = "deepseek/deepseek-r1-0528:free"
        self.parser = PydanticOutputParser(pydantic_object=AnswerOutput)

    def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> AnswerOutput:
        context = json.dumps(retrieved_chunks, indent=0)
        format_instructions = self.parser.get_format_instructions()
        prompt = get_prompt_template(question, context, format_instructions)

        max_retries = 5

        for attempt in range(1, max_retries + 1):
            try:
                print(f"[Attempt {attempt}] Calling LLM...")
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )

                raw_text = response["choices"][0]["message"]["content"]
                raw_text = remove_think_content(raw_text)

                #print(f"[Attempt {attempt}] Raw LLM Output:\n{raw_text}\n")

                parsed = self.parser.parse(raw_text)

                return parsed

            except ValidationError as ve:
                print(f"[Attempt {attempt}] Pydantic validation failed:\n{ve}")
            except Exception as e:
                print(f"[Attempt {attempt}] LLM API call failed:\n{e}")

            # Retry logic
            if attempt == max_retries:
                raise RuntimeError("LLM failed after maximum retries.")
            wait_time = 2 ** (attempt - 1)
            print(f"Retrying in {wait_time} seconds...\n")
            time.sleep(wait_time)
