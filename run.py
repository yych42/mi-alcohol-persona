import os
from dotenv import load_dotenv

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import PushToHub, KeepColumns

from steps.load_data import LoadDataFromHub
from steps.background_sketch import BackgroundSketch
from steps.profile import ProfileGeneration
from steps.save_data import PushToHub

load_dotenv()


with Pipeline(
    name="mi-alcohol-persona-dev",
    description="",
) as pipeline:
    load_dataset = LoadDataFromHub(
        repo_id="ychen/diverse-persona-10k-no-minors",
        split="train",
        output_mappings={"persona": "persona"},
        num_examples=1000,
        shuffle=True,
    )

    llm = OpenAILLM(
        model="anthropic:claude-3-5-sonnet-20241022",
        api_key=os.getenv("OPENPIPE_API_KEY"),
        base_url="https://api.openpipe.ai/api/v1",
        generation_kwargs={
            "temperature": 0.9,
            "max_new_tokens": 4096,
        },
    )

    background_sketch = BackgroundSketch(
        llm=llm,
        column="persona",
        input_batch_size=5,
    )

    alcohol_profile = ProfileGeneration(
        llm=llm,
        columns=["persona", "background_sketch"],
        input_batch_size=5,
    )

    keep_columns = KeepColumns(columns=["persona", "background_sketch", "profile"])

    push_to_hub = PushToHub(
        repo_id="ychen/mi-alcohol-persona-1k-claude",
        private=True,
        token="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",
        dropna=True,
    )

    load_dataset >> background_sketch >> alcohol_profile >> keep_columns >> push_to_hub


if __name__ == "__main__":
    distiset = pipeline.run(
        use_cache=True,
    )
