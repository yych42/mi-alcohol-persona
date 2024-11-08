import os
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from dotenv import load_dotenv
from steps.background_sketch import BackgroundSketch
from steps.profile import ProfileGeneration

from steps.keep import KeepColumns

load_dotenv()


with Pipeline(
    name="mi-alcohol-persona-dev",
    description="",
) as pipeline:
    load_dataset = LoadDataFromHub(
        repo_id="ychen/diverse-persona-10k-no-minors",
        split="train",
        output_mappings={"persona": "persona"},
    )

    llm = OpenAILLM(
        model="openpipe:alcohol-persona-pipe",
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

    keep_columns = KeepColumns(
        columns=["persona", "background_sketch", "profile"], dropna=True
    )

    load_dataset >> background_sketch >> alcohol_profile >> keep_columns


if __name__ == "__main__":
    # Create and run the pipeline
    distiset = pipeline.run(
        use_cache=False,
    )

    # Push to hub
    distiset.push_to_hub(
        "ychen/mi-alcohol-persona-dev",
        private=True,
        token="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",
        include_script=False,
        generate_card=False,
    )
