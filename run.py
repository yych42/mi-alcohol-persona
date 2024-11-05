import os
from distilabel.llms import AnthropicLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromHub
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv
from steps.yaml_extractor import YamlExtractorStep


def load_template(templates_dir: str, task_name: str) -> str:
    """
    Load a template from a text file in the specified directory.

    Args:
        templates_dir (str): Path to the directory containing template files
        task_name (str): Name of the task/template to load

    Returns:
        str: Content of the template file

    Raises:
        FileNotFoundError: If the template file doesn't exist
    """
    template_path = os.path.join(templates_dir, f"{task_name}.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def create_pipeline(templates_dir: str) -> Pipeline:
    """
    Create and configure the pipeline with templates loaded from files.

    Args:
        templates_dir (str): Path to the directory containing template files

    Returns:
        Pipeline: Configured pipeline instance
    """
    load_dotenv()

    with Pipeline(
        name="mi-alcohol-persona-dev",
        description="",
    ) as pipeline:
        load_dataset = LoadDataFromHub(
            repo_id="proj-persona/PersonaHub",
            split="train",
            config="persona",
            output_mappings={"persona": "persona"},
            num_examples=10,
        )

        background_sketch = TextGeneration(
            name="background-sketch",
            # llm=AnthropicLLM(
            #     model="claude-3-5-sonnet-20241022",
            #     api_key=os.getenv("ANTHROPIC_API_KEY"),
            #     generation_kwargs={
            #         "temperature": 0.9,
            #         "max_tokens": 4096,
            #     },
            # ),
            llm=OpenAILLM(
                model="anthropic:claude-3-5-sonnet-20241022",
                api_key=os.getenv("OPENPIPE_API_KEY"),
                base_url="https://api.openpipe.ai/api/v1",
                generation_kwargs={
                    "temperature": 0.9,
                    "max_new_tokens": 4096,
                },
            ),
            input_batch_size=3,
            template=load_template(templates_dir, "background-sketch"),
            columns=["persona"],
            output_mappings={"generation": "background_sketch_raw"},
        )

        background_sketch_yaml_extraction = YamlExtractorStep(
            input_mappings={"text": "background_sketch_raw"},
            output_mappings={"extracted_yaml": "background_sketch"},
        )

        alcohol_profile = TextGeneration(
            name="alcohol-profile",
            llm=OpenAILLM(
                model="anthropic:claude-3-5-sonnet-20241022",
                api_key=os.getenv("OPENPIPE_API_KEY"),
                base_url="https://api.openpipe.ai/api/v1",
                generation_kwargs={
                    "temperature": 0.9,
                    "max_new_tokens": 4096,
                },
            ),
            input_batch_size=3,
            template=load_template(templates_dir, "alcohol-profile"),
            columns=["persona", "background_sketch"],
            output_mappings={"generation": "alcohol_profile_raw"},
        )

        alcohol_profile_yaml_extraction = YamlExtractorStep(
            input_mappings={"text": "alcohol_profile_raw"},
            output_mappings={"extracted_yaml": "alcohol_profile"},
        )

        keep_columns = KeepColumns(
            columns=["persona", "background_sketch", "alcohol_profile"],
            use_cache=False,
        )

        (
            load_dataset
            >> background_sketch
            >> background_sketch_yaml_extraction
            >> alcohol_profile
            >> alcohol_profile_yaml_extraction
            >> keep_columns
        )

    return pipeline


if __name__ == "__main__":
    # Specify the directory containing your template files
    templates_directory = "prompts"  # You can change this path as needed

    # Create and run the pipeline
    pipeline = create_pipeline(templates_directory)
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
