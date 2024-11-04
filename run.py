import os
from distilabel.llms import AnthropicLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromHub
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv
from steps.yaml_extractor import YamlExtractorStep

load_dotenv()


with Pipeline(  #
    name="mi-alcohol-persona-dev",
    description="",
) as pipeline:  #
    load_dataset = LoadDataFromHub(
        repo_id="proj-persona/PersonaHub",
        split="train",
        config="persona",
        output_mappings={"persona": "persona"},
        num_examples=3,
    )

    background_sketch = TextGeneration(
        llm=AnthropicLLM(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            generation_kwargs={
                "temperature": 0.9,
                "max_tokens": 4096,
            },
        ),
        template="""You are a screenwriter known for developing rich and realistic characters."""
        """You are invited to contribute to a motivational interviewing research project, where the goal is to develop a set of personas that can be used to train healthcare professionals for motivational interviewing."""
        """For example...

"A recently diagnosed hypertensive patient..."

Could ultimately be extended into "A recently diagnosed hypertensive patient who is hesitant to take prescribed medication and modify salt intake due to disbelief in the diagnosis and concerns about lifelong medication dependence.\""""
        """Here, your current task is simply to design rich and detailed background setups for the personas."""
        """To help with quality control and data collection, you were asked to write with the following YAML template to also demonstrate your step-by-step thought process. The template is as follows:

```yaml
- Personal life: Replace this with a brainstorm of the character's personal life background, based on what we can infer from the description.
- Personality: Replace this with your idea of the character's personality in terms of the BIG FIVE (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) based on the description and their personal life background; pick one high and one low trait that you think are most relevant.
    - <Trait 1>: <High/Low>
    - <Trait 2>: <High/Low>
```

Now, please develop this character by extending the sentence:

"{{persona}}"

Submit your YAML worksheet:""",
        columns=["persona"],
        output_mappings={"generation": "yaml_raw"},
    )

    background_sketch_yaml_extraction = YamlExtractorStep(
        input_mappings={"text": "yaml_raw"},
        output_mappings={"extracted_yaml": "background sketch"},
    )

    keep_columns = KeepColumns(  #
        columns=["persona", "background sketch"],
        use_cache=False,
    )

    (
        load_dataset
        >> background_sketch
        >> background_sketch_yaml_extraction
        >> keep_columns
    )

if __name__ == "__main__":
    distiset = pipeline.run(  #
        use_cache=False,
    )
    distiset.push_to_hub(
        "ychen/mi-alcohol-persona-dev",
        private=True,
        token="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",
        include_script=False,
        generate_card=False,
    )
