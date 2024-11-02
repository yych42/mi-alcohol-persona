import os
from distilabel.llms import AnthropicLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromHub
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv
from steps.quote_extractor import QuoteExtractorStep

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
        num_examples=20,
    )

    background_development = TextGeneration(
        llm=AnthropicLLM(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
        template="""You are a screenwriter known for developing rich and realistic characters. Here, your task is to extend a character's description by imagining an unique personal problem they could possibly face, how they could resolve it, the motivation for them to commit to the change they need to resolve it, but also what's holding them back.

For example...

"A recently diagnosed hypertensive patient..."

Could be extended into "A recently diagnosed hypertensive patient who is hesitant to take prescribed medication and modify salt intake due to disbelief in the diagnosis and concerns about lifelong medication dependence."

Now, please develop this character by extending the sentence:

"{{persona}}"

The extended description in a single sentence:""",
        columns=["persona"],
        output_mappings={"generation": "extended_persona_raw"},
    )

    persona_extraction = QuoteExtractorStep(
        input_mappings={"text": "extended_persona_raw"},
        output_mappings={"extracted_quote": "change_persona"},
        use_cache=False,
    )

    keep = KeepColumns(  #
        columns=["persona", "extended_persona_raw", "change_persona"],
        use_cache=False,
    )

    load_dataset >> text_generation >> persona_extraction >> keep

if __name__ == "__main__":
    distiset = pipeline.run(  #
        # use_cache=False,
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.9,
                    }
                }
            },
        },
    )
    distiset.push_to_hub(
        "ychen/mi-alcohol-persona-dev",
        private=True,
        token="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",
        include_script=False,
        generate_card=False,
    )
