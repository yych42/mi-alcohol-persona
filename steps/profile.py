import re
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from typing_extensions import override

from jinja2 import Template
from pydantic import Field, PrivateAttr

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.utils.dicts import group_dicts

from .utils.markdown_parser import MarkdownParser

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns

BACKGROUND_SKETCH_TEMPLATE: str = (
    """\
You are a screenwriter known for developing rich and realistic characters. You are invited to contribute to a motivational interviewing research project, where the goal is to develop a set of personas that can be used to train healthcare professionals for motivational interviewing.

You are working through a dataset of alcohol-related personas. Your current task is to closely follow the provided "Persona", "Background sketch" to develop a plausible and believable profile of this character's relationship with alcohol.

Now, please develop a profile for this character:

---
Persona: {{persona}}

Background sketch: 
```yaml
{{background_sketch}}
```
---


To help with quality control and data collection, you were asked to write with the following markdown template to also demonstrate your step-by-step thought process. The template is as follows:

```markdown
## Exteded background
<Replace this with an extended background of this character, based on the provided "Background sketch", but adding more details and context that are specific to this character, yet go deeper than the sketch to create a more vivid picture of the character's life without being too limited to their persona.>

## Relationship with alcohol in the past
<Replace this with your character's relationship with alcohol in the past, based on the provided "Background sketch" and the extended background.>

## Problem event
<Most recently, how did alcohol became a problem for this character? What happened? Be creative, specific, and detailed; don't attribute to trite or cliched reasons like a hangover. {{problem_event_suffix}}>

## Change required
<What is a most possible specific change that this character needs to make in their relationship with alcohol? This should be a specific move in the narrative, not a general goal or direction.>

## Obstacles 
<What are the obstacles / conflicts that this character faces in making this change? What is holding them back? This also needs to be specific to the narrative.>

## Naive solutions
<What are some naive solutions that some bystanders might suggest to this character? Why are these solutions naive?>

## Auxiliary challenge
<What is ONE other challenges that this character faces in their life, that are not directly related to alcohol, but might affect their motivation or ability to make a change? This should be specific, and narrative-based as well.>
```

Submit your markdown worksheet---be sure to include the ```markdown ``` wrapper around your content:
""".rstrip()
)

SUFFIXES = [
    "",
    "Use at least 3 different details from the extended background.",
    "Use at least many details from the extended background as possible.",
    "Use at least 2 different aspects from the extended background.",
    "Narrow-in on one unique aspect in the extended background.",
    "Build on one specific detail in the extended background.",
    "This should be based on ONE specific detail in the extended background.",
    "Work closely with the extended background.",
    "In this case, the event should be catastrophic.",
    "In this specific case, the event is related to a professional side of the persona.",
    "In this case, the event is related to a personal side of the persona that is not related to their profession.",
    "For this persona, focus on a drastic shift in drinking habits.",
    "For this character, this event led to a very bad outcome being forced upon them unless they change their drinking habits.",
    "For this case, the character remains unsure if they should change their drinking habits.",
    "In this scenario, the character remains unconvinced that alcohol has caused any problems.",
    "In this case, the character is convinced that alcohol has caused problems.",
]


class ProfileGeneration(Task):
    columns: List[str] = Field(
        ...,
        description="The columns to use as input for the task.",
    )

    _can_be_used_with_offline_batch_generation = True
    _template: Optional["Template"] = PrivateAttr(default=...)

    def load(self) -> None:
        super().load()

        def check_column_in_template(column, template):
            pattern = (
                r"(?:{%.*?\b"
                + re.escape(column)
                + r"\b.*?%}|{{\s*"
                + re.escape(column)
                + r"\s*}})"
            )
            if not re.search(pattern, template):
                raise DistilabelUserError(
                    (
                        f"You required column name '{column}', but is not present in the template, "
                        "ensure the 'columns' match with the 'template' to avoid errors."
                    ),
                )

        for column in self.columns:
            check_column_in_template(column, BACKGROUND_SKETCH_TEMPLATE)

        self._template = Template(BACKGROUND_SKETCH_TEMPLATE)

    def unload(self) -> None:
        super().unload()
        self._template = None

    @property
    def inputs(self) -> "StepColumns":
        return {column: True for column in self.columns}

    def _prepare_message_content(self, input: Dict[str, Any]) -> "ChatType":
        """Prepares the content for the template and returns the formatted messages."""
        fields = {column: input[column] for column in self.columns}
        suffix = random.choice(SUFFIXES)
        print(f"Suffix: {suffix}")
        fields.update(
            {
                "problem_event_suffix": random.choice(SUFFIXES),
            },
        )
        return [{"role": "user", "content": self._template.render(**fields)}]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        for column in self.columns:
            if not isinstance(input[column], str):
                raise DistilabelUserError(
                    f"Input `{column}` must be a string. Got: {input[column]}",
                )

        messages = self._prepare_message_content(input)

        return messages  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["profile", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `background_sketch`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"profile": MarkdownParser.extract_markdown(output)}
