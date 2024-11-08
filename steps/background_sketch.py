import re
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

For example...

"A recently diagnosed hypertensive patient..." could ultimately be extended into "A recently diagnosed hypertensive patient who is hesitant to take prescribed medication and modify salt intake due to disbelief in the diagnosis and concerns about lifelong medication dependence."

Here, your current task is simply to design rich and detailed background setups for the personas. To help with quality control and data collection, you were asked to write with the following markdown template to also demonstrate your step-by-step thought process. The template is as follows:

```markdown
## Personal life
<Replace this with a brainstorm of the character's personal life background, based on what we can infer from the description.>

## Personality
Openness: <High/Middle/Low>
Conscientiousness: <High/Middle/Low>
Extraversion: <High/Middle/Low>
Agreeableness: <High/Middle/Low>
Neuroticism: <High/Middle/Low>
```

Now, please develop this character by extending the sentence:

"{{persona}}"

Submit your markdown worksheet---be sure to include the ```markdown ``` wrapper around your content:
""".rstrip()
)


class BackgroundSketch(Task):
    column: str = Field(
        default="persona",
        description=(
            "The column name for the persona. This is used to check if the column "
            "is present in the input data."
        ),
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

        check_column_in_template(self.column, BACKGROUND_SKETCH_TEMPLATE)

        self._template = Template(BACKGROUND_SKETCH_TEMPLATE)

    def unload(self) -> None:
        super().unload()
        self._template = None

    @property
    def inputs(self) -> "StepColumns":
        return [self.column]

    def _prepare_message_content(self, input: Dict[str, Any]) -> "ChatType":
        """Prepares the content for the template and returns the formatted messages."""
        fields = {self.column: input[self.column]}
        return [{"role": "user", "content": self._template.render(**fields)}]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        if not isinstance(input[self.column], str):
            raise DistilabelUserError(
                f"Input `{self.column}` must be a string. Got: {input['persona']}.",
            )

        messages = self._prepare_message_content(input)

        return messages  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["background_sketch", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `background_sketch`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"background_sketch": MarkdownParser.extract_markdown(output)}
