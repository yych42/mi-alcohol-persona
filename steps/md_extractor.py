from typing import Optional
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepColumns, StepOutput
from distilabel.errors import DistilabelUserError


class MarkdownExtractorStep(Step):
    """A step that extracts text between ```markdown and ``` in a string.

    This step takes an input text field and produces an output field containing
    the text found between the first occurrence of ```markdown and the following ``` delimiter.
    If no such delimiters are found, it returns None.

    Example:
        Input: {'text': 'Here is some code ```markdown\nkey: value\n``` goodbye'}
        Output: {'text': 'Here is some code ```markdown\nkey: value\n``` goodbye', 'extracted_markdown': 'key: value'}
    """

    @property
    def inputs(self) -> StepColumns:
        """Define the required input field."""
        return ["text"]

    @property
    def outputs(self) -> StepColumns:
        """Define the output field that will contain the extracted Markdown content."""
        return ["extracted_markdown"]

    def extract_markdown(self, text: str) -> Optional[str]:
        """Helper method to extract text between ```markdown and ``` delimiters.

        Args:
            text: Input string to process

        Returns:
            The text between ```markdown and ``` if found, None otherwise
        """
        # Find the opening ```markdown delimiter
        start_marker = "```markdown"
        end_marker = "```"

        start_index = text.find(start_marker)
        if start_index == -1:
            return None

        # Find the closing ``` delimiter after ```markdown
        end_index = text.find(end_marker, start_index + len(start_marker))
        if end_index == -1:
            return None

        # Extract text between the markers
        return text[start_index + len(start_marker) : end_index].strip()

    def process(self, inputs: StepInput) -> StepOutput:
        """Process the input data to extract Markdown content.

        Takes each input dictionary, extracts Markdown from the 'text' field,
        and adds it to a new 'extracted_markdown' field.

        Args:
            inputs: List of dictionaries containing 'text' field

        Yields:
            Input dictionaries with added 'extracted_markdown' field
        """
        for input in inputs:
            input["extracted_markdown"] = self.extract_markdown(input["text"])
            if input["extracted_markdown"] is None:
                input["extracted_markdown"] = ""
        yield inputs
