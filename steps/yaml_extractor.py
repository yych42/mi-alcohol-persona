from typing import Optional
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepColumns, StepOutput
from distilabel.errors import DistilabelUserError


class YamlExtractorStep(Step):
    """A step that extracts text between ```yaml and ``` in a string.

    This step takes an input text field and produces an output field containing
    the text found between the first occurrence of ```yaml and the following ``` delimiter.
    If no such delimiters are found, it returns None.

    Example:
        Input: {'text': 'Here is some code ```yaml\nkey: value\n``` goodbye'}
        Output: {'text': 'Here is some code ```yaml\nkey: value\n``` goodbye', 'extracted_yaml': 'key: value'}
    """

    @property
    def inputs(self) -> StepColumns:
        """Define the required input field."""
        return ["text"]

    @property
    def outputs(self) -> StepColumns:
        """Define the output field that will contain the extracted YAML content."""
        return ["extracted_yaml"]

    def extract_yaml(self, text: str) -> Optional[str]:
        """Helper method to extract text between ```yaml and ``` delimiters.

        Args:
            text: Input string to process

        Returns:
            The text between ```yaml and ``` if found, None otherwise
        """
        # Find the opening ```yaml delimiter
        start_marker = "```yaml"
        end_marker = "```"

        start_index = text.find(start_marker)
        if start_index == -1:
            return None

        # Find the closing ``` delimiter after ```yaml
        end_index = text.find(end_marker, start_index + len(start_marker))
        if end_index == -1:
            return None

        # Extract text between the markers
        return text[start_index + len(start_marker) : end_index].strip()

    def process(self, inputs: StepInput) -> StepOutput:
        """Process the input data to extract YAML content.

        Takes each input dictionary, extracts YAML from the 'text' field,
        and adds it to a new 'extracted_yaml' field.

        Args:
            inputs: List of dictionaries containing 'text' field

        Yields:
            Input dictionaries with added 'extracted_yaml' field
        """
        for input in inputs:
            input["extracted_yaml"] = self.extract_yaml(input["text"])
            if input["extracted_yaml"] is None:
                raise DistilabelUserError(
                    "No YAML content found in the input text: " + input["text"]
                )
        yield inputs
