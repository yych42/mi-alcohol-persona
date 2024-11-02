from typing import Optional
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepColumns, StepOutput


class QuoteExtractorStep(Step):
    """A step that extracts text between the first pair of double quotes in a string.

    This step takes an input text field and produces an output field containing
    the text found between the first pair of double quotes. If no quotes are found,
    it returns None.

    Example:
        Input: {'text': 'Hello "world" goodbye'}
        Output: {'text': 'Hello "world" goodbye', 'extracted_quote': 'world'}
    """

    @property
    def inputs(self) -> StepColumns:
        """Define the required input field."""
        return ["text"]

    @property
    def outputs(self) -> StepColumns:
        """Define the output field that will contain the extracted quote."""
        return ["extracted_quote"]

    def extract_quote(self, text: str) -> Optional[str]:
        """Helper method to extract text between first pair of double quotes.

        Args:
            text: Input string to process

        Returns:
            The text between first pair of quotes if found, None otherwise
        """
        # Find first quote
        first_quote = text.find('"')
        if first_quote == -1:
            return None

        # Find second quote after the first one
        second_quote = text.find('"', first_quote + 1)
        if second_quote == -1:
            return None

        # Extract text between quotes
        return text[first_quote + 1 : second_quote]

    def process(self, inputs: StepInput) -> StepOutput:
        """Process the input data to extract quotes.

        Takes each input dictionary, extracts quote from the 'text' field,
        and adds it to a new 'extracted_quote' field.

        Args:
            inputs: List of dictionaries containing 'text' field

        Yields:
            Input dictionaries with added 'extracted_quote' field
        """
        for input in inputs:
            input["extracted_quote"] = self.extract_quote(input["text"])
        yield inputs
