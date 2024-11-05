import random
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepColumns, StepOutput


class RandomizeProblemEventSuffixStep(Step):
    """A step that assigns a random problem event suffix from a predefined list.

    This step takes input records and adds a new field 'problem_event_suffix' containing
    a randomly selected value from a specified list of suffixes. If no suffix is selected,
    it assigns an empty string.

    Example:
        Input: [{'text': 'Sample input text'}]
        Output: [{'text': 'Sample input text', 'problem_event_suffix': 'Build on one specific detail in the extended background.'}]
    """

    @property
    def inputs(self) -> StepColumns:
        """Define the required input fields."""
        return []

    @property
    def outputs(self) -> StepColumns:
        """Define the output field that will contain the randomly selected suffix."""
        return ["problem_event_suffix"]

    def process(self, inputs: StepInput) -> StepOutput:
        """Process the input data to assign a random problem event suffix.

        For each input dictionary, a random suffix from the predefined list is selected
        and assigned to the 'problem_event_suffix' field.

        Args:
            inputs: Iterable of dictionaries containing at least the 'text' field.

        Yields:
            Input dictionaries with an added 'problem_event_suffix' field.
        """
        SUFFIXES = [
            "",
            "Narrow-in on one unique aspect in the extended background.",
            "Build on one specific detail in the extended background.",
            "This should be based on ONE specific detail in the extended background.",
            "Work closely with the extended background.",
            "In this case, the event should be catastrophic.",
            "In this specific case, the event is related to a professional side of the persona.",
            "In this case, the event is related to a personal side of the persona.",
            "For this persona, focus on a drastic shift in drinking habits.",
            "For this character, this event led to an outcome being forced upon them.",
        ]
        for input_record in inputs:
            selected_suffix = random.choice(SUFFIXES)
            input_record["problem_event_suffix"] = selected_suffix
        yield inputs
