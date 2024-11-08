from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class KeepColumns(Step):
    """Keeps selected columns in the dataset and optionally drops rows with empty values.

    `KeepColumns` is a `Step` that implements the `process` method that keeps only the columns
    specified in the `columns` attribute. Also `KeepColumns` provides an attribute `columns` to
    specify the columns to keep which will override the default value for the properties `inputs`
    and `outputs`. Optionally, it can drop rows where any of the specified columns have empty values.

    Note:
        The order in which the columns are provided is important, as the output will be sorted
        using the provided order, which is useful before pushing either a `dataset.Dataset` via
        the `PushToHub` step or a `distilabel.Distiset` via the `Pipeline.run` output variable.

    Attributes:
        columns: List of strings with the names of the columns to keep.
        dropna: If True, drops rows where any of the specified columns have empty values.
               Empty values are considered as None, empty strings, or strings with only whitespace.

    Input columns:
        - dynamic (determined by `columns` attribute): The columns to keep.

    Output columns:
        - dynamic (determined by `columns` attribute): The columns that were kept.

    Categories:
        - columns

    Examples:
        Select the columns to keep:

        ```python
        from distilabel.steps import KeepColumns

        # Keep columns and drop rows with empty values
        keep_columns = KeepColumns(
            columns=["instruction", "generation"],
            dropna=True
        )
        keep_columns.load()

        result = next(
            keep_columns.process(
                [
                    {"instruction": "What's the brightest color?", "generation": "white", "model_name": "my_model"},
                    {"instruction": "", "generation": "blue", "model_name": "my_model"},  # This row will be dropped
                ],
            )
        )
        # >>> result
        # [{'instruction': "What's the brightest color?", 'generation': 'white'}]
        ```
    """

    columns: List[str]
    dropna: bool = False

    @property
    def inputs(self) -> "StepColumns":
        """The inputs for the task are the column names in `columns`."""
        return self.columns

    @property
    def outputs(self) -> "StepColumns":
        """The outputs for the task are the column names in `columns`."""
        return self.columns

    def _is_empty(self, value: Optional[str]) -> bool:
        """Check if a value is considered empty.

        Args:
            value: The value to check.

        Returns:
            bool: True if the value is None, an empty string, or contains only whitespace.
        """
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method keeps only the columns specified in the `columns` attribute
        and optionally drops rows with empty values.

        Args:
            *inputs: A list of dictionaries with the input data.

        Yields:
            A list of dictionaries with the output data.
        """
        for input in inputs:
            outputs = []
            for item in input:
                # Create the filtered dictionary with only the specified columns
                filtered_item = {col: item[col] for col in self.columns}

                # If dropna is True, check for empty values
                if self.dropna:
                    if any(self._is_empty(filtered_item[col]) for col in self.columns):
                        continue

                outputs.append(filtered_item)
            yield outputs
