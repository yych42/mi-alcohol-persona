import re


class MarkdownParser:
    # Class-level constant for the pattern
    MARKDOWN_PATTERN = r"[ ]*```markdown\n([\s\S]*?)\n[ ]*```"

    @staticmethod
    def extract_markdown(text):
        """
        Extracts text between ```markdown and ``` markers

        Args:
            text (str): Input text containing markdown block

        Returns:
            str: Extracted markdown text or None if no match found
        """
        match = re.search(MarkdownParser.MARKDOWN_PATTERN, text)
        if match:
            # Get the captured content and dedent it
            content = match.group(1)
            # Split into lines, strip common leading whitespace
            lines = content.split("\n")
            # Find minimum indentation (excluding empty lines)
            min_indent = min(
                (len(line) - len(line.lstrip()) for line in lines if line.strip()),
                default=0,
            )
            # Remove that amount of indentation from each line
            dedented_lines = [
                line[min_indent:] if line.strip() else "" for line in lines
            ]
            return "\n".join(dedented_lines)
        return None

    @staticmethod
    def parse_to_json(markdown_text):
        """
        Converts markdown text to JSON where h2 headers (##) are keys
        and following text are values. Preserves paragraph structure.

        Args:
            markdown_text (str): Markdown text to parse

        Returns:
            dict: JSON representation of the markdown
        """
        if not markdown_text:
            return {}

        # Split text into lines but preserve empty lines that separate paragraphs
        lines = markdown_text.split("\n")

        result = {}
        current_key = None
        current_value = []
        previous_line_empty = False

        for line in lines:
            line = line.rstrip()  # Remove trailing whitespace but preserve line breaks

            if line.startswith("##"):
                # If we were building a previous section, save it
                if current_key is not None:
                    result[current_key] = MarkdownParser._clean_value(
                        "\n".join(current_value)
                    )

                # Start new section
                current_key = line[2:].strip()
                current_value = []
                previous_line_empty = False
            else:
                if current_key is not None:
                    # Handle paragraph breaks (empty lines)
                    if not line and not previous_line_empty and current_value:
                        # Add empty line only if we have content and previous line wasn't empty
                        current_value.append(line)
                    elif line:
                        # Add non-empty lines
                        current_value.append(line)

                    previous_line_empty = not line

        # Don't forget to save the last section
        if current_key is not None and current_value:
            result[current_key] = MarkdownParser._clean_value("\n".join(current_value))

        return result

    @staticmethod
    def _clean_value(text):
        """
        Cleans the final value string by:
        1. Removing leading/trailing whitespace
        2. Replacing 3 or more consecutive newlines with 2 newlines
        3. Ensuring consistent paragraph separation

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Replace multiple consecutive newlines with double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def process_text(input_text):
        """
        Complete pipeline to extract markdown and convert to JSON

        Args:
            input_text (str): Input text containing markdown block

        Returns:
            dict: JSON representation of the markdown
        """
        markdown_text = MarkdownParser.extract_markdown(input_text)
        return MarkdownParser.parse_to_json(markdown_text)


# Example usage:
if __name__ == "__main__":
    sample_text = """```markdown\n## Personal life\n- Mid-40s professional who has spent 15+ years in corporate law\n- Works at a medium to large firm, likely in a major city\n- Married with two children in private school\n- Lives in an affluent suburb\n- Frequently works long hours, often bringing work home\n- Has witnessed numerous cases where poor documentation led to legal issues\n- Personally knows colleagues who lost their jobs due to documentation failures\n- Maintains a wide professional network but limited personal friends\n- Struggles with work-life balance but justify it as \"taking care of business\"\n- Proud of their attention to detail and systematic approach\n- Has a reputation for being thorough and reliable among colleagues\n\n## Personality\nOpenness: Low\n- Values traditional legal approaches\n- Prefers proven methods over innovation\n- Respects established procedural standards\n- Appreciates structure and consistency\n\nConscientiousness: High\n- Extremely detail-oriented\n- Strong sense of responsibility\n- Organized and methodical\n- Places high value on preparation and planning\n\nExtraversion: Middle\n- Comfortable in professional social situations\n- Prefers small group interactions\n- Energized by intellectual discourse\n- Can be reserved in unfamiliar settings\n\nAgreeableness: Middle\n- Generally cooperative but firm in maintaining professional standards\n- Can be skeptical when verifying information\n- Values harmony but not at the expense of accuracy\n- Sometimes perceived as inflexible\n\nNeuroticism: Middle-High\n- Tends to worry about potential legal exposures\n- Feels personal responsibility for client outcomes\n- Experiences anxiety about potential mistakes\n- Manages stress through preparation and control\n```"""

    # Now we can use it without instantiating
    result = MarkdownParser.process_text(sample_text)
    print(result)
