import json
import re

import pandas as pd

from openhands.core.logger import openhands_logger as logger


class LocalPreConditionsModel:
    def __init__(self, model_path):
        # Load the inference results CSV instead of checklists
        df = pd.read_excel('submission.xlsx')

        # Find the row matching the instance_id (model_path parameter)
        matching_rows = df.loc[df['instance_id'] == model_path]

        if matching_rows.empty:
            logger.warning(f'No matching instance_id found for: {model_path}')
            self.questions: str = ""
        else:
            # Get the extracted_questions
            self.questions = matching_rows['clarification_questions_gpt5_3'].values[0]

        logger.info(
            f'Loaded questions for instance_id: {model_path}'
        )

    def format_questions_as_numbered_list(self, questions: list[str]) -> str:
        """Format questions as a numbered list"""
        if not questions:
            return (
                'No questions needed - all required information appears to be present.'
            )

        formatted_questions = []
        for i, question in enumerate(questions, 1):
            # Clean up the question if it already has numbering
            clean_question = question.strip()
            # Remove existing numbering if present
            if clean_question and clean_question[0].isdigit():
                # Find the first non-digit, non-dot, non-space character
                match = re.match(r'^\d+\.?\s*', clean_question)
                if match:
                    clean_question = clean_question[match.end() :].strip()

            formatted_questions.append(f'{i}. {clean_question}')

        return '\n'.join(formatted_questions)

    async def generate_preconditions(self, prompt: str) -> str:
        """Return the numbered list of questions for the given instance"""
        return "First, I have some clarifying questions:\n" + self.questions

    def get_questions_list(self):
        """Return the raw list of questions"""
        return self.questions

    def get_questions_count(self):
        """Return the number of questions"""
        return len(self.questions)
