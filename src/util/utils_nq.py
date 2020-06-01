import json
import logging
import os
import collections
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np

from allennlp.common.file_utils import cached_path

from transformers.tokenization_bert import whitespace_tokenize

from .utils_squad import SquadExample

logger = logging.getLogger(__name__)

# NQExample = collections.namedtuple("NQExample", [
#     "qas_id", "question_text", "doc_tokens",
#     "short_orig_answer_text", "long_orig_answer_text",
#     "short_start_position", "short_end_position",
#     "long_start_position", "long_end_position",
#     "short_is_impossible", "long_is_impossible", "crop_start"])


LongAnswerCandidate = collections.namedtuple('LongAnswerCandidate', ['start_token', 'end_token'])


class ContinueIteration(Exception):
    pass


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_nq_examples(input_file_or_data, stage, long_short_strategy):
    """
    Read a NQ json file into a list of NQExample. Refer to `nq_to_squad.py`
    to convert the `simplified-nq-*.jsonl` files to NQ json.
    """
    # Training: get start and end token position
    # Validation: get start and end token position
    # Testing: no need to get start and end token position
    assert stage in ('training', 'validation', 'testing')
    is_training, is_validation, is_testing = stage == 'training', stage == 'validation', stage == 'testing'
    assert long_short_strategy in ('long_short', 'short_long', 'only_short', 'only_long')

    if isinstance(input_file_or_data, str):
        input_file_or_data = cached_path(input_file_or_data)

        with open(input_file_or_data, "r", encoding='utf-8') as f:
            input_data = json.load(f)["data"]

    else:
        input_data = input_file_or_data

    for entry_index, entry in enumerate(tqdm(input_data, total=len(input_data))):
        assert len(entry["paragraphs"]) == 1
        paragraph = entry["paragraphs"][0]
        paragraph_text = paragraph["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        assert len(paragraph["qas"]) == 1
        qa = paragraph["qas"][0]

        # Start to extract short answer and long answer information.
        if is_training or is_validation:
            short_is_impossible = qa["short_is_impossible"]
            short_answers = qa["short_answers"]
            if len(short_answers) >= 2:
                # logger.info(f"Choosing leftmost of "
                #     f"{len(short_answers)} short answer")
                short_answers = sorted(short_answers, key=lambda sa: sa["answer_start"])
                short_answers = short_answers[0: 1]

            long_is_impossible = qa["long_is_impossible"]
            long_answers = qa["long_answers"]
            if (len(long_answers) != 1) and not long_is_impossible:
                raise ValueError(f"For training, each question should have exactly 1 long answer.")

            def set_long():
                return long_is_impossible, long_answers

            def set_short():
                return short_is_impossible, short_answers

            if long_short_strategy == 'long_short':
                is_impossible, answers = set_long() if not long_is_impossible else set_short()
            elif long_short_strategy == 'short_long':
                is_impossible, answers = set_short() if not short_is_impossible else set_long()
            elif long_short_strategy == 'only_long':
                is_impossible, answers = set_long()
            elif long_short_strategy == 'only_short':
                is_impossible, answers = set_short()
            else:
                raise Exception('long_short_strategy value not found in list')

            if not is_impossible:
                answer = answers[0]
                orig_answer_text = answer['text']
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                if answer_offset + answer_length - 1 >= len(char_to_word_offset):
                    end_position = char_to_word_offset[-1]
                else:
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly
                # recovered from the document. If this CAN'T
                # happen it's likely due to weird Unicode stuff
                # so we will just skip the example.
                #
                # Note that this means for training mode, every
                # example is NOT guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position: end_position + 1])
                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    # logger.warning(
                    #    "Could not find answer: '%s' vs. '%s'",
                    #    actual_text, cleaned_answer_text)
                    # Continue to outer loop for next qas example
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

            example = SquadExample(qas_id=qa['id'],
                                   question_text=qa['question'],
                                   doc_tokens=doc_tokens,
                                   orig_answer_text=orig_answer_text,
                                   start_position=start_position,
                                   end_position=end_position,
                                   is_impossible=is_impossible)
        else:
            example = SquadExample(qas_id=qa['id'],
                                   question_text=qa['question'],
                                   doc_tokens=doc_tokens)
        yield example
