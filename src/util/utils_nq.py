import json
import logging
import os
import collections
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

from allennlp.common.file_utils import cached_path

from transformers.tokenization_bert import whitespace_tokenize


logger = logging.getLogger(__name__)

NQExample = collections.namedtuple("NQExample", [
    "qas_id", "question_text", "doc_tokens",
    "short_orig_answer_text", "long_orig_answer_text",
    "short_start_position", "short_end_position",
    "long_start_position", "long_end_position",
    "short_is_impossible", "long_is_impossible", "crop_start"])


LongAnswerCandidate = collections.namedtuple('LongAnswerCandidate', ['start_token', 'end_token'])


class ContinueIteration(Exception):
    pass


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_nq_examples(input_file_or_data, is_training):
    """
    Read a NQ json file into a list of NQExample. Refer to `nq_to_squad.py`
    to convert the `simplified-nq-*.jsonl` files to NQ json.
    """
    if isinstance(input_file_or_data, str):
        input_file_or_data = cached_path(input_file_or_data)
        with open(input_file_or_data, "r", encoding='utf-8') as f:
            input_data = json.load(f)["data"]

    else:
        input_data = input_file_or_data

    for entry_index, entry in enumerate(tqdm(input_data, total=len(input_data))):
        # if entry_index >= 2:
        #     break
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
        # Use ContinueIteration Exception to jump out of inner loop and continue next qas.
        try:
            example_answer_info = {"short_is_impossible": False,
                                   'short_orig_answer_text': None,
                                   'short_start_position': None,
                                   'short_end_position': None,
                                   'long_is_impossible': False,
                                   'long_orig_answer_text': None,
                                   'long_start_position': None,
                                   'long_end_position': None}
            if is_training:
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

                for prefix, is_impossible, answers in [('short', short_is_impossible, short_answers),
                                                       ('long', long_is_impossible, long_answers)]:
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
                            raise ContinueIteration
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                    example_answer_info[f'{prefix}_is_impossible'] = is_impossible
                    example_answer_info[f'{prefix}_orig_answer_text'] = orig_answer_text
                    example_answer_info[f'{prefix}_start_position'] = start_position
                    example_answer_info[f'{prefix}_end_position'] = end_position

                if not short_is_impossible and not long_is_impossible:
                    assert example_answer_info['long_start_position'] <= \
                           example_answer_info['short_start_position']

                if not short_is_impossible and long_is_impossible:
                    assert False, f'Invalid pair short, long pair'
        except ContinueIteration:
            continue

        example = NQExample(
            qas_id=qa["id"],
            question_text=qa["question"],
            doc_tokens=doc_tokens,
            crop_start=qa["crop_start"],
            **example_answer_info
        )
        """
        example_answer_info: {
            'short_is_impossible': False,
            'short_orig_answer_text': None,
            'short_start_position': None,
            'short_end_position': None,
            'long_is_impossible': False,
            'long_orig_answer_text': None,
            'long_start_position': None,
            'long_end_position': None
        }
        """

        yield example


DocSpan = collections.namedtuple("DocSpan", ["start", "length"])


def get_spans(doc_stride, max_tokens_for_doc, max_len):
    doc_spans = []
    start_offset = 0
    while start_offset < max_len:
        length = max_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == max_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
