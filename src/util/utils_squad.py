import logging
import collections
import six
import math
import numpy as np
import json
from typing import List, Dict

from .tokenization_squad import BasicTokenizer

logger = logging.getLogger(__name__)

UNMAPPED = -123
CLS_INDEX = 0


def printable_text(text):
    """Returns text encoded in a way suitable for print or logger."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class SquadExample(object):
    """A single training/test example for simple sequence classification.
       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.start_position:
            s += ", end_position: %d" % self.end_position
        if self.start_position:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class SquadCrop(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 doc_span_index,
                 token_to_orig_map=None,
                 token_is_max_context=None,
                 paragraph_len=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        """
        A single crop.
        `unique_id`, `example_index`, `tokens`, `input_ids`, `attention_mask`, `token_type_ids`, `doc_span_index` 
        are needed at both training/validation and inference time.
        `start_position`, `end_position`, `is_impossible` are needed only at training/validation time.
        `token_to_orig_map`, `token_is_max_context` are needed only at inference time.
        :param unique_id:
        :param example_index:
        :param tokens:
        :param doc_span_index:
        :param token_to_orig_map:
        :param token_is_max_context:
        :param paragraph_len:
        :param start_position:
        :param end_position:
        :param is_impossible:
        """
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.doc_span_index = doc_span_index
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SquadRawResult(object):
    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits


class SquadPrelimPrediction(object):
    def __init__(self, unique_id, start_index, end_index, start_logit, end_logit):
        self.unique_id = unique_id
        self.start_index = start_index
        self.end_index = end_index
        self.start_logit = start_logit
        self.end_logit = end_logit


class SquadNbestPrediction(object):
    def __init__(self,
                 text,
                 start_index,
                 end_index,
                 start_logit,
                 end_logit,
                 orig_doc_start,
                 orig_doc_end,
                 unique_id):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.start_logit = start_logit
        self.end_logit = end_logit
        self.orig_doc_start = orig_doc_start
        self.orig_doc_end = orig_doc_end
        self.unique_id = unique_id


def convert_examples_to_crops(examples_gen, tokenizer, max_seq_length,
                              doc_stride, max_query_length, stage,
                              cls_token='[CLS]', sep_token='[SEP]', pad_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              mask_padding_with_zero=True,
                              p_keep_impossible=None,
                              sep_token_extra=False):
    """Loads a data file into a list of `InputBatch`s."""
    assert p_keep_impossible is not None, '`p_keep_impossible` is required'
    assert stage in ('training', 'validation', 'testing')
    is_training, is_validation, is_testing = \
        (stage == 'training', stage == 'validation', stage == 'testing')

    unique_id = 1000000000
    num_pos, num_neg = 0, 0
    sub_token_cache = {}

    for example_index, example in enumerate(examples_gen):
        if example_index % 1000 == 0 and example_index > 0:
            logger.info('Converting %s examples: num_pos %s num_neg %s',
                        example_index, num_pos, num_neg)

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # This takes the longest!
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = sub_token_cache.get(token)
            if sub_tokens is None:
                sub_tokens = tokenizer.tokenize(token)
                sub_token_cache[token] = sub_tokens
            tok_to_orig_index.extend([i for _ in range(len(sub_tokens))])
            all_doc_tokens.extend(sub_tokens)

        tok_start_position = None
        tok_end_position = None
        if is_training or is_validation:
            if example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            else:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[
                        example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

        # For Bert: [CLS] question [SEP] paragraph [SEP]
        special_tokens_count = 3
        if sep_token_extra:
            # For Roberta: <s> question </s> </s> paragraph </s>
            special_tokens_count += 1
        max_tokens_for_doc = max_seq_length - len(query_tokens) - special_tokens_count
        assert max_tokens_for_doc > 0
        # We can have documents that are longer than the maximum
        # sequence length. To deal with this we do a sliding window
        # approach, where we take chunks of the up to our max length
        # with a stride of `doc_stride`.
        doc_spans = get_spans(doc_stride, max_tokens_for_doc, len(all_doc_tokens))
        for doc_span_index, doc_span in enumerate(doc_spans):
            # Tokens are constructed as: CLS Query SEP Paragraph SEP

            # Needed by testing
            token_to_orig_map = UNMAPPED * np.ones((max_seq_length, ), dtype=np.int32)
            # Needed by testing
            token_is_max_context = np.zeros((max_seq_length, ), dtype=np.bool)
            # Needed by training, validation and testing
            is_impossible = example.is_impossible
            # Needed by training, validation
            start_position = None
            # Needed by training, validation
            end_position = None

            special_tokens_offset = special_tokens_count - 1
            doc_offset = len(query_tokens) + special_tokens_offset
            if (is_training or is_validation) and not is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    start_position = 0
                    end_position = 0
                    is_impossible = True
                else:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # drop impossible samples
            if is_impossible:
                if np.random.rand() > p_keep_impossible:
                    continue

            # Needed by training, validation and testing
            tokens = []
            # Needed by training, validation and testing
            token_type_ids = []

            # CLS token at the beginning
            tokens.append(cls_token)
            token_type_ids.append(cls_token_segment_id)

            # Query
            tokens += query_tokens
            token_type_ids += [sequence_a_segment_id] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_a_segment_id)
            if sep_token_extra:
                tokens.append(sep_token)
                token_type_ids.append(sequence_a_segment_id)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                if is_testing:
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    token_is_max_context[len(tokens)] = \
                        check_is_max_context(doc_spans, doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                token_type_ids.append(sequence_b_segment_id)

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            token_type_ids.append(sequence_b_segment_id)
            # p_mask.append(1)  # can not be answer

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            # Needed by training, validation and testing
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_id)
                attention_mask.append(0 if mask_padding_with_zero else 1)
                token_type_ids.append(pad_token_segment_id)

            # reduce memory, only input_ids needs more bits
            input_ids = np.array(input_ids, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.bool)
            token_type_ids = np.array(token_type_ids, dtype=np.uint8)

            if (is_training or is_validation) and is_impossible:
                start_position = CLS_INDEX
                end_position = CLS_INDEX

            if is_impossible:
                num_neg += 1
            else:
                num_pos += 1

            if is_training or is_validation:
                crop = SquadCrop(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
            else:
                crop = SquadCrop(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    paragraph_len=paragraph_len,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context)
            yield crop
            unique_id += 1


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


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _clean_text(tok_text):
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def get_example_nbest(prelim_predictions: List[SquadPrelimPrediction],
                      crops: Dict[int, SquadCrop],
                      n_best_size: int):
    """
    Get n best predictions from `prelim_predictions` for one single example. The crops are stored in `crops`.
    :param prelim_predictions: [SquadPrelimPrediction]. The list is sorted in descending for logits score.
    :param crops: Dict[int, SquadCrop]. Contains crops information for the predictions. Mapping from unique_id to crop.
    :param n_best_size: int.
    :return: [SquadNbestPrediction]
    """
    seen, nbest = set(), []
    assert len(prelim_predictions) >= 1
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        crop = crops[pred.unique_id]
        orig_doc_start, orig_doc_end = -1, -1
        if pred.start_index > 0:
            tok_tokens = crop.tokens[pred.start_index: pred.end_index + 1]
            tok_text = " ".join(tok_tokens)
            tok_text = _clean_text(tok_text)

            orig_doc_start = int(crop.token_to_orig_map[pred.start_index])
            orig_doc_end = int(crop.token_to_orig_map[pred.end_index])

            final_text = tok_text
            if final_text in seen:
                continue
        else:
            final_text = ""

        seen.add(final_text)
        nbest.append(SquadNbestPrediction(
            text=final_text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred.start_index, end_index=pred.end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            unique_id=pred.unique_id
        ))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if len(nbest) == 0:
        nbest.insert(0, SquadNbestPrediction(
            text=None,
            start_logit=0.0, end_logit=0.0,
            start_index=-1, end_index=-1,
            orig_doc_start=-1, orig_doc_end=-1,
            unique_id=None
        ))
    assert len(nbest) >= 1
    return nbest


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=True):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def write_predictions(all_examples, all_crops, all_results, n_best_size, max_answer_length,
                      output_prediction_file, output_nbest_file, output_null_log_odds_file,
                      null_score_diff_threshold, with_negative_samples=True):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)
    logger.info("Writing null log odds to: %s" % output_null_log_odds_file)

    example_index_to_crops = collections.defaultdict(dict)
    for crop in all_crops:
        example_index_to_crops[crop.example_index][crop.unique_id] = crop

    unique_id_to_result = {result.unique_id: result for result in all_results}

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    score_diff_json = collections.OrderedDict()
    num_empty_prediction = 0
    example_index = 0
    for example_index, example in enumerate(all_examples):
        crops = example_index_to_crops[example.qas_id]

        prelim_predictions = []
        for unique_id, crop in enumerate(crops):
            result = unique_id_to_result[unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                if start_index >= len(crop.tokens):
                    continue
                # this skips [CLS] i.e. null prediction
                if crop.token_to_orig_map[start_index] == UNMAPPED:
                    continue
                if not crop.token_is_max_context[start_index]:
                    continue

                for end_index in end_indexes:
                    if end_index >= len(crop.tokens):
                        continue
                    if crop.token_to_orig_map[end_index] == UNMAPPED:
                        continue
                    if end_index not in crop.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        SquadPrelimPrediction(
                            unique_id=unique_id,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        nbest = get_example_nbest(prelim_predictions, crops, n_best_size)

        # The predictions are in descending order for logits. The first non-null one is the best.
        best_non_null_entry = None
        for entry in nbest:
            if best_non_null_entry is None:
                if entry.text != "":
                    best_non_null_entry = entry

        # total_scores = []
        # best_non_null_entry = None
        # for entry in nbest:
        #     total_scores.append(entry.start_logit + entry.end_logit)
        #     if not best_non_null_entry:
        #         if entry.text:
        #             best_non_null_entry = entry
        #
        # probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            # output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_index"] = entry.start_index
            output["end_index"] = entry.end_index
            output["orig_doc_start"] = entry.orig_doc_start
            output["orig_doc_end"] = entry.orig_doc_end
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        # We use the [CLS] score of the crop that has the maximum positive score
        # long_score_diff = min_long_score_null - long_best_non_null.start_logit
        # Predict "" if null score - the score of best non-null > threshold

        if with_negative_samples:
            # try:
            #     crop_unique_id = crops[best_non_null_entry.crop_index].unique_id
            #     start_score_null = unique_id_to_result[crop_unique_id].start_logits[CLS_INDEX]
            #     end_score_null = unique_id_to_result[crop_unique_id].end_logits[CLS_INDEX]
            #     score_null = start_score_null + end_score_null
            #     score_diff = score_null - (best_non_null_entry.start_logit + best_non_null_entry.end_logit)
            #
            #     if score_diff > null_score_diff_threshold:
            #         final_pred = ("", -1, -1)
            #         num_empty_prediction += 1
            #     else:
            #         final_pred = (best_non_null_entry.text,
            #                       best_non_null_entry.orig_doc_start,
            #                       best_non_null_entry.orig_doc_end)
            # except Exception as e:
            #     print(e)
            #     final_pred = ("", -1, -1)
            #     num_empty_prediction += 1
            crop_unique_id = best_non_null_entry.unique_id
            start_score_null = unique_id_to_result[crop_unique_id].start_logits[CLS_INDEX]
            end_score_null = unique_id_to_result[crop_unique_id].end_logits[CLS_INDEX]
            score_null = start_score_null + end_score_null
            score_diff = score_null - (best_non_null_entry.start_logit + best_non_null_entry.end_logit)
            score_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                final_pred = ("", -1, -1)
                num_empty_prediction += 1
            else:
                final_pred = (best_non_null_entry.text,
                              best_non_null_entry.orig_doc_start,
                              best_non_null_entry.orig_doc_end)
        else:
            final_pred = (best_non_null_entry.text,
                          best_non_null_entry.orig_doc_start,
                          best_non_null_entry.orig_doc_end)

        all_predictions[example.qas_id] = final_pred
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=2))

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=2))

    if output_null_log_odds_file is not None:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(score_diff_json, indent=2))

    logger.info(f'{num_empty_prediction} empty of {example_index}')
    return all_predictions
