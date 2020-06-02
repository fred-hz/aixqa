from typing import Dict, List, Iterable
import logging
import collections
from overrides import overrides
import math
import numpy as np
import os
import json
from multiprocessing import Pool

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.fields import (Field, TextField, IndexField, LabelField, ListField,
                                  MetadataField, SequenceLabelField, SpanField, ArrayField)

from ..util.utils_squad import SquadCrop, SquadExample
from ..util.utils_squad import get_spans, check_is_max_context
from ..util.utils_nq import NQExample, read_nq_examples
from ..util.utils_common import SplitAndCache

logger = logging.getLogger(__name__)


@DatasetReader.register('nq')
class NQDatasetReader(DatasetReader):
    """
    DatasetReader for google's natural questions Q&A dataset.
    """
    def __init__(self,
                 tokenizer: PretrainedTransformerTokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model_name: str = None,
                 is_training: bool = None,
                 max_seq_length: int = 512,
                 doc_stride: int = 256,
                 max_query_length: int = 64,
                 p_keep_impossible: float = 0.1,
                 long_short_strategy: str = 'short_long',
                 sequence_a_segment_id: int = 0,
                 sequence_b_segment_id: int = 1,
                 cls_token_segment_id: int = 0,
                 pad_token_segment_id: int = 0,
                 mask_padding: bool = True,
                 lazy: bool = False,
                 cache_directory: str = None) -> None:
        super().__init__(lazy, cache_directory)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        if model_name is None:
            raise Exception("Need to specify model_name for NQDatasetReader")
        self.model_name = model_name
        if self.model_name.startswith('roberta'):
            self.model_type = 'roberta'
            self.cls_token = '<s>'
            self.sep_token = '</s>'
            self.pad_token = '<pad>'
            self.sep_token_extra = True
        elif self.model_name.startswith('bert'):
            self.model_type = 'bert'
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
            self.pad_token = '[PAD]'
            self.sep_token_extra = False
        else:
            raise Exception('Only bert and roberta models are supported now for NQDatasetReader.model_name')
        self.is_training = is_training
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.p_keep_impossible = p_keep_impossible
        self.long_short_strategy = long_short_strategy
        self.sequence_a_segment_id = sequence_a_segment_id
        self.sequence_b_segment_id = sequence_b_segment_id
        self.cls_token_segment_id = cls_token_segment_id
        self.pad_token_segment_id = pad_token_segment_id
        self.mask_padding = mask_padding

        # example crops generation stats:
        self.num_short_pos, self.num_long_pos, self.num_neg = 0, 0, 0

        # unique_id for crop generation
        self.unique_id = 1000000000

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        The other piece we have to implement is _read, which takes a filename and produces a stream of Instances.
        Most of the work has already been done in text_to_instance
        :param file_path:
        :return:
        """
        examples_gen = read_nq_examples(file_path, self.is_training)

        for example_index, example in enumerate(examples_gen):
            if example_index % 10000 == 0 and example_index > 0:
                logger.info('Converting %s examples: short_pos %s long_pos %s neg %s',
                            example_index,
                            self.num_short_pos, self.num_long_pos, self.num_neg)
            crops = self.convert_example_to_crops(example)
            # logger.info(f'saving crops into cache at {crops_cache_path}')
            # crops_cache.dump(crops)

            """
            Crop = collections.namedtuple("Crop", ["unique_id", "doc_span_index",
                                       "tokens", "token_to_orig_map", "token_is_max_context",
                                       "input_ids", "attention_mask", "token_type_ids",
                                       # "p_mask",
                                       "paragraph_len", "answer_start_position", "answer_end_position",
                                       "answer_is_impossible"])
            )
            """
            for crop_index, crop in enumerate(crops):

                # reduce memory, only input_ids needs more bits
                input_ids = np.array(crop.input_ids, dtype=np.int32)
                # attention_mask = np.array(attention_mask, dtype=np.bool)
                attention_mask = np.array(crop.attention_mask, dtype=np.int32)
                # subtoken_type_ids = np.array(subtoken_type_ids, dtype=np.uint8)
                token_type_ids = np.array(crop.token_type_ids, dtype=np.int32)

                instance = self.text_to_instance(input_ids,
                                                 token_type_ids,
                                                 attention_mask,
                                                 crop.answer_start_position,
                                                 crop.answer_end_position)
                yield instance

    @overrides
    def text_to_instance(self,
                         input_ids: np.array = None,
                         token_type_ids: np.array = None,
                         attention_mask: np.array = None,
                         answer_start: int = None,
                         answer_end: int = None
                         ) -> Instance:
        # Long Tensor needed for embedding layer
        fields = {'input_ids': ArrayField(input_ids, dtype=np.dtype('int64')),
                  'token_type_ids': ArrayField(token_type_ids, dtype=np.dtype('int64')),
                  'attention_mask': ArrayField(attention_mask, dtype=np.dtype('int8')),
                  'start_positions': LabelField(answer_start, skip_indexing=True),
                  'end_positions': LabelField(answer_end, skip_indexing=True)}
        return Instance(fields)

    def _tokenize(self, text: str):
        """
        Tokenize text into corresponding tokens. No special tokens added
        :param text:
        :return: List[str]
        """
        return self.tokenizer.tokenizer.tokenize(text)

    def _convert_tokens_to_ids(self, tokens: List[str]):
        """
        Convert list of tokens into corresponding ids.
        :param tokens:
        :return: List[int]
        """
        return self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)

    def convert_example_to_crops(self,
                                 example: NQExample
                                 ) -> [Crop]:

        query_subtokens = self._tokenize(example.question_text)
        if len(query_subtokens) > self.max_query_length:
            query_subtokens = query_subtokens[:self.max_query_length]

        # Generate a mappings from original tokens to subtokens, and vice verse
        subtoken_to_token_map = []
        token_to_subtoken_map = []
        doc_subtokens = []
        subtoken_cache = {}
        for token_index, token in enumerate(example.doc_tokens):
            token_to_subtoken_map.append(len(doc_subtokens))
            subtokens = subtoken_cache.get(token)
            if subtokens is None:
                subtokens = self._tokenize(token)
                subtoken_cache[token] = subtokens
            subtoken_to_token_map.extend([token_index for _ in range(len(subtokens))])
            doc_subtokens.extend(subtokens)

        """
        NQExample definition:
        NQExample = collections.namedtuple("NQExample", [
            "qas_id", "question_text", "doc_tokens",
            "short_orig_answer_text", "long_orig_answer_text",
            "short_start_position", "short_end_position",
            "long_start_position", "long_end_position",
            "short_is_impossible", "long_is_impossible", "crop_start"])
        """

        # Find start token position and end token position according to self.long_short_strategy
        token_start_position, token_end_position = None, None
        sample_is_long_or_short = None
        if self.is_training:
            if self.long_short_strategy == 'short_long':
                if not example.short_is_impossible:
                    token_start_position, token_end_position = \
                        example.short_start_position, example.short_end_position
                    sample_is_long_or_short = 'short'
                elif not example.long_is_impossible:
                    token_start_position, token_end_position = \
                        example.long_start_position, example.long_end_position
                    sample_is_long_or_short = 'long'
            elif self.long_short_strategy == 'long_short':
                if not example.long_is_impossible:
                    token_start_position, token_end_position = \
                        example.long_start_position, example.long_end_position
                    sample_is_long_or_short = 'long'
                elif not example.short_is_impossible:
                    token_start_position, token_end_position = \
                        example.short_start_position, example.short_end_position
                    sample_is_long_or_short = 'short'
            else:
                raise Exception('Not long_short_strategy allowed except short_long and long_short')

        subtoken_start_position, subtoken_end_position = None, None
        if token_start_position is not None and token_end_position is not None:
            subtoken_start_position = token_to_subtoken_map[token_start_position]
            if token_end_position < len(example.doc_tokens) - 1:
                subtoken_end_position = token_to_subtoken_map[token_end_position + 1] - 1
            else:
                subtoken_end_position = len(doc_subtokens) - 1

        # For Bert: [CLS] question [SEP] context [SEP]
        special_subtokens_count = 3
        if self.sep_token_extra:
            # For Roberta: <s> question </s> </s> context </s>
            special_subtokens_count += 1
        max_subtokens_for_doc = self.max_seq_length - len(query_subtokens) - special_subtokens_count
        assert max_subtokens_for_doc > 0

        # We can have documents that are longer than the maximum
        # sequence length. To deal with this we do a sliding window
        # approach, where we take chunks of the up to our max length
        # with a stride of `doc_stride`.

        UNMAPPED = -123
        # doc_spans are in the subtoken level
        doc_spans = get_spans(self.doc_stride, max_subtokens_for_doc, len(doc_subtokens))
        # crops = []
        for doc_span_index, doc_span in enumerate(doc_spans):
            span_sample_subtokens = []
            subtoken_to_passage_map = UNMAPPED * np.ones((self.max_seq_length, ), dtype=np.int32)
            subtoken_is_max_content = np.zeros((self.max_seq_length, ), dtype=np.bool)
            subtoken_type_ids = []

            # subtoken start and end position for current span
            if subtoken_start_position is None and subtoken_end_position is None:
                span_subtoken_start_position, span_subtoken_end_position = 0, 0
            else:
                special_subtokens_offset = special_subtokens_count - 1
                doc_offset = len(query_subtokens) + special_subtokens_offset
                doc_start, doc_end = doc_span.start, doc_span.start + doc_span.length - 1
                if not (subtoken_start_position >= doc_start and subtoken_end_position <= doc_end):
                    # Answer not in current span. Thus answer_is_impossible=True for this span.
                    span_subtoken_start_position, span_subtoken_end_position = 0, 0
                else:
                    span_subtoken_start_position = subtoken_start_position - doc_start + doc_offset
                    span_subtoken_end_position = subtoken_end_position - doc_start + doc_offset

            span_answer_is_positive = True
            if span_subtoken_start_position == 0 and span_subtoken_end_position == 0:
                # Answer is impossible for this span. Drop impossible samples by probability.
                span_answer_is_positive = False
                if self.is_training and np.random.rand() > self.p_keep_impossible:
                    continue

            # Generate model inputs
            # CLS token at the beginning
            span_sample_subtokens.append(self.cls_token)
            subtoken_type_ids.append(self.cls_token_segment_id)

            # Query
            span_sample_subtokens += query_subtokens
            subtoken_type_ids += [self.sequence_a_segment_id] * len(query_subtokens)

            # SEP token
            span_sample_subtokens.append(self.sep_token)
            subtoken_type_ids.append(self.sequence_a_segment_id)
            if self.sep_token_extra:
                span_sample_subtokens.append(self.sep_token)
                subtoken_type_ids.append(self.sequence_a_segment_id)

            # Context/Paragraph
            for i in range(doc_span.length):
                subtoken_index_in_passage = doc_span.start + i
                # |<--example.crop_start-->|<--doc_span_1-->|
                # |<--example.crop_start-->|.......|<--doc_span_2-->|
                # |<--example.crop_start-->|................|<--doc_span_3-->|
                subtoken_to_passage_map[len(span_sample_subtokens)] = subtoken_to_token_map[
                    subtoken_index_in_passage] + example.crop_start

                subtoken_is_max_content[len(span_sample_subtokens)] = check_is_max_context(doc_spans,
                                                                                           doc_span_index,
                                                                                           subtoken_index_in_passage)
                span_sample_subtokens.append(doc_subtokens[subtoken_index_in_passage])
                subtoken_type_ids.append(self.sequence_b_segment_id)

            paragraph_len = doc_span.length

            # SEP token
            span_sample_subtokens.append(self.sep_token)
            subtoken_type_ids.append(self.sequence_b_segment_id)

            # Final input_ids for subtokens
            input_ids = self._convert_tokens_to_ids(span_sample_subtokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if self.mask_padding else 0] * len(input_ids)

            # Zero-pad up to the sequence length
            pad_id = self._convert_tokens_to_ids([self.pad_token])[0]
            while len(input_ids) < self.max_seq_length:
                input_ids.append(pad_id)
                attention_mask.append(0 if self.mask_padding else 1)
                subtoken_type_ids.append(self.pad_token_segment_id)

            if span_answer_is_positive:
                if sample_is_long_or_short == 'long':
                    self.num_long_pos += 1
                elif sample_is_long_or_short == 'short':
                    self.num_short_pos += 1
                else:
                    raise Exception('Sample should be either long or short for positive samples.')
            else:
                self.num_neg += 1

            crop = Crop(
                unique_id=self.unique_id,
                # TODO
                example_index=example.qas_id,
                doc_span_index=doc_span_index,
                tokens=span_sample_subtokens,
                token_to_orig_map=subtoken_to_passage_map.tolist(),
                token_is_max_context=subtoken_is_max_content.tolist(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=subtoken_type_ids,
                paragraph_len=paragraph_len,
                answer_start_position=span_subtoken_start_position,
                answer_end_position=span_subtoken_end_position,
                answer_is_impossible=not span_answer_is_positive
            )
            self.unique_id += 1

            yield crop
        #     crops.append(crop)
        # return crops
