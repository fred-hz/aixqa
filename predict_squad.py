import argparse
import collections
import logging
import os

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer
import torch
import json
from tqdm import tqdm

from src.data.data_reader_nq import NQDatasetReader
from src.model.bert_nq import BertForNaturalQuestionAnswering
from src.util.utils_nq import is_whitespace, NQExample
from src.util.utils_squad import RawResult, write_predictions

logger = logging.getLogger(__name__)


def predict_squad(args):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    dataset_reader = NQDatasetReader(tokenizer=PretrainedTransformerTokenizer(model_name=model_name),
                                     token_indexers={'tokens': PretrainedTransformerIndexer(model_name=model_name)},
                                     model_name=model_name,
                                     is_training=False)
    model = BertForNaturalQuestionAnswering(None, model_name)
    with open(args.model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, torch.device('cpu')))

    with open(args.squad_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)['data']
    all_examples = []
    for entry_index, entry in enumerate(tqdm(input_data, total=len(input_data))):
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
        for qa in paragraph['qas']:
            example = NQExample(
                qas_id=qa['id'],
                question_text=qa['question'],
                doc_tokens=doc_tokens,
                crop_start=0,
                **{"short_is_impossible": False,
                   'short_orig_answer_text': None,
                   'short_start_position': None,
                   'short_end_position': None,
                   'long_is_impossible': False,
                   'long_orig_answer_text': None,
                   'long_start_position': None,
                   'long_end_position': None}
            )
            all_examples.append(example)

    n_best_size = 10
    max_answer_length = 512
    all_crops, all_results = [], []
    for example_index, example in enumerate(all_examples):
        crops = dataset_reader.convert_example_to_crops(example)
        for crop_index, crop in enumerate(crops):
            all_crops.append(crop)
            input_ids = torch.tensor([crop.input_ids], dtype=torch.long)
            token_type_ids = torch.tensor([crop.token_type_ids], dtype=torch.long)
            attention_mask = torch.tensor([crop.attention_mask])
            output = model(input_ids, token_type_ids, attention_mask)
            start_logits = output['start_logits'].squeeze(-1)[0]
            end_logits = output['end_logits'].squeeze(-1)[0]

            all_results.append(RawResult(
                unique_id=crop.unique_id,
                start_logits=start_logits,
                end_logits=end_logits
            ))

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    output_prediction_file = os.path.join(args.output_path, 'prediction.json')
    output_nbest_file = os.path.join(args.output_path, 'nbest.json')
    output_null_log_odds_file = os.path.join(args.output_path, 'odds.json')
    write_predictions(all_examples, all_crops, all_results, n_best_size,
                      max_answer_length, True, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.null_score_diff_threshold,
                      True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='file path of the model')
    parser.add_argument('--squad_path', type=str, help='file path of the squad data')
    parser.add_argument('--output_path', type=str, help='output path for the prediction json')
    parser.add_argument('--null_score_diff_threshold', type=float)
    _args = parser.parse_args()
    predict_squad(_args)
