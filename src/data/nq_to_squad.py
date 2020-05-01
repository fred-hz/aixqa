import json
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm


def get_sample_count(fn):
    count = 0
    with open(fn) as f:
        for line_count, line in enumerate(f):
            if line != '':
                count += 1
    f.close()
    return count


def convert_nq_to_squad(args):
    np.random.seed(123)

    # Train. Otherwise it's Test.
    is_train = 'train' in args.fn
    if is_train:
        train_fn = f'{args.prefix}-train-{args.version}.json'
        if args.generate_val:
            val_fn = f'{args.prefix}-val-{args.version}.json'
            print(f'Converting {args.fn} to {train_fn} & {val_fn} ... ')
        else:
            print(f'Converting {args.fn} to {train_fn} ... ')
    else:
        # To generate test set or test-purpose validation set.
        test_fn = f'{args.prefix}-test-{args.version}.json'
        print(f'Converting {args.fn} to {test_fn} ... ')

    if args.generate_val and args.val_ids:
        val_ids = set(str(x) for x in pd.read_csv(args.val_ids)['val_ids'].values)
    else:
        val_ids = set()

    # Store every single result entry
    entries = []
    # Smooth variable to record general stats of document pre-processing
    smooth = 0.999
    # total_split_token_len: 'average' length of all the document tokens after split by space
    # long_split_token_len: 'average' length of all the long answer tokens after split by space
    total_split_token_len, long_split_token_len = 0, 0
    # long_end: 'average' index of end token of long answers
    # max_end_token: max index of end token
    long_end, max_end_token = 0, -1
    # num_very_long: number of document with too long long-answer (longer than crop end)
    # num_yes_no: number of document with yes/no question
    # num_short_dropped: number of short answer being dropped because it exceeds the crop end
    # (crop start is fixed by long answer)
    # num_trimmed: number of document trimmed because of the limit of crop
    num_very_long, num_yes_no, num_short_dropped, num_trimmed = 0, 0, 0, 0
    num_short_possible, num_long_possible = 0, 0
    origin_data = {}

    if args.num_samples == -1:
        # Generate all samples
        num_samples = get_sample_count(args.fn)
    else:
        num_samples = args.num_samples

    # TODO: delete html tags
    with open(args.fn) as f:
        progress = tqdm(f, total=num_samples)
        entry = {}
        for sample_count, line in enumerate(progress):
            if sample_count >= num_samples:
                break

            data = json.loads(line)

            # keep original data
            data_cpy = data.copy()
            example_id = str(data_cpy.pop('example_id'))
            data_cpy['document_text'] = ''
            origin_data[example_id] = data_cpy

            document_text = data['document_text']
            document_text_split = document_text.split(' ')

            # trim super long document
            if len(document_text_split) > args.num_max_tokens:
                num_trimmed += 1
                document_text_split = document_text_split[:args.num_max_tokens]

            question = data['question_text']
            annotations = [None] if not is_train else data['annotations']
            assert len(annotations) == 1, annotations

            example_id = str(data['example_id'])
            long_candidates = data['long_answer_candidates']
            if not is_train:
                # if not training data, context will be document after trimmed
                qa = {'question': question, 'id': example_id, 'crop_start': 0}
                context = ' '.join(document_text_split)
            else:
                # get stats for document and long answers
                long_answer = annotations[0]['long_answer']
                long_answer_len = long_answer['end_token'] - long_answer['start_token']
                total_split_token_len = smooth * total_split_token_len + (1. - smooth) * len(document_text_split)
                if long_answer_len != 0:
                    long_split_token_len = smooth * long_split_token_len + (1. - smooth) * long_answer_len
                if long_answer['end_token'] > 0:
                    long_end = smooth * long_end + (1. - smooth) * long_answer['end_token']
                if long_answer['end_token'] > max_end_token:
                    max_end_token = long_answer['end_token']

                progress.set_postfix({'document_split_len': int(total_split_token_len),
                                      'long_answer_split_len': int(long_split_token_len),
                                      'long_answer_end_token_index': round(long_end, 2)})

                short_answers = annotations[0]['short_answers']
                yes_no_answer = annotations[0]['yes_no_answer']
                # skip yes/no answer for now
                if yes_no_answer != 'NONE':
                    num_yes_no += 1
                    continue

                long_is_impossible = long_answer['start_token'] == -1
                if long_is_impossible:
                    # random pick an candidate as negative sample
                    long_answer_candidate_index = np.random.randint(len(long_candidates))
                else:
                    long_answer_candidate_index = long_answer['candidate_index']

                long_start_token = long_candidates[long_answer_candidate_index]['start_token']
                long_end_token = long_candidates[long_answer_candidate_index]['end_token']
                # generate crop based on tokens. Note that if validation samples are to be generated,
                # they should not be cropped as this won't reflect test set performance.

                sample_in_train = True
                if args.generate_val:
                    if args.val_ids and example_id in val_ids:
                        sample_in_train = False
                    else:
                        if np.random.random_sample() < args.val_prob:
                            sample_in_train = False
                            val_ids.add(example_id)
                if not sample_in_train:
                    # not cropped if sample not in training set
                    crop_start_token = 0
                    crop_start_len = -1
                    crop_end_token = 10_000_000
                else:
                    crop_start_token = long_start_token - np.random.randint(int(args.crop_len * 0.75))
                    if crop_start_token <= 0:
                        crop_start_token = 0
                        crop_start_len = -1
                    else:
                        crop_start_len = len(' '.join(document_text_split[:crop_start_token]))
                    crop_end_token = crop_start_token + args.crop_len



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, default='simplified-nq-train.jsonl',
                        help='Json file path for training or evaluation from nq dataset')
    parser.add_argument('--output_path', type=str, default='./', help='Output folder path for squad-like data')
    parser.add_argument('--version', type=str, default='v1.0.0', help='Output version of squad-like data')
    parser.add_argument('--prefix', type=str, default='nq', help='Output data name prefix')
    parser.add_argument('--p_val', type=float, default=0.1)
    parser.add_argument('--crop_len', type=int, default=2_500, help='Crop length for too long document')
    parser.add_argument('--num_samples', type=int, default=1_000_000,
                        help='Number of samples generated. -1 to generate all the samples')
    parser.add_argument('--generate_val', action='store_true', help='If appears, generate a val file from input data')
    parser.add_argument('--val_ids', type=str,
                        help='Provide a val id file to the train data if val is to be generated from training data')
    parser.add_argument('--val_prob', type=float, default=0.2,
                        help='If val_ids file is not provided, generate the validation sample with probability')
    parser.add_argument('--num_max_tokens', type=int, default=400_000, help='Max number of tokens allowed in document')
    _args = parser.parse_args()
    convert_nq_to_squad(_args)
