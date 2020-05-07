import json
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import os


def token_is_table_start(token):
    return token.startswith('<Table') and token.endswith('>')


def token_is_table_end(token):
    return token.startswith('</Table') and token.endswith('>')


def token_is_html_tag(token):
    return token.startswith('<') and token.endswith('>')


def find_table_end(document_text_split, index):
    """
    Find the end token (</Table>) index for a table start token (<Table>)
    :param document_text_split: [str]. List of tokens of a document.
    :param index: int. Start index of <Table>
    :return: End index of </Table>
    """
    if not token_is_table_start(document_text_split[index]):
        raise Exception("index %d is not a table start tag for document %s" % (index, " ".join(document_text_split)))
    table_count = 0
    for i in range(index, len(document_text_split)):
        if token_is_table_start(document_text_split[i]):
            table_count += 1
            continue
        if token_is_table_end(document_text_split[i]):
            table_count -= 1
            if table_count == 0:
                return index
    raise Exception("can't find table end for start %d in document %s" % (index, "".join(document_text_split)))


def remove_tags(remove_option, document_text_split):
    start, index_mapping, new_document_split = 0, [0] * len(document_text_split), []
    if remove_option == 'keep':
        return list(range(len(document_text_split))), document_text_split
    while start < len(document_text_split):
        if remove_option == 'no_html_and_table':
            if token_is_table_start(document_text_split[start]):
                try:
                    table_end = find_table_end(document_text_split, start)
                except Exception as e:
                    print(e)
                    return None, None
                for i in range(start, table_end + 1):
                    index_mapping[i] = len(new_document_split)
                start = table_end + 1
            elif token_is_html_tag(document_text_split[start]):
                index_mapping[start] = len(new_document_split)
                start += 1
            else:
                new_document_split.append(document_text_split[start])
                index_mapping[start] = len(new_document_split) - 1
                start += 1
        elif remove_option == 'no_html':
            if token_is_html_tag(document_text_split[start]):
                index_mapping[start] = len(new_document_split)
                start += 1
            else:
                new_document_split.append(document_text_split[start])
                index_mapping[start] = len(new_document_split) - 1
                start += 1
        else:
            raise Exception("remove_option can only be in ['keep', 'no_html', 'no_html_and_table']."
                            "%s not found" % remove_option)
    # print(f"Removed html tags by option {remove_option}")
    # print(f"Result is {' '.join(new_document_split)}")
    return index_mapping, new_document_split


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

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Train. Otherwise it's Test.
    is_train = 'train' in args.fn

    train_fn, val_fn, test_fn = None, None, None
    if is_train:
        train_fn = os.path.join(args.output_path, f'{args.prefix}-train-{args.version}.json')
        if args.generate_val:
            val_fn = os.path.join(args.output_path, f'{args.prefix}-val-{args.version}.json')
            print(f'Converting {args.fn} to {train_fn} & {val_fn} ... ')
        else:
            print(f'Converting {args.fn} to {train_fn} ... ')
    else:
        # To generate test set or test-purpose validation set.
        test_fn = os.path.join(args.output_path, f'{args.prefix}-test-{args.version}.json')
        print(f'Converting {args.fn} to {test_fn} ... ')
    print(f'Option: {args.html_tag}')

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
    # num_dropped_for_table: number of document dropped because the answer is in <Table>...</Table>
    num_dropped_for_table = 0
    num_short_possible, num_long_possible = 0, 0
    origin_data = {}

    print(f'total lines in file: {get_sample_count(args.fn)}; to generate {args.num_samples} samples')

    if args.num_samples == -1:
        # Generate all samples
        num_samples = get_sample_count(args.fn)
    else:
        num_samples = args.num_samples

    with open(args.fn) as f:
        progress = tqdm(f, total=num_samples)
        entry = {}
        sample_count = 0
        for _, line in enumerate(progress):
            if sample_count >= num_samples:
                break

            data = json.loads(line)

            document_text = data['document_text']
            url = 'MISSING' if not is_train else data['document_url']
            orig_document_text_split = document_text.split(' ')

            # new_document_split: document split after removing tags according to args.html_tag
            # index_mapping: mapping from original tokens to new_document_split
            index_mapping, document_text_split = remove_tags(args.html_tag, orig_document_text_split)

            # keep original data
            data_cpy = data.copy()
            example_id = str(data_cpy.pop('example_id'))
            data_cpy['document_text'] = ''
            origin_data[example_id] = data_cpy

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
                orig_long_answer = annotations[0]['long_answer']
                orig_long_answer_start, orig_long_answer_end = \
                    orig_long_answer['start_token'], orig_long_answer['end_token']
                long_answer_start = index_mapping[orig_long_answer_start]
                long_answer_end = index_mapping[orig_long_answer_end]
                long_answer_len = long_answer_end - long_answer_start
                total_split_token_len = smooth * total_split_token_len + (1. - smooth) * len(document_text_split)
                if long_answer_len != 0:
                    long_split_token_len = smooth * long_split_token_len + (1. - smooth) * long_answer_len
                if long_answer_end > 0:
                    long_end = smooth * long_end + (1. - smooth) * long_answer_end
                if long_answer_end > max_end_token:
                    max_end_token = long_answer_end

                progress.set_postfix({'document_split_len': int(total_split_token_len),
                                      'long_answer_split_len': int(long_split_token_len),
                                      'long_answer_end_token_index': round(long_end, 2)})

                short_answers = annotations[0]['short_answers']
                yes_no_answer = annotations[0]['yes_no_answer']
                # skip yes/no answer for now
                if yes_no_answer != 'NONE':
                    num_yes_no += 1
                    continue

                long_is_impossible = (long_answer_start == -1) or (long_answer_len == 0)
                if long_is_impossible:
                    # random pick an candidate as negative sample
                    long_answer_candidate_index = np.random.randint(len(long_candidates))
                else:
                    long_answer_candidate_index = orig_long_answer['candidate_index']

                orig_long_start_token = long_candidates[long_answer_candidate_index]['start_token']
                orig_long_end_token = long_candidates[long_answer_candidate_index]['end_token']
                long_start_token = index_mapping[orig_long_start_token]
                long_end_token = index_mapping[orig_long_end_token]

                if long_end_token - long_start_token == 0:
                    # for a random candidate negative sample, if it's in a table span, drop it.
                    continue

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

                is_very_long = False
                if long_end_token > crop_end_token:
                    num_very_long += 1
                    is_very_long = True

                document_text_crop_split = document_text_split[crop_start_token: crop_end_token]
                context = ' '.join(document_text_crop_split)

                # create long answer
                long_answer_ = []
                if not long_is_impossible:
                    long_answer_pre_split = document_text_split[:long_start_token]
                    long_answer_start_char = len(' '.join(long_answer_pre_split)) - crop_start_len
                    long_answer_split = document_text_split[long_start_token: long_end_token]
                    long_answer_text = ' '.join(long_answer_split)
                    if not is_very_long:
                        assert context[long_answer_start_char: long_answer_start_char + len(long_answer_text)] \
                            == long_answer_text, long_answer_text
                    long_answer_ = [{'text': long_answer_text, 'answer_start': long_answer_start_char}]

                # create short answers
                short_is_impossible = len(short_answers) == 0
                short_answers_ = []
                if not short_is_impossible:
                    for short_answer in short_answers:
                        orig_short_start_token = short_answer['start_token']
                        orig_short_end_token = short_answer['end_token']
                        short_start_token = index_mapping[orig_short_start_token]
                        short_end_token = index_mapping[orig_short_end_token]
                        if short_start_token >= crop_start_token + args.crop_len:
                            num_short_dropped += 1
                            continue
                        short_answer_pre_split = document_text_split[:short_start_token]
                        short_answer_start_char = len(' '.join(short_answer_pre_split)) - crop_start_len
                        short_answer_split = document_text_split[short_start_token: short_end_token]
                        short_answer_text = ' '.join(short_answer_split)
                        assert short_answer_text != ''

                        # this happens if we crop and parts of the short answer overflow
                        short_from_context = context[short_answer_start_char:
                                                     short_answer_start_char + len(short_answer_text)]
                        # if short_from_context != short_answer_text:
                        #     print(f'short diff {short_from_context} vs {short_answer_text}')
                        short_answers_.append({'text': short_from_context, 'answer_start': short_answer_start_char})

                if len(short_answers_) == 0:
                    short_is_impossible = True

                if not short_is_impossible:
                    num_short_possible += 1
                if not long_is_impossible:
                    num_long_possible += 1

                qa = {'question': question,
                      'short_answers': short_answers_,
                      'long_answers': long_answer_,
                      'id': example_id,
                      'short_is_impossible': short_is_impossible,
                      'long_is_impossible': long_is_impossible,
                      'crop_start': crop_start_token}

            paragraph = {'qas': [qa], 'context': context}
            entry = {'title': url, 'paragraphs': [paragraph]}
            entries.append(entry)
            sample_count += 1

    progress.write('    ------------ STATS ------------------')
    progress.write(f'   Found {num_yes_no} yes/no, {num_very_long} very long'
                   f' and {num_short_dropped} short of {sample_count} and trimmed {num_trimmed}')
    progress.write(f'   #short {num_short_possible} #long {num_long_possible}'
                   f' of {len(entries)}')

    if is_train:
        train_entries, val_entries = [], []
        for entry in entries:
            if entry['paragraphs'][0]['qas'][0]['id'] not in val_ids:
                train_entries.append(entry)
            else:
                val_entries.append(entry)

        if train_fn:
            with open(train_fn, 'w') as f:
                json.dump({'version': args.version, 'data': train_entries}, f)
                progress.write(f'Wrote {len(train_entries)} entries to {train_fn}')
                f.close()
        if args.generate_val and val_fn and len(val_ids) > 0:
            with open(val_fn, 'w') as f:
                json.dump({'version': args.version, 'data': val_entries}, f)
                progress.write(f'Wrote {len(val_entries)} entries to {val_fn}')
                f.close()
        # Didn't provide a validation id set. Will dump the validation ids generated.
        if args.generate_val and args.val_ids is None:
            val_df = pd.DataFrame({'example_id': list(val_ids)})
            val_csv_fn = os.path.join(args.output_path, f'{args.prefix}-val-ids-{args.version}.csv')
            val_df.to_csv(val_csv_fn, index=False, columns=['example_id'])
            progress.write(f'Wrote validation ids to {val_csv_fn}')
    else:
        with open(test_fn, 'w') as f:
            json.dump({'version': args.version, 'data': entries}, f)
            progress.write(f'Wrote {len(entries)} entries to {test_fn}')
            f.close()

    if args.val_ids:
        print(f'Using val ids from: {args.val_ids}')
    return entries


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
    parser.add_argument('--val_prob', type=float, default=0.1,
                        help='If val_ids file is not provided, generate the validation sample with probability')
    parser.add_argument('--num_max_tokens', type=int, default=400_000, help='Max number of tokens allowed in document')
    # keep: keep all html tags
    # no_html: remove all html tags
    # no_html_and_table: remove all html tags and content between <Table>...</Table>
    parser.add_argument('--html_tag', choices=['keep', 'no_html', 'no_html_and_table'],
                        default='keep',
                        const='keep',
                        nargs='?',
                        help='Option to deal with html tags')
    _args = parser.parse_args()
    convert_nq_to_squad(_args)
