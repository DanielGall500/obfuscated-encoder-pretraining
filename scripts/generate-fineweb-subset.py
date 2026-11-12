import argparse

from datasets import load_dataset, Dataset, DatasetDict
from lightning import seed_everything
from transformers import AutoTokenizer


def main(dataset_repo, dataset_name, tokenizer_name, output_name, train_size, validation_fraction,
         save_to_disk=False, seed=42):
    ds = load_dataset(dataset_repo, name=dataset_name, streaming=True)

    ds = ds.filter(lambda x: x['language'] == 'en')

    # remove unnecessary columns
    columns_to_remove = ['id', 'dump', 'url', 'date', 'file_path', 'language',
                         'language_score', 'token_count', 'score', 'int_score']
    ds = ds.remove_columns([col for col in columns_to_remove if col in ds['train'].column_names])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # shuffle once
    train_shuffled = ds['train'].shuffle(seed=seed, buffer_size=10000)

    # step 1: Count tokens until we hit target, get document count
    print(f"Counting tokens to reach {train_size:,}...")
    num_docs = count_docs_for_tokens(train_shuffled, tokenizer, train_size)

    # step 2: Take exactly that many documents
    print(f"Taking first {num_docs:,} documents...")
    train = train_shuffled.take(num_docs)

    # convert to Dataset
    train = Dataset.from_generator(lambda: (yield from train), features=train.features)

    ds_dict = {'train': train}

    if validation_fraction > 0:
        print(f"Splitting validation set ({validation_fraction * 100}%)...")
        ds_tmp = train.train_test_split(test_size=validation_fraction, seed=seed)
        train = ds_tmp['train']
        valid = ds_tmp['test']

        ds_dict['train'] = train
        ds_dict['valid'] = valid

    ds_out = DatasetDict(ds_dict)

    # save to disk or push to hub
    if save_to_disk:
        print(f"Saving to disk: {output_name}")
        ds_out.save_to_disk(output_name)
    else:
        print(f"Pushing to hub: {output_name}")
        ds_out.push_to_hub(output_name, private=True, max_shard_size="2048MB")

    print("Done!")


def count_docs_for_tokens(ds, tokenizer, target_tokens, text_column='text', max_length=1024):
    """Count how many documents we need to reach target token count."""
    num_tokens = 0
    num_docs = 0

    for i, doc in enumerate(ds):
        tokens = tokenizer.tokenize(
            doc[text_column],
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )

        doc_tokens = len(tokens)
        if num_tokens + doc_tokens > target_tokens:
            break

        num_tokens += doc_tokens
        num_docs += 1

        if (i + 1) % 1000 == 0:
            print(f'  Processed {i + 1:,} documents, {num_tokens:,} tokens')

    print(f'Need {num_docs:,} documents for {num_tokens:,} tokens')
    return num_docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='create_fineweb_subsets',
                                     description='Creates subset with the specified number of train / test documents '
                                                 'for fineweb dataset.')

    parser.add_argument('--dataset_repo',
                        help='name of the dataset repo',
                        default='HuggingFaceFW/fineweb-edu')
    parser.add_argument('--dataset_name',
                        help='name of the fineweb dataset configuration (sub dataset)',
                        default='sample-10BT')
    parser.add_argument('--output_name',
                        help='name for output (hub repo name or disk path)',
                        default='cschroeder/fineweb-edu-10B')
    parser.add_argument('--save_to_disk',
                        action='store_true',
                        help='save to disk instead of pushing to hub')

    parser.add_argument('--tokenizer_name', help='tokenizer name', required=True)
    parser.add_argument('--train_size', default=10_000_000_000, type=int,
                        help='size of the training set in tokens')
    parser.add_argument('--validation_fraction', default=0.001, type=float,
                        help='percentage of the training set that is used as validation data')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args.dataset_repo,
         args.dataset_name,
         args.tokenizer_name,
         args.output_name,
         args.train_size,
         args.validation_fraction,
         args.save_to_disk,
         args.seed)
