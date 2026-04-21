import argparse
import csv
import os

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel


# model
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('hidden_size is none')

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# encode
def encode_pair(tokenizer, text_a, text_b, max_length):
    try:
        encoding = tokenizer(
            text=text_a,
            text_pair=text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        try:
            encoding = tokenizer(
                text_a,
                text_b,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
            )
        except TypeError:
            encoding = tokenizer(
                text=text_a,
                text_pair=text_b,
                max_seq_len=max_length,
                pad_to_max_seq_len=True,
                truncation=True,
                return_attention_mask=True,
            )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# dataset
class RowInferenceDataset(Dataset):
    def __init__(self, test_dir, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
        for filename in tqdm(csv_files, desc='scan table row'):
            table_id = filename[:-4]
            file_path = os.path.join(test_dir, filename)
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
                df.columns = [str(col).strip() for col in df.columns]

                if len(df.columns) < 2:
                    continue

                s_col = 'Subject' if 'Subject' in df.columns else df.columns[0]
                o_col = 'Object' if 'Object' in df.columns else df.columns[1]

                temp_data = df[[s_col, o_col]].dropna()
                for idx, row in temp_data.iterrows():
                    self.samples.append({
                        'id': f'{table_id}_{idx}',
                        's': str(row[s_col]),
                        'o': str(row[o_col]),
                    })
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids, attention_mask = encode_pair(
            self.tokenizer,
            item['s'],
            item['o'],
            self.max_length,
        )
        return {
            'row_id': item['id'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def collate_fn(samples):
    return {
        'row_id': [s['row_id'] for s in samples],
        'input_ids': np.stack([s['input_ids'] for s in samples]).astype('int64'),
        'attention_mask': np.stack([s['attention_mask'] for s in samples]).astype('int64'),
    }


# device
def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    print(f'devices valid: {custom_types}')

    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            print(f'use device: {dev}')
            return dev
        except Exception as e:
            print(f'device use error: {e}')

    dev = paddle.set_device('cpu')
    print('use device: CPU')
    return dev


# inference
def run_inference(args):
    device = resolve_device(args.device)

    # load labels
    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    # model init
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    model = CPAModel(args.shortcut_name, len(classes))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'can't find: {args.model_path}')

    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    model.eval()

    # load data
    dataset = RowInferenceDataset(args.test_dir, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        return_list=True,
    )

    print(f'start infer: {len(dataset)}')
    results = []
    use_amp = args.use_amp and str(device) != 'cpu'

    with paddle.no_grad():
        for batch in tqdm(dataloader, desc='inference'):
            ids = paddle.to_tensor(batch['input_ids'], dtype='int64')
            mask = paddle.to_tensor(batch['attention_mask'], dtype='int64')
            row_ids = batch['row_id']

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(ids, mask)
            else:
                logits = model(ids, mask)

            preds = paddle.argmax(logits, axis=1).numpy().tolist()
            for r_id, p_idx in zip(row_ids, preds):
                results.append((r_id, id2label[p_idx]))

    # save into csv
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

    print(f'infer finish，save into: {args.output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="./dataset/Test_Set")
    parser.add_argument('--labels_path', type=str, default="./labels.txt")
    parser.add_argument('--model_path', type=str, default="./cpa_output/cpa_20260418_151913/best_model.pdparams")
    parser.add_argument('--output_file', type=str, default='./cpa_row_predictions.txt')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()
    run_inference(args)
