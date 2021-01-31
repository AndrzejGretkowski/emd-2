import argparse
import contextlib
import sys
from pathlib import Path

import spacy
import torch
import tqdm

from emd_2.data import load_data, normalize_review_weight, preprocess
from emd_2.neural import BasicNet, LSTMNet


@contextlib.contextmanager
def smart_open(filename, mode):
    if filename and filename != '-':
        fh = open(filename, mode)
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

def output_to_label(out):
    num = out.argmax(1)
    return str(float(num) + 1)

def tokenize(nlp, text):
    return nlp(text.lower()).tensor

def parse_args():
    parser = argparse.ArgumentParser(description='Predict review score.')
    parser.add_argument('--input_path', type=str,
                        help='path to the .csv file similar to train')
    parser.add_argument('--model_path', type=str,
                        default=str(Path(__file__).parent.absolute() / 'model/lstm-net.pt'),
                        help='path to pytorch model')
    parser.add_argument('--output_path', type=str, default='',
                        help='path where to save the output, review per line')
    parser.add_argument('--device', type=str, default='cpu')

    args, _ = parser.parse_known_args()
    return args

def load_model(model_path, basic = False):
    if basic:
        model = BasicNet((96, 96, 1), 5)
    else:
        model = LSTMNet((96, 96, 1), 5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_output(dataframe, model_path = 'model/lstm-net.pt', basic = False):
    nlp = spacy.load('en_core_web_md')

    model = load_model(model_path, basic)
    predictions = []

    for i, row in tqdm.tqdm(dataframe.iterrows(), total=len(dataframe)):
        if basic:
            out = model.forward(
                torch.mean(torch.from_numpy(tokenize(nlp, row['reviewText'])), dim=0).unsqueeze(0),
                torch.mean(torch.from_numpy(tokenize(nlp, row['summary'])), dim=0).unsqueeze(0),
                torch.tensor([normalize_review_weight(row['helpful'])], dtype=torch.float),
                device='cpu')
        else:
            out = model.forward(
                torch.from_numpy(tokenize(nlp, row['reviewText'])).unsqueeze(0),
                torch.from_numpy(tokenize(nlp, row['summary'])).unsqueeze(0),
                torch.tensor([normalize_review_weight(row['helpful'])], dtype=torch.float),
                device='cpu')
        label = output_to_label(out)
        predictions.append(label)

    return predictions

def main(args):
    data = preprocess(load_data(args.input_path))
    nlp = spacy.load('en_core_web_md')

    model = load_model(args.model_path)

    with smart_open(args.output_path, 'wt') as output_file:
        for i, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            out = model.forward(
                torch.from_numpy(tokenize(nlp, row['reviewText'])).unsqueeze(0),
                torch.from_numpy(tokenize(nlp, row['summary'])).unsqueeze(0),
                torch.tensor([normalize_review_weight(row['helpful'])], dtype=torch.float),
                device=args.device)
            label = output_to_label(out)
            output_file.write(f'{label}\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
