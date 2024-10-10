# A lot of the code here is based on the code Jiaxin developed
# I used taco server to run this code (GPU)

import pandas as pd
from os.path import join as opj
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import math
import numpy as np
from multiprocessing import Pool, current_process, Queue
import textstat
import os
import gc
import sys
os.environ["CUDA_HOME"] = "/usr/local/cuda" # this one is due to an error I get when running from the command line
# if the above does not work, you have to run "CUDA_HOME=/usr/local/cuda" in bash
sys.path.append("/shared/2/projects/jiaxin/hf_models/politeness")
from politeness_inference import PolitenessEstimator
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
queue = Queue()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# def get_prediction(data):
#     gpu_id = queue.get()
#     # model = models[gpu_id]
#     try:
#         # run processing on GPU <gpu_id>
#         ident = current_process().ident
#         print('{}: starting process on GPU {}'.format(ident, gpu_id), flush=True)
#
#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#         # you can easily switch this with other inference objects like topic, sentiment and sexual content
#         inti = Estimator(cuda=True)
#         res = inti.predict(data)
#         print('{}: finished'.format(ident), flush=True)
#         return res
#     finally:
#         queue.put(gpu_id)
#         # print("queue updated")


def chunks(lst, num):
    n = int(len(lst) / num)
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# class multi_gpu_inference():
#     def __init__(self, gpus=[0], batch_size=256):
#         # self.queue = Queue()
#         self.gpus = gpus
#         self.batch_size = batch_size
#         # self.m = Manager()
#         # self.queue = self.m.Queue()
#         # self.models = {}
#         # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#         # os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus)[1:-1]
#         print(str(gpus)[1:-1], flush=True)
#         # initialize the queue with the GPU ids
#         for gpu_id in gpus:
#             # for _ in range(PROC_PER_GPU):
#             queue.put(gpu_id)
#             # self.models[gpu_id] = PolitenessEstimator(cuda = gpu_id)
#
#     def predict(self, data):
#         pool = Pool(processes=len(self.gpus))
#         inferred = []
#         for res in pool.map(get_prediction, chunks(data, len(self.gpus))):
#             inferred += res
#         pool.close()
#         pool.join()
#         return inferred


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Estimator():
    def __init__(self, tokenizer, model, num_labels, cuda=True, batch_size=256):

        TOKENIZER = tokenizer
        MODEL = model
        BATCH_SIZE = batch_size
        self.chunk_size = 0.02

        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
        self.cuda = cuda
        if cuda:
            self.model.cuda()

        self.training_args = TrainingArguments(
            output_dir='/shared/3/projects/relationship-aware-networks-wikipedia',  # output directory
            num_train_epochs=1,  # total number of training epochs
            per_device_eval_batch_size=BATCH_SIZE,  # batch size for evaluation
        )

    def data_iterator(self, train_x, chunk_size=500000):
        if chunk_size < 500000:
            chunk_size = 500000
        n_batches = math.ceil(len(train_x) / chunk_size)
        for idx in range(n_batches):
            x = train_x[idx * chunk_size:(idx + 1) * chunk_size]
            yield x

    # eval_data is a list of input or a pandas frame
    def prepare_dataset(self, eval_data, max_length=100):
        if type(eval_data) == list:
            print(f'start tokenizing {len(eval_data)} lines of text', flush=True)
            eval_encodings = self.tokenizer(eval_data, truncation=True, max_length=max_length, padding=True)
            eval_dataset = MyDataset(eval_encodings, [0] * len(eval_data))
            return eval_dataset

    def predict(self, eval_data, max_length=512):
        eval_iterator = self.data_iterator(eval_data, chunk_size=int(len(eval_data) * self.chunk_size))
        eval_preds = []

        for x in tqdm(eval_iterator):
            trainer = Trainer(
                model=self.model,  # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,  # training arguments, defined above
            )
            eval_dataset = self.prepare_dataset(x, max_length)
            eval_preds_raw, eval_labels, _ = trainer.predict(eval_dataset)
            if self.num_labels == 1:
                eval_preds += [it[0] for it in eval_preds_raw]
            else:
                #print(np.argmax(eval_preds_raw, axis=-1))
                eval_preds += list(np.argmax(eval_preds_raw, axis=-1))
        return eval_preds

    def predict_proba(self, eval_data, max_length=512):
        eval_iterator = self.data_iterator(eval_data, chunk_size=int(len(eval_data) * self.chunk_size))
        eval_preds = []

        for x in tqdm(eval_iterator):
            trainer = Trainer(
                model=self.model,  # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,  # training arguments, defined above
            )
            eval_dataset = self.prepare_dataset(x, max_length)
            eval_preds_raw, eval_labels, _ = trainer.predict(eval_dataset)
            # convert into probabilities
            probabilities = F.softmax(torch.tensor(eval_preds_raw), dim=-1).numpy()
            if self.num_labels == 1:
                eval_preds += [it[0] for it in probabilities]
            else:
                #print(np.argmax(eval_preds_raw, axis=-1))
                eval_preds.extend(probabilities.tolist())
        return eval_preds


def arguments():
    parser = ArgumentParser()
    parser.set_defaults(show_path=False, show_similarity=False)
    parser.add_argument('--predict_data_path', default=None)
    parser.add_argument('--text_key', default='text')
    # parser.add_argument('--saving_path', default=None)

    return parser.parse_args()


def read_file(path):
    with open(path) as r:
        lines = r.readlines()
        lines = [it.strip() for it in lines]
    return lines


def write_file(path, lines):
    with open(path, 'w') as w:
        for line in tqdm(lines):
            w.writelines(str(line) + '\n')
    print(f'file saved at: {path}', flush=True)


if __name__ == '__main__':
    # loading the dataset for prediction
    data_folder = '/shared/3/projects/relationship-aware-networks-wikipedia/wiki_generated_data/'
    orig_data = pd.read_csv(opj(data_folder, 'talkpage_discussions_sustained_articles',
                                'all_comments_posted_for_dl_models.csv'))
    print(f"Data has been loaded, rows to predict: {orig_data.shape[0]}", flush=True)

    # now we need to run the prediction models and extract the prediciton
    text_for_prediction_df = orig_data['clean_text']
    text_for_prediction_df.fillna("", inplace=True)
    text_for_prediction = list(text_for_prediction_df)
    text_for_prediction = text_for_prediction

    # A - sentiment
    model = Estimator(tokenizer="siebert/sentiment-roberta-large-english",
                      model="siebert/sentiment-roberta-large-english", num_labels=2, cuda=True)
    print('Sentiment model loaded', flush=True)
    sentiment_score = model.predict_proba(text_for_prediction)
    sentiment_score = [s[1] for s in sentiment_score] # extracting the prob(positive_score)
    torch.cuda.empty_cache()
    gc.collect()

    # B - formality
    model = Estimator(tokenizer='s-nlp/roberta-base-formality-ranker', model='s-nlp/roberta-base-formality-ranker',
                      num_labels=2, cuda=True)
    print('Formality model loaded', flush=True)
    formality_score = model.predict_proba(text_for_prediction)
    formality_score = [fs[1] for fs in formality_score]
    torch.cuda.empty_cache()
    gc.collect()

    # C - politeness
    inti = PolitenessEstimator(cuda=True)
    print('Politeness model loaded', flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    politeness_scores = inti.predict(text_for_prediction)

    # D - toxicity
    model = Estimator(tokenizer="s-nlp/roberta_toxicity_classifier", model="s-nlp/roberta_toxicity_classifier",
                      num_labels=2, cuda=True)
    print('Toxicity model loaded', flush=True)
    toxicity_scores = model.predict_proba(text_for_prediction)
    # taking only the second value, as this is a binary classification, so only the prob(toxic) is relevant
    toxicity_scores = [s[1] for s in toxicity_scores]
    torch.cuda.empty_cache()
    gc.collect()

    # E - certainty
    model = Estimator(tokenizer='pedropei/sentence-level-certainty', model='pedropei/sentence-level-certainty',
                      num_labels=1, cuda=True)
    certainty_scores = model.predict(text_for_prediction)
    torch.cuda.empty_cache()
    gc.collect()

    # converting the list of lists we have into a pandas df
    orig_data['sentiment'] = sentiment_score
    orig_data['formality'] = formality_score
    orig_data['politeness'] = politeness_scores
    orig_data['toxicity'] = toxicity_scores
    orig_data['certainty'] = certainty_scores

    # saving the df back to disk, with all new features
    orig_data.to_csv(opj(data_folder, 'talkpage_discussions_sustained_articles',
                         'all_comments_posted_with_dl_predictions_take2.csv'), index=False)
    scores_as_df = pd.DataFrame({
        'sentiment': sentiment_score,
        'formality': formality_score,
        'politeness': politeness_scores,
        'toxicity': toxicity_scores,
        'certainty': certainty_scores
    })
    print(f"Prediction ended for all tags. Number of predicted values: {scores_as_df.shape[0]}. "
          f"Measures summary:\n {scores_as_df.describe()}", flush=True)

