import logging
import argparse
import math
import os

import sys
from time import strftime, localtime
import random
import numpy
import copy
from sklearn import metrics
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import Process_topic

from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

from transformers import AutoTokenizer, AutoModel
from MyModel import MyIdenticator
import pickle as pk


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        cache = 'cache/topic_{}_{}.pk'.format(self.opt.dataset, self.opt.plm)
        if os.path.exists(cache):
            d = pk.load(open(cache, 'rb'))
            self.trainset = d['train']
        else:
            tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name, cache_dir='/workspace/plm/')
            self.trainset = Process_topic(opt.data_file['train'], tokenizer, opt.max_seq_len, opt.dataset)

            if not os.path.exists('cache'):
                os.mkdir('cache')
            d = {'train': self.trainset}
            pk.dump(d, open(cache, 'wb'))

        logging.info('training: {}, PLM: {}'.format(len(self.trainset), self.opt.plm))
        self.model = MyIdenticator(opt)

        self.model.to(opt.device)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def _train(self, optimizer, train_data_loader, t_total):

        global_step = 0
        path = None

        if not os.path.exists('state_dict/topic/'):
            os.mkdir('state_dict/')
            os.mkdir('state_dict/topic/')

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            loss_total = 0
            n_total = 0
            self.model.train()
            min_loss = float('inf')

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                inputs = sample_batched['text']
                n_total += len(inputs)

                loss = self.model(inputs)
                loss_total += loss.detach().item()
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
                    if (loss_total / n_total) < min_loss:
                        min_loss = (loss_total / n_total)
                        if not os.path.exists('state_dict/topic'):
                            os.mkdir('state_dict/topic')
                        path = copy.deepcopy(self.model.state_dict())

            logger.info('epoch : {}, loss: {:.4f}'.format(epoch, loss_total / n_total))

            if (loss_total / n_total) < min_loss:
                path = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(path)

            path_ = 'state_dict/topic/{}_{}_topic_.bm'.format(self.opt.dataset, self.opt.plm)
            torch.save(self.model.state_dict(), path_)

        return path

    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)

        t_total = int(len(train_data_loader) * self.opt.num_epoch)

        self._train(optimizer, train_data_loader, t_total)


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='unlabeled', type=str, help='unlabeled')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='')
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='')
    parser.add_argument('--num_epoch', default=6, type=int, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--batch_size_val', default=256, type=int, help='')
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--negative_sampling', default=20, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--n_clusters', default=15, type=int)
    parser.add_argument('--pretrained_bert_name', default='sbert', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)  #
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--log_step', default=2000, type=int)

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    plm = {
        'sbert': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
        'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    }

    opt.plm = opt.pretrained_bert_name
    opt.pretrained_bert_name = plm[opt.pretrained_bert_name]

    dataset_files = {
        'train': '/workspace/NLP_ADI/datasets/large_corpus/unlabeled_corpus.txt',
    }
    input_colses = ['input_ids', 'segments_ids', 'input_mask', 'text']
    opt.optimizer = torch.optim.Adam

    opt.inputs_cols = input_colses
    opt.data_file = dataset_files
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    log_file = '{}-{}.log'.format(opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
