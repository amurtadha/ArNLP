
import logging
import argparse
import math
import os

import sys
from time import strftime, localtime
import random
import numpy
import  copy
from sklearn import metrics
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import   Process_topic

from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

from transformers import  AutoTokenizer, AutoModel
from MyModel import MyIdenticator
import pickle as pk
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print(opt.pretrained_bert_name)
        cache = 'cache/topic_{}_{}.pk'.format(self.opt.dataset, self.opt.plm)
        print(cache)
        if os.path.exists(cache):
            d = pk.load(open(cache, 'rb'))
            self.trainset = d['train']

        else:
            tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)
            self.trainset = Process_topic(opt.data_file['train'], tokenizer, opt.max_seq_len, opt.dataset)


            if not os.path.exists('cache'):
                os.mkdir('cache')
            d = {'train': self.trainset}
            pk.dump(d, open(cache, 'wb'))

        print(len( self.trainset), self.opt.plm)
        self.model = MyIdenticator(opt)
        
        self.model.to(opt.device)
        print(opt.device)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

   

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
    def _train(self, optimizer, train_data_loader,t_total):

        global_step = 0
        path = None

        if not os.path.exists('state_dict/topic/'):
            os.mkdir('state_dict/topic/')

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))

            loss_total=0
            n_total = 0
            self.model.train()
            min_loss = float('inf')

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                inputs= sample_batched['text']
                n_total += len(inputs)

                
             
                loss= self.model(inputs)
                loss_total+=loss.detach().item()
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
                    if  (loss_total/n_total) < min_loss:
                        min_loss = (loss_total/n_total)
                        if not os.path.exists('state_dict/topic'):
                            os.mkdir('state_dict/topic')
                        path = copy.deepcopy(self.model.state_dict())

            logger.info('epoch : {}, loss: {:.4f}'.format(epoch, loss_total/n_total))

            if (loss_total/n_total) < min_loss:
                path = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(path)

            path_=  'state_dict/topic/{}_{}_topic_.bm'.format(self.opt.dataset, self.opt.plm)
            torch.save(self.model.state_dict(), path_)

        return path
   
    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)


        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)

        t_total= int(len(train_data_loader) * self.opt.num_epoch)


        self._train( optimizer, train_data_loader, t_total)




def main(max_s=None,lr = None):
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='unlabeled', type=str, help='unlabeled')
    #parser.add_argument('--dataset', default='Corpus-6', type=str, help='Corpus-6, Corpus-26')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    # parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch', default=6, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=256, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--batch_size_val', default=256, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=20000, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--negative_sampling', default=20, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--n_clusters', default=15, type=int)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/camelbert-mix', type=str)
    #parser.add_argument('--pretrained_bert_name', default="D:/models/camelbert-msa", type=str)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/camelbert-da', type=str)
    #parser.add_argument('--pretrained_bert_name', default='E:/ADI_6_26/LaBSE', type=str)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/AraBERTv0.2b', type=str)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/AraBERTv0.1', type=str)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/camelbert-ca', type=str)
    # parser.add_argument('--pretrained_bert_name', default='UBC-NLP/ARBERT', type=str)
   # parser.add_argument('--pretrained_bert_name', default='D:/models/Arabic_BERT_Larg', type=str)
    #parser.add_argument('--pretrained_bert_name', default='bashar-talafha/multi-dialect-bert-base-arabic', type=str)
    #parser.add_argument('--pretrained_bert_name', default='D:/models/mutli_dialect', type=str)
    # parser.add_argument('--pretrained_bert_name', default='/workspace/plm/models--sentence-transformers--distiluse-base-multilingual-cased-v1/snapshots/ae9f5e096840ab325a30464d284f293fbea761a8/', type=str)
    parser.add_argument('--pretrained_bert_name', default='/workspace/plm/mabert', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int) #
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()


    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if lr is not None :
        opt.dropout = lr
    if max_s is not None :
        opt.max_seq_len = max_s



    plm = {
            'ours': 'OurArNLP/',
            # 'camel':'CAMeL-Lab/bert-base-arabic-camelbert-msa',
            'camel': 'bert-base-arabic-camelbert-mix',
            'labse': 'sentence-transformers/LaBSE',
            'arabert': 'aubmindlab/bert-base-arabertv2',
            'mdbert': 'bashar-talafha/multi-dialect-bert-base-arabic',
            'mbert': 'models--google-bert--bert-base-multilingual-cased',
            'peotbert': 'models--faisalq--bert-base-arapoembert',
            'arbert': 'arbert',
            'mabert': 'mabert',
            'sbert': 'sbertv2/snapshots/03a0532331151aeb3e1d2e602ffad62bb212a38d/',

        }
    opt.plm= opt.pretrained_bert_name
    opt.pretrained_bert_name= '/workspace/plm/{}'.format(plm[opt.pretrained_bert_name])


    dataset_files = {
        'train': '/workspace/NLP_ADI/datasets/large_corpus/unlabeled_corpus.txt',

    }
    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'text']

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.AdamW,  # default lr=0.001
        # 'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.inputs_cols = input_colses
    opt.data_file=dataset_files
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    # opt.device='cuda:5'
    log_file = '{}-{}.log'.format(opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
