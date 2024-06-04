
import logging
import argparse
import math
import os

from sklearn.metrics import accuracy_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import sys
from time import strftime, localtime
import random
import numpy
import  copy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
# torch.nn.CrossEntropyLoss(ignore_index=-1)(torch.rand((16,2)), torch.ones((16), dtype=int))
from data_utils import   Process_Corpus, Process_Corpus_from_json
import copy
from tqdm import tqdm
import json
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# from transformers import  RobertaTokenizer, RobertaModel
from transformers import  AutoTokenizer
# from roberta_model_ranker import EmoRanker
from MyModel import pure_plm_sbert,pure_plm

# from MyModel_biLSTM import Pure_Roberta
import pickle as pk

# os.environ['CUDA_VISIBLE_DEVICES'] ="1,2"
class Instructor:
    def __init__(self, opt):
        self.opt = opt


        cache = 'cache/TADI_{}_{}{}.pk'.format(opt.dataset, opt.plm, opt.plm_base)

        opt.lebel_dim = len(json.load(open('/'.join(opt.dataset_file['train'].split('/')[:-1])+'/labels.json')))

        if os.path.exists(cache):
            d = pk.load(open(cache, 'rb'))
            self.trainset = d['train']
            self.testset = d['test']
            self.valset = d['dev']
        else:
            tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)
            tokenizer_base_plm = AutoTokenizer.from_pretrained(opt.baseline_plm)

            self.trainset = Process_Corpus_from_json(opt.dataset_file['train'], tokenizer,tokenizer_base_plm, opt.max_seq_len, opt.dataset)
            self.testset = Process_Corpus_from_json(opt.dataset_file['test'], tokenizer,tokenizer_base_plm, opt.max_seq_len, opt.dataset)
            self.valset = Process_Corpus_from_json(opt.dataset_file['dev'], tokenizer,tokenizer_base_plm, opt.max_seq_len, opt.dataset)

            if not os.path.exists('cache'):
                os.mkdir('cache')
            d = {'train': self.trainset, 'test': self.testset, 'dev': self.valset}
            pk.dump(d, open(cache, 'wb'))

        logger.info('Train: {}, Test: {}, Dev: {}, Labels: {}, '.format(len( self.trainset), len( self.testset), len( self.valset), opt.lebel_dim ))


        print(self.opt.plm_base)
        if 'sbert' in self.opt.plm_base:
            self.model = pure_plm_sbert(opt)
        else:
            self.model = pure_plm(opt)
        #self.model = nn.DataParallel(self.model,device_ids=[0,1])
        self.model.to(opt.device)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        # self._print_args()


    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != RobertaModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def _train(self, criterion, optimizer,scheduler, train_data_loader, val_data_loader, test_data_loader, t_total):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            # logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            targets_all, outputs_all = None, None
            # switch model to training mode
            loss_total = []
            self.model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                # self.model.train()
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = inputs[-1]
                if 'sbert' in self.opt.topic_model:
                    inputs.append(sample_batched['text'])

                outputs = self.model(inputs)
                # targets = sample_batched['label'].to(self.opt.device)

                loss = criterion(outputs, targets)

                # logger.info(outputs.shape)
                loss.sum().backward()

                optimizer.step()
                # scheduler.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

            # logger.info('epoch : {}'.format(epoch))
            # logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc = self._evaluate_acc_f1(val_data_loader)
            logger.info(
                'epoch : {}/{}, loss: {:.4f},  > val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f}'.format( epoch,self.opt.num_epoch, np.mean(loss_total), pres, recall,
                                                                                                       f1_score, acc))
            if f1_score > max_val_acc:
                max_val_acc = f1_score
                # if not os.path.exists('state_dict'):
                #     os.mkdir('state_dict')
                # if not os.path.exists('state_dict/models/'):
                #     os.mkdir('state_dict/models/')
                # path = f"state_dict/models/{opt.dataset}_{opt.datetime}_{opt.seed}_{opt.pretrained_bert_name.split('/')[-1]}.pkl"
                path = copy.deepcopy(self.model.state_dict())

                # torch.save(self.model.state_dict(), path)

            lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,
                                                                       self.opt.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_inputs[-1]
                if 'sbert' in self.opt.topic_model:
                    t_inputs.append(t_sample_batched['text'])
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true = t_targets_all.cpu().detach().numpy().tolist()
            pred = torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()
            f = metrics.f1_score(true, pred, average='macro')
            r = metrics.recall_score(true, pred, average='macro')
            p = metrics.precision_score(true, pred, average='macro')
            acc = metrics.accuracy_score(true, pred)

        return p, r, f, acc



    def run(self):

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        #
        # optimizer = AdamW(self.model.parameters(), lr=2e-5)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
        #                                             num_training_steps=len(train_data_loader) * self.opt.num_epoch)
        #
        #
        # # optimizer = AdamW(self.model.parameters(), lr=5e-5)
        #
        # # Define loss function
        # loss_fn = nn.CrossEntropyLoss()
        #
        # num_epochs = 3
        # for epoch in range(self.opt.num_epoch):
        #     self.model.train()
        #     total_loss = 0
        #     for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
        #         optimizer.zero_grad()
        #
        #         inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
        #         labels = inputs[-1]
        #         if 'sbert' in self.opt.topic_model:
        #             inputs.append(sample_batched['text'])
        #
        #         preds = self.model(inputs)
        #
        #         # input_ids = batch['input_ids']
        #         # attention_mask = batch['attention_mask']
        #         # labels = batch['label']
        #         # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #         # logits = outputs.last_hidden_state[:, 0, :]
        #         # preds = model.classifier(logits)
        #         loss = loss_fn(preds, labels)
        #         total_loss += loss.item()
        #         loss.backward()
        #         optimizer.step()
        #         scheduler.step()
        #
        #     # Evaluate
        #     self.model.eval()
        #     all_preds = []
        #     all_labels = []
        #     with torch.no_grad():
        #         for i_batch, sample_batched in enumerate(tqdm(val_data_loader)):
        #
        #             inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
        #             labels = inputs[-1]
        #             if 'sbert' in self.opt.topic_model:
        #                 inputs.append(sample_batched['text'])
        #
        #             preds = self.model(inputs)
        #             preds = torch.argmax(preds, dim=1)
        #             # input_ids = batch['input_ids']
        #             # attention_mask = batch['attention_mask']
        #             # labels = batch['label']
        #             # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #             # logits = outputs.last_hidden_state[:, 0, :]
        #             # preds = model.classifier(logits)
        #             # preds = torch.argmax(preds, dim=1)
        #             all_preds.extend(preds.tolist())
        #             all_labels.extend(labels.tolist())
        #
        #     # print(all_preds)
        #     # print(all_labels)
        #
        #     f1 = metrics.f1_score(all_labels, all_preds, average='macro')
        #     r = metrics.recall_score(all_labels, all_preds, average='macro')
        #     p = metrics.precision_score(all_labels, all_preds, average='macro')
        #     acc = metrics.accuracy_score(all_labels, all_preds)
        #
        #
        #     accuracy = accuracy_score(all_labels, all_preds)
        #     print(f"Epoch {epoch + 1}/{num_epochs}: Loss: {total_loss}, Accuracy: {accuracy}, f1:{f1}")

        optimizer = AdamW(self.model.parameters(), lr=self.opt.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
        #                                             num_training_steps=len(train_data_loader) * self.opt.num_epoch)

        scheduler=None
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer = self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        # train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        # val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        t_total = int(len(train_data_loader) * self.opt.num_epoch)

        best_model_path = self._train(criterion, optimizer,scheduler, train_data_loader, val_data_loader, test_data_loader,
                                      t_total)
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(best_model_path)
        self.model.eval()
        pres, recall, f1_score, acc = self._evaluate_acc_f1(test_data_loader)
        logger.info(
            '>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(pres, recall,f1_score, acc))
        with open('results.txt', 'a+') as f :
            f.write('task:{}, t-plm:{}, plm:{}, prec:{}, rec:{}, f1:{}, acc:{}\n'.format(opt.dataset, opt.plm_base, opt.plm,pres, recall,f1_score, acc) )


def main(opt, seed= None,max_s=None,lr = None):
    # label_dims = {'Corpus-6': 6,'Corpus-26': 26}
    # opt.lebel_dim = label_dims[opt.dataset]
    if seed is not None:
        opt.seed = seed
    else:
        opt.seed = random.randint(20, 300)

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



    # dataset_files = {
    #     'train': '/workspace/ArNLP/datasets/{0}/MADAR-{0}-train.tsv'.format(opt.dataset),
    #     'test': '/workspace/ArNLP/datasets/{0}/MADAR-Corpus-26-test.tsv'.format(opt.dataset),
    #     'dev': '/workspace/ArNLP/datasets/{0}/MADAR-{0}-dev.tsv'.format(opt.dataset),
    # }

    dataset_files = {
        'train': '/workspace/ArNLP/datasets/{0}/train.json'.format(opt.dataset),
        'test': '/workspace/ArNLP/datasets/{0}/test.json'.format(opt.dataset),
        'dev': '/workspace/ArNLP/datasets/{0}/dev.json'.format(opt.dataset),
    }
    # os.environ['CUDA_VISIBLE_DEVICES']=opt.device_group
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group
    input_colses =  ['input_ids', 'segments_ids', 'input_mask','input_ids_r', 'segments_ids_r', 'input_mask_r', 'label']

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses

    logger.info('seed {}'.format(opt.seed))
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
        # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Corpus-26', type=str, help='Corpus-6, Corpus-26')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch', default=12, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 4, 8, 16, 32, 64 for BERT models')
    parser.add_argument('--batch_size_val', default=32, type=int, help='try 4, 8, 16, 32, 64 for BERT models')
    parser.add_argument('--num_runs', default=3, type=int, help='number of runs')
    parser.add_argument('--log_step', default=35500, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--negative_sampling', default=20, type=int)
    parser.add_argument('--n_clusters', default=15, type=int)
    parser.add_argument('--pretrained_bert_name', default='arbert', type=str)  # 26
    parser.add_argument('--baseline_plm', default='mabert', type=str)  # 26

    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--lebel_dim', default=6, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
    opt = parser.parse_args()

    opt.datetime = strftime("%y%m%d-%H%M", localtime())
    log_file = f'{opt.dataset}-{opt.datetime}.log'
    logger.addHandler(logging.FileHandler(log_file))
    plm = {
            'ours': 'OurArNLP/',
            # 'camel':'CAMeL-Lab/bert-base-arabic-camelbert-msa',
            'camel': 'bert-base-arabic-camelbert-mix',
            'labse': 'labse/models--sentence-transformers--LaBSE',
            'arabert': 'arabert/models--aubmindlab--bert-base-arabertv2/',
            'mdbert': 'bashar-talafha/multi-dialect-bert-base-arabic',
            'mbert': 'models--google-bert--bert-base-multilingual-cased',
            'peotbert': 'models--faisalq--bert-base-arapoembert',
            'arbert': 'arbert',
            'mabert': 'mabert',
            'sbert': 'sbertv2/snapshots/03a0532331151aeb3e1d2e602ffad62bb212a38d/',

        }
    opt.plm_base= opt.baseline_plm
    opt.plm= opt.pretrained_bert_name
    opt.pretrained_bert_name= '/workspace/plm/{}'.format(plm[opt.pretrained_bert_name])
    opt.topic_model = 'state_dict/topic/unlabeled_{}_topic_best.bm'.format(opt.baseline_plm)
    opt.baseline_plm= '/workspace/plm/{}'.format(plm[opt.baseline_plm])


    opt.pretrained_bert_name_path ='/workspace/plm/{}'.format(opt.pretrained_bert_name)

    opt.initializer =  torch.nn.init.xavier_uniform_
    opt.optimizer = torch.optim.AdamW
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    for i in range(opt.num_runs):
        logger.info('run: %d\n', i+1)
        main(opt)
