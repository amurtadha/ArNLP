from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np
class Process_Corpus_from_json(Dataset):
    def __init__(self, fname, tokenizer, baseline_plm, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        labels = json.load(open('/'.join(fname.split('/')[:-1])+'/labels.json'))

        # data = open(fname, encoding="utf-8").read().splitlines()
        data = json.load(open(fname, encoding="utf-8"))

        all_data=[]
        for d in tqdm(data):

            # text, label = d.split('\t')
            text, label = d['text'], d['label']
            if label not in labels:continue
            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

           

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]
   

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

         
            data = {
                'text': text,
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)




class Process_topic(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        data = open(fname, encoding="utf-8")

        print(fname)

        all_data=[]
        for line in data:
            if len(line.split())<3:continue

            data = {
                'text': line.strip(),
            }
            all_data.append(data)
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
