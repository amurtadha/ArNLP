import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.functional import normalize, softmax
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer

class MyIdenticator(nn.Module):
    def __init__(self, opt, training=True):
        super(MyIdenticator, self).__init__()
        config = AutoConfig.from_pretrained(opt.pretrained_bert_name,  cache_dir='/workspace/plm/')
        self.config=config
        dim =512
        if training:
            self.encoder = SentenceTransformer(opt.pretrained_bert_name,  cache_dir='/workspace/plm/')
        else:
            self.encoder = SentenceTransformer(opt.baseline_plm,  cache_dir='/workspace/plm/')
        self.opt = opt
        self.n_clusters = self.opt.n_clusters
        self.training = training
        self.embed_dim=config.hidden_size
        self.dense = nn.Linear(dim, self.n_clusters)
        self.T= nn.Embedding(self.n_clusters,dim)


   
    def _get_r(self, text):
        with torch.no_grad():
            outputs = self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)

        outputs = normalize(outputs, p=2, dim=1)
        m_t = outputs
        v_t = softmax(self.dense(m_t), dim=-1)

        r_t = normalize(torch.matmul(v_t, self.T.weight), dim=-1)
        return r_t
    def forward(self, text, training=True):
        with torch.no_grad():
            outputs =  self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)


        outputs = normalize(outputs, p=2, dim=1)
        m_t = outputs        
        v_t= softmax(self.dense(m_t), dim=-1)
        r_t = normalize(torch.matmul( v_t, self.T.weight), dim=-1)
        m_n = m_t[torch.randint(0, m_t.size(0), (m_t.shape[0], self.opt.negative_sampling))]

        return self.loss(m_t, m_n, r_t).mean()

    def loss(self, m_t, m_n, r_t):

        step =  m_n.shape[1]
        m_t= normalize(m_t)
        m_n= normalize(m_n)
        r_t= normalize(r_t)
        pos = torch.sum(r_t * m_t, axis=-1, keepdim=True)

        pos= pos.expand(-1,step)
        r_t = r_t.unsqueeze(1).expand(-1,step, -1)
        neg = torch.sum(r_t * m_n, axis=-1)

        loss =torch.sum(torch.max(torch.zeros(neg.shape).to('cuda') , 1.0-pos+neg),dim=-1)
        return loss


class Pure_plm(nn.Module):
    def __init__(self, args, hidden_size=256):
        super(Pure_plm, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name,  cache_dir='/workspace/plm/')

        self.topic=MyIdenticator(args, training=False)
        self.topic.load_state_dict(torch.load(args.topic_model))
      
        self.encoder = AutoModel.from_pretrained(
            args.pretrained_bert_name, config=config)

        dim = 512
        self.classifier = nn.Linear(config.hidden_size +dim, args.lebel_dim)
    def forward(self, inputs, pred=False):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        with torch.no_grad():
            topic_r = self.topic._get_r(inputs[-1])
       
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        pooled_output = torch.cat([outputs, topic_r], dim=-1)
        logits = self.classifier(pooled_output)
      
        if pred:
            return logits, output
        else:
            return logits


    
