import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.functional import normalize, softmax
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer

class DialTopic_SBERT(nn.Module):
    def __init__(self, opt, training=True):
        super(DialTopic_SBERT, self).__init__()
        config = AutoConfig.from_pretrained(opt.pretrained_bert_name)
        self.config=config
        dim =512
        # self.encoder = AutoModel.from_pretrained(opt.pretrained_bert_name, config=config)
        if training:
            self.encoder = SentenceTransformer(opt.pretrained_bert_name)
        else:
            self.encoder = SentenceTransformer(opt.baseline_plm)
        # self.emb_matrix = embedding_matrix
        self.opt = opt
        self.n_clusters = self.opt.n_clusters
        self.training = training
        self.embed_dim=config.hidden_size
        # self.dense = nn.Linear(config.hidden_size, self.n_clusters)
        self.dense = nn.Linear(dim, self.n_clusters)
        # self.T= nn.Embedding(self.n_clusters,config.hidden_size)
        self.T= nn.Embedding(self.n_clusters,dim)

        self.ortho = 0.1

    def reset_M(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.M.data.uniform_(-stdv, stdv)

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_
        return nn.Embedding.from_pretrained(torch.tensor(clusters, dtype=torch.float))
    def _get_r(self, text):
        with torch.no_grad():
            outputs = self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)

        outputs = normalize(outputs, p=2, dim=1)
        m_t = outputs
        v_t = softmax(self.dense(m_t), dim=-1)

        r_t = normalize(torch.matmul(v_t, self.T.weight), dim=-1)
        return r_t
    def forward(self, text, training=True):
        # input_ids, token_type_ids, attention_mask = inputs[:4]
        # text = inputs[4]
        with torch.no_grad():
            # outputs =  self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            outputs =  self.encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)


        # print(outputs.device)
        # outputs = self.mean_pooling(outputs, attention_mask)
        # outputs = torch.mean(outputs, dim=0, keepdim=True)
        # print(outputs.shape)
        # Normalize embeddings
        outputs = normalize(outputs, p=2, dim=1)

        # outputs = self.roberta(input_ids,  attention_mask=attention_mask)
        m_t = outputs
        # m_t = outputs['last_hidden_state'][:, 0, :]
        # print(m_t.shape)
        # print(self.config.hidden_size)
        v_t= softmax(self.dense(m_t), dim=-1)

        r_t = normalize(torch.matmul( v_t, self.T.weight), dim=-1)

        # create the negative samplings
        m_n = m_t[torch.randint(0, m_t.size(0), (m_t.shape[0], self.opt.negative_sampling))]

        # Compute the loss
        # print(self.loss(m_t, m_n, r_t))
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


    def predict(self, m_t):
        return torch.softmax(
            torch.add(torch.bmm(self.w_a.unsqueeze(0), m_t.unsqueeze(0).transpose(2, 1)).transpose(1, 2).squeeze(),
                      self.b_a.unsqueeze(0)), dim=1)

    def clausters(self):
        E_n = normalize(self.E.weight, dim=1)
        T_n = normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection
class DialTopic(nn.Module):
    def __init__(self, opt, training=True):
        super(DialTopic, self).__init__()
        config = AutoConfig.from_pretrained(opt.pretrained_bert_name)

        self.encoder = AutoModel.from_pretrained(opt.pretrained_bert_name, config=config)
        # self.emb_matrix = embedding_matrix
        self.opt = opt
        self.n_clusters = self.opt.n_clusters
        self.training = training
        self.embed_dim=config.hidden_size
        self.dense = nn.Linear(config.hidden_size, self.n_clusters)
        self.T= nn.Embedding(self.n_clusters,config.hidden_size)

        self.ortho = 0.1
        self.M = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size), requires_grad=True)


        self.reset_M()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def reset_M(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.M.data.uniform_(-stdv, stdv)

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_
        return nn.Embedding.from_pretrained(torch.tensor(clusters, dtype=torch.float))

    def forward(self, inputs, training=True):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        with torch.no_grad():
            outputs =  self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        outputs = self.mean_pooling(outputs, attention_mask)

        # Normalize embeddings
        outputs = normalize(outputs, p=2, dim=1)

        # outputs = self.roberta(input_ids,  attention_mask=attention_mask)
        m_t = outputs
        # m_t = outputs['last_hidden_state'][:, 0, :]

        v_t= softmax(self.dense(m_t), dim=-1)

        r_t = normalize(torch.matmul( v_t, self.T.weight), dim=-1)

        # create the negative samplings
        m_n = m_t[torch.randint(0, m_t.size(0), (m_t.shape[0], self.opt.negative_sampling))]

        # Compute the loss
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


    def predict(self, m_t):
        return torch.softmax(
            torch.add(torch.bmm(self.w_a.unsqueeze(0), m_t.unsqueeze(0).transpose(2, 1)).transpose(1, 2).squeeze(),
                      self.b_a.unsqueeze(0)), dim=1)

    def clausters(self):
        E_n = normalize(self.E.weight, dim=1)
        T_n = normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection




class pure_plm(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256, active_att=False):
        super(pure_plm, self).__init__()
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_group
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model_dir)

        self.topic=DialTopic(args, training=False)
        self.topic.load_state_dict(torch.load(args.topic_model))
        # self.topic.load_state_dict(torch.load(args.topic_model))

        # self.topic.to(args.device)
        self.encoder = AutoModel.from_pretrained(
            args.pretrained_bert_name, config=config)


        self.active_att=active_att
        if active_att:
            self.att =NoQueryAttention(config.hidden_size*2)

            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            layers = [nn.Linear(config.hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(.1),
                      nn.Linear(hidden_size, args.lebel_dim)]
            self.classifier = nn.Sequential(*layers)
        else:

            # Define an intermediate layer
            # self.intermediate_layer = nn.Linear(config.hidden_size, self.hidden_size * 2)
            # self.dropout = nn.Dropout(0.3)  # Optional dropout layer
            # self.relu = nn.ReLU()
            #
            #
            self.classifier = nn.Linear(config.hidden_size * 2, args.lebel_dim)
            # layers = [nn.Linear(config.hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(.1),
            #           nn.Linear(hidden_size, args.lebel_dim)]
            # self.classifier = nn.Sequential(*layers)

    def forward(self, inputs, pred=False):
        input_ids, token_type_ids, attention_mask = inputs[:3]

        # outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0, :]
        # outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['last_hidden_state']
        with torch.no_grad():
            # topic_r = self.topic(inputs[:3])
            topic_r = self.topic(inputs[3:-1])




        # print(outputs.shape,topic_r.shape )
        if self.active_att:
            outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['last_hidden_state']

        # add topic to the representation
            topic_r=topic_r.unsqueeze(1).expand(-1, outputs.shape[1], -1)
            pooled_output = torch.cat([outputs, topic_r], dim=-1)
            pooled_output = self.dropout(pooled_output)
            _, score= self.att(pooled_output)
            output= torch.sum(pooled_output*score.transpose(1, 2), dim=1)
            logits = self.classifier(output)
        # logits = self.classifier(pooled_output)
        else:
            outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
            # Pass through the intermediate layer and apply ReLU

            # outputs=outputs[:, 0, :]
            pooled_output = torch.cat([outputs, topic_r], dim=-1)

            # intermediate_output = self.relu(self.intermediate_layer(pooled_output))
            #
            # # Optionally apply dropout
            # intermediate_output = self.dropout(intermediate_output)

            # pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # logits = self.classifier(intermediate_output)
        if pred:
            return logits, output
        else:
            return logits


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
class pure_plm_sbert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256, active_att=False):
        super(pure_plm_sbert, self).__init__()
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_group
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model_dir)

        self.topic=DialTopic_SBERT(args, training=False)
        self.topic.load_state_dict(torch.load(args.topic_model))
        # self.topic.load_state_dict(torch.load(args.topic_model))

        # self.topic.to(args.device)
        self.encoder = AutoModel.from_pretrained(
            args.pretrained_bert_name, config=config)

        dim = 512
        self.active_att=active_att
        if active_att:
            self.att =NoQueryAttention(config.hidden_size+dim)

            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            layers = [nn.Linear(config.hidden_size +dim, hidden_size), nn.ReLU(), nn.Dropout(.1),
                      nn.Linear(hidden_size, args.lebel_dim)]
            self.classifier = nn.Sequential(*layers)
        else:

            # Define an intermediate layer
            # self.intermediate_layer = nn.Linear(config.hidden_size, self.hidden_size * 2)
            # self.dropout = nn.Dropout(0.3)  # Optional dropout layer
            # self.relu = nn.ReLU()
            #
            #
            self.classifier = nn.Linear(config.hidden_size +dim, args.lebel_dim)
            # self.classifier = nn.Linear(config.hidden_size * 2, args.lebel_dim)
            # layers = [nn.Linear(config.hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(.1),
            #           nn.Linear(hidden_size, args.lebel_dim)]
            # self.classifier = nn.Sequential(*layers)

    def forward(self, inputs, pred=False):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        with torch.no_grad():
            topic_r = self.topic._get_r(inputs[-1])
        # topic_r = self.topic._get_r(inputs[-1])




        # print(outputs.shape,topic_r.shape )
        if self.active_att:
            outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)['last_hidden_state']

        # add topic to the representation
            topic_r=topic_r.unsqueeze(1).expand(-1, outputs.shape[1], -1)
            pooled_output = torch.cat([outputs, topic_r], dim=-1)
            pooled_output = self.dropout(pooled_output)
            _, score= self.att(pooled_output)
            output= torch.sum(pooled_output*score.transpose(1, 2), dim=1)
            logits = self.classifier(output)
        # logits = self.classifier(pooled_output)
        else:
            outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
            # Pass through the intermediate layer and apply ReLU

            # outputs=outputs[:, 0, :]
            pooled_output = torch.cat([outputs, topic_r], dim=-1)

            # intermediate_output = self.relu(self.intermediate_layer(pooled_output))
            #
            # # Optionally apply dropout
            # intermediate_output = self.dropout(intermediate_output)

            # pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # logits = self.classifier(intermediate_output)
        if pred:
            return logits, output
        else:
            return logits


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
