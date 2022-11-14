import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
from CoAttention import MultiHeadAttention
import pickle
from torch_geometric.nn import GCNConv
import scipy.sparse as sp

    
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x
    

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        for param in self.bert.parameters():
            param.requires_grad = not(args.freeze_bert)
        
        # dual multi head attention
        self.co_attention_layer_1 = MultiHeadAttention(
                                n_head=args.config.num_attention_heads, 
                                d_model=args.config.hidden_size, 
                                d_k=(args.config.hidden_size // args.config.num_attention_heads), 
                                d_v=(args.config.hidden_size // args.config.num_attention_heads), 
                                dropout=args.config.attention_probs_dropout_prob
                              )
        self.co_attention_layer_2 = MultiHeadAttention(
                                n_head=args.config.num_attention_heads, 
                                d_model=args.config.hidden_size, 
                                d_k=(args.config.hidden_size // args.config.num_attention_heads), 
                                d_v=(args.config.hidden_size // args.config.num_attention_heads), 
                                dropout=args.config.attention_probs_dropout_prob
                              )
        self.layer_norm = nn.LayerNorm(args.config.hidden_size, eps=args.config.layer_norm_eps)
        
        # GCN
        with open(args.data_file + 'label_graph.g', 'rb') as f:
            label_graph = pickle.load(f)
        ed = sp.coo_matrix(torch.from_numpy(label_graph))
        self.edge_index = torch.from_numpy(np.vstack((ed.row, ed.col))).long().to(args.device)
        self.label_embedding = nn.Parameter(torch.randn(args.label_num, 
                                                        args.label_embedding_size, 
                                                        dtype=torch.float32))
        nn.init.kaiming_normal_(self.label_embedding.data)
        self.gcn_layers = nn.ModuleList([GCNConv(args.label_embedding_size, args.label_embedding_size) \
                          for i in range(args.num_gcn_layer)])
        self.gcn_dropout = nn.Dropout(args.gcn_dropout)
        
        # contarstive learning
        self.sim = Similarity(temp=args.temperature)
        self.mlp = MLPLayer(input_dim = args.config.hidden_size, 
                            output_dim = args.config.hidden_size)
        self.mlp_relation_to_joint = MLPLayer(input_dim = args.config.hidden_size, 
                                              output_dim = args.config.hidden_size)
        self.mlp_label_to_joint = MLPLayer(input_dim = args.label_embedding_size, 
                                              output_dim = args.config.hidden_size)
        
        # classifier
        self.fc_top = nn.Linear(args.config.hidden_size, args.n_top)
        self.fc_sec = nn.Linear(args.config.hidden_size + args.n_top, args.n_sec)
        self.fc_conn = nn.Linear(args.config.hidden_size + args.n_sec, args.n_conn)

    
    # def jaccard(self, A, B):
    #     return len(set(A).intersection(set(B))) / len(set(A).union(set(B)))
    
    def dice(self, A, B):
        return (2 * len(set(A).intersection(set(B)))) / (len(set(A)) + len(set(B)))
    
    def forward(self, x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask, train=False):
        if train:
            return self.train_forward(x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask)
        else:
            return self.evaluate_forward(x, mask, arg1_mask, arg2_mask)
    
    def evaluate_forward(self, x, mask, arg1_mask, arg2_mask):
        ### BERT encoder
        context = x  # (batch, len)
        bert_out = self.bert(context, attention_mask=mask)
        
        
        ### dual multi-head attention
        arg1_mask = arg1_mask[:, None, None, :]
        arg2_mask = arg2_mask[:, None, None, :]
        
        hidden_last = bert_out.last_hidden_state
        for i in range(self.args.num_co_attention_layer):
            arg2_hidden_last, _ = self.co_attention_layer_1(q=hidden_last, 
                                                        k=hidden_last, 
                                                        v=hidden_last, 
                                                        mask=arg1_mask)
            arg1_hidden_last, _ = self.co_attention_layer_2(q=hidden_last, 
                                                            k=hidden_last, 
                                                            v=hidden_last, 
                                                            mask=arg2_mask)
            updated_hidden_last = (arg1_hidden_last * arg1_mask.squeeze().unsqueeze(dim=-1)) \
                                + (arg2_hidden_last * arg2_mask.squeeze().unsqueeze(dim=-1))
            hidden_last = self.layer_norm(updated_hidden_last) # (batch, seq_len, hidden)
        
        
        ### classifier
        pooled = hidden_last[:, 0, :] # (batch, hidden)
        logits_top = self.fc_top(pooled) # (batch, top)
        logits_sec = self.fc_sec(torch.cat([pooled, logits_top], dim=-1)) # (batch, sec)
        logits_conn = self.fc_conn(torch.cat([pooled, logits_sec], dim=-1)) # (batch, conn)
        
        return logits_top, logits_sec, logits_conn
    
    def train_forward(self, x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask):
        ### BERT encoder
        bs = x.shape[0]
        context = torch.cat([x, x], dim=0)  # (batch*2, len)
        mask = torch.cat([mask, mask], dim=0) # (batch*2, len)
        bert_out = self.bert(context, attention_mask=mask)
        
        
        ### dual multi-head attention
        arg1_mask = torch.cat([arg1_mask, arg1_mask], dim=0)[:, None, None, :]
        arg2_mask = torch.cat([arg2_mask, arg2_mask], dim=0)[:, None, None, :]
        
        hidden_last = bert_out.last_hidden_state
        for i in range(self.args.num_co_attention_layer):
            arg2_hidden_last, _ = self.co_attention_layer_1(q=hidden_last, 
                                                        k=hidden_last, 
                                                        v=hidden_last, 
                                                        mask=arg1_mask)
            arg1_hidden_last, _ = self.co_attention_layer_2(q=hidden_last, 
                                                            k=hidden_last, 
                                                            v=hidden_last, 
                                                            mask=arg2_mask)
            updated_hidden_last = (arg1_hidden_last * arg1_mask.squeeze().unsqueeze(dim=-1)) \
                                + (arg2_hidden_last * arg2_mask.squeeze().unsqueeze(dim=-1))
            hidden_last = self.layer_norm(updated_hidden_last) # (batch*2, seq_len, hidden)
      
        
        ### compute sudo label for contrastive learning
        y10 = y1_top
        y11 = y1_sec.cpu().numpy()
        y11 += self.args.n_top
        y11 = torch.from_numpy(y11).to(x.device)
        y12 = y1_conn.cpu().numpy()
        y12 += (self.args.n_top + self.args.n_sec)
        y12 = torch.from_numpy(y12).to(x.device)
        
        # Top
        dice_T = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_T[i, j] = self.dice([y10.cpu().numpy()[i]], [y10.cpu().numpy()[j]])
        dice_T = torch.from_numpy(dice_T).to(x.device)   
        
        # Second
        dice_S = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_S[i, j] = self.dice([y11.cpu().numpy()[i]], [y11.cpu().numpy()[j]])
        dice_S = torch.from_numpy(dice_S).to(x.device)   
        
        # Connective
        dice_C = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_C[i, j] = self.dice([y12.cpu().numpy()[i]], [y12.cpu().numpy()[j]])
        dice_C = torch.from_numpy(dice_C).to(x.device)   
        
        # Top-Second
        dice_TS = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_TS[i, j] = self.dice([y10.cpu().numpy()[i], y11.cpu().numpy()[i]], 
                                                [y10.cpu().numpy()[j], y11.cpu().numpy()[j]])
        dice_TS = torch.from_numpy(dice_TS).to(x.device)   
        
        # Second-Connective
        dice_SC = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_SC[i, j] = self.dice([y11.cpu().numpy()[i], y12.cpu().numpy()[i]], 
                                                [y11.cpu().numpy()[j], y12.cpu().numpy()[j]])
        dice_SC = torch.from_numpy(dice_SC).to(x.device)   
        
        # Top-Second-Connective
        dice_TSC = np.empty(shape=[bs, bs])
        for i in range(bs):
            for j in range(bs):
                dice_TSC[i, j] = self.dice([y10.cpu().numpy()[i], y11.cpu().numpy()[i], y12.cpu().numpy()[i]], 
                                                  [y10.cpu().numpy()[j], y11.cpu().numpy()[j], y12.cpu().numpy()[j]])
        dice_TSC = torch.from_numpy(dice_TSC).to(x.device)
        
        dice_multi_view = (dice_T + dice_S + dice_C + dice_TS + dice_SC + dice_TSC) / 6.0
        
        
        ### local_hierarcial_contrastive_loss
        pooler_output = hidden_last[:, 0, :] # (batch*2, hidden)
        pooled = self.mlp(pooler_output) # (batch*2, hidden)
        z1, z2 = pooled.reshape(bs, 2, -1)[:, 0], pooled.reshape(bs, 2, -1)[:, 1] # (batch, hidden)
        cos_sim_z1_z2 = self.sim(z1.unsqueeze(1), z2.unsqueeze(0)) # (batch, batch)
        LogSoftmax = nn.LogSoftmax(dim=1)
        multi_view_log_softmax = LogSoftmax(cos_sim_z1_z2) # (batch, batch)
        local_hierarcial_contrastive_loss = (dice_multi_view * (-multi_view_log_softmax)).sum() / bs
        
        
        ### add gcn to get label representation
        label_repr = self.label_embedding
        for gcn_layer in self.gcn_layers:
            label_repr = F.relu(gcn_layer(label_repr, self.edge_index))
            label_repr = self.gcn_dropout(label_repr)
        label_repr = self.mlp_label_to_joint(label_repr)
        
        
        ### global_hierarcial_contrastive_loss
        relation = self.mlp_relation_to_joint(pooler_output[:bs, :])
        cos_sim_z1_label_top = self.sim(relation.unsqueeze(1), \
            label_repr[0:self.args.n_top].unsqueeze(0)) # (batch, 4)
        cos_sim_z1_label_sec = self.sim(relation.unsqueeze(1), \
            label_repr[self.args.n_top:(self.args.n_top+self.args.n_sec)].unsqueeze(0)) # (batch, 11)
        cos_sim_z1_label_conn = self.sim(relation.unsqueeze(1), \
            label_repr[(self.args.n_top+self.args.n_sec):].unsqueeze(0)) # (batch, 102)
        loss_fct = nn.CrossEntropyLoss()
        global_hierarcial_contrastive_loss = loss_fct(cos_sim_z1_label_top, y1_top) \
                                            + loss_fct(cos_sim_z1_label_sec, y1_sec) \
                                            + loss_fct(cos_sim_z1_label_conn, y1_conn)
        
        
        ### classification loss
        logits_top = self.fc_top(pooler_output[:bs, :]) # (batch, top)
        logits_sec = self.fc_sec(torch.cat([pooler_output[:bs, :], logits_top], dim=-1)) # (batch, sec)
        logits_conn = self.fc_conn(torch.cat([pooler_output[:bs, :], logits_sec], dim=-1)) # (batch, conn)
        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(logits_top, y1_top) \
                            + loss_fct(logits_sec, y1_sec) \
                            + loss_fct(logits_conn, y1_conn)
        
        
        loss = classification_loss \
                + self.args.lambda_global * global_hierarcial_contrastive_loss \
                + self.args.lambda_local * local_hierarcial_contrastive_loss
        
        return logits_top, logits_sec, logits_conn, loss
    
        
    
    
    
    
    
    
    
    
    
    

    
    