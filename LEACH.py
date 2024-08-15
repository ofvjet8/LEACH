import math
import dgl
import dgl.function as fn
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

import argparse
import math
import urllib.request
import pickle

import numpy as np
import scipy.io


import dgl
from sklearn.metrics import roc_auc_score , f1_score
from sklearn.metrics import roc_curve
import random
# torch.manual_seed(0)

import time
import pandas as pd
import networkx as nx


from collections import Counter, defaultdict

import inspect, os, sys

# random.seed(10)


parser = argparse.ArgumentParser(
    description="Training GNN on ogbn-products benchmark"
)
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--n_hid", type=int, default=128)
parser.add_argument("--n_inp", type=int, default=128)
parser.add_argument("--n_heads", type=int, default=8) 
parser.add_argument("--clip", type=int, default=1.0)
parser.add_argument("--max_lr", type=float, default=1e-3)
parser.add_argument("--n_batch", type=int, default=256)
parser.add_argument("--max_length", type=int, default=7)
parser.add_argument("--dataset_name", type=str, default="amz") 
parser.add_argument("--negative_sampling", type=float, default=1) 
parser.add_argument("--num_patience", type=int, default=5)
parser.add_argument("--HGT_layers", type=int, default=3)
parser.add_argument("--slice_layers", type=int, default=2)
parser.add_argument("--top_n", type=int, default=5)

print("python commanded at" , time.strftime('%x %X'))

args, unknown = parser.parse_known_args()

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):

        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        # print('mask', attn_mask.size())
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask == True, -1e9)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):

        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_V = torch.nn.Linear(d_model, d_v * n_heads)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(d_k)
        self.wrap = torch.nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layerNorm = torch.nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_prod_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        output = self.wrap(context)

        return self.layerNorm(output + residual), attn


class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self, d_model, d_ff):

        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):

        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):

        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(
            enc_outputs
        )  # enc_outputs: [batch_size x len_q x d_model]

        return enc_outputs, attn
    
    
class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)
                

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in  edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """

                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
                    print(new_h[ntype])
            return new_h



class HGT(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        n_inp,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        top_n,
        use_norm=True,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = 1
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(
                    n_hid,
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )
        #######
        self.out = nn.Linear(n_hid, 128)
        self.cnt_layers = 4 if args.slice_layers>=4 else args.slice_layers
        
        self.layers = torch.nn.ModuleList([EncoderLayer(128, 64, 64, 128*4, n_heads) for _ in range(args.slice_layers)]).to(device)
        self.ffn1 = torch.nn.Linear(128 * self.cnt_layers, 512).to(device)
        self.dropout = torch.nn.Dropout(0.1).to(device)
        self.ffn2 = torch.nn.Linear(512, 128).to(device)
        self.special_embed = torch.nn.Embedding(2, 128).weight.data.uniform_(-1, 1)
        self.decoder = DistMult(len(G.etypes), n_hid)
        self.gumbel_linear = torch.nn.Linear(128, 1)
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.top_n = top_n

    def get_attn_pad_mask_gumbel(self, seq_q, seq_k, max_length):
        batch_size, len_q = len(seq_q), len(seq_q[0])
        batch_size, len_k = len(seq_k), len(seq_k[0])
        pad_attn_mask = []
        for itm in seq_k:
            tmp_mask = []
            tmp_mask += [False, False]
            for sub in itm:
                if sub == 1:
                    tmp_mask.append(False)
                else:
                    tmp_mask.append(True)
                if len(tmp_mask) == max_length:
                    break
            while len(tmp_mask) < max_length:
                tmp_mask.append(True)
            pad_attn_mask.append(tmp_mask)
        pad_attn_mask = (torch.ByteTensor(pad_attn_mask)).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.to(device)
        return pad_attn_mask.expand(batch_size, max_length, max_length)
    
    def forward(self, G, out_key_1 , total, r_id, temp):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data["inp"]))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        logits = self.out(h[out_key_1])
        
        sl_embeddings = torch.zeros(len(total),self.top_n+2,128)
        temp_sl_embeddings = torch.zeros(len(total),self.top_n,128).to(device)
        source_target_embedding = torch.zeros(len(total),2,128).to(device)

        for i, v in enumerate(total):
            source, target = v[0], v[-1]
            source_cos_similarity = self.cosine(logits[source], logits)
            target_cos_similarity = self.cosine(logits[target], logits)
            total_cos_similarity = source_cos_similarity + target_cos_similarity
            top100 = torch.topk(total_cos_similarity, self.top_n+2)[1]

            if source in top100:
                indices = torch.where(top100 != source)
                top100 = top100[indices]
            if target in top100:
                indices = torch.where(top100 != target)
                top100 = top100[indices]

            if top100.shape[0] > self.top_n:
                top100 = top100[:self.top_n]

            top100_tensor = logits[top100]
            temp_sl_embeddings[i] = top100_tensor
            source_target_embedding[i] = torch.cat([logits[source].unsqueeze(0), logits[target].unsqueeze(0)])

        gumbel_input = self.gumbel_linear(temp_sl_embeddings)
        gumbel_input = torch.sigmoid(gumbel_input)
        gumbel_input = torch.cat([gumbel_input, 1 - gumbel_input], dim = -1)
        gumbel_output = F.gumbel_softmax(gumbel_input, tau = temp, hard = True)
        gumbel_picked = gumbel_output[:,:,0]
        temp_sl_embeddings = torch.mul(temp_sl_embeddings, gumbel_picked.unsqueeze(-1))

        sl_embeddings = torch.cat([source_target_embedding, temp_sl_embeddings], dim = 1)
        
        output = sl_embeddings.to(device)
        enc_self_attn_mask = self.get_attn_pad_mask_gumbel(gumbel_picked,gumbel_picked,sl_embeddings.shape[1])
            
        for layer in self.layers:   
            output, enc_self_attn = layer(output, enc_self_attn_mask.to(device))
            
            try:
                layer_output = torch.cat((layer_output, output.unsqueeze(1)), 1)
            except NameError:  # FIXME - replaced bare except
                layer_output = output.unsqueeze(1)
        
        ee= max(args.slice_layers, self.cnt_layers)
        ss = ee - self.cnt_layers
        for ii in range(ss, ee):
            source_embed = layer_output[:, ii, 0, :].unsqueeze(1)
            destination_embed = layer_output[:, ii, 1, :].unsqueeze(1)
            try:
                source_embedding = torch.cat(
                    (source_embedding, source_embed), 2
                )
                destination_embedding = torch.cat(
                    (destination_embedding, destination_embed), 2
                )
            except:
                source_embedding = source_embed
                destination_embedding = destination_embed

        src_embedding = torch.relu(self.dropout(self.ffn1(source_embedding)))
        src_embedding = self.ffn2(src_embedding)
        dst_embedding = torch.relu(self.dropout(self.ffn1(destination_embedding)))
        dst_embedding = self.ffn2(dst_embedding)

        dst_embedding = dst_embedding.transpose(1, 2)
        pred_score = self.decoder(src_embedding,dst_embedding, r_id)
        pred_score = torch.sigmoid(pred_score)
        return pred_score
    
class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
#         left_emb = torch.unsqueeze(left_emb, 1)
#         right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

def sp_generator(trainset):
    sp = []
    for i,k in trainset.iterrows():
        u,v = k[1], k[2]
        sp.append([u,v])
    return sp

def val_sp_generator(trainset):
    sp = []
    for i,k in trainset.iterrows():
        u,v = k[1], k[2]
        sp.append([u,v])
    return sp

def neg_sp_generator(negative_u,negative_v):
    neg_short_path = []
    for u,v in zip(negative_u,negative_v):
        neg_short_path.append([u,v])
    return neg_short_path



def get_attn_pad_mask(seq_q, seq_k, max_length = args.max_length):
    batch_size, len_q = len(seq_q), len(seq_q[0])
    batch_size, len_k = len(seq_k), len(seq_k[0])
    # print(batch_size, len_q, len_k)
    pad_attn_mask = []
    for itm in seq_k:
        tmp_mask = []
        for sub in itm:
            tmp_mask.append(False)
            if len(tmp_mask) == max_length:
                break
        while len(tmp_mask) < max_length:
            tmp_mask.append(True)
        pad_attn_mask.append(tmp_mask)
        # print(tmp_mask)
    # print('mask', len(pad_attn_mask), len(pad_attn_mask[0]))
    pad_attn_mask = (torch.ByteTensor(pad_attn_mask)).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.to(device)
    return pad_attn_mask.expand(batch_size, max_length, max_length)



def compute_loss(pos_score, neg_score):
    pos_score = pos_score.to(device)
    neg_score = neg_score.to(device)

    scores = torch.cat([pos_score, neg_score]).to(device)
    # print(scores)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    # print(labels)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    return roc_auc_score(labels, scores)

def compute_f1(labels, scores, threshold):

    y_pred = np.zeros(len(scores), dtype=np.int32)

    for i, _ in enumerate(scores):
        if scores[i] >= threshold:
            y_pred[i] = 1

    return f1_score(labels, y_pred)

def compute_threshold(y_true, y_scores):
    y_scores = np.array([s.cpu().numpy() for s in y_scores])
    y_true = np.array(y_true)
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
 
    # Find the best threshold (e.g., the threshold that maximizes the Youden's J statistic)
    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]
   
    return best_threshold


######################### emb loading
def embloader(emb = [], max_len=10000 , n_hid = 128):
    emb_tensor = torch.zeros(max_len, n_hid)
    for i in emb:
        idx = int(i.split(" ")[0])
        id_len = len(i.split(" ")[0])
        b = i[id_len+1:]
        emb_tensor[idx] = torch.tensor(list(map(float,b.split(' '))))
    return emb_tensor


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 40, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)#to(device)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))#to(device)
        emb = nn.Embedding(max_len,n_hid)#to(device)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)#to(device)
    def forward(self, t):
        return self.lin(self.emb(t))
    
def eval_mrr(confidence,u,v,labels):
    confidence = np.array(confidence)
    labels = np.array(labels)
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    mrr_list, cur_mrr = [], 0
    for i, h_id in enumerate(u):
        t_dict[h_id].append(v[i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(confidence[i])  
    for h_id in t_dict.keys():
        conf_array = np.array(conf_dict[h_id])
        rank = np.argsort(-conf_array)
        sorted_label_array = np.array(labels_dict[h_id])[rank]
        pos_index = np.where(sorted_label_array == 1)[0]
        if len(pos_index) == 0:
            continue
        pos_min_rank = np.min(pos_index)
        cur_mrr = 1 / (1 + pos_min_rank)
        mrr_list.append(cur_mrr)
    mrr = np.mean(mrr_list)
    
    return mrr
    


trainset = pd.read_csv(args.dataset_name+"_train_new.csv", header =None)
testset = pd.read_csv(args.dataset_name+"_test_new.csv", header =None)
validset = pd.read_csv(args.dataset_name+"_valid_new.csv", header =None)
validset = pd.concat([ validset[validset[3]==1], validset[validset[3]==0]])
testset = pd.concat([ testset[testset[3]==1], testset[testset[3]==0]])
nxG = nx.Graph()

testset_mrr = pd.read_csv(args.dataset_name+"_mrr")


for i,k in trainset.iterrows():
    nxG.add_edge(int(k[1]),int(k[2]), rate= k[0])

for i in range(len(trainset[0].unique())):
    i = i + 1
    globals()[f'trainset_{i}'] = trainset[trainset[0] == i]

total = pd.concat([testset[testset[3]==1], validset[validset[3]==1] , trainset])



test_mrr_path = []
for i,k in testset_mrr.iterrows():
    try:
        test_mrr_path.append((nx.shortest_path(nxG,k[1],k[2])))
    except:
        test_mrr_path.append([k[1],k[2]])
        

item_dict = {}
for i in range(len(trainset[0].unique())):
    i = i+1
    item_dict[("node",str(i),"node")] = (globals()[f'trainset_{i}'][1].values, globals()[f'trainset_{i}'][2].values)
    item_dict[("node",str(i)+"_rev","node")] = (globals()[f'trainset_{i}'][2].values, globals()[f'trainset_{i}'][1].values)


G=dgl.heterograph(item_dict)

nnode = []

with open(args.dataset_name+"_n2v.emd" , "r") as f:
    lines = f.readlines()
    for line in lines:
        nnode.append(line[:-1])
del nnode[0]

cnt = 0 

for i in nnode: 
    ori_idx = str(int(i.split(" ")[0]))
    temp = [ori_idx] + i.split(" ")[1:]
    temp = " ".join(temp)
    nnode[cnt] = temp
    cnt += 1

def embloader(emb = [], max_len=10000 , n_hid = 128):
    emb_tensor = torch.zeros(max_len, n_hid)
    for i in emb:
        idx = int(i.split(" ")[0])
        id_len = len(i.split(" ")[0])
        b = i[id_len+1:]
        emb_tensor[idx] = torch.tensor(list(map(float,b.split(' '))))
    return emb_tensor


torch.cuda.empty_cache()
# G=G.to(device)
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

criterion = torch.nn.BCELoss()


def train(model, G):
    print("Dataset name:" , args.dataset_name)
    print("started at" , time.strftime('%x %X'))

    temperature = 1.0
    temp = temperature
    ANNEAL_RATE = 0.013
    temp_min = 0.1

    # print(test_size)
    test_neg_u, test_neg_v= testset[testset[3] == 0][1].values, testset[testset[3] == 0][2].values
    valid_neg_u, valid_neg_v= validset[validset[3] == 0][1].values, validset[validset[3] == 0][2].values
    
    #negative generation
    negative_u = []
    negative_v = []
    negative_pair = []
    ccnt = 0
    try :
        with open(str(args.negative_sampling)+args.dataset_name+"_u.pickle","rb") as fr:
            negative_u = pickle.load(fr)
        with open(str(args.negative_sampling)+args.dataset_name+"_v.pickle","rb") as fr:
            negative_v = pickle.load(fr)
    except:
        negative_u = []
        negative_v = []
        negative_pair = []
        ccnt = 0
        while ccnt < (args.negative_sampling*len(u)):
            a = (random.choice(G.nodes()).item())
            b = (random.choice(G.nodes()).item())
        #         print(a,b)
            if (True in ((total[1]== a ) * (total[2]== b )).values):
                continue
            elif (True in ((total[1]== b ) * (total[2]== a )).values):
                continue
            elif a==b:
                continue
            elif (a,b) in negative_pair or (b,a) in negative_pair:
                continue
            else:
                negative_u.append(a)
                negative_v.append(b)
                negative_pair.append((a,b))
                ccnt += 1
        with open(str(args.negative_sampling)+args.dataset_name+"_u.pickle","wb") as f:
            pickle.dump(negative_u ,  f, pickle.HIGHEST_PROTOCOL)
        with open(str(args.negative_sampling)+args.dataset_name+"_v.pickle","wb") as f:
            pickle.dump(negative_v ,  f, pickle.HIGHEST_PROTOCOL)
    train_neg_u, train_neg_v = negative_u, negative_v
    #########################

    sp = sp_generator(trainset)
    neg_short_path = neg_sp_generator(negative_u, negative_v)
    train_total = (sp + neg_short_path).copy()
    train_total = np.array(train_total)
    train_indicies = list(range(len(train_total)))
    train_labels = np.concatenate((np.ones(len(sp)),np.zeros(len(neg_short_path))),dtype = float)
    train_rid = np.concatenate([trainset[0].values,trainset[0].values])
    random.shuffle(train_indicies)
    train_total = train_total[train_indicies]
    train_labels = train_labels[train_indicies]
    train_rid = train_rid[train_indicies]


    vp = val_sp_generator(validset[validset[3]==1])
    val_neg_short_path = neg_sp_generator(valid_neg_u,valid_neg_v)
    valid_total = (vp + val_neg_short_path).copy()
    valid_total = np.array(valid_total)
    valid_labels = np.concatenate((np.ones(len(vp)),np.zeros(len(val_neg_short_path))),dtype = float)
    valid_indicies = list(range(len(valid_total)))
    valid_rid = validset[0].values
    random.shuffle(valid_indicies)
    valid_total = valid_total[valid_indicies]
    valid_labels = valid_labels[valid_indicies]
    valid_rid = valid_rid[valid_indicies]


    tp = val_sp_generator(testset[testset[3]==1])
    test_neg_short_path = neg_sp_generator(test_neg_u,test_neg_v)
    test_total = (tp + test_neg_short_path).copy()
    test_total = np.array(test_total)
    test_labels = np.concatenate((np.ones(len(tp)),np.zeros(len(test_neg_short_path))),dtype = float)
    test_indicies = list(range(len(test_total)))
    test_rid = testset[0].values
    random.shuffle(test_indicies)
    test_total = test_total[test_indicies]
    test_labels = test_labels[test_indicies]
    test_rid = test_rid[test_indicies]
    
    test_mrr_path = []
    for i,k in testset_mrr.iterrows():
        try:
            if k[1]!=k[2]:
                test_mrr_path.append((nx.shortest_path(nxG,k[1],k[2])))
            else:
                test_mrr_path.append((nx.shortest_path([k[1],k[2]])))
        except:
            test_mrr_path.append([k[1],k[2]])

    train_loss =[]
    val_loss = []
    
    best_loss = 0
    cur_patienece = 0
    for epoch in np.arange(args.n_epoch) + 1:
        print(epoch)
        model.train()
        loss_arr = []
        temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)     

        for j in tqdm((range(len(train_total)//args.n_batch+1))):
            batch = args.n_batch 
#             if j>2: continue
            if j == len(train_total)//args.n_batch:
                start = batch*j
                end = batch*j + len(train_total)%args.n_batch
            else : 
                start = batch*j
                end = batch*(j+1)
            pred_scores = model(G, "node", train_total[start:end], train_rid[start:end],temp)
            
#             print(pred_scores)
            loss = criterion(pred_scores.float(), torch.tensor(train_labels[start:end]).cuda().float())
            loss_arr.append(loss.item())
            optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        print("avg_loss:",np.mean(loss_arr))
        train_loss.append(np.mean(loss_arr))


        if epoch % 5 >= 0:
            print("valid")
            group_pred = torch.tensor([])
            model.eval()
            for j in (range(len(valid_total)//args.n_batch+1)):
                batch = args.n_batch 
                if j == len(valid_total)//args.n_batch:
                    start = batch*j
                    end = batch*j + len(valid_total)%args.n_batch
                else : 
                    start = batch*j
                    end = batch*(j+1)
            
                pred_scores = model(G, "node", valid_total[start:end], valid_rid[start:end],temp).detach().to("cpu")
                group_pred = torch.cat([group_pred,pred_scores])
            loss = criterion(group_pred.float(), torch.tensor(valid_labels).float())
            print("val_loss",loss)
            roc_auc = roc_auc_score(valid_labels, group_pred)
            print("valid_roc_auc",roc_auc)
            threshold = compute_threshold(valid_labels,group_pred)
            print("valid_threshold",threshold)
            f1_score = compute_f1(valid_labels, group_pred, threshold)
            print("valid_f1",f1_score)
            val_loss.append(loss)


            if roc_auc > best_loss :
                best_loss = roc_auc
                best_threshold = []
                best_threshold.append(threshold)
                torch.save(model.state_dict(),args.dataset_name+ "_0_jimin.model")
                print("★★★★★★★★epoch:",epoch,"--save")
                cur_patienece = 0
                best_epoch = epoch
                for pat in range(1,args.num_patience + 1):
                    if os.path.exists(args.dataset_name+"_"+str(pat)+"_jimin.model"):
                        os.remove(args.dataset_name+"_"+str(pat)+"_jimin.model")
            elif cur_patienece < args.num_patience:
                cur_patienece += 1 
                best_threshold.append(threshold)
                torch.save(model.state_dict(),args.dataset_name+"_"+ str(cur_patienece)+"_jimin.model")
                print("🔥🔥🔥🔥🔥🔥epoch:",epoch,"--patience")
            if cur_patienece == args.num_patience: break

    for pat in range(args.num_patience + 1):
        model.eval()
        model.load_state_dict(torch.load(args.dataset_name+"_"+str(pat)+"_jimin.model"))
        print("test --- patience:",pat)
        group_pred = torch.tensor([])
        threshold = best_threshold[pat]
        for j in (range(len(test_total)//args.n_batch+1)):
            batch = args.n_batch 
            if j == len(test_total)//args.n_batch:
                start = batch*j
                end = batch*j + len(test_total)%args.n_batch
            else : 
                start = batch*j
                end = batch*(j+1)

            pred_scores = model(G, "node", test_total[start:end], test_rid[start:end],temp).detach().to("cpu")
            group_pred = torch.cat([group_pred,pred_scores])
        roc_auc = roc_auc_score(test_labels, group_pred)
        print("test_roc_auc",roc_auc)
        f1_score = compute_f1(test_labels, group_pred, threshold)
        print("test_f1",f1_score)
        
        group_mrr_pred = torch.tensor([])
        for j in (range(len(test_mrr_path)//args.n_batch+1)):
            batch = args.n_batch 
            if j == len(test_mrr_path)//args.n_batch:
                start = batch*j
                end = batch*j + len(test_mrr_path)%args.n_batch
            else : 
                start = batch*j
                end = batch*(j+1)

            pred_mrr_scores = model(G, "node", test_mrr_path[start:end], testset_mrr['0'].values[start:end],temp).detach().to("cpu")
            group_mrr_pred = torch.cat([group_mrr_pred,pred_mrr_scores])
            
        mrr = eval_mrr(group_mrr_pred,testset_mrr['1'].values, testset_mrr['2'].values, testset_mrr['3'].values)
        print("MRR:",mrr)
    print("Dataset name:" , args.dataset_name)
    print("finished at" , time.strftime('%x %X'))
    print("train_loss",train_loss)
    print("valid_loss",val_loss)


device = torch.device("cuda:3")

G=dgl.heterograph(item_dict)

node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data["id"] = (
        torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
    )

emblist = []
node_tensor = embloader(nnode, len(nnode))

G.nodes["node"].data["inp"] = node_tensor

#######################################
G = G.to(device)
model = HGT(
    G,
    node_dict,
    edge_dict,
    n_inp=args.n_inp,
    n_hid=args.n_hid,
    n_out=1,#labels.max().item() + 1,
    n_layers=args.HGT_layers,
    n_heads=args.n_heads,
    top_n=args.top_n,
    use_norm=True,
).to(device)
##################


optimizer = torch.optim.AdamW(model.parameters(), )

scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, epochs = args.n_epoch , steps_per_epoch = len(trainset) // args.n_batch + 1, max_lr = args.max_lr
    )
#########################

for i in range(len(trainset[0].unique())):
    i = i+1
    globals()[f'u_{i}'] ,  globals()[f'v_{i}'] = G.edges(etype = str(i))
u = torch.tensor([], dtype = int).to(device)
v = torch.tensor([], dtype = int).to(device)
for i in range(1, len(trainset[0].unique())+1):
    u = torch.cat([u,globals()[f'u_{i}']])
    v = torch.cat([v,globals()[f'v_{i}']])

##########################################


print("Training HGT with #param: %d" % (get_n_params(model)))
# print("Training SLICE with #param: %d" % (get_n_params(slaice)))

train(model, G)
sys.exit()
