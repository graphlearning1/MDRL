import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta



class MDRL_2v(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(MDRL_2v, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)

        self.cut_pos = nhid2//2

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def dot_product_normalize(self, shared_1, shared_2):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        num_of_samples = shared_1.size(0)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        # Dot product
        match_map = torch.mm(shared_1, shared_2.T)
        return match_map

    def distill_loss(self, y_s, y_t, T=1):
        p_s = F.log_softmax(y_s / T, dim=1)
        p_t = F.softmax(y_t / T, dim=1)

        loss = -(p_t * p_s).sum(dim=1).mean()
        return loss


    def forward(self, x, sadj, fadj, s_rec, sim_v):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph

        ##ML loss
        same_adj_loss1, diff1_adj_loss1 = self.rec_A_loss(emb1[:,:self.cut_pos], sadj, fadj)
        same_adj_loss2, diff1_adj_loss2 = self.rec_A_loss(emb2[:,:self.cut_pos], fadj, sadj)
        rec_loss = (same_adj_loss1 + diff1_adj_loss1 + same_adj_loss2 + diff1_adj_loss2)

        similarity_loss = 1 - self.dot_product_normalize(emb1[:,:self.cut_pos], emb2[:,:self.cut_pos]).mean()
        shared_loss = s_rec * rec_loss + sim_v*similarity_loss

        ##SL_loss
        recself_loss1, recdiff_loss1 = self.rec_A_loss(emb1, sadj, fadj)
        recself_loss2, recdiff_loss2 = self.rec_A_loss(emb2, fadj, sadj)
        spec_loss = ((recself_loss1 - recdiff_loss1 + 1) + (recself_loss2 - recdiff_loss2 + 1))
        emb = torch.stack([emb1, emb2], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)

        ##TD_loss
        diss_loss1 = self.distill_loss(emb1, emb)
        diss_loss2 = self.distill_loss(emb2, emb)

        disttill_loss = diss_loss1 + diss_loss2
        return output, att, shared_loss, spec_loss, disttill_loss, emb1, emb2

    def rec_A_loss(self, x, sadj, fadj1, loss_type='cls'):
        sadj = torch.where(sadj.to_dense()>0,1.0,0.0)
        norm_w_s = sadj.shape[0] ** 2 / float((sadj.shape[0] ** 2 - sadj.sum()) * 2)
        pos_weight_s = torch.FloatTensor([float(sadj.shape[0] ** 2 - sadj.sum()) / sadj.sum()]).cuda()

        fadj = torch.where(fadj1.to_dense() > 0, 1.0, 0.0)
        norm_w_f = fadj.shape[0] ** 2 / float((fadj.shape[0] ** 2 - fadj.sum()) * 2)
        pos_weight_f = torch.FloatTensor([float(fadj.shape[0] ** 2 - fadj.sum()) / fadj.sum()]).cuda()

        edge_pre = self.dot_product_normalize(x, x)
        if loss_type == 'cls':
            true_loss = norm_w_s * F.binary_cross_entropy_with_logits(edge_pre, sadj, pos_weight=pos_weight_s)
            true_false_loss = norm_w_f * F.binary_cross_entropy_with_logits(edge_pre, fadj, pos_weight=pos_weight_f)
        return true_loss, true_false_loss
