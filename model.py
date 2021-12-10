import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from basemodel import HFLSTM, HFLinear, HFEmbedding


# LSTM-based language model for HF-CG(not support bi-lstm for now)
class HFLSTMLM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout = 0.5, tie_weights = False):
        super(HFLSTMLM, self).__init__()

        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)
        self.encoder = HFEmbedding(ntoken, ninp)
        self.lstm = HFLSTM(ninp, nhid, nlayers, dropout = dropout)
        self.decoder = HFLinear(nhid, ntoken)
        
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid))

    def forward(self, input, hidden = None):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output).view(-1, self.ntoken)
        return decoded, hidden

    # init v_0, r_0
    def init_cg(self):
        self.encoder.init_cg()
        self.lstm.init_cg()
        self.decoder.init_cg()

    # r^T * r
    def ComputeResidualDotProduct(self):
        rp1 = self.encoder.ComputeResidualDotProduct()
        rp2 = self.lstm.ComputeResidualDotProduct()
        rp3 = self.decoder.ComputeResidualDotProduct()
        return rp1 + rp2 + rp3

    # norm_theta / norm_v
    def ComputeRatioforStableCG(self):
        vp1 = self.encoder.ComputeConjugateDirectionNorm()
        vp2 = self.lstm.ComputeConjugateDirectionNorm()
        vp3 = self.decoder.ComputeConjugateDirectionNorm()
        vp = vp1 + vp2 + vp3
        nt = 0.
        for p in self.parameters():
            nt += float((p.data * p.data).sum())
        return nt / vp

    # v^T * B * v
    def ComputeNormDotProduct(self):
        np1 = self.encoder.ComputeNormDotProduct()
        np2 = self.lstm.ComputeNormDotProduct()
        np3 = self.decoder.ComputeNormDotProduct()
        return np1 + np2 + np3

    # update r, v, theta, delta_theta, best_delta_theta
    def Update(self, alpha, beta, mode):
        self.encoder.Update(alpha, beta, mode)
        self.lstm.Update(alpha, beta, mode)
        self.decoder.Update(alpha, beta, mode)

    # create the graph for modified EBP
    def CreateGraph(self, input, hidden = None):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output).view(-1, self.ntoken)
        return emb, output, decoded

    # modified forward propagation(a.k.a R operator)
    def Rop(self, data, emb, output):
        R_emb = self.encoder.Rop(data)
        R_output = self.lstm.Rop(emb.detach(), R_emb)
        R_decoded = self.decoder.Rop(output.detach(), R_output).view(-1, self.ntoken)
        return R_decoded


# CrossEntropyLoss for modified EBP
class HFCrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, input, Jv):
        pred = F.softmax(input, dim = 1)
        ctx.save_for_backward(pred, Jv)
        
        return pred.mean() # We just use this function to do modified EBP but not to calculate the real loss,
                           # so we return an arbitrary scalar to do loss.backward().
    
    @staticmethod
    def backward(ctx, grad_output):
        pred, Jv = ctx.saved_tensors
        # H^hat * Jv = [diag(p) - p * p^T] * Jv
        input_grad = pred * Jv - pred * (pred * Jv).sum(dim = 1, keepdim = True)
        return input_grad, None


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
