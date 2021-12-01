import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel import HFLSTM, HFLinear, HFEmbedding, HFCrossEntropyLoss


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

    def forward(self, input, hidden):
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

    def evaluate(self, val_iter, bsz, device):
        self.eval()
        total_loss = 0.
        total_len = 0
        val_hidden = self.init_hidden(bsz)
        with torch.no_grad():
            for i, (val_data, val_target, seq_len) in enumerate(val_iter):

                val_data = val_data.to(device)
                val_target = val_target.to(device)

                val_decoded, val_hidden = self.forward(val_data, val_hidden)
                val_hidden = repackage_hidden(val_hidden)
                
                total_loss += seq_len * F.cross_entropy(val_decoded, val_target).item()
                total_len += seq_len
        self.train()
        return total_loss / total_len

    # CG stage
    def cg(self, data, target, val_iter, bsz, M, device):
        best_loss = None
        
        # create the graph for modified EBP
        emb = self.dropout(self.encoder(data))
        output, hidden = self.lstm(emb)
        output = self.dropout(output)
        decoded = self.decoder(output).view(-1, self.ntoken)

        # CG iteration
        for m in range(M):
            self.zero_grad() # set all grads to 0 before each modified EBP
            residualProd = self.ComputeResidualDotProduct()
            ratio = self.ComputeRatioforStableCG() # ratio for stable CG
            R_emb = self.encoder.Rop(data, ratio = ratio)
            R_output = self.lstm.Rop(emb.detach(), R_emb, ratio = ratio)
            R_decoded = self.decoder.Rop(output.detach(), R_output, ratio = ratio).view(-1, self.ntoken) / ratio
            loss = HFCrossEntropyLoss.apply(decoded, R_decoded) # not a real loss, just a trigger for backward()
            if m < M - 1:
                loss.backward(retain_graph = True)
            else:
                loss.backward()
            normProd = self.ComputeNormDotProduct()
            alpha = residualProd / normProd
            self.Update(alpha, None, 0) # update delta_theta, theta
            self.Update(alpha, None, 3) # update r
            beta = self.ComputeResidualDotProduct() / residualProd
            self.Update(None, beta, 4) # update v
            cur_loss = self.evaluate(val_iter, bsz, device) # test current update with dev data
            if best_loss is None or cur_loss < best_loss: # update best_delta_theta
                best_loss = cur_loss
                self.Update(None, None, 2)
            self.Update(None, None, 1) # reset theta

        self.Update(None, None, 5) # update theta with best_delta_theta
        self.Update(None, None, 6) # reset r, v, delta_theta, best_delta_theta
        return best_loss


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)