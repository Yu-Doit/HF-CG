import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# LSTM for HF-CG(not support bi-lstm for now)
class HFLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(HFLSTM, self).__init__(*args, **kwargs)

        # CG variables
        self.v_weight_ih, self.v_weight_hh = [], [] # v
        self.r_weight_ih, self.r_weight_hh = [], [] # r
        self.u_weight_ih, self.u_weight_hh = [], [] # delta_theta
        self.bu_weight_ih, self.bu_weight_hh = [], [] # best_delta_theta
        if self.bias:
            self.v_bias_ih, self.v_bias_hh = [], []
            self.r_bias_ih, self.r_bias_hh = [], []
            self.u_bias_ih, self.u_bias_hh = [], []
            self.bu_bias_ih, self.bu_bias_hh = [], []
    
        # lstm parameters
        self.p_names = ['weight_ih_l{}', 'weight_hh_l{}']
        if self.bias:
            self.p_names += ['bias_ih_l{}', 'bias_hh_l{}']

    # init v_0, r_0
    def init_cg(self):
        for layer in range(self.num_layers):
            pn = [x.format(layer) for x in self.p_names]
            self.v_weight_ih.append(-getattr(self, pn[0]).grad)
            self.v_weight_hh.append(-getattr(self, pn[1]).grad)
            self.r_weight_ih.append(-getattr(self, pn[0]).grad)
            self.r_weight_hh.append(-getattr(self, pn[1]).grad)
            if self.bias:
                self.v_bias_ih.append(-getattr(self, pn[2]).grad)
                self.v_bias_hh.append(-getattr(self, pn[3]).grad)
                self.r_bias_ih.append(-getattr(self, pn[2]).grad)
                self.r_bias_hh.append(-getattr(self, pn[3]).grad)

    # r^T * r
    def ComputeResidualDotProduct(self):
        rp = 0.
        for r in self.r_weight_ih:
            rp += float((r * r).sum())
        for r in self.r_weight_hh:
            rp += float((r * r).sum())
        if self.bias:
            for r in self.r_bias_ih:
                rp += float((r * r).sum())
            for r in self.r_bias_hh:
                rp += float((r * r).sum())
        return rp

    # v^T * v
    def ComputeConjugateDirectionNorm(self):
        vp = 0.
        for v in self.v_weight_ih:
            vp += float((v * v).sum())
        for v in self.v_weight_hh:
            vp += float((v * v).sum())
        if self.bias:
            for v in self.v_bias_ih:
                vp += float((v * v).sum())
            for v in self.v_bias_hh:
                vp += float((v * v).sum())
        return vp

    # modified forward propagation(a.k.a R operator)
    def Rop(self, input, R_input, hx = None, R_hx = None, ratio = None):
        L, N, _ = input.size()
        H = self.hidden_size
        orig_input = input
        orig_R_input = R_input
        # redundant variables
        h_n = []
        R_h_n = []
        c_n = []
        R_c_n = []

        for layer in range(self.num_layers):
            output = []
            R_output = []
            if hx is None:
                h_t, c_t = input.new_zeros(N, H), input.new_zeros(N, H)
                R_h_t, R_c_t = input.new_zeros(N, H), input.new_zeros(N, H)
            else:
                h_t, c_t = hx[0][layer], hx[1][layer]
                R_h_t, R_c_t = R_hx[0][layer], R_hx[1][layer]
            w_ih = getattr(self, 'weight_ih_l{}'.format(layer)).detach()
            R_w_ih = self.v_weight_ih[layer] if ratio is None else self.v_weight_ih[layer] * ratio
            w_hh = getattr(self, 'weight_hh_l{}'.format(layer)).detach()
            R_w_hh = self.v_weight_hh[layer] if ratio is None else self.v_weight_hh[layer] * ratio
            if self.bias:
                b_ih = getattr(self, 'bias_ih_l{}'.format(layer)).detach()
                R_b_ih = self.v_bias_ih[layer] if ratio is None else self.v_bias_ih[layer] * ratio
                b_hh = getattr(self, 'bias_hh_l{}'.format(layer)).detach()
                R_b_hh = self.v_bias_hh[layer] if ratio is None else self.v_bias_hh[layer] * ratio
            for t in range(L):
                x = orig_input[t]
                R_x = orig_R_input[t]
                if self.bias:
                    gates = F.linear(x, w_ih, b_ih) + F.linear(h_t, w_hh, b_hh)
                    R_gates = F.linear(x, R_w_ih) + F.linear(R_x, w_ih) + R_b_ih + F.linear(h_t, R_w_hh) + F.linear(R_h_t, w_hh) + R_b_hh
                else:
                    gates = F.linear(x, w_ih) + F.linear(h_t, w_hh)
                    R_gates = F.linear(x, R_w_ih) + F.linear(R_x, w_ih) + F.linear(h_t, R_w_hh) + F.linear(R_h_t, w_hh)
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :H]),
                    torch.sigmoid(gates[:, H: H * 2]),
                    torch.tanh(gates[:, H * 2: H * 3]),
                    torch.sigmoid(gates[:, H * 3:])
                )
                R_i_t = i_t * (1 - i_t) * R_gates[:, :H]
                R_f_t = f_t * (1 - f_t) * R_gates[:, H: H * 2]
                R_g_t = (1 - g_t * g_t) * R_gates[:, H * 2: H * 3]
                R_o_t = o_t * (1 - o_t) * R_gates[:, H * 3: H * 4]
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                R_c_t = R_f_t * c_t + f_t * R_c_t + R_i_t * g_t + i_t * R_g_t
                R_h_t = R_o_t * torch.tanh(c_t) + o_t * (1 - c_t * c_t) * R_c_t
                output.append(h_t.unsqueeze(0))
                R_output.append(R_h_t.unsqueeze(0))
            orig_input = torch.cat(output, dim = 0)
            orig_R_input = torch.cat(R_output, dim = 0)

        R_output = orig_R_input
        return R_output

    # v^T * B * v
    def ComputeNormDotProduct(self):
        np = 0.
        for layer in range(self.num_layers):
            pn = [x.format(layer) for x in self.p_names]
            np += float((self.v_weight_ih[layer] * getattr(self, pn[0]).grad).sum())
            np += float((self.v_weight_hh[layer] * getattr(self, pn[1]).grad).sum())
            if self.bias:
                np += float((self.v_bias_ih[layer] * getattr(self, pn[2]).grad).sum())
                np += float((self.v_bias_hh[layer] * getattr(self, pn[3]).grad).sum())
        return np
    
    # update r, v, theta, delta_theta, best_delta_theta
    def Update(self, alpha, beta, mode):
        if mode == 0: # update delta_theta, theta
            if len(self.u_weight_ih) == 0: # init delta_theta_0
                for v in self.v_weight_ih:
                    self.u_weight_ih.append(alpha * v)
                for v in self.v_weight_hh:
                    self.u_weight_hh.append(alpha * v)
                if self.bias:
                    for v in self.v_bias_ih:
                        self.u_bias_ih.append(alpha * v)
                    for v in self.v_bias_hh:
                        self.u_bias_hh.append(alpha * v)
            else: # delta_theta += alpha * v
                for layer in range(self.num_layers):
                    self.u_weight_ih[layer] += alpha * self.v_weight_ih[layer]
                    self.u_weight_hh[layer] += alpha * self.v_weight_hh[layer]
                    if self.bias:
                        self.u_bias_ih[layer] += alpha * self.v_bias_ih[layer]
                        self.u_bias_hh[layer] += alpha * self.v_bias_hh[layer]
            # update theta
            for layer in range(self.num_layers):
                pn = [x.format(layer) for x in self.p_names]
                p1 = getattr(self, pn[0])
                p2 = getattr(self, pn[1])
                p1.data.add_(self.u_weight_ih[layer])
                p2.data.add_(self.u_weight_hh[layer])
                if self.bias:
                    p3 = getattr(self, pn[2])
                    p4 = getattr(self, pn[3])
                    p3.data.add_(self.u_bias_ih[layer])
                    p4.data.add_(self.u_bias_hh[layer])
        elif mode == 1: # reset theta
            for layer in range(self.num_layers):
                pn = [x.format(layer) for x in self.p_names]
                p1 = getattr(self, pn[0])
                p2 = getattr(self, pn[1])
                p1.data.add_(self.u_weight_ih[layer], alpha = -1)
                p2.data.add_(self.u_weight_hh[layer], alpha = -1)
                if self.bias:
                    p3 = getattr(self, pn[2])
                    p4 = getattr(self, pn[3])
                    p3.data.add_(self.u_bias_ih[layer], alpha = -1)
                    p4.data.add_(self.u_bias_hh[layer], alpha = -1)
        elif mode == 2: # update best_delta_theta
            if len(self.bu_weight_ih) == 0:
                for layer in range(self.num_layers):
                    self.bu_weight_ih.append(self.u_weight_ih[layer].clone())
                    self.bu_weight_hh.append(self.u_weight_hh[layer].clone())
                    if self.bias:
                        self.bu_bias_ih.append(self.u_bias_ih[layer].clone())
                        self.bu_bias_hh.append(self.u_bias_hh[layer].clone())
            else:
                for layer in range(self.num_layers):
                    self.bu_weight_ih[layer] = self.u_weight_ih[layer].clone()
                    self.bu_weight_hh[layer] = self.u_weight_hh[layer].clone()
                    if self.bias:
                        self.bu_bias_ih[layer] = self.u_bias_ih[layer].clone()
                        self.bu_bias_hh[layer] = self.u_bias_hh[layer].clone()
        elif mode == 3: # update r
            for layer in range(self.num_layers):
                pn = [x.format(layer) for x in self.p_names]
                self.r_weight_ih[layer] -= alpha * getattr(self, pn[0]).grad
                self.r_weight_hh[layer] -= alpha * getattr(self, pn[1]).grad
                if self.bias:
                    self.r_bias_ih[layer] -= alpha * getattr(self, pn[2]).grad
                    self.r_bias_hh[layer] -= alpha * getattr(self, pn[3]).grad
        elif mode == 4: # update v
            for layer in range(self.num_layers):
                self.v_weight_ih[layer] = self.r_weight_ih[layer] + beta * self.v_weight_ih[layer]
                self.v_weight_hh[layer] = self.r_weight_hh[layer] + beta * self.v_weight_hh[layer]
                if self.bias:
                    self.v_bias_ih[layer] = self.r_bias_ih[layer] + beta * self.v_bias_ih[layer]
                    self.v_bias_hh[layer] = self.r_bias_hh[layer] + beta * self.v_bias_hh[layer]
        elif mode == 5: # update theta with best_delta_theta
            for layer in range(self.num_layers):
                pn = [x.format(layer) for x in self.p_names]
                p1 = getattr(self, pn[0])
                p2 = getattr(self, pn[1])
                p1.data.add_(self.bu_weight_ih[layer])
                p2.data.add_(self.bu_weight_hh[layer])
                if self.bias:
                    p3 = getattr(self, pn[2])
                    p4 = getattr(self, pn[3])
                    p3.data.add_(self.bu_bias_ih[layer])
                    p4.data.add_(self.bu_bias_hh[layer])
        elif mode == 6: # reset r, v, delta_theta, best_delta_theta
            self.v_weight_ih, self.v_weight_hh = [], []
            self.r_weight_ih, self.r_weight_hh = [], []
            self.u_weight_ih, self.u_weight_hh = [], []
            self.bu_weight_ih, self.bu_weight_hh = [], []
            if self.bias:
                self.v_bias_ih, self.v_bias_hh = [], []
                self.r_bias_ih, self.r_bias_hh = [], []
                self.u_bias_ih, self.u_bias_hh = [], []
                self.bu_bias_ih, self.bu_bias_hh = [], []


# Linear layer for HF-CG
class HFLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device = None, dtype = None):
        super(HFLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.b = bias
        
        # CG variables
        self.v_weight = None
        self.r_weight = None
        self.u_weight = None
        self.bu_weight = None
        if bias:
            self.v_bias = None
            self.r_bias = None
            self.u_bias = None
            self.bu_bias = None

    # init v_0, r_0
    def init_cg(self):
        self.v_weight = -self.weight.grad
        self.r_weight = -self.weight.grad
        if self.b:
            self.v_bias = -self.bias.grad
            self.r_bias = -self.bias.grad

    # r^T * r
    def ComputeResidualDotProduct(self):
        rp = (self.r_weight * self.r_weight).sum()
        if self.b:
            rp += (self.r_bias * self.r_bias).sum()
        return float(rp)

    # v^T * v
    def ComputeConjugateDirectionNorm(self):
        vp = (self.v_weight * self.v_weight).sum()
        if self.b:
            vp += (self.v_bias * self.v_bias).sum()
        return float(vp)

    # modified forward propagation(a.k.a R operator)
    def Rop(self, input, R_input, ratio = None):
        if ratio is None:
            R_output = F.linear(R_input, self.weight.detach()) + F.linear(input, self.v_weight)
            if self.b:
                R_output += self.v_bias
        else:
            R_output = F.linear(R_input, self.weight.detach()) + F.linear(input, self.v_weight * ratio)
            if self.b:
                R_output += self.v_bias * ratio
        return R_output

    # v^T * B * v
    def ComputeNormDotProduct(self):
        np = (self.v_weight * self.weight.grad).sum()
        if self.b:
            np += (self.v_bias * self.bias.grad).sum()
        return float(np)

    # update r, v, theta, delta_theta, best_delta_theta
    def Update(self, alpha, beta, mode):
        if mode == 0: # update delta_theta, theta
            if self.u_weight is None: # init delta_theta_0
                self.u_weight = alpha * self.v_weight
                if self.b:
                    self.u_bias = alpha * self.v_bias
            else: # delta_theta += alpha * v
                self.u_weight += alpha * self.v_weight
                if self.b:
                    self.u_bias += alpha * self.v_bias
            # update theta
            self.weight.data.add_(self.u_weight)
            if self.b:
                self.bias.data.add_(self.u_bias)
        elif mode == 1: # reset theta
            self.weight.data.add_(self.u_weight, alpha = -1)
            if self.b:
                self.bias.data.add_(self.u_bias, alpha = -1)
        elif mode == 2: # update best_delta_theta
            self.bu_weight = self.u_weight.clone()
            if self.b:
                self.bu_bias = self.u_bias.clone()
        elif mode == 3: # update r
            self.r_weight -= alpha * self.weight.grad
            if self.b:
                self.r_bias -= alpha * self.bias.grad
        elif mode == 4: # update v
            self.v_weight = self.r_weight + beta * self.v_weight
            if self.b:
                self.v_bias = self.r_bias + beta * self.v_bias
        elif mode == 5: # update theta with best_delta_theta
            self.weight.data.add_(self.bu_weight)
            if self.b:
                self.bias.data.add_(self.bu_bias)
        elif mode == 6: # reset r, v, delta_theta, best_delta_theta
            self.v_weight = None
            self.r_weight = None
            self.u_weight = None
            self.bu_weight = None
            if self.b:
                self.v_bias = None
                self.r_bias = None
                self.u_bias = None
                self.bu_bias = None


# Embedding layer for HF-CG
class HFEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx = None,
                 max_norm = None, norm_type = 2., scale_grad_by_freq = False,
                 sparse = False, _weight = None, device = None, dtype = None):
        super(HFEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx,
                                          max_norm, norm_type, scale_grad_by_freq,
                                          sparse, _weight, device, dtype)

        # CG variables
        self.v_weight = None
        self.r_weight = None
        self.u_weight = None
        self.bu_weight = None

    # init v_0, r_0
    def init_cg(self):
        self.v_weight = -self.weight.grad
        self.r_weight = -self.weight.grad

    # r^T * r
    def ComputeResidualDotProduct(self):
        return float((self.r_weight * self.r_weight).sum())

    # v^T * v
    def ComputeConjugateDirectionNorm(self):
        return float((self.v_weight * self.v_weight).sum())

    # modified forward propagation(a.k.a R operator)
    def Rop(self, input, ratio):
        L, N = input.size()
        index = input.new_zeros(L, N, self.num_embeddings, dtype = torch.float).scatter_(2, input.view(L, N, 1), 1)
        if ratio is None:
            R_output = F.linear(index, self.v_weight.t())
        else:
            R_output = F.linear(index, self.v_weight.t() * ratio)
        return R_output

    # v^T * B * v
    def ComputeNormDotProduct(self):
        return float((self.v_weight * self.weight.grad).sum())

    # update r, v, theta, delta_theta, best_delta_theta\
    def Update(self, alpha, beta, mode):
        if mode == 0: # update delta_theta, theta
            if self.u_weight is None: # init delta_theta_0
                self.u_weight = alpha * self.v_weight
            else: # delta_theta += alpha * v
                self.u_weight += alpha * self.v_weight
            # update theta
            self.weight.data.add_(self.u_weight)
        elif mode == 1: # reset theta
            self.weight.data.add_(self.u_weight, alpha = -1)
        elif mode == 2: # update best_delta_theta
            self.bu_weight = self.u_weight.clone()
        elif mode == 3: # update r
            self.r_weight -= alpha * self.weight.grad
        elif mode == 4: # update v
            self.v_weight = self.r_weight + beta * self.v_weight
        elif mode == 5: # update theta with best_delta_theta
            self.weight.data.add_(self.bu_weight)
        elif mode == 6: # reset r, v, delta_theta, best_delta_theta
            self.v_weight = None
            self.r_weight = None
            self.u_weight = None
            self.bu_weight = None


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