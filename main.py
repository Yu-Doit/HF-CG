# coding: utf-8
import argparse
import time
import math
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import HFLSTMLM, repackage_hidden
from data import Corpus, DataIterator


# not support HF-CG for Transformer-based LM for now
# model configuration
parser = argparse.ArgumentParser(description = 'Transformer/LSTM Language Model')
parser.add_argument('--data', type = str, default = './data/ami', help = 'location of the data corpus')
parser.add_argument('--model', type = str, default = 'LSTM', help = 'type of recurrent net (Transformer, LSTM)')
parser.add_argument('--emsize', type = int, default = 512, help = 'size of word embeddings')
parser.add_argument('--nhid', type = int, default = 512, help = 'number of hidden units per layer')
parser.add_argument('--nlayers', type = int, default = 2, help = 'number of layers')
parser.add_argument('--nhead', type = int, default = 8, help = 'the number of heads in the encoder/decoder of the transformer model')
# configuration for 1st-order optimization method
parser.add_argument('--lr', type = float, default = 20, help = 'initial learning rate')
parser.add_argument('--patience', type = int, default = 15, help = 'max lr decay times')
parser.add_argument('--optim', type = str, default = None, help = 'optimizer')
parser.add_argument('--momentum', type = float, default = 0, help = 'momentum')
parser.add_argument('--weight-decay', type = float, default = 0, help = 'L2 norm')
parser.add_argument('--clip', type = float, default = 0.25, help = 'gradient clipping')
# basic configuration
parser.add_argument('--epochs', type = int, default = 80, help = 'upper epoch limit')
parser.add_argument('--batch_size', type = int, default = 32, metavar = 'N', help = 'training batch size of 1st-order method')
parser.add_argument('--eval_bsz', type = int, default = 32, help = 'eval batch size')
parser.add_argument('--bptt', type = int, default = 70, help = 'sequence length')
parser.add_argument('--varlen', action = 'store_true', help = 'use variable bptt')
parser.add_argument('--dropout', type = float, default = 0.2, help = 'dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action = 'store_true', help = 'tie the word embedding and softmax weights')
parser.add_argument('--seed', type = int, default = 1111, help = 'random seed')
parser.add_argument('--cuda', action = 'store_true', help = 'use CUDA')
parser.add_argument('--log-interval', type = int, default = 80, metavar = 'N', help = 'report interval')
parser.add_argument('--save', type = str, default = 'model.pt', help = 'path to save the final model')
parser.add_argument('--No', type = int, default = 0, help = 'experiment No.')
parser.add_argument('--pre', type = str, default = None, help = 'pretrained chekpoint')
parser.add_argument('--dry-run', action = 'store_true', help = 'verify the code and the model')
# CG configuration
parser.add_argument('--hfcg', action = 'store_true', help = 'use HF-CG')
parser.add_argument('--ga_bsz', type = int, default = 500, help = 'Gradient Accumulation Stage batch size')
parser.add_argument('--cg_bsz', type = int, default = 32, help = 'Conjuagte Gradient Stage batch size')
parser.add_argument('--M', type = int, default = 10, help = 'num of CG iterations')
args = parser.parse_args()

# =======================================================================================================================================
# preparations
torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')

corpus = Corpus(args.data, vocab = args.data + '/vocab.txt')
if args.hfcg:
    ga_iter = DataIterator(corpus.train, args.ga_bsz, args.bptt) # data iterator for GA stage
    cg_iter = DataIterator(corpus.train, args.cg_bsz, args.bptt) # data iterator for CG stage
else:
    train_iter = DataIterator(corpus.train, args.batch_size, args.bptt) # data iterator for 1st-order training
val_iter = DataIterator(corpus.valid, args.eval_bsz, args.bptt)
test_iter = DataIterator(corpus.test, args.eval_bsz, args.bptt)

ntokens = len(corpus.vocab)
if args.model == 'Transformer':
    pass # not implement
else:
    model = HFLSTMLM(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()
lr = args.lr
if args.optim is not None: # create optimizer for 1st-order training
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = lr)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = args.momentum, weight_decay = args.weight_decay)

log_path = './log_ami/logging_' + str(args.No) + '.txt'
model_path = './model_ami/' + str(args.No) + '_' + args.save
final_model_path = './final_model_ami/' + str(args.No) + '_' + args.save

# =======================================================================================================================================
# util functions
def logging(s, path):
    print(s)
    with open(path, 'a+', encoding = 'utf8') as f:
        f.write(s + '\n')

def trainHFCG():
    model.train()
    start_time = time.time()

    if args.model == 'LSTM':
        hidden = model.init_hidden(args.ga_bsz)
    
    for batch, (data, target, seq_len) in enumerate(ga_iter):

        model.zero_grad()

        # GA stage
        data = data.to(device)
        target = target.to(device)
        if args.model == 'Transformer':
            pass
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, target)
        loss.backward()
        # CG stage
        cg_sample = random.randint(0, len(cg_iter) - args.bptt)
        cg_data, cg_target, _ = cg_iter.get_batch(cg_sample)
        cg_data = cg_data.to(device)
        cg_target = cg_target.to(device)
        model.init_cg()
        best_val_loss = model.cg(cg_data, cg_target, val_iter, args.eval_bsz, args.M, device)

        elapsed = time.time() - start_time
        log_info = '| epoch {:3d} | batch {:2d} | s/batch {:5.2f} | GA loss {:5.2f} | CG val loss {:5.2f} | CG val ppl {:8.2f} |'.format(
            epoch, batch, elapsed, loss.item(), best_val_loss, math.exp(best_val_loss))
        logging(log_info, log_path)
        start_time = time.time()

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.vocab)

    if args.model == 'LSTM':
        hidden = model.init_hidden(args.batch_size)

    tr_iter = train_iter.get_varlen_iter() if args.varlen else train_iter
    for batch, (data, target, seq_len) in enumerate(tr_iter):
        
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()
        if args.model == 'Transformer':
            pass # not implement
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.optim is None:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)
        else:
            optimizer.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            log_info = '| epoch {:3d} | batch {} | seq_len {} | lr {} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, seq_len, lr, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))
            logging(log_info, log_path)
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def evaluate(data_iter):
    model.eval()
    total_loss = 0.
    total_len = 0
    ntokens = len(corpus.vocab)
    
    if args.model == 'LSTM':
        hidden = model.init_hidden(args.eval_bsz)

    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(data_iter):

            data = data.to(device)
            target = target.to(device)

            if args.model == 'Transformer':
                pass # not implement
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += seq_len * criterion(output, target).item()
            total_len += seq_len

    return total_loss / total_len

best_val_loss = None
decay = 0

def train_config():
    log_info = '| Experiment {} starts | Model type: {} | Dataset: {} | Optim method: {} |'.format(
        args.No, args.model, args.data, 'HF-CG' if args.hfcg else '1st-order method')
    logging(log_info, log_path)
    if args.model == 'Transformer':
        pass # not implement
    else:
        log_info = '| embed size: {} | hidden size: {} | layer num: {} | dropout: {} | tied: {} |'.format(
            args.emsize, args.nhid, args.nlayers, args.dropout, 'Y' if args.tied else 'N')
    logging(log_info, log_path)
    if args.hfcg:
        log_info = '| GA bsz: {} | CG bsz: {} | bptt: {} | varlen: {} |'.format(
            args.ga_bsz, args.cg_bsz, args.bptt, 'Y' if args.varlen else 'N')
    else:
        if args.optim.lower == 'sgd':
            log_info = '| bsz: {} | bptt: {} | varlen: {} | optimizer: {} | init lr: {} | momentum: {} | L2: {} |'.format(
                args.batch_size, args.bptt, 'Y' if args.varlen else 'N', args.optim, args.lr, args.momentum, args.weight_decay)
        else:
            log_info = '| bsz: {} | bptt: {} | varlen: {} | optimizer: {} | init lr: {} |'.format(
                args.batch_size, args.bptt, 'Y' if args.varlen else 'N', args.optim, args.lr)
    logging(log_info, log_path)

# =======================================================================================================================================
# train
try:
    train_config()

    if args.pre is not None: # load checkpoint
        with open(args.pre, 'rb') as f:
            state = torch.load(f)
        model.load_state_dict(state['model'])
        if args.optim is not None:
            optimizer.load_state_dict(state['optimizer'])

        if args.model == 'LSTM':
            model.lstm.flatten_parameters()

        log_info = 'Continue training from ckeckpoint: {}'.format(args.pre)
        logging(log_info, log_path)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        if args.hfcg:
            trainHFCG()
        else:
            train()
            val_loss = evaluate(val_iter)
        
            log_info = '-' * 89
            logging(log_info, log_path)
            log_info = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))
            logging(log_info, log_path)
            log_info = '-' * 89
            logging(log_info, log_path)

            if not best_val_loss or val_loss < best_val_loss:
                if args.optim is not None:
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                else:
                    state = {'model': model.state_dict()}
                with open(model_path, 'wb') as f:
                    torch.save(state, f)
                best_val_loss = val_loss

                log_info = 'save model. valid loss {:5.2f} valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss))
                logging(log_info, log_path)
            else:
                decay += 1
                if args.optim is None:
                    lr /= 4.0
                else:
                    lr /= 2.0
                    for param in optimizer.param_groups:
                        param['lr'] = lr
                if decay > args.patience:
                    break
    
    if args.optim is not None:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    else:
        state = {'model': model.state_dict()}
    with open(final_model_path, 'wb') as f:
        torch.save(state, f)

    log_info = 'save final model.'
    logging(log_info, log_path)

except KeyboardInterrupt:
    log_info = '-' * 89
    logging(log_info, log_path)
    log_info = 'Exiting from training early'
    logging(log_info, log_path)

# =======================================================================================================================================
# test the best model
if not args.hfcg:
    with open(model_path, 'rb') as f:
        state = torch.load(f)
    model.load_state_dict(state['model'])
    if args.model == 'LSTM':
        model.lstm.flatten_parameters()

    test_loss = evaluate(test_iter)

    log_info = '=' * 89
    logging(log_info, log_path)
    log_info = '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss))
    logging(log_info, log_path)
    log_info = '=' * 89
    logging(log_info, log_path)

# =======================================================================================================================================
# test the final model
with open(final_model_path, 'rb') as f:
    state = torch.load(f)
model.load_state_dict(state['model'])
if args.model == 'LSTM':
    model.lstm.flatten_parameters()

test_loss = evaluate(test_iter)

log_info = '=' * 89
logging(log_info, log_path)
log_info = '| Final model performance | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss))
logging(log_info, log_path)
log_info = '=' * 89
logging(log_info, log_path)