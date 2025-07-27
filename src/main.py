"""
PAEFF: Code is adapted from FOP
"""

from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from scipy import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import online_evaluation
from tqdm import tqdm
from torch.optim import lr_scheduler
from retrieval_model import *


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def clip_loss(img_feats, aud_feats):
    """
    img_feats: [B, L]
    aud_feats: [B, L]
    """
    logits = (img_feats @ aud_feats.T) * torch.exp(torch.tensor(np.log(1./0.07))).to(img_feats.device)
    labels = torch.arange(img_feats.shape[0], device=img_feats.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_t + loss_i) / 2.

    return loss



def read_data(train_file='./voxceleb1_feats/faceTrain.csv', train_file_voice='./voxceleb1_feats/voiceTrain.csv'):
        
    print('Reading Train Faces')
    print('----------------------------')
    img_train = pd.read_csv(train_file, header=None)
    #img_train = img_train.iloc[1: , :] # drop first row
    
    train_label = img_train[4096]       # modify according to face encoder features dimension
    img_train = np.asarray(img_train)
    img_train = img_train[:, 0:4096]    # modify according to face encoder features dimension
    train_label = np.asarray(train_label)
    
    print('Reading Voices')
    print('----------------------------')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)
    voice_train = voice_train[:, 0:512] # modify according to audio encoder features dimension
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    print("Train file length", len(img_train))
    print(train_label)
        
    print('\nShuffling\n')
    print('----------------------------')
    combined = list(zip(img_train, voice_train, train_label))
    img_train = []
    voice_train = []
    train_label = []
    random.shuffle(combined)
    img_train[:], voice_train, train_label[:] = zip(*combined)
    combined = [] 
    img_train = np.asarray(img_train).astype(np.float32)
    voice_train = np.asarray(voice_train).astype(np.float32)
    train_label = np.asarray(train_label)
    
    
    return img_train, voice_train, train_label

 
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(face_train, voice_train, train_label):

    model = PAEFF(FLAGS, face_train.shape[1], voice_train.shape[1])
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()

    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLAGS.max_num_epoch, eta_min=1e-8)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    #for alpha in FLAGS.alpha_list:
    eer_list = []
    epoch=1
    num_of_batches = (len(train_label) // FLAGS.batch_size)
    loss_plot = []
    auc_list = []
    loss_per_epoch = 0
    save_dir = FLAGS.save_dir
    txt = './output/ce_opl_%03d.txt'%(FLAGS.max_num_epoch)

    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    with open(txt,'w+') as f:
        f.write('EPOCH\tLOSS\tEER\tAUC\n')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_best = 'best_%s'%(save_dir)
    
    if not os.path.exists(save_best):
        os.mkdir(save_best)

    with open(txt,'a+') as f:
        while (epoch < FLAGS.max_num_epoch):
            print('Epoch %03d'%(epoch))

            for idx in tqdm(range(num_of_batches)):
                face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
                voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                
                loss_tmp, loss_opl, loss_soft, _, _ = train(face_feats, voice_feats, 
                                                             batch_labels, 
                                                             model, optimizer, ce_loss, opl_loss)
                
                loss_per_epoch += loss_tmp

            scheduler.step()

            loss_per_epoch = loss_per_epoch/num_of_batches
            loss_plot.append(loss_per_epoch)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d.pth.tar'%(epoch))
            print('==> Epoch: %d/%d Loss: %0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch))
            
            eer, auc = online_evaluation.test(FLAGS, model, face_test, voice_test)
            eer_list.append(eer)
            auc_list.append(auc)

            if eer <= min(eer_list):
                min_eer = eer
                max_auc = auc
                save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()}, save_best, 'checkpoint_%04d.pth.tar'%(epoch))

            epoch += 1
            f.write('%04d\t%0.4f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc))
            loss_per_epoch = 0
        
        plt.figure(1)
        plt.title('Total Loss')
        plt.plot(loss_plot)
        plt.savefig('output/im_total_loss.jpg', dpi=600)
        
        plt.figure(2)
        plt.title('EER')
        plt.plot(eer_list)
        plt.savefig('output/im_eer.jpg', dpi=600)
        
        plt.figure(3)
        plt.title('AUC')
        plt.plot(auc_list)
        plt.savefig('output/im_auc.jpg', dpi=600)
                
        return loss_plot, min_eer, max_auc



def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, lw=[0.35, 0.35, 0.3]):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
    comb, face_embeds, voice_embeds, face_f, voice_f = model(face_feats, voice_feats)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)

    loss_align = clip_loss(face_f[0], voice_f[0])

    loss = loss_soft*lw[0] + loss_opl*lw[1] + loss_align*lw[2]

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac


class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average

def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints.')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--max_num_epoch', type=int, default=60, help='Max number of epochs to train, number')
    parser.add_argument('--dim_embed', type=int, default=256, help='Embedding Size')
    parser.add_argument('--n_class', type=int, default=901, help='Number of classes')
    parser.add_argument('--train_path_face', type=str, default='../face_voice/voxceleb1_feats/faceTrain.csv', help='Directory for train file of face features')
    parser.add_argument('--train_path_voice', type=str, default='../face_voice/voxceleb1_feats/voiceTrain.csv', help='Directory for train file of voice features')
    parser.add_argument('--test_path_face', type=str, default='../face_voice/voxceleb1_feats/face_veriflist_test_random_unseenunheard.csv', help='Directory for train file of face features')
    parser.add_argument('--test_path_voice', type=str, default='../face_voice/voxceleb1_feats/voice_veriflist_test_random_unseenunheard.csv', help='Directory for train file of voice features')
    

    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    global FLAGS

    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)

    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    face_train, voice_train, train_label = read_data(train_file=FLAGS.train_path_face, train_file_voice=FLAGS.train_path_voice)
    face_test, voice_test = online_evaluation.read_data(FLAGS.test_path_face, FLAGS.test_path_voice)

    print('Training')

    loss_tmp, eer_tmp, auc_tmp = main(face_train, voice_train, train_label)
