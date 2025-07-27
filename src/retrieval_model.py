import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import ToPoincare, ReLU_hyperbolic


'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.normalize(x) 
        return x


def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.4),
                        nn.Linear(f_out, f_out))


class Fusion_Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv1d(1, 1, 5, padding=2)

    def forward(self, face, voice):
        ff = nn.functional.gelu(face)
        vf = nn.functional.gelu(voice)

        attn = ff * vf
        attn = self.conv(attn.unsqueeze(1)).squeeze(1)
        attn = torch.sigmoid(attn)

        x = face*attn + (1-attn)*voice

        return x, ff, vf


##################################################################

class PAEFF(nn.Module):
    def __init__(self, args, face_feat_dim, voice_feat_dim):
        super(PAEFF, self).__init__()
        
        self.voice_branch = EmbedBranch(voice_feat_dim, args.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, args.dim_embed)
        
        self.fusion_layer = Fusion_Attn(dim=args.dim_embed)
        self.res_mix = nn.Linear(args.dim_embed, args.dim_embed)
        
        self.logits_layer = nn.Linear(args.dim_embed, args.n_class)
        self.tp = ToPoincare(c=0.05)

        if args.cuda:
            self.cuda()

    def forward(self, faces, voices):
        voices_feats = self.voice_branch(voices)
        faces_feats = self.face_branch(faces)

        fused_feats, faces_feats_e, voices_feats_e = self.fusion_layer(faces_feats, voices_feats) # fusion of features
        fused_feats = torch.tanh(self.res_mix(fused_feats)) #.squeeze(1)

        logits = self.logits_layer(fused_feats)

        faces_feats_a = self.tp(faces_feats)
        voices_feats_a = self.tp(voices_feats)
        
        return [fused_feats, logits], faces_feats_e, voices_feats_e, [faces_feats_a], [voices_feats_a]
    