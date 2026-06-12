import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class CARE(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 equiv_weight=0.01, num_equiv_chunks=8):
        super().__init__()
        self.loss_function = loss_function
        self.device = device
        self.equiv_weight = equiv_weight
        self.num_equiv_chunks = num_equiv_chunks

    def forward(self, image_features1, image_features2, logit_scale, 
                aug_features1=None, aug_features2=None):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        infonce_loss = (self.loss_function(logits_per_image1, labels) + 
                       self.loss_function(logits_per_image2, labels)) / 2
        
        if aug_features1 is not None and aug_features2 is not None:
            aug_features1 = F.normalize(aug_features1, dim=-1)
            aug_features2 = F.normalize(aug_features2, dim=-1)
            
            equiv_loss1 = self.equivariance_loss(image_features1, aug_features1)
            equiv_loss2 = self.equivariance_loss(image_features2, aug_features2)
            equiv_loss = (equiv_loss1 + equiv_loss2) / 2
        else:
            equiv_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = infonce_loss + self.equiv_weight * equiv_loss
        return total_loss

    def equivariance_loss(self, original_features, augmented_features):
        batch_size = original_features.shape[0]
        chunk_size = batch_size // self.num_equiv_chunks
        loss = 0.0
        
        for i in range(self.num_equiv_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_equiv_chunks - 1 else batch_size
            
            orig_chunk = original_features[start_idx:end_idx]
            aug_chunk = augmented_features[start_idx:end_idx]
            chunk_bs = orig_chunk.shape[0]
            
            orig_inner = torch.mm(orig_chunk, orig_chunk.t().contiguous())
            aug_inner = torch.mm(aug_chunk, aug_chunk.t().contiguous())
            
            mask = torch.ones(chunk_bs, chunk_bs, device=self.device) - torch.eye(chunk_bs, device=self.device)
            mask = mask.bool()
            
            orig_flat = orig_inner[mask].view(chunk_bs, -1)
            aug_flat = aug_inner[mask].view(chunk_bs, -1)
            
            loss += 2 * torch.norm(orig_flat - aug_flat, p='fro', dim=-1).pow(2).mean()
        
        return loss / self.num_equiv_chunks

class BarlowTwins(nn.Module):
    def __init__(self, bs, lamda):
        super().__init__()
        self.bs = bs
        self.lamda = lamda
        self.bn = nn.BatchNorm1d(1024, affine=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, image_features1, image_features2):
        z1 = image_features1
        z2 = image_features2

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        #c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.bs)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamda * off_diag
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

