import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sadm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8, margin=0.003):
    """
    Additive Angular Margin Matching Loss
    """
    image_feture = image_fetures.mean(dim=1)
    batch_size = image_feture.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    image_norm = image_feture / image_feture.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()
    
    mask = (labels != 0)
    t2i_cosine_theta[mask] += (torch.cos(torch.arccos(t2i_cosine_theta[mask]) + margin) - t2i_cosine_theta[mask]).detach()
    i2t_cosine_theta[mask] += (torch.cos(torch.arccos(i2t_cosine_theta[mask]) + margin) - i2t_cosine_theta[mask]).detach()
    
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    # softmax based loss function normally utilize three functions, which are F.softmax, F.log_softmax and torch.log
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    
    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2