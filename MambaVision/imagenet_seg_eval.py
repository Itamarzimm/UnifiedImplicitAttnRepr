import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
from tqdm import tqdm
import sys

import sys

from timm.models import create_model
import render
import torch
import numpy as np

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
import vim.models_mamba as models_mamba

import cv2
import torch
import numpy as np

from iou import IoU

from Imagenet import Imagenet_Segmentation
from saver import Saver

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn

from timm.models import create_model

import os

def load_model():
    device = torch.device('cuda')
    cudnn.benchmark = True
    resume = r'/media/data1/ameenali/LRPFusion/FusionMambaXAI-ido/vim_b_midclstok_81p9acc.pth'
    model = create_model('vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',pretrained=False,num_classes=1000,drop_rate=0,drop_path_rate=0,drop_block_rate=None,img_size=224)
    checkpoint = torch.load(resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model.to(device)
plt.switch_backend('agg')


import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve

SMOOTH = 1e-6
__all__ = ['get_f1_scores', 'get_ap_scores', 'batch_pix_accuracy', 'batch_intersection_union', 'get_iou', 'get_pr',
           'get_roc', 'get_ap_multiclass']


def get_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.cpu().numpy()


def get_f1_scores(predict, target, ignore_index=-1):
    # Tensor process
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target.data.cpu().numpy().reshape(-1)
    pb = predict[target != ignore_index].reshape(batch_size, -1)
    tb = target[target != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(np.nan_to_num(f1_score(t, p)))

    return total


def get_roc(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(roc_curve(t, p))

    return total


def get_pr(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(precision_recall_curve(t, p))

    return total


def get_ap_scores(predict, target, ignore_index=-1):
    total = []
    for pred, tgt in zip(predict, target):
        target_expand = tgt.unsqueeze(0).expand_as(pred)
        target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)

        # Tensor process
        x = torch.zeros_like(target_expand)
        t = tgt.unsqueeze(0).clamp(min=0).long()
        target_1hot = x.scatter_(0, t, 1)
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = target_1hot.data.cpu().numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))

    return total


def get_ap_multiclass(predict, target):
    total = []
    for pred, tgt in zip(predict, target):
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = tgt.data.cpu().numpy().reshape(-1)

        total.append(np.nan_to_num(average_precision_score(target_flat, predict_flat)))

    return total


def batch_precision_recall(predict, target, thr=0.5):
    """Batch Precision Recall
    Args:
        predict: input 4D tensor
        target: label 4D tensor
    """
    # _, predict = torch.max(predict, 1)

    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    precision = float(np.nan_to_num(tp / (tp + fp)))
    recall = float(np.nan_to_num(tp / (tp + fn)))

    return precision, recall


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 3D tensor
        target: label 3D tensor
    """

    # for thr in np.linspace(0, 1, slices):

    _, predict = torch.max(predict, 0)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 3D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 0)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]



def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def generate_ours(model, image):
    image.requires_grad_()
    logits = modified_model(image)
    
    index = np.argmax(logits.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    
    attn_vecs = []

    all_layer_attentions = []
    cls_pos = 98
    start_layer=15

    for i in range(len(model.layers)):
        attn_vecs_a = modified_model.layers[i].mixer.attn_vec.detach().cpu()
        attn_vecs_b = modified_model.layers[i].mixer.attn_vec_b.detach().cpu()
        attn_vecs_a[:,:,99:] = attn_vecs_b.flip([-1])[:,:,99:]
        
        s = model.layers[i].get_gradients().squeeze().detach() #[1:, :].clamp(min=0).max(dim=1)[0].unsqueeze(0)

        attn_vecs_a = attn_vecs_a.clamp(min=0)
        # attn_vecs_a = (attn_vecs_a - attn_vecs_a.min()) / (attn_vecs_a.max() - attn_vecs_a.min())
        s = s.clamp(min=0).max(dim=1)[0].unsqueeze(0).detach().cpu()
        s = (s - s.min()) / (s.max() - s.min())

        avg_heads = (attn_vecs_a.sum(dim=1) / attn_vecs_a.shape[1]).detach()
        # avg_heads = (avg_heads - avg_heads.min()) / (avg_heads.max() - avg_heads.min())
        avg_heads = avg_heads * s
        avg_heads = (avg_heads - avg_heads.min()) / (avg_heads.max() - avg_heads.min())

        all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    p = rollout[0 , cls_pos , :].unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0)


# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='ours',
                    choices=[ 'ours', 'lrp','transformer_attribution', 'full_lrp', 'lrp_last_layer',
                              'attn_last_layer', 'attn_gradcam'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=True,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')
parser.add_argument('--imagenet-seg-path', type=str, default='/media/data1/ameenali/WorkingIdo/FusionMambaXAI-ido/gtsegs_ijcv.mat')
args = parser.parse_args()

args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, 'results')
if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model


#you need to also download checkpoint from https://huggingface.co/hustvl/Vim-tiny-midclstok 


model_name="vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

model_ = create_model(
    model_name,
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
)

checkpoint_path = './vim_s_midclstok_80p5acc.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_.load_state_dict(checkpoint["model"])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_.to(device)
model_.eval()

for i in range(24):
    model_.layers[i].mixer.saveAttnVec = True


Test = False #works only when both saveAttnMat and saveAttnVec are True
modified_model = model_# ModifiedVisionMamba(model_, zero_bias, layer_transforms, saveAttnMat, saveAttnVec, Test)
modified_model.eval()
# model.patch_embed.requires_grad = False #TODO: needed?


metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

modified_model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

    image.requires_grad = True

    image = image.requires_grad_()
    # predictions = evaluator(image)
    import time
    # segmentation test for the rollout baseline
    Res = generate_ours(evaluator, image).reshape(batch_size, 1, 14, 14)#baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
    Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
    after = time.time()
    # Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0


    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):

    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()
    # print("image", image.shape)
    # print("lables", labels.shape)

    correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, modified_model, batch_idx)

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)
pr, rc, thr = precision_recall_curve(targets, predictions)
np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

plt.figure()
plt.plot(rc, pr)
plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
# txtfile = 'result_mIoU_%.4f.txt' % mIoU
fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()


# #230 : pixAcc: 0.7089, mIoU: 0.4980, mAP: 0.7627, mF1: 0.3580