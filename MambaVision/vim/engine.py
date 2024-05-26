# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.diff_maps:
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].mixer.compute_diff_maps = True
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # debug
    # count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count += 1
        # if count > 20:
        #     break

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 


            if args.diff_maps:
                sum_maps  = 0 
                for layer_idx in range(len(model.layers)):
                    if layer_idx == 0:
                        sum_maps = model.layers[layer_idx].mixer.diff_map.abs().mean(dim=1)#abs and mean over channels
                    else: 
                        sum_maps = sum_maps +model.layers[layer_idx].mixer.diff_map.abs().mean(dim=1)#abs and mean over channels
                sum_maps = sum_maps.squeeze(1) #[2,1, 197] -> [2, 197]
                scalr_maps = sum_maps.mean(1).mean(0)
                loss = loss + scalr_maps
        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    saveAttnMat = args.attnmatrices_path is not None
    saveAttnVec = args.attnvector_path is not None
    if saveAttnMat:
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].mixer.saveAttnMat = True
    if saveAttnVec:
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].mixer.saveAttnVec = True
    if args.old_attention:
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].mixer.old_attention = True
    if args.compute_attn_gradients:
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].mixer.compute_attn_gradients = True
        for name, param in model.named_parameters():
            param.requires_grad = True

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)
            if args.compute_attn_gradients:
                loss.backward()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.exit_after_1_batch: 
            if saveAttnMat:
                for layer_idx in range(len(model.layers)):
                    HiddenAttnMat = model.layers[layer_idx].mixer.attn_mat.abs()
                    for b_idx in range(HiddenAttnMat.shape[0]):
                        if b_idx != 0: continue
                        for c_idx in range(HiddenAttnMat.shape[1]):
                            currHiddenAttnMat = HiddenAttnMat[b_idx,c_idx]
                            min_val = torch.min(currHiddenAttnMat)
                            max_val = torch.max(currHiddenAttnMat)
                            currHiddenAttnMat = (currHiddenAttnMat - min_val) / (max_val - min_val)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(6,6))
                            plt.imshow(currHiddenAttnMat.detach().cpu().numpy(), cmap='viridis', aspect='auto')  # Convert tensor to numpy array and use a grayscale colormap
                            plt.colorbar()  # Optional: Adds a colorbar to the side showing the scale
                            plt.axis('off')  # Turn off the axis numbers and ticks
                            # Save the figure as an image file
                            plt.savefig(args.attnmatrices_path + "b-" + str(b_idx) + "!c-" + str(c_idx) + "!layer_idx-" + str(layer_idx)+'attnmap.png', bbox_inches='tight', pad_inches=0.1)
                            plt.close()

            if saveAttnVec and not args.compute_attn_gradients:
                for layer_idx in range(len(model.layers)):
                    HiddenAttnVec = model.layers[layer_idx].mixer.attn_vec.abs()
                    for b_idx in range(HiddenAttnVec.shape[0]):
                        if b_idx != 0: continue
                        for c_idx in range(HiddenAttnVec.shape[1]):
                            currHiddenAttnVec = HiddenAttnVec[b_idx,c_idx]
                            min_val = torch.min(currHiddenAttnVec)
                            max_val = torch.max(currHiddenAttnVec)
                            currHiddenAttnVec = (currHiddenAttnVec - min_val) / (max_val - min_val)
                            currHiddenAttnVec = torch.cat((currHiddenAttnVec[:97], currHiddenAttnVec[98:]))
                            currHiddenAttnVec = currHiddenAttnVec.view(14, 14)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(6,6))
                            plt.imshow(currHiddenAttnVec.detach().cpu().numpy(), cmap='viridis', aspect='auto')  # Convert tensor to numpy array and use a grayscale colormap
                            plt.colorbar()  # Optional: Adds a colorbar to the side showing the scale
                            plt.axis('off')  # Turn off the axis numbers and ticks
                            # Save the figure as an image file
                            plt.savefig(args.attnvector_path + "b-" + str(b_idx) + "!c-" + str(c_idx) + "!layer_idx-" + str(layer_idx)+'attnmap.png', bbox_inches='tight', pad_inches=0.1)
                            plt.close()

            if saveAttnVec and args.compute_attn_gradients:
                for layer_idx in range(len(model.layers)):
                    out_grad = model.layers[layer_idx].mixer.out.grad[:,:,98].unsqueeze(-1) #BH1
                    x = model.layers[layer_idx].mixer.x #BHL 
                    attn_grads = (out_grad*x).abs()
                    for b_idx in range(attn_grads.shape[0]):
                        if b_idx != 0: continue
                        for c_idx in range(attn_grads.shape[1]):
                            currattn_grads = attn_grads[b_idx,c_idx]
                            min_val = torch.min(currattn_grads)
                            max_val = torch.max(currattn_grads)
                            currattn_grads = (currattn_grads - min_val) / (max_val - min_val)
                            currattn_grads = torch.cat((currattn_grads[:97], currattn_grads[98:]))
                            currattn_grads = currattn_grads.view(14, 14)
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(6,6))
                            plt.imshow(currattn_grads.detach().cpu().numpy(), cmap='viridis', aspect='auto')  # Convert tensor to numpy array and use a grayscale colormap
                            plt.colorbar()  # Optional: Adds a colorbar to the side showing the scale
                            plt.axis('off')  # Turn off the axis numbers and ticks
                            # Save the figure as an image file
                            plt.savefig(args.attnvector_path + "gradsMap-b-" + str(b_idx) + "!c-" + str(c_idx) + "!layer_idx-" + str(layer_idx)+'attnmap.png', bbox_inches='tight', pad_inches=0.1)
                            plt.close()

            print("The first batch has been processed")
            exit()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
