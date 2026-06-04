"""
convnext_cls.py: single epoch train loop + evaluation for convnextv2 classifier

logits = backbone(img)                 # forward_features -> GAP+LayerNorm -> head
loss   = CrossEntropyLoss()(logits, class_id)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ..models.convnext.utils import MetricLogger, SmoothedValue, adjust_learning_rate


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of correctly-classified images (plain top-1)."""
    preds = logits.argmax(dim=1)
    total = target.numel()
    if total == 0:
        return 0.0
    return int((preds == target).sum()) / total


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
) -> dict[str, float]:
    
    """Train the classifier backbone for one epoch; returns averaged-stat dict.

    `data_loader` yields `(image, label, name)` (from `TrainLabeledDataset`);
    only `image` and `label` are used. `args` must expose `lr`, `min_lr`,
    `warmup_epochs`, `epochs` for the per-iteration cosine schedule.
    """
    
    model.train(True)

    # logging...?
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq = 20
    if args:
        update_freq = args.update_freq
        use_amp = args.use_amp # type: ignore
    else:
        update_freq = 1
        use_amp = None

    optimizer.zero_grad()

    for data_iter_step, (samples, label, _name) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # per-iteration (not per-epoch) LR schedule; fractional epoch
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        target = label.to(device, non_blocking=True)

        # TODO: missing mixup

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(samples)
                loss = criterion(logits, target)
        else: # full precision
            logits = model(samples)
            loss = criterion(logits, target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order # type: ignore
            loss /= update_freq

            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, 
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step+ 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                
                # TODO: model ema
        else:
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # TODO: model ema

        torch.cuda.synchronize()

        # TODO: mixup fn

        class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Evaluate top-1 and macro (class-balanced) accuracy on a labeled split.

    `data_loader` may yield either `(image, label, name)` (TrainLabeledDataset)
    or `(image, label, seg_id, name, orig_size)` (ValDataset) — we only read the
    first two elements, so both contracts work.

    Macro accuracy = mean per-class recall (each class weighted equally), which
    is what the leaderboard's classification metric rewards (class-balanced), so
    don't rely on plain top-1 for model selection.
    """
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        torch.cuda.synchronize()
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]