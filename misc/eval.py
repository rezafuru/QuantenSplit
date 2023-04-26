import copy
from functools import partial

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchdistill.common.constant import def_logger
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import MetricLogger

from misc.util import check_if_module_exits, compute_bitrate, extract_entropy_bottleneck_module
from model.franken_net import CQHybridFrankenNet

logger = def_logger.getChild(__name__)


class EvaluationMetric:
    def __init__(self,
                 eval_func,
                 init_best_val,
                 comparator):
        self.eval_func = eval_func
        self.best_val = init_best_val
        self.comparator = comparator

    def compare_with_curr_best(self, result) -> bool:
        return self.comparator(self.best_val, result)


EVAL_METRIC_DICT = dict()


def get_eval_metric(metric_name, **kwargs) -> EvaluationMetric:
    if metric_name not in EVAL_METRIC_DICT:
        raise ValueError("Evaluation metric with name `{}` not registered".format(metric_name))
    return EVAL_METRIC_DICT[metric_name]()


def register_eval_metric(cls: EvaluationMetric):
    EVAL_METRIC_DICT[cls.__name__] = cls
    return cls


@torch.inference_mode()
def evaluate_accuracy(model,
                      data_loader,
                      device,
                      device_ids=None,
                      distributed=False,
                      log_freq=1000,
                      title=None,
                      header='Test:',
                      no_dp_eval=True,
                      accelerator=None,
                      include_top_k=False,
                      evaluate_ensemble=False,
                      **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if evaluate_ensemble:
            output = model.forward_ensemble(image)
        else:
            output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if include_top_k:
        return top1_accuracy, top5_accuracy
    return top1_accuracy


@torch.inference_mode()
def evaluate_accuracy_binary(model,
                             data_loader,
                             device,
                             device_ids=None,
                             distributed=False,
                             log_freq=1000,
                             title=None,
                             header='Test:',
                             no_dp_eval=True,
                             accelerator=None,
                             predictor: str = None,
                             include_top_k=False,
                             evaluate_ensemble=False,
                             **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if evaluate_ensemble:
            output = model.forward_ensemble(image)
        else:
            output = model(image)
        acc1 = compute_accuracy(output, target, topk=(1,))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    logger.info(' * Acc@1 {:.4f}\n'.format(top1_accuracy))
    if include_top_k:
        return top1_accuracy, top1_accuracy
    else:
        return top1_accuracy


@torch.inference_mode()
def evaluate_bpp(model,
                 data_loader,
                 device,
                 device_ids=None,
                 distributed=False,
                 log_freq=1000,
                 title=None,
                 header='Test:',
                 no_dp_eval=True,
                 extract_bottleneck=True,
                 **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    model.eval()
    bottleneck_module = extract_entropy_bottleneck_module(model)
    has_hyperprior = False
    if check_if_module_exits(bottleneck_module, 'gaussian_conditional'):
        has_hyperprior = True
    metric_logger = MetricLogger(delimiter='  ')
    for image, _ in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        _, likelihoods = bottleneck_module(image, return_likelihoods=True)
        if has_hyperprior:
            likelihoods_y, likelihoods_z = likelihoods.values()
            bpp_z, _ = compute_bitrate(likelihoods_z, image.shape)
            bpp_y, _ = compute_bitrate(likelihoods_y, image.shape)
            metric_logger.meters['bpp_z'].update(bpp_z.item(), n=image.size(0))
            metric_logger.meters['bpp_y'].update(bpp_y.item(), n=image.size(0))
            bpp = bpp_z + bpp_y
        else:
            bpp, _ = compute_bitrate(likelihoods["y"], image.shape)
        metric_logger.meters['bpp'].update(bpp.item(), n=image.size(0))
    metric_logger.synchronize_between_processes()
    avg_bpp = metric_logger.bpp.global_avg
    logger.info(' * Bpp {:.5f}\n'.format(avg_bpp))
    if has_hyperprior:
        avg_bpp_z = metric_logger.bpp_z.global_avg
        avg_bpp_y = metric_logger.bpp_y.global_avg
        logger.info(' * Bpp_z {:.5f}\n'.format(avg_bpp_z))
        logger.info(' * Bpp_y {:.5f}\n'.format(avg_bpp_y))
    return avg_bpp


@torch.inference_mode()
def evaluate_filesize_and_accuracy(model,
                                   data_loader,
                                   device,
                                   device_ids,
                                   distributed,
                                   log_freq=1000,
                                   title=None,
                                   header='Test:',
                                   no_dp_eval=True,
                                   test_mode=False,
                                   use_hnetwork=False,
                                   **kwargs) -> float:
    model.to(device)
    if not no_dp_eval:
        if distributed:
            model = DistributedDataParallel(model, device_ids=device_ids)
        elif device.type.startswith('cuda'):
            model = DataParallel(model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    analyzable = False
    if test_mode:
        if check_if_analyzable(model):
            model.activate_analysis()
            analyzable = True
        else:
            logger.warning("Requesting analyzing compressed size but model is not analyzable")
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    avg_filesize = None
    if analyzable:
        avg_filesize = model.summarize()[0]
        model.deactivate_analysis()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg, avg_filesize


@register_eval_metric
class Accuracy(EvaluationMetric):
    def __init__(self):
        super().__init__(
            eval_func=evaluate_accuracy,
            init_best_val=0,
            comparator=lambda curr_top_val, epoch_val: epoch_val > curr_top_val,
        )

@register_eval_metric
class Bpp(EvaluationMetric):
    def __init__(self):
        super().__init__(
            eval_func=evaluate_bpp,
            init_best_val=float("inf"),
            comparator=lambda curr_top_val, epoch_val: epoch_val < curr_top_val,
        )

