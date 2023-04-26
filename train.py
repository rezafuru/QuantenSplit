import datetime
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchdistill.common import yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import TrainingBox, get_training_box
from torchdistill.misc.log import MetricLogger, SmoothedValue
from torchdistill.datasets import util

from misc.eval import EvaluationMetric
from misc.loss import BppLossOrig
from misc.train_util import get_argparser
from misc.util import extract_entropy_bottleneck_module, prepare_log_file, save_ckpt_fs
from model.compression.analysis import SimpleResidualAnalysisNetwork
from model.compression.synthesis import SynthesisNetworkSwinTransform
from model.cq_hybrid.circuits import alternating_rotation_circuit_a2
from model.cq_hybrid.classifiers import SimpleCClassifier
from model.timm_models import get_timm_model

logger = def_logger.getChild(__name__)


def _train_one_epoch(training_box: TrainingBox,
                     device: str,
                     epoch: int,
                     log_freq: int,
                     apply_aux_loss: bool):
    model = training_box.student_model if hasattr(training_box, 'student_model') else training_box.model
    model.to(device)
    model.train()
    if apply_aux_loss:
        entropy_bottleneck_module = extract_entropy_bottleneck_module(model)
    else:
        entropy_bottleneck_module = None
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter(f'lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        start_time = time.time()

        sample_batch, targets = sample_batch.to(device), targets.to(device)
        batch_size = sample_batch.shape[0]
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if entropy_bottleneck_module:
            aux_loss = entropy_bottleneck_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        if aux_loss is None:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(),
                                 aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])

        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError("Detected faulty loss = {}".format(loss))


def train(teacher_model: Optional[nn.Module],
          student_model: nn.Module,
          dataset_dict: Mapping[str, Dataset],
          ckpt_file_path: os.PathLike,
          device: str,
          train_config: Mapping[str, Any],
          eval_metrics: List[Mapping[str, EvaluationMetric]],
          apply_aux_loss: bool,
          args: Any):
    log_freq = train_config['log_freq']
    device = torch.device(device)
    if teacher_model is None:
        training_box = get_training_box(student_model,
                                        dataset_dict,
                                        train_config,
                                        device,
                                        device_ids=None,
                                        distributed=False,
                                        lr_factor=1)
    else:
        training_box = get_distillation_box(teacher_model,
                                            student_model,
                                            dataset_dict,
                                            train_config,
                                            device,
                                            None,
                                            False,
                                            1)

    logger.info('Start training')
    training_box.current_epoch = args.start_epoch
    # single stage training only

    results = {'accuracy': 0,
               'bpp': float('-inf')}
    stage_validations = eval_metrics[0]
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        _train_one_epoch(training_box=training_box,
                         device=device,
                         epoch=epoch,
                         log_freq=log_freq,
                         apply_aux_loss=apply_aux_loss)

        for metric, evaluation in stage_validations.items():
            result = evaluation.eval_func(student_model,
                                          training_box.val_data_loader,
                                          device,
                                          None,
                                          False,
                                          log_freq=log_freq,
                                          header=f'Validation-{metric}:')
            results[metric] = result
            if evaluation.compare_with_curr_best(result):
                logger.info('Best {}: {:.4f} -> {:.4f}'.format(metric, evaluation.best_val, result))
                evaluation.best_val = result
                logger.info('Updating ckpt at {}'.format(ckpt_file_path))
                save_ckpt_fs(model=student_model,
                             output_file_path=ckpt_file_path,
                             store_backbone=args.store_backbone)

        training_box.post_process()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()
    return results


def train_main(description: str, task: str, train_func: Callable[[Mapping[str, Any], Any], None]):
    args = get_argparser(description=description, task=task).parse_args()
    prepare_log_file(test_only=args.test_only,
                     log_file_path=args.log_path,
                     config_path=args.config,
                     start_epoch=args.start_epoch,
                     overwrite=False)
    if args.device != args.device:
        torch.cuda.empty_cache()

    cudnn.benchmark = True
    cudnn.deterministic = False
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    logger.info(json.dumps(config))

    train_func(config, args)
