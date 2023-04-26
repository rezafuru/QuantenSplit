import argparse
from typing import Any, List, Mapping

from misc.eval import EvaluationMetric, get_eval_metric


def get_no_stages(train_config) -> int:
    return sum(map(lambda x: "stage" in x, train_config.keys()))


def get_eval_metrics(train_config: Mapping[str, Any]) -> List[Mapping[str, EvaluationMetric]]:
    stages = get_no_stages(train_config)
    eval_metrics = []
    if stages == 0:
        stage_eval_metrics = {}
        metrics = train_config.get("eval_metrics")
        for metric in metrics:
            stage_eval_metrics[metric] = get_eval_metric(metric)
        eval_metrics.append(stage_eval_metrics)
    else:
        for stage in range(stages):
            stage_eval_metrics = {}
            stage_metrics = train_config.get(f"stage{stage + 1}").get("eval_metrics")
            for metric in stage_metrics:
                stage_eval_metrics[metric] = get_eval_metric(metric)
            eval_metrics.append(stage_eval_metrics)
    return eval_metrics


def common_args(parser) -> argparse.ArgumentParser:
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log_path', help='log file folder path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, default=0, help='seed in random number generator')
    parser.add_argument('--test_only', action='store_true', help='only test the models')
    parser.add_argument('--student_only', action='store_true', help='test the student model only')
    parser.add_argument('--validation_metric', default="Accuracy",
                        help="Which Validation metric should be favored when updating the training checkpoint")
    parser.add_argument('--skip_ckpt', action='store_true', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Will only run and log profilers')
    parser.add_argument('--load_last_after_train', action='store_true')
    parser.add_argument('--aux_loss_stage', default=1, type=int)
    parser.add_argument('--load_best_after_stage', action='store_true',
                        help='Load the best performing model after each stage in multi-stage training')
    parser.add_argument('--store_backbone',
                        action='store_true',
                        help='Store backbone weights of bottleneck injected model. ''Typically, not necessary since we '
                             'attach pretrained backbones from some repository (timm, torchvision')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def train_compressor_args(parser) -> argparse.ArgumentParser:
    parser.add_argument('--eval_teacher', action='store_true')

    return parser


def train_predictors_args(parser):
    parser.add_argument('--result_file', required=False, help='result file path')
    parser.add_argument('--json_overwrite' ,required=False, help='json object to overwrite the config file')
    return parser


def get_argparser(description: str, task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser = common_args(parser)
    if task == 'train_compressor':
        parser = train_compressor_args(parser)
    elif task == 'train_predictors':
        parser = train_predictors_args(parser)
    else:
        raise ValueError()

    return parser
