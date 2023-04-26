from typing import Any, Dict, Mapping, Tuple

from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.datasets import util

from misc.eval import evaluate_accuracy, get_eval_metric
from misc.train_util import get_eval_metrics
from misc.util import calc_compression_module_sizes, load_model
from model.franken_net import CQHybridFrankenNet
from train import train, train_main

logger = def_logger.getChild(__name__)


def _load_models(models_config: Dict[str, Any], device: str, skip_ckpt: bool) -> Tuple[CQHybridFrankenNet, nn.Module]:
    assert "student_model" in models_config and "teacher_model" in models_config, "Invalid models config"
    student_model = load_model(models_config["student_model"],
                               device,
                               skip_ckpt=skip_ckpt)
    teacher_model = load_model(models_config['teacher_model'],
                               device,
                               skip_ckpt=False)
    return student_model, teacher_model


def _train_compressor(config: Mapping[str, Any], args: Any):
    models_config = config["models"]
    device = args.device
    student, teacher = _load_models(models_config, device, args.skip_ckpt)
    ckpt_file_path = models_config["student_model"]["ckpt"]
    summary_str, _, _ = calc_compression_module_sizes(
        bnet_injected_model=student,
        device=device,
        input_size=(1, 3, 224, 224))
    logger.info(summary_str)
    datasets_config = config['datasets']
    dataset_dict = util.get_all_datasets(datasets_config)

    if not args.test_only:
        eval_metrics = get_eval_metrics(config["train"])
        train(teacher_model=teacher,
              student_model=student,
              dataset_dict=dataset_dict,
              ckpt_file_path=ckpt_file_path,
              device=device,
              train_config=config["train"],
              eval_metrics=eval_metrics,
              args=args,
              apply_aux_loss=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config,
                                              distributed=False)

    log_freq = test_config.get('log_freq', 1000)
    # check if test has multiple datasets
    metrics = test_config.get("eval_metrics")

    # if args.eval_teacher:
    #     evaluate_accuracy(teacher,
    #                       data_loader=test_data_loader,
    #                       device=device,
    #                       device_ids=None,
    #                       distributed=False,
    #                       log_freq=log_freq,
    #                       title="[Teacher: ]")
    #
    # for metric in metrics:
    #     get_eval_metric(metric).eval_func(student,
    #                                       data_loader=test_data_loader,
    #                                       device=device,
    #                                       device_ids=None,
    #                                       distributed=False,
    #                                       log_freq=log_freq,
    #                                       title="[Student: ]",
    #                                       test_mode=True,
    #                                       use_hnetwork=True)
    #


if __name__ == "__main__":
    train_main(description="Train Compressor", task="train_compressor", train_func=_train_compressor)
