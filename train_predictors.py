import json
import os
import warnings
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

from torchdistill.common.constant import def_logger

from misc.train_util import get_eval_metrics
from misc.util import calc_compression_module_sizes, load_fs_weights, load_model, overwrite_config, short_uid
from model.franken_net import CQHybridFrankenNet
from train import train, train_main
from torchdistill.datasets import util

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

logger = def_logger.getChild(__name__)


def _prepare_df(col_names: List[str], path: Optional[Union[str, bytes, os.PathLike]] = None):
    if path and os.path.isfile(path):
        df = pd.read_csv(path, names=col_names, index_col=None, sep=";", skiprows=[0])
    else:
        df = pd.DataFrame(columns=col_names, index=None)
    return df


def _train_predictors(config: Mapping[str, Any], args: Any):
    qubits = int(os.environ.get('QUBITS', 6))
    q_depth = int(os.environ.get("Q_DEPTH", 8))
    q_device = os.environ.get("Q_DEVICE", "default.qubit")

    if args.json_overwrite:
        logger.info('Overwriting config')
        overwrite_config(config, json.loads(args.json_overwrite))

    models_config = config["models"]["student_model"]
    device = args.device
    student: CQHybridFrankenNet = load_model(models_config,
                                             device,
                                             skip_ckpt=args.skip_ckpt)
    logger.info(f"Running simulations with {qubits} qubits and circuit depth {q_depth} with {q_device}")
    cols = ["predictor", "model", "@1 Acc", "qubits", "circuit depth", "q_device"]
    result_file = args.result_file \
                  or f"resources/results_{short_uid()}/predictor_results_{qubits}_{q_depth}_{q_device}.csv"
    result_file = Path(result_file)
    result_path = result_file.parent
    os.makedirs(result_path, exist_ok=True)
    df = _prepare_df(cols, result_file)

    summary_str, _, _ = calc_compression_module_sizes(
        bnet_injected_model=student,
        device=device,
        input_size=(1, 3, 224, 224))
    logger.info(summary_str)
    student.update(force=True)
    student.delete_backbone_predictor()
    ckpt_file_path = Path(models_config["ckpt_pred"])
    ckpt_file_path = Path(ckpt_file_path.parent) / f"{ckpt_file_path.stem}_{q_depth}_{qubits}.pt"
    datasets_config = config['datasets']
    dataset_dict = util.get_all_datasets(datasets_config)
    predictors = student.predictors.keys()
    groups = ["hybrid", "classical"]
    if not args.test_only:
        for predictor in predictors:
            logger.info(f"Training predictor: {predictor}")
            train_config_name = f"train_{predictor}"
            train_config = config[train_config_name]
            student.active_predictor = predictor
            for group in groups:
                if group == "classical":
                    student.toggle_classical()
                student.use_ensemble = False
                logger.info(f"Training {group} predictor")
                results = train(teacher_model=None,
                                student_model=student,
                                dataset_dict=dataset_dict,
                                ckpt_file_path=ckpt_file_path,
                                device=device,
                                train_config=train_config,
                                eval_metrics=get_eval_metrics(train_config),
                                args=args,
                                apply_aux_loss=False)
                logger.info("Loading best performing predictor")
                df = pd.concat([df, pd.DataFrame({"predictor": [predictor],
                                                  "model": [group],
                                                  "@1 Acc": [results["Accuracy"]],
                                                  "qubits": [qubits],
                                                  "circuit depth": [q_depth],
                                                  "q_device": [q_device]})])
                df.to_csv(result_file, index=False, sep=";")


if __name__ == "__main__":
    train_main(description="Train Predictors", task="train_predictors", train_func=_train_predictors)
