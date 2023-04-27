# QuantenSplit
 Repository of Prototype for IEEE Services Submission _Architectural Vision for Quantum Computing in the Edge-Cloud Continuum_


## Notes
- This is a simple extension of [_FrankenSplit_](https://github.com/rezafuru/FrankenSplit) to support Hybrid Classical-Quantum Predictors. I tried to remove all redundant references that were not necessary for the experiments in the paper, but I may have missed some things
- I may include a more convenient way to run all experiments, however, I don't plan on actively maintaing this repository/monitor issues. In case you need my assitance, or you notice some problems (e.g. missed reference, broken implementation) please contact me at: a.furutanpey@dsg.tuwien.ac.at
- You can easily change the configuration in the `yaml` file and implement your own Backbones, Compression Models and Ansatz composition

## Preparation
1. Download and Prepare the ILSVRC2012 dataset from the official site (Apparently, I'm not allowed to include a direct download link)
2. Set the repository root as content root and run `misc/create_subsets.py`
3. Run `train_compressor.py` with --config `config/compressor/FP-baseline_compressor-l032.yaml` (See `train_util` for optional arguments)
4. Add as many seeds for the number of runs in `train_predictors.sh`
5. Run `bash train_preidctors.sh cuda 8 HybridQNN alternating_rotation_circuit default.qubit` (replace `cuda` with `cpu` if necessary)

# Citation
Will be added within a few days

# References
- Furutanpey, Alireza, Philipp Raith, and Schahram Dustdar. "FrankenSplit: Saliency Guided Neural Feature Compression with Shallow Variational Bottleneck Injection." arXiv preprint arXiv:2302.10681 (2023).
- Bergholm, Ville, et al. "Pennylane: Automatic differentiation of hybrid quantum-classical computations." arXiv preprint arXiv:1811.04968 (2018).
- Mari, Andrea, et al. "Transfer learning in hybrid classical-quantum neural networks." Quantum 4 (2020): 340.
- Matsubara, Yoshitomo. "torchdistill: A modular, configuration-driven framework for knowledge distillation." Reproducible Research in Pattern Recognition: Third International Workshop, RRPR 2021, Virtual Event, January 11, 2021, Revised Selected Papers. Cham: Springer International Publishing, 2021.
- Wightman, Ross. "Pytorch image models." (2019).
- Bégaint, Jean, et al. "Compressai: a pytorch library and evaluation platform for end-to-end compression research." arXiv preprint arXiv:2011.03029 (2020).
- Ballé, Johannes, Valero Laparra, and Eero P. Simoncelli. "End-to-end optimized image compression." arXiv preprint arXiv:1611.01704 (2016).
