## EFM3D Benchmark

We provide three evaluation datasets for the EFM3D benchmarks. For more details on the benchmark see the [EFM3D](https://arxiv.org/abs/2406.10224) paper.

### ASE - 3D object detection and mesh reconstruction
Aria Synthetic Environments (ASE) is a synthetic dataset, created from procedurally-generated interior layouts filled with 3D objects, simulated with the sensor characteristics of Aria glasses. We use ASE for both surface reconstruction and 3D object detection evaluation.

First follow instructions in the dataset [README.md](data/README.md) to download the ASE eval set and eval meshes, then run the following

```
python eval.py --ase
```

Running the full evaluation on 100 eval sequences takes a long time (>10 hrs on a single GPU). To see the eval results on a reduced set, use `--num_seqs` and `--num_snips` to specify number of sequences and number of snippets per sequence to speed up evaluation, for example

```
# run eval on the first 10 sequences of ASE, each running for 100 snippets (10s)
python eval.py --ase --num_seqs 10 --num_snips 100
```

### ADT - mesh reconstruction
ADT is the benchmark data for surface reconstruction, containing 6 sequences.
Download the ADT data and mesh files following the data instruction. Then run

```
python eval.py --adt
```

The provided script provides an end-to-end solution to run EVL model with the default checkpoint,
finding the right GT mesh path for ASE and ADT dataset, then run the evaluation metrics for mesh-to-mesh distance.
If you have your own model that generates a ply file, check [eval_mesh_to_mesh](efm3d/utils/mesh_utils.py) for how to evaluate against surface GT directly.

### AEO - 3D object detection
AEO is the benchmark data for 3D object detection, with 25 sequences.
Download the AEO dataset following the data instruction. Then run

```
python eval.py --aeo
```

This will run the EVL model inference using the default model checkpoint path.
If you have your own model for inference, check [eval.py](efm3d/inference/eval.py) for how to evaluate against 3D object GT directly.
