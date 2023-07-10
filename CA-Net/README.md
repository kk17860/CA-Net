### Requirements

The code is built with the following libraries:

- Python 3.6 or higher
- CUDA 10.0 or higher
- [PyTorch](https://pytorch.org/) 1.2 or higher
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scipy](https://www.scipy.org/)

Besides, you need to install a custom module for bounding box NMS and overlap calculation.
```
cd build/box
python setup.py install
```

### Testing

Run the following scripts to evaluate the model and obtain the results of FROC analysis.
```
python test.py
```

### Training
This implementation supports multi-gpu, `data_parallel` training.

Change training configuration and data configuration in `config.py`, especially the path to preprocessed data.

Run the training script:
```
python train.py
```

### Contact

For any questions, please contact me.
