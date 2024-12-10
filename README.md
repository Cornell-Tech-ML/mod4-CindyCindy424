# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py

----

## Task 4.5

1. Commands:
```
python project/run_sentiment.py > sentiment.txt
python project/run_mnist_multiclass.py > mnist.txt
```

2. Result File:

Please refer to `'mnist.txt'` and `'sentiment.txt'` files in the root folder of this repo.

Result Preview:
1. For Sentiment (SST2) task, the model achieved best validation accuracy of 77.00%

> Epoch 213, loss 8.480409707886865, train accuracy: 85.78%

> Validation accuracy: 72.00%

> Best Valid accuracy: 77.00%

2. For Digit Classification (MNIST) task, achieve final accuracy of 16/16.

> Epoch 21 loss 0.010621765468021294 valid acc 16/16