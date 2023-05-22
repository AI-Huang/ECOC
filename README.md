# ECOC

ECOC: Error-Correcting Output Codes

## ECOC: Error-Correcting Output Codes

## Usage

```bash
python ./pytorch/train.py --model-name lenet5 --lr 0.001 --codebook_name=hunqun_deng_c10_n5 --do-train --do-eval --epochs 400 --batch-size 32 --loss binary_cross_entropy

# ResNet18
python ./pytorch/train.py --model-name resnet18 --lr 0.1 --codebook_name=hunqun_deng_c10_n5 --do-train --do-eval --epochs 200 --batch-size 32 --loss binary_cross_entropy
```

### Finetune

```bash
python ./pytorch/train.py --codebook_name=hunqun_deng_c10_n5 --do-train --do-eval --epochs 200 --batch-size 32 --loss binary_cross_entropy --model-path ./output/ECOC-LeNet-5_MNIST_sgd_20230520-020335/ECOC-LeNet-5.pt
```

### Baseline

```bash
# LeNet5
python ./pytorch/train.py --do-train --do-eval --epochs 200 --batch-size 32  --loss cross_entropy

# ResNet18
python ./pytorch/train.py --model-name resnet18 --lr 0.1 --do-train --do-eval --epochs 200 --batch-size 32 --loss cross_entropy
```

### Bug fixxing

For `ModuleNotFoundError: No module named 'ecoc'` or `ModuleNotFoundError: No module named 'pytorch'` etc, add '.' to PYTHONPATH:

```bash
export PYTHONPATH=.:PYTHONPATH
```

Windows:

```cmd
set PYTHONPATH=%PYTHONPATH%;.
```

### Requirements

## Error-Correcting Code Design

See `examples/code_sets.md`

## Research log

| Date         | log                                          |
| ------------ | -------------------------------------------- |
| Jun 20, 2020 | Initially implemented ECOC generation codes. |

## Acknowledgement

Thanks to Dr. Yunxiang Yao's (from ECE Dept. of HKUST) explaination on correcting codes.

## References

- [1] [Applying Error-Correcting Output Coding to Enhance Convolutional Neural Network for Target Detection and Pattern Recognition](https://ieeexplore.ieee.org/document/5597751)
- [2] [Solving Multiclass Learning Problems via Error-Correcting Output Codes](https://www.jair.org/index.php/jair/article/view/10127)
