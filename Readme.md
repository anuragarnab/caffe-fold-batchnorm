Folding batch normalisation layers
==

This tool folds batch normalisation and the following scale layer into a single scale layer for networks trained in Caffe. This can be done at inference time to reduce memory consumption, as well as to speed-up computation. This is particularly useful in network architectures which use a lot of batch normalisation (such as ResNet).

For an input, `x`, the batch normalisation and scale layers at test time, perform

```
\gamma * (x - \mu) / \sigma + \beta
```

This can be converted to a single scale layer

```
(\gamma / \sigma) * x + (\beta - \gamma * \mu / \sigma)
```

Here, `\mu` is the mean, `\sigma` the standard deviation, `\gamma` the learned scale, and `\beta` the learned bias.

## Usage
This is for Caffe models, and requires Caffe to be installed.

```
python fold_batchnorm.py 
--model_def_original <path to original input prototxt>
--model_weights_original <path to input weights> 
--model_def_folded <path to save folded prototxt to>
--model_weights_folded <path to save folded weights to>
```