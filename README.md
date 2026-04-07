# Using paraboloid neurons to train DLA models on CIFAR10 with PyTorch

Paraboloid neuron demonstration of the [GeoND Library](https://geond.tech) for [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Requirements
- Linux only
- Python 3.9+, use of a virtual environment recommended
- Install the rest of the requirements by running:
```
pip install -r requirements.txt
```
- (Optional) Download the pre-trained models by running:
```
wget -i models.txt -P checkpoint
```

## Models
### dla
Our baseline Deep Layer Aggregation model.
### dla_paraboloidout
A DLA model with a layer of paraboloid neurons as the output layer. In terms of code, first we import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```

Then we replace the existing output layer:
```
#self.linear = nn.Linear(512, num_classes)
self.paraboloid = gpt.ParaboloidOutput(512, num_classes, h_factor = 0.01, lr_factor = 1., wd_factor = 0.1, grad_factor = 1., input_factor = 0.4, output_factor = 0.1, init='spotlight')
```
Note that ```ParaboloidOutput``` is the same as ```Paraboloid```, but using a base configuration more appropriate for output layers. We use a slightly different configuration here.

Remember to update the forward function:
```
out = self.layer6(out)
out = F.avg_pool2d(out, 4)
out = out.view(out.size(0), -1)
out = self.paraboloid(out)
#out = self.linear(out)
```

### DLA_paraconv_quarter
A DLA model with the first convolutional layer replaced with a paraboloid convolutional layer. In terms of code, again, we first import the Library:
```
try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt
```
Then we find the line with the first convolutional layer:
```
nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
```
and replace it with:
```
gpt.ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1, wd_factor = 2., lr_factor = 1., output_factor = 0.1, h_factor = 0.01),
```
Again, ```ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1, wd_factor = 2., lr_factor = 1., output_factor = 0.1, h_factor = 0.01)``` is equivalent to ```ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1)```. We include the assignments here to show which parameters can be changed to fine tune the ParaConv2d layer.

In this case, we do not need to update the forward function, as we replaced an existing layer.


## IMPORTANT
Including a layer with paraboloid neurons requires a specialized optimizer:
```
optimizer = gpt.GeoNDSGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.wd, nesterov = args.nesterov)
```

## Evaluation and Training

### Evaluate the models by running: 
```
python main.py --model dla --eval dla.pth
python main.py --model dla_paraboloidout --eval dla_paraboloidout.pth
python main.py --model dla_paraconv_quarter --eval dla_paraconv_quarter.pth
```
### Train a model from scratch by ommitting the --eval argument, e.g.:
```
python main.py --model dla_paraconv_half
```
### You can resume the training with:
```
python main.py --model dla_paraconv_half --resume
```

## Training Loss and Accuracy of pretrained models
|   Model           | Training Loss        | Accuracy |
| ----------------- | -------------        | -------- |
| [dla]  - BASELINE       | 0.001210       | 96.03% |
| [dla_paraboloidout]        | 0.000218       | 96.08% |
| [dla_paraconv_quarter] (200 epochs) | 0.001007       | 96.06% |

Note that, due to numerical issues and data augmentation, the results may not always line up exactly.

## References
- Original repository: [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- GeoND Library: [https://geond.tech/download/](https://geond.tech/download/)
- Paraboloid Neurons: [https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf](https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf)
