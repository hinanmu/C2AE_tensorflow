# dataset
|dataset |labels |instances |feature |cardinality|
| ------ | ------ | ------ |------ |---|
| mirflickr| 38 | 25000 |1000 |4.7|

# Performance
|type|dataset |Micro F1 |Macro F1 |
| ------| ------ | ------ | ------ |
|this code| mirflickr | 0.540699| 0.39643 |
|paper| mirflickr | 0.54| 0.39 |

# Parameter
- alpha(for output loss para): 0.5
- learning rate: 0.0001
- learning rate decay: 0.98
- momentum: 0.99
- optimizer decay: 0.9
- l2penalty：0.001
- maxepoch: 50
- lagrange para:0.5
- batch size: 500
- hidden units: 512
- latent embedding units: (0, 1) of label dims, default 0.8

# Description
- learning rate: decay by new_lr = lr_init * decay^epoch
- custom optimizer: similar to RMSProp, decay 0.9, momentum 0.99

    init rrr = 0, delta = 0
    rrr=sqrt((rrr.^2)*0.9+(grad.^2)*0.1)
    grad=grad/rrr
    delta=momentum*delta-eta*grad
    new_weight=old_weight + delta

- lagrange para: for caculate embedding loss

# Requrements
- tensorflow 1.12.0
- numpy 1.16.2

# Thanks
the paper's author code matlab [https://github.com/chihkuanyeh/C2AE](https://github.com/chihkuanyeh/C2AE)

other implement(I think there some mistakes in this code) [https://github.com/dhruvramani/C2AE-Multilabel-Classification](https://github.com/dhruvramani/C2AE-Multilabel-Classification)

# Reference
Yeh, C. K., Wu, W. C., Ko, W. J., & Wang, Y. C. F. (2017). Learning deep latent space for multi-label classification. In AAAI (pp. 2838–2844).
