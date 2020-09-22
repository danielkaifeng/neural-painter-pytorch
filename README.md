# neural-painter-pytorch
This is pytorch version of painting with neural network

It is a trial of NN art generation several years ago. With the state of art NN struture, maybe we can create more beautiful patterns.

### demo usage
```
python3 net_pytorch.py  --image_size 500x500 --hidden_size 100 --nr_hidden 4 --nonlin random_every_time --nr_channel 3 --output_nonlin identity --coord_bias --output 00.png
```

### demo example
![1](./example/1.png)
![2](./example/2.png)
![3](./example/3.png)
![4](./example/4.png)



### Reference 
1. [neural-painter](https://github.com/zxytim/neural-painter)
2. [npainter](https://github.com/rupeshs/npainter)
