
# Semi-Supervised Semantic Segmentation With Region Relevance


## Requirements

>📋  The code is developed using Python 3.6 with PyTorch 1.10.1. The code is developed and tested using 2 NVIDIA TITAN RTX GPUs.

 

## Training && Evaluation

To train and eval the model in the paper, run this command:

```train
cd ./voc/voc8.res50v3+.CPS+CutMix
bash script_mod.sh
```

>📋  In script_mod.sh, you need to specify some variables, such as the path to your data dir, the path to your snapshot dir that stores checkpoints, etc.

## Our Models DownLoad && Test

We have released the weight models on PASCAL VOC in our experiment, you can download here:

- [My voc2 model](https://drive.google.com/file/d/12ub3q4-W_3gBPcUcsoOqbe3fOVBngEDt/view?usp=share_link) trained on PASCAL VOC 2012 at 1/2 partition protocol. 
- [My voc4 model](https://drive.google.com/file/d/1GErx-JndoqM1pzuQ-33KUJ21gdatH055/view?usp=share_link) trained on PASCAL VOC 2012 at 1/4 partition protocol.
- [My voc8 model](https://drive.google.com/file/d/1olSbRjWSMFckkhewc6qWM13D-oSeoFG3/view?usp=share_link) trained on PASCAL VOC 2012 at 1/8 partition protocol.
- [My voc16 model](https://drive.google.com/file/d/14WJBhoo1NpPb1PyPzoIwgEUdFWmtadHH/view?usp=share_link) trained on PASCAL VOC 2012 at 1/16 partition protocol.


>📋  The command for test is in script_mod.sh, you only need to annotate the command for training and modify the parameters appropriately.
