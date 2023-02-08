# Source code for WGSE

* train and evaluate

To train the network for WGSE,

``` bash
python train.py -c 0 -b 16 -e 600 --dataroot "rootpath/of/dataset"  -l "folder/for/save/logfiles"
```

`--dataroot` is the root path of the dataset and `-l` is a folder path where you want to save the log files and weight file of the model.