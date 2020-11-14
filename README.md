# QuickThoughts

Pytorch reimplementation of quickthoughts paper: https://arxiv.org/pdf/1803.02893.pdf. I've refactored code from 
original pytorch [implementation](https://github.com/jcaip/quickthoughts). 

## Changes

I've refactored original code a bit. Made Learner class for simpler model creation. 
### Train

1. download data and clean it (lowercase, add space between punctuation)
2. change path to the training data file in conf (`conf/conf_dev.py` key `data_path`)
3. if you want to evaluate during training on some downstream task (like classification), do following:

      - download downstream dataset, clean it like training data)
      - change script in `src_custom/eval.py`, especially in `load_encode_data` function so that it knows 
      how to load your dataset. You have to gie it a name
      - add the name of your dataset to your conf (`conf/conf_dev.py`) in the key `downstream_eval_datasets` 
      (you can add multiple downstream tasks, just make sure you add them to the conf and `load_encode_data` 
      knows how to read in the data). During training downstream performance is saved/displayed
4. train model (example script `train_dev.py` or if you want to train from checkpoint `train_dev_from_checkpoint.py`)

### Use model

If you wan to use model, use script `get_vectors_dev.py` as an example
     


