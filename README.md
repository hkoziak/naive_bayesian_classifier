## Naive bayesian classifier
  This classifier can predict the sentiment of the sentence as neutral, negative or neutral.
### Some code details

`pandas` is used for cleaning dataset before training the model.
You can split the structure - so don't forget to change path variables `TEST_DATA` and `TRAIN_DATA`.

Notice: if you are going to run a score test for this dataset, check if datasets have similar proportion.
For example, in attached `test.csv` set vast number of values are negative, and it differs by proportion with `train.csv`.

### Usage
Just run the main module:
```shell script
python3 main.py
```
