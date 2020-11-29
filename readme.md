# DeepOutbreak

## Requirements

Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f requirements.yml
```

## Training ILI model

Enter to ILI folder. To train the model, run this command:

```bash
python ./TrainPredict.py
```

You can set up your own hyperparameter values and data/modules to remove. See help on ```TrainPredict.py``` for more details.

e.g.
```bash
python ./TrainPredict.py --data linelist --recon_weight 0.005 --alpha 0.3 --_beta 0.01
```

To evaluate the results, go to ```evaluate.py``` and change line 77 for the name of results file (saved in folder ```rmse_results```). Then, run.

```bash
python ./evaluate.py
```

## Training COVID model

Enter to COVID folder. To train the model, run this command:

```bash
python ./src/training/testbed.py --infile1 ./scripts/specs/base.json --infile2 ./scripts/specs/m1_weekly.json --target death --runs 20 --data_ew 202046 --pred_ew 202046
```

If you want to get predictions for previous weeks, say epidemic week 40, set prediction week (pred_ew):

e.g.
```bash
python ./src/training/testbed.py --infile1 ./scripts/specs/base.json --infile2 ./scripts/specs/m1_weekly.json --target death --runs 20 --data_ew 202046 --pred_ew 202040
```

To parse and evaluate the results (change line 101 to update prediction week):

```bash
python ./src/training/parse_results.py
```
