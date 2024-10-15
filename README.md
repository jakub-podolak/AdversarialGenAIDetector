# AdversarialGenAIDetector

Project for DL4NLP at UVA (2024/2025)

## 1. Running on Lisa / Snellius

Install environment using

```bash
sbatch install_environment.job
```

Run interactive session

```bash
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=04:00:00 --pty bash -i
```

And later

```bash
module purge
module load 2022
module load Anaconda3/2022.05

source activate dl4nlp_gpu
```

To run notebooks run this command in the terminal:

```bash
chmod a+x run_notebook.sh

./run_notebook.sh
```

And follow the displayed instructions



# 2. Downloading the datasets and model checkpoints

To download datasets and model checkpoints, please download them from [Google drive](https://drive.google.com/drive/folders/14dnW7LNL-qzdCqjKOV-wS-MNNVs9YA60?usp=sharing), and put the downloaded folders in the root directory of the project, so it looks like this:

```
output_roberta_augmentations
output_roberta_twibot22
RADAR
raw_data
testing_stuff
src
scripts
README.md
... (rest of the files)
```



**Important note** The [Twibot](https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs) dataset is not present in our repo or Google drive folder as it is proprietary. Please request an access to the dataset, and then in raw data create files:

```
raw_data
	twibot22
		label.csv
		tweet_0.json
		tweet_1.json
```


# 3. Demo notebook

In `demo.ipynb`, you can find a brief demonstration on how to use all the models, as well as how to perform the interpretability experiments using the `transformers-interpret library`.

NOTE: Please move the notebook file into `RADAR/` directory, once you download it from the drive.


# 3. Evaluate the models

To run the augmentations analysis, please refer to `test_models-JT.ipynb`.

