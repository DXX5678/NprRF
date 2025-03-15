# NprRF

We provide the necessary scripts and data for training.

```
|--Generator: patch generator training and evaluation code
​	|--CodeBert
​	|--CodeT5
​	|--UniXcoder
|--Reward: patch assessment model training and evaluation code
​	|--APPT
​	|--Metric (CodeBLEU+BERTScore)
|--RL: implementation of reinforcement learning algorithm
​	|--PPO
​	|--REINFORCE
|--Defects4J_method_singlehunk.csv: list of method level buggy projects in Defects4J
|--Defects4J_method_singlehunk.json: list of method level buggy projects in Defects4J
|--CodeQL.sh: vulnerability detection script
|--RL_run_metric_REINFORCE.sh
|--RL_run_REINFORCE.sh
|--RL_run_metric.sh
|--RL_run.sh
|--Defects4J_30_测试结果.xlsx: Defects4J test results
```

## Dataset

The raw dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1vKyABQbdvH8SuQc23VihB2INj_brrdnv?usp=sharing).

## Environment

- python 3.8
- numpy==1.24.3
- pandas==2.0.3
- scikit_learn==1.3.0
- torch==2.4.1+cu121
- transformers==4.42.3

**Defects4J: please configure the [Defects4J environment](https://github.com/rjust/defects4j/tree/master) yourself.**

## Train & Evaluate

Model_type: CodeBert / CodeT5 / UniXcoder

Reward_type: APPT / Metric

### Train

1. #### **Repair Training**

- Please modify the `Generator/Model_type/train.sh`, this file has some parameters needed to train our model.

  ```bash
  bash Generator/Model_type/train.sh
  ```

2. #### **Reinforcement Learning**

- Please modify the `RL_run.sh` (for APPT) or  `RL_run_metric.sh` (for Metric), this file has some parameters needed to train our model.

  ```bash
  bash RL_run.sh
  bash RL_run_metric.sh
  ```

### Evaluate (for Defects4J)

- Please modify the Generator/Model_type/Defects4J_experiment.sh, reset the model loading path and some parameters required for the model.

```bash
bash Generator/Model_type/Defects4J_experiment.sh
```

