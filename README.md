# Why Gradient Subspace? Identifying and Mitigating LoRA's Bottlenecks in Federated Fine-Tuning of Large Language Models

This is the official repository of the paper "Why Gradient Subspace? Identifying and Mitigating LoRA's Bottlenecks in Federated Fine-Tuning of Large Language Models" (link [here](https://arxiv.org/abs/2410.23111)). 

## How to Setup and Run the Repo:

I usually use `virtualenv`, you can use `conda` or any other package management system.

```shell
virtualenv venv
source venv/bin/activate
pip install -r requirments.txt
```

Before running the experiments, make sure you have all the datasets ready. The dataset samples are given in the folder `save_paths`. The scripts for pre-processing the datasets will be added on later. All the details regarding dataset will be added on later. 

`config.yaml` contains the configuration. Details and necessary information regarding the configuration arguments are listed in the file itself. 

```shell
python3 main.py --config config.yaml
```

