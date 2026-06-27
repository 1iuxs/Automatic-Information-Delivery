
import torch
import numpy as np
from train_eval import train,test
from importlib import import_module
import argparse
from utils import build_dataset,build_iterator,build_time


# parser = argparse.ArgumentParser(description="Chinese Text Classification")
# parser.add_argument("--model", type=str, required=True, help="choose a model: bert")
# args = parser.parse_args()
# args.model = "bert"


if __name__ == '__main__':
    dataset = "data"
    # if args.model == "bert":
    model_name = "bert"
    x = import_module("models." + model_name)

    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    print("loading data")
    train_data, dev_data, test_data = build_dataset.build_dataset(config)




    train_iter = build_iterator.build_iterator(train_data, config)
    dev_iter = build_iterator.build_iterator(dev_data, config)
    test_iter = build_iterator.build_iterator(test_data, config)
    print(config.num_classes)
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter)
    test(config, model, test_iter)






















