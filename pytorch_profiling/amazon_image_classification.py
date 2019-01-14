import argparse
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from collections import defaultdict
from PIL import Image

# Image transformation
composed = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
     transforms.ToTensor()])


class AmazonTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=composed):
        """
        @param root_dir: path to "kaggleamazon/" root data directory
        @param transform: torchvision transformation object
        """
        self.labels = pd.read_csv(
            root_dir + "labels.csv", sep=",", index_col="id")
        self.train = pd.read_csv(root_dir + "train.csv", sep=",")
        self.img_dir = root_dir + "/train-jpg/"

        self.num_labels = len(self.labels)
        self.num_samples = len(self.train)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, ix):
        """
        Load and process image samples

        @param ix: index of training image in train.csv
        @return img: transformed image
        @return labels: label tensor
        """
        img_name = self.train.iloc[ix]["image_name"]
        img_tags = [int(tag) for tag in self.train.iloc[ix]["tags"].split()]

        # Resize image
        img = Image.open(self.img_dir + img_name + ".jpg")
        img = img.convert("RGB")  # Important to obtain 3 instead of 4 channels
        img = self.transform(img)

        # Create label tensor
        labels = torch.zeros([self.num_labels], dtype=torch.float32)
        index = torch.tensor(img_tags)
        labels.index_fill_(0, index, 1)  # Replace 0 with 1 for all true labels

        return img, labels


class Net(nn.Module):

    def __init__(self, num_labels=17):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        """
            1: torch.Size([250, 3, 32, 32])
            2: torch.Size([250, 32, 30, 30])    # 32 -> 30  (convolutions - 3)
            3: torch.Size([250, 32, 15, 15])    # 15 = 30/2 (max_pool - 2)
            4: torch.Size([250, 64, 13, 13])    # 15 -> 13  (convolutions - 3)
            5: torch.Size([250, 64, 6, 6])      # 6 = 13/2  (max_pool - 2)
            6: torch.Size([250, 2304])          # 2304 = 64 * 6 * 6
            7: torch.Size([250, 256])
            8: torch.Size([250, 256])
            9: torch.Size([250, 17])
        """
        # print("1:", x.shape)
        # print("2:", self.conv1(x).shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("3:", x.shape)
        # print("4:", self.conv2(x).shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print("5:", x.shape)
        x = x.view(-1, 2304)
        # print("6:", x.shape)
        x = F.relu(self.fc1(x))
        # print("7:", x.shape)
        x = F.dropout(x, training=self.training)
        # print("8:", x.shape)
        # print("9:", self.fc2(x).shape)
        return torch.sigmoid(self.fc2(x))


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Training function

    @param args: command line arguments
    @param model: Net()
    @param device: device type ("cpu" or "cuda")
    @param train loader: DataLoader()
    @param optimizer: optimizer object
    @param epoch: current epoch (1-5)

    @return: average batch metrics (loss value, precision@1, precision@3)
    """
    model.train()
    criterion = torch.nn.BCELoss()

    sample_cnt = 0  # Counter of samples processed so far
    running_loss = 0.0  # Running sum of batch loss value
    running_prec_1 = 0.0  # Running sum of batch precision@1
    running_prec_3 = 0.0  # Running sum of batch precision@3

    last_batch_ix = len(train_loader) - 1  # Last batch index

    load_time = 0.0
    batch_time = 0.0

    ts = time.monotonic()  # Helper timestamp

    # Iterate over mini-batches
    for batch_ix, (data, target) in enumerate(train_loader):

        # Get data load time
        load_time += (time.monotonic() - ts)

        assert(len(data) == len(target))
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)  # torch.Size([250, 17])

        loss = criterion(output.view(-1), target.view(-1))  # Flatten
        loss.backward()
        optimizer.step()

        # Get batch computation time
        batch_time += (time.monotonic() - ts)

        # Evaluation
        sample_cnt += len(data)
        loss_ = loss.item()
        running_loss += loss_ * len(data)
        prec_1 = precision_at_k(output, target, k=1)
        prec_3 = precision_at_k(output, target, k=3)
        running_prec_1 += prec_1 * len(data)
        running_prec_3 += prec_3 * len(data)

        # Logging
        if ((batch_ix % args.log_interval == 0) or (batch_ix == last_batch_ix)):
            print("batch :: ix: {:>3}; loss: {:.5f}; prec@1: {:.5f}; prec@3: {:.5f}".format(
                batch_ix, loss_, prec_1, prec_3))

        ts = time.monotonic()  # Reset timestamp

    # Average evaluation metrics across the entire training set
    avg_loss = running_loss / sample_cnt
    avg_prec_1 = running_prec_1 / sample_cnt
    avg_prec_3 = running_prec_3 / sample_cnt

    return avg_loss, avg_prec_1, avg_prec_3, load_time, batch_time


def precision_at_k(output, target, k):
    """
    Compute precision accuracy per batch

    @param output: batch_size x labels tensor
    @param target: batch_size x labels tensor
    @param k: integer of predictions to count

    @return running_sum: sum of precision accuracy of all batch samples
    """
    running_sum = 0.0
    with torch.no_grad():
        # Get topk predictions across all samples
        _, pred = output.topk(k, dim=-1)
        # t = vector of len(labels); p = indexes to check
        for t, p in zip(target, pred):
            running_sum += (t[p]).mean()
    return running_sum / len(target)


def construct_optimizer(args, model):
    """
    Choose optimization algorithm and construct optimizer object

    @param name: name of optimizer to use
    """
    valid = ["sgd", "sgd_nesterov", "adagrad", "adadelta", "adam"]
    if args.optim not in valid:
        print("Invalid optimizer, exiting")
        exit()

    if args.optim == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum)

    if args.optim == "sgd_nesterov":
        return optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum, nesterov=True)

    if args.optim == "adagrad":
        return optim.Adagrad(model.parameters())

    if args.optim == "adadelta":
        return optim.Adadelta(model.parameters())

    if args.optim == "adam":
        return optim.Adam(model.parameters())


def main(args):
    """
    Main function
    """
    # Use CUDA
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Fix random seed
    torch.manual_seed(args.seed)

    print("*" * 5)
    print("NUM_WORKERS: {:>3}".format(args.num_workers))
    print("OPTIMIZER: {}".format(args.optim))

    # Log times for number of workers
    logging = defaultdict(list)

    # Create and load data set
    params = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers,
    }
    train_set = AmazonTrainingDataset(root_dir="/scratch/mt3685/kaggleamazon/")
    # print(len(train_set))  # 30000
    train_loader = torch.utils.data.DataLoader(train_set, **params)

    # Model
    model = Net().to(device)

    # Optimizer
    optimizer = construct_optimizer(args, model)

    # Main training loop
    for epoch in range(1, args.epochs + 1):

        print("\n{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))

        start = time.monotonic()  # Start time
        loss, prec_1, prec_3, load_time, batch_time = train(
            args, model, device, train_loader, optimizer, epoch)
        end = time.monotonic()  # End time

        epoch_time = end - start
        logging["load"].append(load_time)
        logging["batch"].append(batch_time)
        logging["epoch"].append(epoch_time)

        print("\nagg. load exectime:\t {:.2f}".format(load_time))
        print("agg. batch exectime:\t {:.2f}".format(batch_time))
        print("agg. epoch exectime:\t {:.2f}".format(epoch_time))

        print("\navg. loss after {} epoch(s):\t {:.5f}".format(epoch, loss))
        print("avg. prec@1 after {} epoch(s):\t {:.5f}".format(epoch, prec_1))
        print("avg. prec@3 after {} epoch(s):\t {:.5f}".format(epoch, prec_3))

    print("*" * 5 + "\n")
    return logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch image classification")
    parser.add_argument("--batch-size", type=int, default=250, metavar="N",
                        help="mini-batch size for training (default: 250)")
    parser.add_argument("--num-workers", type=int, default=1, metavar="N",
                        help="number of worker threads (default: 1)")
    parser.add_argument("--shuffle", type=int, default=0, metavar="S",
                        help="shuffle training data (default: 0)")
    parser.add_argument("--epochs", type=int, default=5, metavar="E",
                        help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="momentum (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="L",
                        help="training log interval (default: 10)")
    parser.add_argument("--use-cuda", type=int, default=1, metavar="C",
                        help="use CUDA (default: 1)")
    parser.add_argument("--optim", type=str, default="sgd", metavar="O",
                        help="optimizer (default: sgd)")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/mt3685/kaggleamazon/", metavar="D",
                        help="path to data directory \
                        (default: /scratch/mt3685/kaggleamazon/)")
    args = parser.parse_args()

    logging = main(args)
    for key in sorted(logging.keys()):
        print("{}: {}\n".format(key, logging[key]))

"""
python = /home/am9031/anaconda3/bin/python

##### C1 and C2

$ sbatch launch_cpu.C1_C2.s  # or
$ python lab2.py

##### C3

$ sbatch launch_cpu.C3.epoch_1.s  # or
$ python lab2.py --num-workers 8 --epochs 1

##### C4

$ sbatch launch_cpu.C4.s  # or
$ python -m cProfile -o lab2.num-workers_1.prof lab2.py --num-workers 1
$ python -m cProfile -o lab2.num-workers_8.prof lab2.py --num-workers 8

$ pip install snakeviz
$ snakeviz lab2.num-workers_1.prof
$ snakeviz lab2.num-workers_8.prof

##### C5

$ sbatch launch_cpu.C5.s
$ sbatch launch_gpu.C5.s
$ sbatch launch_gpu.C5.optimizers.s
"""
