from PIL import Image
import time
import csv
import pandas as pd
import operator
import functools

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import argparse


DEBUG = False
TOTAL_EPOCHS = 5


def debug_print(*strings):
    if not DEBUG:
        return
    if debug_print.rank == 0:
        print("[server]", *strings)
    else:
        print("[work{:2d}]".format(debug_print.rank), *strings)


class KaggleAmazonDataset(Dataset):
    def __init__(
        self, csv_path, img_path, img_ext, transform=None,
        rank=None, total_workers=None
    ):

        # decrement total_workers for the server; keep ranks between 0 ..
        # total_workers-1 because the math is easier that way
        if rank:
            self.rank = rank - 1
            self.total_workers = total_workers - 1
        else:
            self.rank = self.total_workers = None

        tmp_df = pd.read_csv(csv_path)

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags']
        debug_print("Done loading data")

        self.num_labels = 17

    def __getitem__(self, index):
        if self.rank is not None:
            # only get every Nth image
            index = self.total_workers * index + self.rank

        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_ids = self.y_train[index].split()
        label_ids = [int(s) for s in label_ids]
        label = torch.zeros(self.num_labels)
        label[label_ids] = 1
        return img, label

    def __len__(self):
        if self.rank is None:
            return len(self.X_train)
        else:
            # only get every Nth image
            n = len(self.X_train) // self.total_workers
            # might have some leftover...
            if self.rank < len(self.X_train) % self.total_workers:
                n += 1
            return n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, train_loader, model, criterion):
    losses = AverageMeter()
    precisions_1 = AverageMeter()
    precisions_3 = AverageMeter()
    topk = 3

    model.train()

    t_train = time.monotonic()
    num_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # zero out the gradients
        for p in model.parameters():
            p.grad.data.zero_()

        data = data.to(device=device)
        target = target.to(device=device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # send the gradients to the server
        send_gradient_to_server(model)

        _, predicted = output.topk(topk, 1, True, True)
        batch_size = target.size(0)
        num_samples += batch_size
        for i in range(batch_size):
            prec_3 = target[i][predicted[i, :3]].sum()
            prec_1 = target[i][predicted[i][0]]
            count_3 = min(target[i].sum(), 3)
            precisions_1.update(prec_1)
            precisions_3.update(prec_3 / count_3)

        # Update of averaged metrics
        losses.update(loss.item(), num_samples)

    train_time = time.monotonic() - t_train
    return (
        train_time,
        num_samples,
        losses.avg,
        precisions_1.avg,
        precisions_3.avg
    )


@functools.lru_cache(None)
def get_total_size(model):
    total_size = 0
    for t in model.parameters():
        t_size = functools.reduce(operator.mul, t.shape)
        total_size += t_size
    return total_size


def flatten_many_tensors(list_of_tensors, total_size):
    data = torch.zeros(total_size)
    idx = 0
    for t in list_of_tensors:
        t_size = functools.reduce(operator.mul, t.shape)
        data[idx:idx + t_size] = t.data.view(-1)
        idx += t_size
    assert idx == total_size
    return data


def unflatten_into_tensors(data, list_of_tensors):
    idx = 0
    for t in list_of_tensors:
        t_size = functools.reduce(operator.mul, t.shape)
        t.data = data[idx:idx + t_size].view_as(t.data)
        idx += t_size
    assert idx == data.size(0)


def send_model_to_worker(model, worker_id=None):
    assert worker_id is not None
    # flatten it all so we can send in one go
    data = flatten_many_tensors(
        [p for p in model.parameters()],
        get_total_size(model)
    )
    # data now contains the entire model flattened
    dist.send(data, worker_id)


def receive_model_from_server(model):
    debug_print("waiting on data")
    data = torch.zeros(get_total_size(model))
    dist.recv(data, src=0)
    debug_print("got data, dumping into model")
    unflatten_into_tensors(
        data,
        [p for p in model.parameters()]
    )


def receive_gradients(model):
    debug_print("waiting on msg from anyone")
    msg = torch.zeros(1)
    worker_id = dist.recv(msg)
    debug_print("got msg from", worker_id, "waiting on grad")
    data = torch.zeros(get_total_size(model))
    dist.recv(data, src=worker_id)
    debug_print("got grad from", worker_id)
    unflatten_into_tensors(
        data,
        [p.grad for p in model.parameters()],
    )
    return worker_id, msg.item()


def send_gradient_to_server(model):
    # send the server the msg saying we're not done yet
    dist.send(torch.zeros(1), 0)
    data = flatten_many_tensors(
        [p.grad for p in model.parameters()],
        get_total_size(model)
    )
    dist.send(data, 0)
    receive_model_from_server(model)


def force_model_update(model, is_last_epoch=False):
    # let the server know if this is the last one
    debug_print("asking for module_update from server")
    dist.send(torch.Tensor([is_last_epoch]), 0)
    debug_print("sending server zero grad")
    fake_grad = torch.zeros(get_total_size(model))
    dist.send(fake_grad, 0)
    debug_print("receiving model from server")
    receive_model_from_server(model)
    debug_print("finished updating model")


def init_model(device):
    model = Net().to(device=device)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter.data)
    return model


def main_parameter_server(args):
    debug_print("Starting server")
    worker_group = dist.new_group(ranks=list(range(1, args.world_size)))
    debug_print("made group")
    # initialize the model, and zero out the grads
    model = init_model(args.device)

    debug_print('Optimizer:', args.opt)
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif args.opt == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    elif args.opt == 'nesterov':
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=True
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=False
        )

    # Send a zero gradient and receive the inital model.
    optimizer.zero_grad()

    finished_workers = set()

    while len(finished_workers) < args.world_size - 1:
        optimizer.zero_grad()
        worker_id, worker_finished = receive_gradients(model)
        debug_print("received msg from ", worker_id, worker_finished)
        optimizer.step()
        debug_print("calling send_model_to_worker ", worker_id)
        send_model_to_worker(model, worker_id)
        if worker_finished:
            assert worker_id not in finished_workers
            finished_workers.add(worker_id)
    debug_print("finishing")


def main_worker(args):
    debug_print("starting worker:", args.rank, list(range(1, args.world_size)))
    worker_group = dist.new_group(ranks=list(range(1, args.world_size)))
    debug_print("found worker group")

    model = init_model(args.device)

    DATA_PATH = args.data_path
    IMG_PATH = DATA_PATH + 'train-jpg/'
    IMG_EXT = '.jpg'
    TRAIN_DATA = DATA_PATH + 'train.csv'
    batch_size = 250

    torch.manual_seed(123)

    criterion = nn.BCELoss().to(device=args.device)
    transformations = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor()]
    )
    dset_train = KaggleAmazonDataset(
        TRAIN_DATA,
        IMG_PATH,
        IMG_EXT,
        transformations,
        rank=args.rank,
        total_workers=args.world_size
    )
    train_loader = DataLoader(
        dset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers  # 1 for CUDA
        # pin_memory=True # CUDA only
    )

    # first get the model from the server
    force_model_update(model)
    # make sure we all start with the same parameters
    dist.barrier(worker_group)

    # Do the actual training.
    total_start_time = time.monotonic()
    train_times = []
    for epoch in range(TOTAL_EPOCHS):
        train_time, num_samples, loss, prec1, prec3 = train(
            epoch, train_loader, model, criterion
        )
        train_times.append(train_time)
        print(
            '{}, {}, {}, {}, {}'.format(
                args.rank,
                loss,
                prec1,
                prec3,
                train_time
            )
        )
        # synchronize across workers
        debug_print("Syncing ", args.rank)
        dist.barrier(worker_group)
        force_model_update(model, is_last_epoch=(epoch + 1 == TOTAL_EPOCHS))

    # final step, compute allreduce across workers
    dist.barrier(worker_group)
    stats = torch.Tensor([
        num_samples,
        loss * num_samples,
        prec1 * num_samples,
        prec3 * num_samples,
    ])
    dist.all_reduce(stats, group=worker_group)
    # make sure we average over the dataset, not by worker
    total_samples, total_loss, total_prec1, total_prec3 = stats
    full_loss = total_loss / total_samples
    full_prec1 = total_prec1 / total_samples
    full_prec3 = total_prec3 / total_samples

    total_time = (time.monotonic() - total_start_time)
    print(
        'final {}, {}, {}, {}, {}'.format(
            args.rank,
            full_loss,
            full_prec1,
            full_prec3,
            total_time,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--use_cuda', type=str, default='false',
                        help='Use CUDA if available')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str,
                        default='/scratch/am9031/CSCI-GA.3033-022/lab2/kaggleamazon/',
                        help='Data path')
    parser.add_argument('--opt', type=str, default='adam',
                        help='NN optimizer (Examples: adam, rmsprop, ...)')

    args = parser.parse_args()
    device = None
    if args.use_cuda.lower() == 'true' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args.device = device

    dist.init_process_group(
        backend="mpi",
    )

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    debug_print.rank = args.rank

    if args.rank == 0:
        main_parameter_server(args)
    else:
        main_worker(args)
