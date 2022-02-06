
# import torch.nn.init
from colorama import Fore
from dataset import EurosatDataset
from model import Classifier
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import warnings
# suppress warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    """
    Object for parsing command line strings into Python objects.
    """
    arg = argparse.ArgumentParser()

    arg.add_argument("--epoch", type=int, default=10)
    arg.add_argument(
        "--device", type=int, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'])
    arg.add_argument("--save_dir", type=str, default="./saved_models")
    arg.add_argument("--data_dir", type=str, default="data/EuroSAT/")

    arg.add_argument("--batch_size", type=int, default=128,
                     help="total number of batch size of labeled data")

    arg.add_argument("--eval_batch_size", type=int, default=256,
                     help="batch size of evaluation data loader")

    arg.add_argument("--criterion", type=str,
                     default="crossentropy", choices=['crossentropy'])
    arg.add_argument("--optimizer", type=str,
                     default="sgd", choices=['sgd', 'adam'])

    arg.add_argument("--learning_rate", type=float, default=0.01)
    arg.add_argument("--momentum", type=float, default=0.9)
    arg.add_argument("--weight_decay", type=float, default=5e-4)
    arg.add_argument("--dropout", type=float, default=0.0,
                     choices=[0.0, 0.5, 0.7, 0.9, 0.99, 0.999])
    arg.add_argument("--num_workers", type=int, default=3)
    arg.add_argument("--seed", type=int, default=42)

    return arg.parse_args()


def softmax(x):
    """
    Sofmax function.
    """
    return np.exp(x)/sum(np.exp(x))


def accuracy(gt_S, pred_S):
    """
    Get the accuracy of the model.

    Args:
        gt_S : Ground truth.
        pred_S : Prediction.

    Returns:
        Accuracy classification score.
    """
    _, alp = torch.max(torch.from_numpy(pred_S), 1)
    return accuracy_score(gt_S, np.asarray(alp))


def main(args):
    """
    Image Classification Model Training.
    """
    # pylint: disable=not-callable

    # Seed for reproducibility

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    writer = SummaryWriter(
        comment=f"Learn_{args.learning_rate}_Drop{args.dropout}")

    # Construct Dataset

    train_dataset = EurosatDataset(
        is_train=True, seed=args.seed, root_dir=args.data_dir)

    print(
        f'Number of data on Train Dataset is {len(train_dataset)}')

    test_dataset = EurosatDataset(
        is_train=False, seed=args.seed, root_dir=args.data_dir)

    print(
        f'Number of data on Test Dataset is {len(test_dataset)}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print(
        f'The images split in train and test is based on the seed {args.seed}')

    config = {
        "input_size": train_dataset.size[0],
        "labels": train_dataset.label_encoding,
        "loss_function": args.criterion,
        'dropout': args.dropout,
    }
    model = Classifier(config=config)
    model = model.to(args.device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer')

    test_losses = []
    train_losses = []
    test_acc = []
    train_acc = []

    best_acc = 0
    correct_pred = {classname: 0 for classname in train_dataset.label_encoding}
    total_pred = {classname: 0 for classname in train_dataset.label_encoding}

    print(f'Start training with {args.epoch} epochs')

    for e in range(1, args.epoch + 1):
        with tqdm(train_loader, unit="batch", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)) as train_epoch:
            model.train()
            for data, target in train_epoch:
                # get the inputs; train_epoch is a list of [data, target]
                data, target = (
                    Variable(data.to(args.device)),
                    Variable(target.to(args.device)),
                )
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model(data)
                loss = model.loss(output, target)
                loss.backward()
                optimizer.step()

                train_losses = np.append(train_losses, loss.item())
                pred = output.data.cpu().numpy()  # [0]
                pred = softmax(pred)
                gt = target.data.cpu().numpy()  # [0]
                train_acc = np.append(train_acc, accuracy(gt, pred))

                train_epoch.set_description(f"Epoch {e}")
                train_epoch.set_postfix(
                    loss=loss.item(), acc=train_acc[-1], mean_loss=np.mean(train_losses), mean_acc=np.mean(train_acc))

            writer.add_scalar('Loss/train', np.mean(train_losses), e)
            writer.add_scalar('Accuracy/train', np.mean(train_acc), e)

        with tqdm(test_loader, unit="batch", bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)) as test_epoch:
            model.eval()
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data, target in test_epoch:
                    data, target = (
                        Variable(data.to(args.device)),
                        Variable(target.to(args.device)),
                    )

                    # calculate outputs by running images through the network
                    output = model(data)
                    loss = model.loss(output, target)
                    test_losses = np.append(test_losses, loss.item())

                    # the class with the highest conf is what we choose as prediction
                    _, pred = torch.max(output, 1)

                    for label, prediction in zip(target, pred):
                        if label == prediction:
                            correct_pred[train_dataset.label_encoding[
                                label]] += 1
                        total_pred[train_dataset.label_encoding[label]] += 1

                    pred = output.data.cpu().numpy()  # [0]
                    pred = softmax(pred)
                    gt = target.data.cpu().numpy()  # [0]
                    test_acc = np.append(test_acc, accuracy(gt, pred))

                    test_epoch.set_description(f"Test")
                    test_epoch.set_postfix(
                        loss=loss.item(), acc=test_acc[-1], mean_loss=np.mean(test_losses), mean_acc=np.mean(test_acc))

            writer.add_scalar('Loss/test', np.mean(test_losses), e)
            writer.add_scalar('Accuracy/test', np.mean(test_acc), e)
            state = {
                'config': config,
                'state_dict': model.state_dict(),
            }
            if np.mean(test_acc) > best_acc:
                best_acc = np.mean(test_acc)
                torch.save(state,
                           f"{args.save_dir}/model_best.pth")

            torch.save(state,
                       f"{args.save_dir}/model_epoch{e}_acc{round(np.mean(test_acc),2)}.pth")
            for classname, correct_count in correct_pred.items():
                t_accuracy = 100 * float(correct_count) / total_pred[classname]
                # print(f'Accuracy for class: {classname:5s} is {t_accuracy:.1f} %')
                writer.add_scalar(f"Accuracy/{classname}", t_accuracy, e)

    print('Finished Training')
    print(f'Saving the model on {args.save_dir}')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
