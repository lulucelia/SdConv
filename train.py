from dataset import video_train_loader, video_test_loader
import argparse
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from model.Models import SDConv
from model.Optim import ScheduledOptim
import numpy as np
from eval import visualize_pred, edit_score, f_score


# visualise the video segmentation result
def eval_pred(model, test_loader, device, n_class, dataset):
    was_training = model.training
    print(was_training)
    model.eval()

    current_video = 0
    overall_acc = 0.0
    edit = 0

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    ground_truth = []
    predicted_result = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            _, input_len, _ = inputs.size()
            idx = torch.arange(1, input_len + 1)
            attn = torch.arange(1, (input_len // 4) + 1)
            idx = idx.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)
            outputs = model(inputs, idx, attn)

            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels).long()
            _, predicted = torch.max(outputs, 1)

            running_corrects = (predicted == labels).sum()
            running_acc = running_corrects.double() / labels.size(0)
            overall_acc += running_acc

            # transfer the torch tensor to numpy array first, then normalized to p[0,1]
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()

            ground_truth.extend(np.array(labels.data))
            predicted_result.extend(np.array(predicted.data))

            # edit score
            edit += edit_score(predicted, labels)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(predicted, labels, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
            # visualize the comparison between the prediction and gt
            visualize_pred(predicted, labels, running_acc, current_video, n_class, dataset)
            current_video += 1

        eval_acc = overall_acc / len(test_loader)
        print('Evaluation Accuray: {:.4f}'.format(eval_acc))
        print('Edit: {:.4f}'.format((1.0 * edit) / len(test_loader)))

        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            print('F1@{:.2f} : {:.4f}'.format(overlap[s], f1))

        model.train(mode=was_training)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 0):
        # Iterate over data
        inputs, labels = data[0].to(device), data[1].to(device)
        _, input_len, _ = inputs.size()
        idx = torch.arange(1, input_len + 1)
        attn = torch.arange(1, (input_len // 4) + 1)
        idx = idx.unsqueeze(0).to(device)
        attn = attn.unsqueeze(0).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs, idx, attn)
        outputs = torch.squeeze(outputs)
        labels = torch.squeeze(labels).long()

        _, predicted = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step_and_update_lr()
        running_loss += loss.item()
        # count the correct result
        running_corrects = (predicted == labels).sum()
        # calculate the accuray each step and accumulate
        running_acc += running_corrects.double() / labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = running_acc / len(train_loader)

    return train_loss, train_accuracy


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, input_len, _ = inputs.size()
            idx = torch.arange(1, input_len + 1)
            attn = torch.arange(1, (input_len // 4) + 1)
            idx = idx.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)
            outputs = model(inputs, idx, attn)
            outputs = torch.squeeze(outputs)
            labels = torch.squeeze(labels).long()

            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects = (predicted == labels).sum()
            running_acc += running_corrects.double() / labels.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_acc / len(test_loader)

    return test_loss, test_accuracy


def train(model, train_loader, test_loader, criterion1, optimizer, device, config):
    since = time.time()
    best_acc = 0.0

    for epoch in range(config.epoch):
        print('Epoch {}/{}'.format(epoch, config.epoch - 1))
        print('-' * 10)

        train_loss, train_accu = train_epoch(
            model, train_loader, criterion1, optimizer, device
        )
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
            train_loss, train_accu))

        test_loss, test_accu = test_epoch(
            model, test_loader, criterion1, device
        )
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(
            test_loss, test_accu))

        if test_accu > best_acc:
            best_acc = test_accu
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def main():
    parser = argparse.ArgumentParser(description='video parsing')
    parser.add_argument('--dataset', default="split1", help='visualization path')

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    parser.add_argument('--cuda', default=True, help='use cuda?')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate. Default=0.01')

    parser.add_argument('--d_model', type=int, default=128, help='model dimension')
    parser.add_argument('--d_inner_hid', type=int, default=512, help='hidden_state dim')
    parser.add_argument('--d_k', type=int, default=16, help='key')
    parser.add_argument('--d_v', type=int, default=16, help='value')
    parser.add_argument('--n_classes', type=int, default=10, help='no.of surgical gestures')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_position', type=int, default=5000, help='max sequence len')

    # parameters for the dilated conv net
    parser.add_argument('--n_dlayers', type=int, default=10, help='no.of dilated layers')
    parser.add_argument('--num_f_maps', type=int, default=128, help='dilated layer output')

    parser.add_argument('--n_head', type=int, default=1, help='no.of attention head')
    parser.add_argument('--n_layers', type=int, default=1, help='no.of encoder layers')
    parser.add_argument('--n_warmup_steps', type=int, default=4000, help='optimization')

    parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')
    parser.add_argument('--data_root', type=str, default='JIGSAWS/data', help='data root path.')
    parser.add_argument('--train_label', type=str, default='JIGSAWS/split/train_split1.txt', help='train label path.')
    parser.add_argument('--test_label', type=str, default='JIGSAWS/split/test_split1.txt', help='test label path.')

    config = parser.parse_args()
    print(config)

    # ========= Loading Dataset =========#
    train_loader = video_train_loader(config)
    test_loader = video_test_loader(config)

    # Using Cuda
    device = torch.device("cuda" if config.cuda else "cpu")
    if config.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    print(device)

    # ========= Preparing Model =========#
    sdConv = SDConv(
        n_position=config.n_position,
        n_classes=config.n_classes,
        n_dlayers=config.n_dlayers,
        num_f_maps=config.num_f_maps,
        d_model=config.d_model,
        d_inner=config.d_inner_hid,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, sdConv.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        config.d_model, config.n_warmup_steps)

    sdConv = train(sdConv, train_loader, test_loader, criterion, optimizer, device, config)
    # save the best model
    # torch.save(transformer.state_dict(), './model/'+ config.dataset+ '.pth')

    eval_pred(sdConv, test_loader, device, config.n_classes, config.dataset)


if __name__ == '__main__':
    main()
