import os
import random
from tqdm import tqdm
import numpy as np
import torch.utils.data
from torch import nn
from sklearn.metrics import f1_score
from my_dataset import combine
from my_resnet18 import MyResnet18
from my_resvit import NewVit
from my_resvit_leaky import NewVit1
from my_reslstmvit import NewVit2
from my_resvit_leaky_3L import NewVit3
from my_resvit_leaky_6L import NewVit4

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(epoch, model, train_loader, test_loader):
    # 训练模式
    correct = 0
    total = 0
    sum_loss = 0

    model.train()
    loop = tqdm(train_loader, desc='Train')
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_ = torch.argmax(y_pred, dim=1)
            correct += (y_ == y).sum().item()
            total += y.size(0)
            # sum_loss += loss.item()
            running_loss = loss.item()
            running_acc = correct / total

        # 更新训练信息
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        loop.set_postfix(loss=running_loss, acc=running_acc)

    epoch_loss = running_loss
    epoch_acc = correct / total

    # 测试模式
    test_correct = 0
    test_total = 0
    test_loss = 0
    y_true_list = []  # 用于存储真实标签
    y_pred_list = []  # 用于存储预测标签
    model.eval()
    with torch.no_grad():

        loop2 = tqdm(test_loader, desc='Test')
        for x, y in loop2:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_ = torch.argmax(y_pred, dim=1)
            test_correct += (y_ == y).sum().item()
            test_total += y.size(0)
            test_loss = loss.item()
            test_running_loss = test_loss
            test_running_acc = test_correct / test_total

            y_true_list.extend(y.cpu().numpy())
            y_pred_list.extend(y_.cpu().numpy())
            f1 = f1_score(y_true_list, y_pred_list, average='macro')
            # 更新测试信息
            loop2.set_postfix(loss=test_running_loss, acc=test_running_acc, f1=f1)

    f1 = f1_score(y_true_list, y_pred_list, average='macro')
    test_epoch_loss = test_loss
    test_epoch_acc = test_correct / test_total

    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc, f1


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    seed_torch()

    train_set = combine(root_dir="testing_set")
    test_set = combine(root_dir="validation_set")

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, 32, num_workers=8, pin_memory=True, shuffle=True)

    # 搭建模型
    model = MyResnet18()
    model = model.to(device)

    train_size = len(train_set)
    test_size = len(test_set)
    print("训练集长度为{}".format(train_size))
    print("测试集长度为{}".format(test_size))

    lr = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # 训练
    epochs = 100
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = 0
    best_test_f1 = 0
    for epoch in range(epochs):
        epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc, f1 = train_one_epoch(epoch, model, train_dataloader, test_dataloader)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        if f1 > best_test_f1:
            best_test_f1 = f1
            best_test_acc = test_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}/{epochs}, best_acc: {best_test_acc:.4f}, best_f1: {best_test_f1:.4f}')



