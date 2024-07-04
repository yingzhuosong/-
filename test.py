import torch
from sklearn.metrics import f1_score
from my_dataset import combine
from torch.utils.data import DataLoader
from my_resvit_leaky_3L import NewVit3
from my_resvit_leaky_6L import NewVit4
import torch.nn.functional as F

test_set = combine(root_dir="testing_set")
test_dataloader = DataLoader(test_set, 32, shuffle=False, num_workers=8, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = NewVit3().to(device)
weights1 = torch.load("best_model1.pth", map_location=device)
model1.load_state_dict(weights1)

model2 = NewVit3().to(device)
weights2 = torch.load("best_model2.pth", map_location=device)
model2.load_state_dict(weights2)

model3 = NewVit4().to(device)
weights3 = torch.load("best_model3.pth", map_location=device)
model3.load_state_dict(weights3)

test_correct = 0
test_total = 0
test_loss = 0
y_true_list = []  # 用于存储真实标签
y_pred_list = []  # 用于存储预测标签

model1.eval()
model2.eval()
model3.eval()
with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = 1*model1(x[:, 0:3, :])+0.15*model2(x[:, 3:6, :])+model3(x[:, 6:12, :])
        y_ = torch.argmax(y_pred, dim=1)
        test_correct += (y_ == y).sum().item()
        test_total += y.size(0)
        test_running_acc = test_correct / test_total

        y_true_list.extend(y.cpu().numpy())
        y_pred_list.extend(y_.cpu().numpy())
    f1 = f1_score(y_true_list, y_pred_list, average='macro')
    print(test_running_acc, f1)
