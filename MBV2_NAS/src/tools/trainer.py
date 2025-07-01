import logging
import torch.nn as nn
import torch.optim as optim
from tools.transform import Cutout, mixup_data, mixup_criterion
from tools.evaluation import evaluate

# ============ 7) 最终训练 (启用 Cutout / Mixup / Label Smoothing / Cosine LR) ============


def train_and_eval(model, train_loader, test_loader, device,
                   epochs=50, lr=0.01,
                   mixup_alpha=1.0,
                   label_smoothing=0.1, weight_decay=5e-4):
    """
    对最终搜索到的结构进行完整训练并评估其在测试集上的Top-1/Top-5准确率。
    - 启用 mixup & label smoothing & cutout(在DataLoader里) & CosineAnnealingLR
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # ============ mixup处理 ============
            if mixup_alpha > 0.:
                pass
                # mixed_x, y_a, y_b, lam = mixup_data(
                #     inputs, labels, alpha=mixup_alpha)
                # outputs = model(mixed_x)
                # loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                # 不使用mixup
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            # 仅用于查看train top1（对mixup只是近似统计）
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc_top1 = correct_top1 / total if total > 0 else 0.

        # 在测试集上计算Top-1 / Top-5
        test_top1, test_top5 = evaluate(model, test_loader, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Loss={train_loss:.3f}, "
                     f"Train@1={train_acc_top1*100:.2f}%, "
                     f"Test@1={test_top1*100:.2f}%, Test@5={test_top5*100:.2f}%"
                     f"lr: {current_lr:.3f}")

    final_top1, final_top5 = evaluate(model, test_loader, device)
    logging.info(
        f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    return final_top1
