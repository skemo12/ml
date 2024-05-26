import torch
from torch import nn
from torch import optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class MyTrainer:
    def __init__(
        self,
        classifier,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
        epochs_no=20,
        optimizer='Adam',
        optimizer_kwargs=None,
    ):
        self.classifier = classifier
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        if optimizer == 'SGD':
            if optimizer_kwargs is None:
                optimizer_kwargs = {}
                optimizer_kwargs['lr'] = 0.001
                optimizer_kwargs['momentum'] = 0.9
            self.optimizer = optim.SGD(self.classifier.parameters(), **optimizer_kwargs)
        else:
            if optimizer_kwargs is None:
                self.optimizer = optim.Adam(self.classifier.parameters())
            else:
                self.optimizer = optim.Adam(self.classifier.parameters(), **optimizer_kwargs)

        self.epochs_no = epochs_no

    def train_model(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Training loop
        best_model_wts = self.classifier.state_dict()
        best_acc = 0.0
        iteration = 0
        train_losses = []
        for epoch in range(self.epochs_no):  # Number of epochs
            if self.epochs_no <= 20 or (epoch + 1) % 10 == 0:
                print("Epoch {}/{}".format(epoch + 1, self.epochs_no))
                print("-" * 10)
            train_loss = 0.0
            train_corrects = 0.0
            self.classifier = self.classifier.train()
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.classifier(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                iter_loss = loss.item()
                train_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                train_corrects += iter_corrects
                iteration += 1
            epoch_loss = train_loss / len(self.train_dataset)
            epoch_acc = train_corrects / len(self.train_dataset)
            epoch_acc = epoch_acc * 100
            train_losses.append(epoch_loss)

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = self.classifier.state_dict()
            if self.epochs_no <= 20 or (epoch + 1) % 10 == 0:
                print("epoch train loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
        return best_model_wts, train_losses

    def load_wts(self, best_model_wts):
        self.classifier.load_state_dict(best_model_wts)

    def test_model(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier.eval()
        predlist = torch.zeros(0, dtype=torch.long, device="cpu")
        lbllist = torch.zeros(0, dtype=torch.long, device="cpu")
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                predlist = torch.cat([predlist, predicted.view(-1).cpu()])
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

        return predlist, lbllist

    def metrics_calculator(self, predictions_list, labels_list):

        accuracy = accuracy_score(predictions_list.numpy(), labels_list.numpy())
        precision = precision_score(
            predictions_list.numpy(), labels_list.numpy(), average="weighted"
        )
        recall = recall_score(
            predictions_list.numpy(), labels_list.numpy(), average="weighted"
        )
        f1 = f1_score(predictions_list.numpy(), labels_list.numpy(), average="weighted")

        return accuracy, precision, recall, f1
