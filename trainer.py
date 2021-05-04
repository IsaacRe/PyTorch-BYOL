from tqdm.auto import tqdm
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import _create_model_training_folder


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader, val_loader, n_class_epochs=1):
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
        #                          num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        # manually set out dim for CIFAR10
        classification_head = nn.Linear(self.online_network.feature_dim, 10).cuda(self.device)
        xent = nn.CrossEntropyLoss()
        class_optim = torch.optim.SGD(classification_head.parameters(), lr=0.01, momentum=0.9)

        train_losses = []
        val_losses = []
        eval_iters = []
        acc = []
        for epoch_counter in range(self.max_epochs):

            pbar = tqdm(total=len(train_loader))
            for _, (batch_view_1, batch_view_2), _, in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                train_losses += [loss.item()]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

                pbar.update(1)
            pbar.close()

            total_loss = 0
            total_batches = 0
            with torch.no_grad():
                for _, (batch_view_1, batch_view_2), _ in val_loader:
                    total_loss += self.update(batch_view_1.to(self.device), batch_view_2.to(self.device)).item()
                    total_batches += 1

            val_losses += [total_loss / total_batches]
            eval_iters += [niter]

            np.savez('BYOL-loss.npz', train=np.array(train_losses), val=np.array(val_losses),
                     eval_iters=np.array(eval_iters))

            # train classification layer and evaluate accuracy
            print('Training classification head for %d epochs...' % n_class_epochs, end='')
            pbar = tqdm(total=n_class_epochs * len(train_loader))
            losses = []
            for class_epoch in range(n_class_epochs):
                for _, (x, _), y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.no_grad():
                        features = self.online_network.encoder(x)
                    out = classification_head(features.flatten(start_dim=1))
                    loss = xent(out, y)
                    losses += [loss.item()]
                    loss.backward()
                    class_optim.step()
                    class_optim.zero_grad()

                    pbar.update(1)
            pbar.close()

            print('done')
            print('Evaluating validation accuracy...')

            with torch.no_grad():
                correct = total = 0
                for _, (x, _), y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = classification_head(self.online_network.encoder(x).flatten(start_dim=1))
                    correct += (out.argmax(dim=1) == y).sum().item()
                    total += len(y)

                acc += [correct / total * 100.]

            print('done')
            np.savez('BYOL-acc.npz', val=np.array(acc), eval_iters=np.array(eval_iters))

            print("Completed {}/{} epochs. Acc={}, Loss={}".format(epoch_counter,
                                                                         self.max_epochs,
                                                                         acc[-1],
                                                                         val_losses[-1]))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def train_incr(self, train_loaders, val_loaders, n_class_epochs=1, train_class_dataloader=None,
                   eval_class_dataloader=None):
        #train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
        #                          num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        # manually set out dim for CIFAR10
        classification_head = nn.Linear(self.online_network.feature_dim, 10).cuda(self.device)
        xent = nn.CrossEntropyLoss()
        class_optim = torch.optim.SGD(classification_head.parameters(), lr=0.01, momentum=0.9)

        task_losses = []
        val_losses = []
        eval_iters = []
        task_accs = []
        for task_id, train_loader in enumerate(train_loaders):
            task_losses += [[]]
            for epoch_counter in range(self.max_epochs):

                pbar = tqdm(total=len(train_loader))
                for _, (batch_view_1, batch_view_2), _, in train_loader:

                    batch_view_1 = batch_view_1.to(self.device)
                    batch_view_2 = batch_view_2.to(self.device)

                    if niter == 0:
                        grid = torchvision.utils.make_grid(batch_view_1[:32])
                        self.writer.add_image('views_1', grid, global_step=niter)

                        grid = torchvision.utils.make_grid(batch_view_2[:32])
                        self.writer.add_image('views_2', grid, global_step=niter)

                    loss = self.update(batch_view_1, batch_view_2)
                    self.writer.add_scalar('loss', loss, global_step=niter)

                    task_losses[-1] += [loss.item()]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self._update_target_network_parameters()  # update the key encoder
                    niter += 1

                    pbar.update(1)
                pbar.close()

                total_loss = 0
                total_batches = 0
                with torch.no_grad():
                    for val_loader in val_loaders:
                        for _, (batch_view_1, batch_view_2), _ in val_loader:
                            total_loss += self.update(batch_view_1.to(self.device), batch_view_2.to(self.device)).item()
                            total_batches += 1

                val_losses += [total_loss / total_batches]
                eval_iters += [niter]

                np.savez('BYOL-incr-loss.npz', same_task=np.array(task_losses), val=np.array(val_losses),
                         eval_iters=np.array(eval_iters))

                # train classification layer and evaluate accuracy
                print('Training classification head...', end='')
                pbar = tqdm(total=n_class_epochs * len(train_class_dataloader))
                losses = []
                for class_epoch in range(n_class_epochs):
                    for _, (x, _), y in train_class_dataloader:
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.no_grad():
                            features = self.online_network.encoder(x)
                        out = classification_head(features.flatten(start_dim=1))
                        loss = xent(out, y)
                        losses += [loss.item()]
                        loss.backward()
                        class_optim.step()
                        class_optim.zero_grad()

                        pbar.update(1)
                pbar.close()

                print('done')
                print('Evaluating validation accuracy...')

                with torch.no_grad():
                    task_accs += [[]]
                    for val_loader in val_loaders:
                        correct = total = 0
                        for _, (x, _), y in val_loader:
                            x, y = x.to(self.device), y.to(self.device)
                            out = classification_head(self.online_network.encoder(x).flatten(start_dim=1))
                            correct += (out.argmax(dim=1) == y).sum().item()
                            total += len(y)

                        task_accs[-1] += [correct / total * 100.]

                print('done')
                np.savez('BYOL-incr-acc.npz', val=np.array(task_accs), eval_iters=np.array(eval_iters))

                print("Completed {}/{} epochs for task {}/{}".format(epoch_counter, self.max_epochs, task_id,
                                                                     len(train_loaders)))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
