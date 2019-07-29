import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

class Slider:
    def __init__(self, data, n):
        assert len(data) >= n, 'length of input must be bigger than or equal the slider width'
        self.data = data
        self.n = n
        self.i = 0

    def __len__(self):
        return len(self.data) - self.n + 1

    def __iter__(self):
        return self

    def __next__(self):
        while self.i < len(self):
            self.i += 1
            return self.data[self.i - 1: self.i - 1 + self.n]
        self.i = 0
        raise StopIteration


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            skip_first_n_epochs (int): How many eopchs to skip at the beginning
                            Default: 0
        """
        # kwargs
        self.skip_first_n_epochs = kwargs.pop('skip_first_n_epochs', 0)
        self.times_called = 0

        if kwargs:
            print('Unknown kwarg', kwargs)
            raise KeyError

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    @staticmethod
    def get_model_name(model):
        try:
            return model.model_name
        except AttributeError:
            return ''

    def __repr__(self):
        return f'EarlyStopping: patience={self.patience}, verbose:{self.verbose}'

    def __call__(self, val_loss, model):
        if self.times_called < self.skip_first_n_epochs:
            print('\n\tSkipping the check for early stopping', end='')
            self.times_called += 1
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'\n\tEarlyStopping counter: {self.counter} out of {self.patience}',
                  end='')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'\n\tValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',
                  end='')
        torch.save(model.state_dict(), self.get_model_name(model) + 'checkpoint.pt')
        self.val_loss_min = val_loss


class Train:

    def __init__(self, model, **kwargs):
        """
        kwargs:            Type    Default
            print_every   (int)    [100]
            quiet_mode    (bool)   [False]
            early_stop    (bool)   [False]
            max_norm      (float)  [5.0]
            epochs        (int)    [10]
            optimizer ............ [Adam(lr=0.001)]
            crit      ............ [CrossEntropyLoss]
            device    ............ ["cuda:0" if available else "cpu"]
        """
        self.new_epoch = False
        self.model = model

        # kwargs
        self.print_every = kwargs.pop('print_every', 100)
        self.quiet_mode = kwargs.pop('quiet_mode', False)
        self.early_stop = kwargs.pop('early_stop', False)
        self.max_norm = kwargs.pop('max_norm', 5.)
        self.epochs = kwargs.pop('epochs', 10)
        self.optimizer = kwargs.pop('optimizer', torch.optim.Adam(self.model.parameters(), lr=0.001))
        self.crit = kwargs.pop('crit', nn.CrossEntropyLoss())
        self.device = kwargs.pop('device', self.__device__())

        if kwargs:
            print('Unknown kwargs', kwargs)

    def __check_params__(self, train_loader):
        assert len(train_loader) >= self.print_every, f'{len(train_loader)} must be bigger than {self.print_every}'

    @staticmethod
    def __device__():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def optim_step(self, xs, ys, h0):
        """performs backprop and returns loss, accuracy, hn"""
        self.optimizer.zero_grad()
        ys_hat, hn = self.model(xs, h0)
        hn = tuple([h.detach() for h in hn])
        loss = self.crit(ys_hat, ys)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        return loss.item(), (ys == ys_hat.max(dim=1)[1]).to(torch.float).mean(), hn

    def print_train_msg(self, epoch, running_loss, running_accuracy,
                        test_loss, test_accuracy):
        print_msg = (f'\rEpoch: {epoch + 1}/{self.epochs} ' +
                     f'Train Loss: {running_loss / self.print_every:.3f} ' +
                     f'Train Acc.: {running_accuracy / self.print_every:.3f} ' +
                     f'Test loss: {test_loss:.3f} ' +
                     f'Test Acc.: {test_accuracy:.3f}')
        if not self.quiet_mode:
            if self.new_epoch:
                print('\n' + print_msg, end='')
                self.new_epoch = False
            else:
                print(print_msg, end='')

    def validation(self, test_loader):
        running_loss, running_acc = 0., 0.
        h = self.model.init_hidden(test_loader.nBatchSize)
        for xs, ys in test_loader:
            xs, ys = xs.to(self.device), ys.to(self.device)
            ys_hat, h = self.model(xs, h)
            h = tuple([_.detach() for _ in h])
            running_loss += self.crit(ys_hat, ys).item()
            running_acc += (ys == ys_hat.max(dim=1)[1]).to(torch.float).mean()
        return running_loss / len(test_loader), \
               running_acc / len(test_loader)

    def check_print(self, test_loader, epoch, running_loss, running_acc):
        self.model.eval()
        with torch.no_grad():
            test_loss, test_accuracy = self.validation(test_loader)
        self.model.train()
        self.print_train_msg(epoch, running_loss, running_acc, test_loss, test_accuracy)
        # test_accuracy is consumed for printing only
        # we return test_loss for history of test_loss
        return test_loss

    def __call__(self, train_loader, test_loader):
        self.__check_params__(train_loader)
        self.model.to(self.device)
        running_loss, running_acc = 0.0, 0.0
        train_loss_history, test_loss_history = [], []
        steps = 0
        for epoch in range(self.epochs):
            self.new_epoch = True
            h = self.model.init_hidden(train_loader.nBatchSize)
            for xs, ys in train_loader:
                steps += 1
                xs, ys = xs.to(self.device), ys.to(self.device)
                loss, accuracy, h = self.optim_step(xs, ys, h)
                running_loss += loss
                running_acc += accuracy
                if steps % self.print_every == 0:
                    test_loss = self.check_print(test_loader, epoch, running_loss, running_acc)
                    train_loss_history.append(running_loss / self.print_every)
                    test_loss_history.append(test_loss)
                    running_loss, running_acc = 0., 0.
            #                     if epoch == 0:
            #                         print('\n memory consumption is', torch.cuda.memory_allocated()/1e9)

            if self.early_stop:  # we want to check for early stop at the end of each epoch
                self.early_stop(test_loss, self.model)
                if self.early_stop.early_stop:
                    print('\nearly stopping', end='')
                    break
        print('\nFinished training')
        return train_loss_history, test_loss_history


# def predictChar(model, seq:str, h, encoder:dict, top_k=None) -> str:
#     decoder = {i: char for (char, i) in encoder.items()}
#     xs = one_hot_encode(np.array([[dSet.encoder[ch] for ch in seq]]), self.nLabel)
#     y = model(xs, h).squeeze()
#     assert y.size() == torch.Size([1])
#     return self.decoder[]
