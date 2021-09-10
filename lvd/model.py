import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, n_layers, hidden_size, 
                 dropout_i=0, dropout_h=0):
        """
        An LSTM model with Variational Dropout applied to the inputs and
        model activations. For details see Eq. 7 of

        A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks. Gal & Ghahramani, 2016.

        Note that this is equivalent to the weight-dropping scheme they
        propose in Eq. 5 (but not Eq. 6). 

        Args:
            input_size (int): input feature size.
            n_layers (int): number of LSTM layers.
            hidden_size (int): hidden layer size of all layers.
            dropout_i (float): dropout rate of the inputs (t).
            dropout_h (float): dropout rate of the state (t-1). 
        """
        super(LSTMModel, self).__init__()

        assert all([0 <= x < 1 for x in [dropout_i, dropout_h]])
        assert all([0 < x for x in [input_size, n_layers, hidden_size]])

        self._input_size = input_size
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._dropout_i = dropout_i
        self._dropout_h = dropout_h

        cells = []
        for i in range(n_layers):
            cells.append(nn.LSTMCell(input_size if i == 0 else hidden_size,
                                     hidden_size,
                                     bias=True))

        self._cells = nn.ModuleList(cells)
        self._input_drop = SampleDrop(dropout=self._dropout_i)
        self._state_drop = SampleDrop(dropout=self._dropout_h)

    @property
    def input_size(self):
        return self._input_size

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def dropout_i(self):
        return self._dropout_i

    @property
    def dropout_h(self):
        return self._dropout_h

    def _new_state(self, batch_size):
        """Initalizes states."""
        h = Variable(torch.zeros(batch_size, self._hidden_size))
        c = Variable(torch.zeros(batch_size, self._hidden_size))

        return (h, c)

    def forward(self, X):
        """Forward pass through the LSTM.

        Args:
            X (tensor): input with dimensions seq_len, batch_size, input_size

        Returns: Output dictionary containing ht from the final LSTM, and all 
            intermediate states.
        """
        seq_len, batch_size, input_size = X.shape
        states = []

        for cell in self._cells:
            ht, ct = [], []
            h, c = self._new_state(batch_size)

            self._input_drop.set_weights(X[0, ...])
            self._state_drop.set_weights(h)

            for sample in X:

                h, c = cell(self._input_drop(sample), (self._state_drop(h), c))
                ht.append(h)
                ct.append(c)

            ht = torch.stack(ht, dim=0)
            ct = torch.stack(ct, dim=0)
            states.append((ht, ct))

            X = ht

        return {'ht': ht, 'states': states}


class SampleDrop(nn.Module):
    """Applies dropout to input samples with a fixed mask."""
    def __init__(self, dropout=0):
        super().__init__()

        assert 0 <= dropout < 1
        self._mask = None
        self._dropout = dropout

    def set_weights(self, X):
        """Calculates a new dropout mask."""
        assert len(X.shape) == 2
        
        mask = Variable(torch.ones(X.size(0), X.size(1)), requires_grad=False)
        
        if X.is_cuda:
            mask = mask.cuda()

        self._mask = F.dropout(mask, p=self._dropout, training=self.training)

    def forward(self, X):
        """Applies dropout to the input X."""
        if not self.training or not self._dropout:
            return X
        else:
            return X * self._mask
