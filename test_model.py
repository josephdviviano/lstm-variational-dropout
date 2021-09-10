from model import LSTMModel, SampleDrop
import torch

BATCH_SIZE = 32
SEQ_LEN = 20
N_FEATURES = 7
HID_SIZE = 5
DATA_SHAPE_BATCH = (BATCH_SIZE, SEQ_LEN, N_FEATURES)


def test_sampledrop():

    dropper = SampleDrop(dropout=0.5)

    def run(dropper):
        X = torch.autograd.Variable(torch.randn(DATA_SHAPE_BATCH))
        dropper.set_weights(X[:, -1, :])  # Remove time dimension.
        samples = []
        X = X.permute(1, 0, 2)  # [seq_len, batch_size, n_features].

        for x in X:
            samples.append(dropper(x))

        return (torch.stack(samples, dim=0).sum(0), dropper._mask)

    run_1, mask_1 = run(dropper)
    run_2, mask_2 = run(dropper)

    # All non-zero elements are constant for all timesteps.
    assert sum(run_1[mask_1 == 0]) == 0  # All timesteps are consistnetly
    assert sum(run_2[mask_2 == 0]) == 0  # masked.
    assert sum(sum(mask_1 != mask_2)) != 0  # The masks must be different.


def test_lstm_mask():

    model = LSTMModel(input_size=N_FEATURES, n_layers=1, hidden_size=HID_SIZE,
    	              dropout_i=0.5, dropout_h=0.5)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    mse = torch.nn.MSELoss()

    def run(model, optim, loss_fxn):
        optim.zero_grad()
        target = torch.autograd.Variable(torch.zeros(BATCH_SIZE, HID_SIZE))
        ht, _ = model(torch.autograd.Variable(torch.randn(DATA_SHAPE_BATCH)))

        i_mask = model._input_drop._mask
        h_mask = model._state_drop._mask

        h = ht[:, -1, :]  # Final time point: [batch_size, n_features].
        loss = loss_fxn(h, target)
        loss.backward()
        optim.step()

        return (i_mask, h_mask)

    i0_mask, h0_mask = run(model, optim, mse)
    i1_mask, h1_mask = run(model, optim, mse)

    assert sum(sum(i1_mask != i0_mask)) > 0
    assert sum(sum(h1_mask != h0_mask)) > 0


if __name__ == "__main__":
    test_sampledrop()
    test_lstm_mask()
