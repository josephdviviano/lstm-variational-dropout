from lvd.model import LSTMModel, SampleDrop
import torch

DATA_SHAPE = (20, 5, 10)


def test_sampledrop():

    dropper = SampleDrop(dropout=0.5)

    def run(dropper):
        data = torch.autograd.Variable(torch.randn(DATA_SHAPE))
        dropper.set_weights(data[0, ...])
        samples = []

        for x in data:
            samples.append(dropper(x))

        return (torch.stack(samples, dim=0).sum(0), dropper._mask)

    run_1, mask_1 = run(dropper)
    run_2, mask_2 = run(dropper)

    # All non-zero elements are constant for all timesteps.
    assert sum(run_1[mask_1 == 0]) == 0  # All timesteps are consistnetly
    assert sum(run_2[mask_2 == 0]) == 0  # masked.
    assert sum(sum(mask_1 != mask_2)) != 0  # The masks must be different.


def test_lstm_mask():

    model = LSTMModel(input_size=10, n_layers=1, hidden_size=10,
    	              dropout_i=0.5, dropout_h=0.5)

    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    mse = torch.nn.MSELoss()

    def run(model, optim, loss_fxn):
        optim.zero_grad()
        target = torch.autograd.Variable(torch.zeros(DATA_SHAPE[1:]))
        outputs = model(torch.autograd.Variable(torch.randn(DATA_SHAPE)))
        i_mask = model._input_drop._mask
        h_mask = model._state_drop._mask
        h = outputs['ht'][-1, ...]
        loss = loss_fxn(h, target)
        loss.backward()
        optim.step()

        return (i_mask, h_mask)

    i0_mask, h0_mask = run(model, optim, mse)
    i1_mask, h1_mask = run(model, optim, mse)

    assert sum(sum(i1_mask != i0_mask)) > 0
    assert sum(sum(h1_mask != h0_mask)) > 0


def test_portfolio():

    data = torch.autograd.Variable(torch.randn(DATA_SHAPE_BATCH))
    model = LSTMModel(
        input_size=10, n_layers=1, hidden_size=10, output_size=7,
        dropout_i=0.5, dropout_h=0.5)

    output = model(data)

    h = output['ht'][-1, :, :]  # Output of the final timepoint.
    mask = torch.tensor([0, 1, 0, 0, 1, 1, 0]).float()  # 7 elements.
    portfolio = model.predict_portfolio(h, mask)

    def _round(data, n_digits):
        return (data * 10**n_digits).round() / (10**n_digits)

    assert all([x == 1. for x in _round(portfolio.sum(1), 5)])
    assert len(portfolio.sum(1) == 5)  # Batch dimension.
    assert all((mask == 0) == (portfolio.sum(0) == 0))


if __name__ == "__main__":
    test_sampledrop()
    test_lstm_mask()
