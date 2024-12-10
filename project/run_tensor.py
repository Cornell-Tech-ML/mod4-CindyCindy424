"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import time

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.

class Network(minitorch.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = Linear(2, hidden_size)
        self.layer2 = Linear(hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, 1)

    def forward(self, x):
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()


# class Linear(minitorch.Module):
#     def __init__(self, input_size: int, output_size: int) -> None:
#         super().__init__()
#         self.weights = RParam(input_size, output_size)
#         self.bias = RParam(output_size)

#     def forward(self, input_tensor: minitorch.Tensor) -> minitorch.Tensor:
#         output = input_tensor @ self.weights.value + self.bias.value
#         return output


class Linear(minitorch.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.weights = RParam(input_size, output_size)
        self.bias = RParam(output_size)
        self.output_size = output_size

    def forward(self, input_tensor: minitorch.Tensor) -> minitorch.Tensor:
        batch_size, input_size = input_tensor.shape
        weights_reshaped = self.weights.value.view(1, input_size, self.output_size)
        input_reshaped = input_tensor.view(batch_size, input_size, 1)
        output = (weights_reshaped * input_reshaped).sum(1)
        output = output.view(batch_size, self.output_size)
        bias_reshaped = self.bias.value.view(self.output_size)

        return output + bias_reshaped


def default_log_fn(epoch, total_loss, correct, losses, epoch_time):
    # print("Epoch ", epoch, " loss ", total_loss, "correct", correct)
    print(f"Epoch {epoch}: Total Loss: {total_loss:.4f} | Correct : {correct} | Epoch Runtime: {epoch_time:.2f} seconds")



class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()

            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
