import time
import torch
import matplotlib.pyplot as plt
import numpy as np




def normalize(x, y):
    m = np.array([x, y])

    mean = m.mean()
    variance = m.var()

    m_normalized = (m - mean) / variance

    x_normalized = m_normalized[0]
    y_normalized = m_normalized[1]

    return x_normalized, y_normalized, mean, variance


# We will build nn to replace this model
def model(x, w, b):
    """The model to predict temperature"""
    y = w * x + b
    return y


# Pytorch comes with a number of predifined loss functions you can just import
# For example: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
# But MSE is basically what you see below

def loss_fn(y, y_hat):
    """The loss function used. Less loss is what we want"""
    squared_diffs = (y - y_hat) ** 2
    mse = squared_diffs.mean()
    return mse


# Before learning about backprop, we are going to do things manually. Concretely
# we are going to calculate the partial derivatives analytically

def dloss_fn(y, y_hat):
    dLoss_y = 2 * (y - y_hat)
    return dLoss_y


def dmodel_dw(x, w, b):
    return x


def dmodel_db(x, w, b):
    return 1.0


# Using the partial derivatives and the chain rule, we can compute a gradient, again
# analytically. We will never do this again. We will use backprop algorithm, using pytorch.


def grad_fn(x_hat, y_hat, my_model, w, b):
    dloss_dtp = dloss_fn(my_model, y_hat)
    dloss_dw = dloss_dtp * dmodel_dw(x_hat, w, b)
    dloss_db = dloss_dtp * dmodel_db(x_hat, w, b)
    return torch.stack([dloss_dw.sum() / my_model.size(0), dloss_db.sum() / my_model.size(0)])


# Now the training loop, which contains the main steps. Once we learn to do this in pytorch
# we will:
# 1. Use pytorch modules to build network architectures
# 2. Use backprop to calculate partial derivatives
# 3. Use built in loss functions
# 4. Use built in optimizers (gradient descent but also others).

losses = []


def training_loop(n_epochs, learning_rate, params, x_hat, y_hat, print_params=True):
    for epoch in range(1, n_epochs + 1):

        # Params we want to fit
        w, b = params

        # Setting the model we will use (a simple linear equation)
        y = model(x_hat, w, b)

        # Setting our gradient vector
        gradient_vector = grad_fn(x_hat, y_hat, y, w, b)

        # Setting our Gradient Descent step
        params = params - learning_rate * gradient_vector

        loss = loss_fn(y, y_hat)
        losses.append(loss)

        if epoch in {1, 2, 3, 10, 11, 99, 100, 1000, 4000, 5000}:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', gradient_vector)
                print('    Loss:  ', loss)
        if epoch in {4, 12, 101}:
            print('...')

        if not torch.isfinite(loss).all():
            break  # <3>

    return params


def main():
    x_hat = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2,
             2.3, 2.4, 2.5]
    y_hat = [0.27, 0.3, 0.38, 0.36, 0.39, 0.42, 0.45, 0.40, 0.45, 0.54, 0.57, 0.6, 0.60, 0.66, 0.69, 0.72, 0.75, 0.78,
             0.81, 0.84, 0.87, 0.9, 0.93, 0.96, 0.99]

    x_hat = torch.tensor(x_hat)
    y_hat = torch.tensor(y_hat)

    params = training_loop(
        n_epochs=1000,
        learning_rate=1e-2,
        params=torch.rand(2),
        x_hat=x_hat,
        y_hat=y_hat)

    my_model = model(x_hat, *params)

    fig = plt.figure(dpi=600)
    plt.xlabel("Inputs observed")
    plt.ylabel("Outputs observed")
    plt.plot(x_hat.numpy(), my_model.detach().numpy())
    plt.plot(x_hat.numpy(), y_hat.numpy(), 'o')
    plt.savefig("temp_unknown_plot.png", format="png")  # bookskip

    fig = plt.figure(dpi=600)
    plt.xlabel("Epoch")
    plt.ylabel("Computed Loss")
    plt.plot(losses)

    test_val = torch.tensor([2.8])

    # prediction = model(test_val,params[0],params[1])
    prediction = model(test_val, *params)

    print("\n Our model estimates a w = {0:.3f} and b = {1:.3f}".format(params[0], params[1]))

    print("\n Our model predicts for the input {0:.3f} the output {1:.3f} ".format(float(test_val.numpy()),
                                                                                   float(prediction.numpy())))


if __name__ == '__main__':
    torch.set_printoptions(edgeitems=2, linewidth=75)
    start = time.time()
    main()
    print(f"\nEllapsed time: {time.time() - start} s")
