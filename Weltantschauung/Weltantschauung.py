import torch

class Weltantschauung:
    """
    Weltantschauung implements a worldview (wv) and can use the world view
    to react to statements and to learn from replies to its own worldview
    """
    # dimension of input and output vector
    default_wv_dimension = 16
    device = torch.device('cpu')

    def __init__(self, name, learning_rate, wv_dimension = None, wv_model = None):
        self.learning_rate = learning_rate
        self.name = name

        if wv_dimension is None:
            self.wv_dimension = self.default_wv_dimension
        else:
            self.wv_dimension = wv_dimension
        if wv_model is None:
            self.wv_model = torch.nn.Sequential(
                      torch.nn.Linear(self.wv_dimension, self.wv_dimension),
                      torch.nn.ReLU(),
                      torch.nn.Linear(self.wv_dimension, self.wv_dimension),
                    ).to(self.device)
            # self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
            self.loss_fn = torch.nn.L1Loss(reduction='none')
        else:
            self.wv_model = wv_model

    def reply(self,statement):
        return self.wv_model(statement)

    def listenAndLearn(self,s,r):
        r_pred = self.wv_model(s)
        # Compute and print loss. We pass Tensors containing the predicted reply and actual reply
        # and the loss function returns a Tensor containing the loss.
        loss = self.loss_fn(r_pred, r)
        #print(self.name,loss.item())

        # Zero the gradients before running the backward pass.
        self.wv_model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward(retain_graph=True)

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its data and gradients like we did before.
        with torch.no_grad():
            for param in self.wv_model.parameters():
                param.data -= self.learning_rate * param.grad

        return loss.item()

    def random_thought(self):
        return torch.randn(self.wv_dimension)
