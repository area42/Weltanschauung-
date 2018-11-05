import torch
from curses import wrapper

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

headerrows = 5
cols = 5
colw = 35

class wv:
    """
    Weltantschauung implements a worldview (wv) and can use the world view
    to react to statements and to learn from replies to its own worldview
    """
    # dimension of input and output vector
    default_wv_dimension = 16

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
                    ).to(device)
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


def learning_loop(stdscr):
    stdscr.clear()
    height,width = stdscr.getmaxyx()
    ewidth = divmod(width,colw)[0]*colw
    #inital statement
    start = "alice"
    for c in range(500):
        s = torch.randn(alice.wv_dimension)
        turn = start
        if start == "alice":
            ar = alice.reply(s)
            br = bob.reply(s)
            start = "bob"
        else:
            ar = alice.reply(s)
            br = bob.reply(s)
            start = "alice"

        stdscr.addstr(1,1,"conversation: %-5d started by %5s, s distance %s" % (c,turn,ar.dist(br).__str__()))

        for t in range(100):
            stdscr.addstr(2,1,"iteration   : %-5d (%5s)" % (t,turn))
            if turn == "alice":
                br = bob.reply(ar)
                lossalice = alice.listenAndLearn(ar,br)
                stdscr.addstr(3,1,"alice loss  : %-25s" % (lossalice.__str__()))
                turn = "bob"
            else:
                ar = alice.reply(br)
                lossbob = bob.listenAndLearn(br,ar)
                stdscr.addstr(4,1,"bob loss    : %-25s" % (lossbob.__str__()))
                turn = "alice"

            if t == 2:
                my,mx=divmod(c,height-headerrows)
                wc,cy=divmod(my*colw,ewidth)

                cmy,cmx=divmod(c+1,height-headerrows)
                cwc,ccy=divmod(cmy*colw,ewidth)

                if lossbob > lossalice:
                    loser = "b"
                else:
                    loser = "a"
                stdscr.addstr(headerrows+mx,cy,"%1s %s %d : %-10f (%10f)" % (turn[0],loser,c,max(lossbob,lossalice),abs(lossbob-lossalice)))
                stdscr.addstr(headerrows+cmx,ccy,"%35s" % (" "))
            stdscr.refresh()

learning_rate = 1e-6
alice = wv("alice",learning_rate)
bob = wv("bob",learning_rate)

wrapper(learning_loop)


s = torch.randn(alice.wv_dimension)
ar = alice.reply(s)
br = bob.reply(s)
print(ar,br,ar.dist(br))
