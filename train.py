class SimpleLossCompute:
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        loss = self.criterion(x.view(-1, x.size(-1)),
                              y.view(-1))
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item()
