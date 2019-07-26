class sgd_nesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates
