import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from random import shuffle
from time import time

"""Feedforward neural network.

"""


class LabelConflict(Exception):
    """Raised when multilabel inputs are passed to single-label model.
    The opposite situation is not exceptional as it is possible all datapoints
    just happen to belong to exactly one class each.

    """
    pass


class QuadraticCost():  # done
    """Quadratic cost function Σ[(a-y)²].

    """
    @staticmethod
    def fn(a, y):
        return ((a - y)**2).sum(axis=1).ravel()

    @staticmethod
    def prime(a, y):
        return (2 * (a - y)).transpose((0, 2, 1))


class CrossEntropyCost():
    """Cross-Entropy cost function -y*log(a) - (1-y)*log(1-a).

    """

    @staticmethod
    def fn(a, y):
        return np.nan_to_num(
            - y * np.log(a)
            - (1.0 - y) * np.log(1.0 - a)
        ).sum(axis=1).ravel()

    @staticmethod
    def prime(a, y):
        return (a - y).transpose((0, 2, 1))


class Softmax():  # done
    """Softmax activation function eᶻ/(Σeᶻ)

    """
    @staticmethod
    def fn(z):
        z = _format_inputs(z)
        denom = np.exp(z).sum(axis=1)
        return np.exp(z)/denom.reshape(-1, 1, 1)  # (m,n,1)

    @staticmethod
    def prime(z, a):  # z is left here for the sake of consistency
        DS = -a @ a.transpose((0, 2, 1))
        for i in range(DS.shape[1]):  # faster than adding diag(a)
            DS[:, i, i] += a[:, i, 0]
        return DS


class Sigmoid():
    """Logistic sigmoid 1/(1+e⁻ᶻ).

    """
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z, a):
        return a*(1.0 - a)


class ReLU():
    """Linear rectification unit max(0,z).

    """
    @staticmethod
    def fn(z):
        return z*(z > 0.0)

    @staticmethod
    def prime(z, a):
        return (z > 0.0).astype(np.float64)


class MLP:

    def __init__(self,
                 layers,
                 cost=CrossEntropyCost,
                 activation=Sigmoid,
                 activation_last=Sigmoid):
        if len(layers) < 2:
            print(f"Not enough layers; gotta have 2 or more; ya got {len(layers)}.")
            exit(1)
        self.layers = layers
        self.weights = [  # rows correspond to out-units, columns to in-units
            np.random.randn(y, x) / np.sqrt(x)  # https://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
            for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        self.mean, self.std = None, None
        self.multilabel = None

        self.cost = cost
        self.last_sigma = activation_last  # last layer activation function
        self.sigma = activation  # all other layers

    @staticmethod
    def _infer_multilabel(labels):
        """Attempt to infer if labels are single-class or multi-class.
        Expects a list if ints OR a list of ndarrays filled with ones and zeros.
        Returns: True/False - is multi-label

        """
        if not isinstance(labels[0], np.ndarray):  # not an array, therefore int?
            return False
        for v in labels:
            if (v == 1).sum() != 1:  # no labels is possible only in multilabel
                return True
        return False

    def _normalize(self, features):
        """Normalizes ``features``, setting mean and std if necessary.

        """
        if self.mean is None or self.std is None:
            size = len(features) * self.layers[0]
            self.mean = sum(ex.mean() for ex in features) / len(features)
            self.std = np.sqrt(1 / (size) * sum([((ex - self.mean)**2).sum() for ex in features]))
        features = [((ex - self.mean) / self.std).ravel() for ex in features]
        return features

    def _unpack_data(self, data):
        """Separates input data to features and labels;
        Sets/uses normalization parameters.
        Returns: ``(m, features, labels)`` - tuple containing the number of
        data entries, a list of feature arrays and a list of target labels.

        """
        features, labels = zip(*data)
        # set normalization variables, normalize
        features = self._normalize(features)
        # infer if labels are multi-label
        multilabel = self._infer_multilabel(labels)
        if self.multilabel is False and multilabel is True:
            # that's a conflict, though the other way around would not be.
            raise LabelConflict("Multilabel data was passed to single-label model.")
        if self.multilabel is None:
            self.multilabel = multilabel
        if not isinstance(labels[0], np.ndarray):
            labels_new = []
            for l in labels:
                label = np.zeros(self.layers[-1], dtype=np.float64)
                label[l] = 1.0
                labels_new.append(label)
        else:
            labels_new = [l.ravel() for l in labels]

        return len(data), list(zip(features, labels_new))

    def SGD(self,
            data,
            eta,
            epochs=10,
            batch_size=1,
            test_data=None,
            training_evaluation=False,
            printout=True,
            plot=False):
        """Stochastic gradient descent.
        ``data`` - list of tuples, each tuple containing two ndarrays:
        training example and (target vector OR target int).
        ``eta`` - gradient step size.
        ``epochs`` - number of times to iterate through the entirety of
        training dataset.
        ``batch_size`` - size of the batch for a single step of stochastic
        descent.
        ``test_data`` - data to use to evaluate performance on previously
        unseen data; same format requirements as ``data``.
        ``training_evaluation`` - whether to evaluate performance on training
        data; same format requirements as ``data``.
        ``printout`` -  whether to print evaluation results.
        ``plot`` -  whether to plot evaluation results.

        """
        m, training_data = self._unpack_data(data)
        train_eval_progress = []

        if test_data is not None:  # SINGLE CLASS ONLY
            test_eval_progress = []
            _, test_data = self._unpack_data(test_data)
            eval_queries, eval_labels = map(list, zip(*test_data))

        # step 0: batching
        for epoch in range(epochs):
            start = time()

            shuffle(training_data)
            examples, targets = map(lambda x: np.array(list(x)), zip(*training_data))  # SAFETY

            example_views = (
                examples[offset:offset+batch_size]
                for offset in range(0, m, batch_size))
            target_views = (
                targets[offset:offset+batch_size]
                for offset in range(0, m, batch_size))

            # gradient descent
            for ex_v, target_v in zip(example_views, target_views):
                grads = self.backprop(ex_v, target_v)  # star of the show
                for l in range(len(grads)):
                    self.weights[l] -= eta * grads[l][0]
                    self.biases[l] -= eta * grads[l][1].T
            # at this point the descent step is done.

            # evaluate performance
            if test_data is not None:
                test_correct = self._evaluate(eval_queries, eval_labels, normalized=True)
                if printout:
                    print(f"Correct from testing dataset:  {test_correct}")
                test_eval_progress.append(test_correct)

            if training_evaluation:
                training_correct = self._evaluate(examples, targets, normalized=True)
                if printout:
                    print(f"Correct from training dataset: {training_correct}")
                train_eval_progress.append(training_correct)
            print(f"Epoch {epoch + 1}/{epochs}: {(time()-start):.2f}s\n")

        if plot and (training_evaluation or test_data is not None):
            fig, ax = plt.subplots()
            ax.set_xlim(1, epochs)
            x = range(1, epochs+1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid()
            ax.set_title("Network performance")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Prediction accuracy")
            if training_evaluation and test_data is not None:
                ax.set_ylim(min(train_eval_progress + test_eval_progress) // 0.1 * 0.1, 1)
                ax.plot(x, test_eval_progress, 'k-', label='Test data')
                ax.plot(x, train_eval_progress, 'k--', label='Training data')
            elif test_data is not None:
                ax.set_ylim(min(test_eval_progress) // 0.1 * 0.1, 1)
                ax.plot(x, test_eval_progress, 'k-', label='Test data')
            else:
                ax.set_ylim(min(train_eval_progress) // 0.1 * 0.1, 1)
                ax.plot(x, train_eval_progress, 'k--', label='Training data')
            ax.legend(loc='lower right')
            plt.show()

    def backprop(self, exs, targets):
        """Returns gradients for all weights and biases in order from first
        to last layer.

        """
        # print([w.min() for w in self.weights])
        # step 1: feedforward
        a = _format_inputs(exs)  # 3D stack of column-vector examples (m,||l₀||,1).
        targets = _format_inputs(targets)
        Z, A = [], []  # to be populated by ``self.feedforward()``.
        a = self.feedforward(a, Z, A)

        # step 2: backprop itself
        Grads = []

        # I'm separating this piece out for a separate activation function for
        # the last layer.
        nabla_C = self.cost.prime(a, targets)  # really a one-row Jacobian.
        # (m,1,||L||)
        DS = self.last_sigma.prime(Z[-1], A[-1])
        # (m,1,||L||) OR (m,||L||,||L||), depends on whether AF is (s->s)ⁿ or v->v
        if DS.shape[1] == DS.shape[2]:  # true v->v AF like Softmax.
            error = nabla_C @ DS  # (m,1,||L||)
        else:  # (s->s)ⁿ AF like logistic Sigmoid / ReLU.
            error = nabla_C * DS.transpose((0, 2, 1))  # Element-wise product is
            # equivalent to a matmul here for (s->s)ⁿ AF because the Jacobian
            # would be a diagonal matrix.

        grad_w = ((A[-2] @ error).transpose((0, 2, 1))).mean(axis=0)
        grad_b = error.mean(axis=0)
        # dC/dwᴸ, dC/dbᴸ
        Grads.append((
            grad_w,  # (1, ||L||, ||L-1||)
            grad_b   # (1, ||L||)
        ))
        # the A[-2] bit is meant to be a jacobian of z's w.r.t. weights of L,
        # and J. of z's w.r.t. biases is an identity matrix, so it's
        # simplified. Reversed order of multiplication is a consequence of
        # wacky dimensions, it's still valid.

        # now on to the rest of the layers
        for l in range(2, len(self.layers)):
            # starting at last layer
            # prepare error: tack on two new jacobians
            Dz = self.weights[-(l-1)]  # Jacobian of z's w.r.t. prev. layer a's.
            # just straight weights for current layer.

            DS = self.sigma.prime(Z[-l], A[-l])

            error = error @ Dz

            if DS.shape[1] == DS.shape[2]:  # v->v AF
                error = error @ DS
            else:  # (s->s)ⁿ AF
                error = error * DS.transpose((0, 2, 1))

            grad_w = ((A[-(l+1)] @ error).transpose((0, 2, 1))).mean(axis=0)
            grad_b = error.mean(axis=0)

            Grads.append((
                grad_w,  # (1, ||L||*||L-1||)
                grad_b   # (1, ||L||)
            ))
        # all donezo
        return Grads[::-1]  # originally it's last-to-first layer

    def feedforward(self, a, Z=None, A=None, weights=None, biases=None):
        """Feedforward the ``a``.
        ``a`` is a row-vector w/ a single query or a row-matrix of queries.
        Returns vector or array of output layer activations respectively.
        """
        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases
        if isinstance(Z, list) and isinstance(A, list):
            populate = True
        else:
            populate = False
        a = _format_inputs(a)
        L = len(self.layers) - 1
        # 3D stack of column-vector examples (m,||l₀||,1)
        if populate:
            Z.append(None)  # Z[0] is a dummy element to keep indexing consistent
            A.append(a)
        for l, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + b  # (m, ||l₀+1||, 1)
            a = self.last_sigma.fn(z) if l+1 == L else self.sigma.fn(z)
            if populate:
                Z.append(z)
                A.append(a)
        return a

    def _evaluate(self, queries, labels, normalized=False):
        """Classify input ``queries`` and compare to truths in ``labels``.
        Classification is considered correct iff all values in ``ans`` match
        all respective values in ``labels``.
        Returns a fraction representing the total number of correct guesses to
        number of inputs.

        """
        ans = self.classify(queries, normalized)
        right = ((ans == np.array(labels)).all(axis=1)).sum()
        return right/len(labels)

    def classify(self, queries, normalized=False):
        """Generate predictions and process them as single- or multi-label.
        Returns a list of vectors with predicted classes.

        """
        if normalized is False:
            # idempotent and all, but think of performance
            queries = self._normalize(queries)
        queries = _format_inputs(np.array(queries))
        a = self.feedforward(queries)
        a = a.reshape(a.shape[0], a.shape[1])
        ans = np.zeros_like(a)
        if self.multilabel:
            ans[a > 0.5] = 1
        else:
            ans[np.arange(a.shape[0]), a.argmax(axis=1)] = 1
        return [r for r in ans]


# internal use formatting shenanigans
def _format_inputs(x):
    """Transform inputs into (m,n,1) format for cost/activation functions.
    Returns formatted array.

    """
    if x.ndim == 1:  # vector
        return x.reshape(1, -1, 1)
    elif x.ndim == 2:
        if x.shape[0] == 1:  # either single-row matrix or 1x1 matrix
            return x.T.reshape(1, -1, 1)
        else:  # multirow
            return x.reshape(x.shape[0], -1, 1)
    elif x.shape[1] == 1:  # mx1x1 or row-stacks
        return x.transpose((0, 2, 1))
    return x
