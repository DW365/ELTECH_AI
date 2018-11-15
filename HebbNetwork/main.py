import random
from typing import List

samples = [
    "####"
    " ## "
    " ## "
    "####",

    "#  #"
    "## #"
    "# ##"
    "#  #",

    "####"
    "##  "
    "  ##"
    "####",

    "####"
    "#  #"
    "#  #"
    "####"]


class InputNeuron:
    def __init__(self):
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Neuron:
    def __init__(self, inputs: List[InputNeuron], weights: List[float]):
        self._inputs = inputs
        self._weights = weights

    @property
    def value(self):
        u_in = sum(inp * weight for inp, weight in zip([n.value for n in self._inputs], self._weights))
        return 1 if u_in > 0 else -1


class Network:
    def __init__(self):
        self.input_layer = [InputNeuron() for i in range(len(samples[0]))]
        self.output_layer = [Neuron(self.input_layer, [0 for i in self.input_layer]) for j in range(2)]

    def print_weights(self):
        for n in self.output_layer:
            print(n._weights)

    def train(self, sample, expectations):
        for cell, inp_neuron in zip(sample, self.input_layer):
            inp_neuron.value = 1 if cell != " " else -1

        for out_neuron, expectation in zip(self.output_layer, expectations):
            for i, inp_neuron in zip(range(len(self.input_layer)), self.input_layer):
                out_neuron._weights[i] += expectation * inp_neuron.value

    def run(self, sample):
        for cell, inp_neuron in zip(sample, self.input_layer):
            inp_neuron.value = 1 if cell != " " else -1

        return [n.value for n in self.output_layer]


n = Network()
n.print_weights()


def train(n):
    print("Training...")
    f = True
    counter = 0
    while f:
        counter += 1
        print(counter)
        f = False
        while n.run(samples[0]) != [-1, -1]:
            f = True
            n.train(samples[0], [-1, -1])
        while n.run(samples[1]) != [-1, 1]:
            f = True
            n.train(samples[1], [-1, 1])
        while n.run(samples[2]) != [1, -1]:
            f = True
            n.train(samples[2], [1, -1])
        while n.run(samples[3]) != [1, 1]:
            f = True
            n.train(samples[3], [1, 1])
    print("Training complete")


train(n)


def print_sample(sample):
    print("\n".join(sample[i * 4:i * 4 + 4] for i in range(4)))


def noisy(example, level):
    new_example = ""
    for i in range(0, len(example)):
        if random.random() < level:
            new_example += random.choice([" ", "#"])
        else:
            new_example += example[i]
    return new_example


for sample in samples:
    for i in range(3):
        print("_____________")
        print("Test:\r\n")
        n_sample = noisy(sample, i * 0.2)
        print_sample(n_sample)
        print("\r\nResult:\r\n")
        result = n.run(n_sample)
        if result == [-1, -1]:
            print_sample(samples[0])
        if result == [-1, 1]:
            print_sample(samples[1])
        if result == [1, -1]:
            print_sample(samples[2])
        if result == [1, 1]:
            print_sample(samples[3])
        print("_____________")
