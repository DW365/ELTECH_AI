import math
import random
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1, rand_init=False):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 0.00000001 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.history = []

        self.dist_matrix = self.to_dist_matrix(coords)
        self.nodes = [i for i in range(self.N)]

        self.cur_solution = self.initial_solution(rand_init)
        self.best_solution = list(self.cur_solution)

        self.cur_fitness = self.fitness(self.cur_solution)
        self.initial_fitness = self.cur_fitness
        self.best_fitness = self.cur_fitness

        self.fitness_list = [self.cur_fitness]

    def initial_solution(self, rand=False):
        cur_node = random.choice(self.nodes)
        solution = [cur_node]

        free_list = list(self.nodes)
        free_list.remove(cur_node)

        while free_list:
            if not rand:
                closest_dist = min([self.dist_matrix[cur_node][j] for j in free_list])
                cur_node = self.dist_matrix[cur_node].index(closest_dist)
            else:
                cur_node = random.choice(free_list)
            free_list.remove(cur_node)
            solution.append(cur_node)

        return solution

    def dist(self, coord1, coord2):
        return round(math.sqrt(math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)), 4)

    def to_dist_matrix(self, coords):
        n = len(coords)
        mat = [[self.dist(coords[i], coords[j]) for i in range(n)] for j in range(n)]
        return mat

    def fitness(self, sol):
        return round(sum([self.dist_matrix[sol[i - 1]][sol[i]] for i in range(1, self.N)]) +
                     self.dist_matrix[sol[0]][sol[self.N - 1]], 4)

    def p_accept(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness = candidate_fitness
            self.cur_solution = candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate

        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness = candidate_fitness
                self.cur_solution = candidate

    def anneal(self):
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i:(i + l)] = reversed(candidate[i:(i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            if self.iteration % 100 == 0:
                self.history.append([[self.best_solution], self.coords])
            self.fitness_list.append(self.cur_fitness)

        print('Best fitness obtained: ', self.best_fitness)
        print('Improvement: ',
              round(self.best_fitness / self.initial_fitness, 2))

    def build_plot(self, step):
        self.visualize_routes(*self.history[step-1])

    def visualize_routes(self, paths=None, points=None):
        if paths is None:
            paths = [self.best_solution]
        if points is None:
            points = self.coords
        num_iters = 1
        x = []
        y = []
        for i in paths[0]:
            x.append(points[i][0])
            y.append(points[i][1])
        plt.plot(x, y, 'co')
        a_scale = float(max(x)) / float(100)
        if num_iters > 1:
            for i in range(1, num_iters):
                xi = []
                yi = []
                for j in paths[i]:
                    xi.append(points[j][0])
                    yi.append(points[j][1])

                plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                          head_width=a_scale, color='r',
                          length_includes_head=True, ls='dashed',
                          width=0.001 / float(num_iters))
                for i in range(0, len(x) - 1):
                    plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
                              head_width=a_scale, color='r', length_includes_head=True,
                              ls='dashed', width=0.001 / float(num_iters))
        plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
                  color='g', length_includes_head=True)
        for i in range(0, len(x) - 1):
            plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=a_scale,
                      color='g', length_includes_head=True)
        plt.xlim(min(x) * 1.1, max(x) * 1.1)
        plt.ylim(min(y) * 1.1, max(y) * 1.1)
        plt.show()

    def plot_learning(self):
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.show()
