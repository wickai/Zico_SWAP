import logging
import torch
import torch.nn as nn
import random
import numpy as np
from tools.evaluation import set_seed, count_parameters_in_MB

# ============ 6) EvolutionarySearch (SWAP作为评估) ============


class EvolutionarySearch:
    """
    简易进化算法: init population -> 评估 -> 排序 -> 选择 + 交叉 + 变异 -> 重复
    用 SWAP.evaluate(...) 作为适应度 (fitness).
    """

    def __init__(self, population_size, mutation_rate, n_generations,
                 swap_metric, search_space, device,
                 num_inits=3):
        """
        :param num_inits: 对同一个结构多次随机初始化并计算SWAP平均，以减少初始化差异
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.swap_metric = swap_metric
        self.search_space = search_space
        self.device = device
        self.num_inits = num_inits

    def search(self, inputs):
        # 1) 初始化种群
        population = []
        for _ in range(self.population_size):
            op_codes = self.search_space.random_op_codes()
            width_codes = self.search_space.random_width_codes()
            population.append({
                "op_codes": op_codes,
                "width_codes": width_codes,
                "fitness": None
            })

        for gen in range(self.n_generations):
            logging.info(f"=== Generation {gen+1} / {self.n_generations} ===")

            # 评估适应度
            for i, indiv in enumerate(population):
                if indiv["fitness"] is None:
                    scores = []
                    for _ in range(self.num_inits):
                        model = self.search_space.get_model(
                            indiv["op_codes"],
                            indiv["width_codes"]
                        ).to(self.device)
                        # 简单初始化
                        for p in model.parameters():
                            if p.dim() > 1:
                                nn.init.kaiming_normal_(p)
                        score = self.swap_metric.evaluate(model, inputs)
                        scores.append(score)
                    param_mb = count_parameters_in_MB(model)
                    indiv["fitness"] = float(np.mean(scores))
                    indiv["model_size"] = param_mb

                logging.info(
                    f"  [Ind-{i+1}] fitness(SWAP): {indiv['fitness']:.3f}, model_size: {indiv['model_size']:.3f}MB")

            # 排序 (从大到小)
            population.sort(key=lambda x: x["fitness"], reverse=True)

            logging.info(
                f"  Best in Gen{gen+1}: fitness={population[0]['fitness']:.3f}")

            # 选择前 half
            next_gen = population[: self.population_size // 2]

            abandon_gen = population[self.population_size // 2:]
            # clear gpu memory
            for model in abandon_gen:
                del model
                torch.cuda.empty_cache()

            # 交叉 + 变异, 直到恢复到 population_size
            while len(next_gen) < self.population_size:
                p1 = random.choice(next_gen)
                p2 = random.choice(next_gen)

                # 交叉
                child_op_codes = self.crossover(p1["op_codes"], p2["op_codes"])
                child_width_codes = self.crossover(
                    p1["width_codes"], p2["width_codes"])

                # 变异
                child_op_codes = self.search_space.mutate_op_codes(
                    child_op_codes, self.mutation_rate)
                child_width_codes = self.search_space.mutate_width_codes(
                    child_width_codes, self.mutation_rate)

                next_gen.append({
                    "op_codes": child_op_codes,
                    "width_codes": child_width_codes,
                    "fitness": None
                })

            # 下一代
            population = next_gen
            # population.sort(key=lambda x: x["fitness"], reverse=True)
            # logging.info(
            #     f"  Best in Gen{gen+1} after crossover&mutation: fitness={population[0]['fitness']:.3f}")

        # 最后一代再做一遍评估
        for indiv in population:
            if indiv["fitness"] is None:
                scores = []
                for _ in range(self.num_inits):
                    model = self.search_space.get_model(
                        indiv["op_codes"],
                        indiv["width_codes"]
                    ).to(self.device)
                    for p in model.parameters():
                        if p.dim() > 1:
                            nn.init.kaiming_normal_(p)
                    score = self.swap_metric.evaluate(model, inputs)
                    scores.append(score)

                    # clear gpu memory
                    del model
                    torch.cuda.empty_cache()
                indiv["fitness"] = float(np.mean(scores))

        population.sort(key=lambda x: x["fitness"], reverse=True)
        logging.info(
            f"  Best in Gen{gen+1} in the end: fitness={population[0]['fitness']:.3f}")
        best = population[0]
        abandon_gen = population[1:]
        # clear gpu memory
        for model in abandon_gen:
            del model
            torch.cuda.empty_cache()
        return best

    @staticmethod
    def crossover(codes1, codes2):
        length = len(codes1)
        point1 = random.randint(0, length - 1)
        point2 = random.randint(point1, length - 1)
        child = codes1[:point1] + codes2[point1:point2] + codes1[point2:]
        return child
