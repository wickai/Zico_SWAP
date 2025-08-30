import logging
import torch
import torch.nn as nn
import random
import numpy as np
from tools.evaluation import set_seed, count_parameters_in_MB, get_model_complexity_info, get_model_complexity_info_fvcore
import time

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
        self.dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

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
                    durations = []
                    for _ in range(self.num_inits):
                        s_time = time.time()
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
                        e_time = time.time()
                        durations.append(e_time - s_time)
                    indiv["swap_time_seconds"] = np.mean(durations)
                    
                    # param_mb = count_parameters_in_MB(model)
                    model_info = get_model_complexity_info_fvcore(model, self.dummy_input)
                    indiv["swap"] = float(np.mean(scores))
                    indiv["fitness"] = indiv["swap"] # float(np.mean(scores)) / model_info['flops']
                    indiv["model_size"] = model_info['params'] / 1e6
                    indiv["flops"] = model_info['flops'] / 1e6
                    indiv["flops_cnn"] = model_info['flops_cnn'] / 1e6
                    indiv["comlexity_time_seconds"] = model_info['time_seconds']
                    del model
                    # s_time = time.time()
                    # pt_macs, pt_params = get_model_complexity_info_ptflops(model, (3, 224, 224), as_strings=True,
                    #     print_per_layer_stat=False, verbose=False)
                    # logging.info(f"[*][Ind-{i+1}] ptflops: {pt_macs}, flops: {indiv['flops']:.3f}MFLOPs, params: {pt_params}, duration: {time.time() - s_time}")


                logging.info(
                    f"  [Ind-{i+1}] fitness(SWAP/flops): {indiv['fitness']:.3f}, model_size: {indiv['model_size']:.3f}MB, flops_cnn: {indiv['flops_cnn']:.3f}MFLOPs, flops: {indiv['flops']:.3f}MFLOPs, comlexity_time_seconds: {indiv['comlexity_time_seconds']:.3f}s, swap_time_seconds: {indiv['swap_time_seconds']:.3f}s")

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
                model = self.search_space.get_model(
                            child_op_codes,
                            child_width_codes
                        ).to(self.device)
                model_info = get_model_complexity_info_fvcore(model, self.dummy_input)
                flops_cnn = model_info['flops_cnn'] / 1e6
                if flops_cnn > 625:
                    logging.info(f"[****] flops_cnn: {flops_cnn:.3f}MFLOPs greater than 600, skip~~")
                    continue
                del model
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

                # indiv["fitness"] = float(np.mean(scores))
                
                model_info = get_model_complexity_info_fvcore(model, self.dummy_input)
                indiv["swap"] = float(np.mean(scores))
                indiv["fitness"] = indiv["swap"] #float(np.mean(scores)) / model_info['flops']
                indiv["model_size"] = model_info['params'] / 1e6
                indiv["flops"] = model_info['flops'] / 1e6
                indiv["flops_cnn"] = model_info['flops_cnn'] / 1e6
                indiv["comlexity_time_seconds"] = model_info['time_seconds']
        logging.info("[*] sort by swap")
        population.sort(key=lambda x: x["swap"], reverse=True)
        logging.info(
            f"  Best in Gen{gen+1} in the end: swap={population[0]['swap']:.3f}, fitness={population[0]['fitness']:.3f}")
        best = population[0]
        abandon_gen = population[1:]
        # clear gpu memory
        for model in abandon_gen:
            del model
            torch.cuda.empty_cache()
        return best, population

    @staticmethod
    def crossover(codes1, codes2):
        length = len(codes1)
        point1 = random.randint(0, length - 1)
        point2 = random.randint(point1, length - 1)
        child = codes1[:point1] + codes2[point1:point2] + codes1[point2:]
        return child
