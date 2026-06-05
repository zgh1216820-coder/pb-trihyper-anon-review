import csv
import time
import numpy as np
import random

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.client.trimoe_nofish_nofuzzy import EmbedNetwork, TaskEmbedding, TriBranchHyperNet, trimoe_nofish_nofuzzyClient
from src.server.fedavg import FedAvgServer

from sklearn.cluster import KMeans
from src.utils.metrics import Metrics
from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

from scipy.stats import entropy
import seaborn as sns

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import font_manager
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class trimoe_nofish_nofuzzyServer(FedAvgServer):
    algorithm_name: str = "trimoe_nofish_nofuzzy"
    all_model_params_personalized = True
    return_diff = False
    client_cls = trimoe_nofish_nofuzzyClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        # parser.add_argument("--embed_dim", type=int, default=-1)
        parser.add_argument("--embed_dim", type=int, default=256)
        parser.add_argument("--embed_y", type=int, default=1)
        parser.add_argument("--embed_num_kernels", type=int, default=16)
        parser.add_argument("--embed_num_batches", type=int, default=1)

        parser.add_argument("--hyper_embed_lr", type=float, default=2e-4)
        # parser.add_argument("--hyper_hidden_dim", type=int, default=100)
        parser.add_argument("--hyper_hidden_dim", type=int, default=512)
        parser.add_argument("--hyper_num_hidden_layers", type=int, default=3)
        parser.add_argument("--clip_norm", type=float, default=50.0)

        parser.add_argument("--chunk_size", type=float, default=512)
        parser.add_argument("--effective_block_size_q", type=int, default=0)
        parser.add_argument("--num_K", type=float, default=-1)
        parser.add_argument(
            "--routing_granularity",
            type=str,
            default="block",
            choices=["block", "layer_tied", "model_tied", "adaptive_smooth"],
        )
        parser.add_argument("--routing_intervention", type=str, default="none")
        parser.add_argument("--routing_intervention_seed", type=int, default=0)
        parser.add_argument("--eval_checkpoint_dir", type=str, default="")
        parser.add_argument(
            "--routing_interventions",
            type=str,
            default="none,layer_mean,model_mean,uniform,permute_blocks,reverse_blocks,second_best",
        )
        parser.add_argument(
            "--ablation_mode",
            type=str,
            default="full",
            choices=["full", "no_global", "no_task", "no_client"],
        )

        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        if args.common.buffers == "global":
            raise NotImplementedError("trimoe_nofish_nofuzzy doesn't support global buffers.")
        super().__init__(args, False)

        if self.args.trimoe_nofish_nofuzzy.embed_dim <= 0:
            self.args.trimoe_nofish_nofuzzy.embed_dim = int(1 + self.client_num / 4)
        self.args.trimoe_nofish_nofuzzy.num_K = int(self.client_num * 0.3)
        self.args.trimoe_nofish_nofuzzy.client_num = int(self.client_num)

        # 初始化三大子网络
        self.embed_net = EmbedNetwork(self.args)
        self.hyper_net = TriBranchHyperNet(self.model, self.args)
        self.task_net = TaskEmbedding(self.args)

        self.embed_hyper_optimizer = torch.optim.Adam(
            list(self.embed_net.parameters()) +
            list(self.task_net.parameters()) +
            list(self.hyper_net.parameters()),
            lr=self.args.trimoe_nofish_nofuzzy.hyper_embed_lr,
            )

        # [[3, 83, 63, 37, 33, 53, 51, 49, 94, 4], [74, 90, 73, 24, 43, 92, 91, 28, 87, 68], [81, 60, 85, 45, 64, 38, 21, 84, 71, 23], [37, 12, 60, 16, 94, 33, 92, 71, 97, 23], [87, 42, 83, 11, 28, 45, 95, 39, 53, 47], [33, 74, 37, 59, 15, 60, 6, 81, 79, 9], [60, 24, 64, 15, 87, 48, 67, 38, 52, 6], [19, 17, 25, 43, 52, 72, 58, 18, 40, 91], [23, 99, 10, 62, 42, 80, 22, 40, 81, 7], [0, 57, 35, 26, 97, 99, 21, 73, 20, 62], [97, 99, 11, 16, 78, 55, 81, 54, 51, 61], [49, 0, 4, 68, 25, 93, 47, 1, 41, 96], [67, 24, 2, 86, 0, 80, 93, 31, 28, 88], [44, 39, 16, 13, 49, 64, 75, 21, 8, 5], [38, 37, 58, 93, 66, 76, 67, 43, 55, 87], [17, 43, 62, 45, 96, 24, 21, 51, 2, 29], [28, 93, 16, 27, 2, 75, 64, 21, 15, 46], [91, 83, 4, 48, 81, 32, 98, 68, 79, 6], [76, 6, 84, 13, 83, 2, 88, 14, 53, 56], [48, 15, 70, 32, 60, 89, 19, 26, 88, 87], [80, 1, 38, 53, 84, 12, 66, 34, 79, 77], [89, 17, 53, 13, 65, 92, 79, 15, 36, 14], [13, 63, 25, 77, 99, 33, 66, 45, 90, 52], [37, 20, 5, 70, 63, 26, 88, 61, 42, 30], [0, 1, 85, 11, 14, 73, 63, 19, 99, 65], [9, 92, 12, 32, 82, 29, 58, 37, 33, 59], [6, 12, 22, 5, 37, 46, 86, 40, 54, 94], [14, 12, 97, 5, 1, 17, 84, 82, 21, 42], [45, 56, 79, 34, 93, 11, 47, 43, 23, 14], [51, 58, 34, 49, 88, 61, 53, 83, 21, 14], [16, 89, 92, 7, 20, 13, 53, 75, 61, 72], [86, 56, 23, 78, 48, 99, 45, 3, 87, 93], [16, 61, 62, 14, 52, 56, 5, 8, 33, 83], [40, 1, 90, 86, 67, 98, 95, 73, 92, 72], [28, 85, 43, 66, 88, 89, 79, 12, 55, 84], [93, 31, 61, 44, 84, 86, 82, 49, 19, 71], [78, 6, 1, 82, 20, 64, 60, 63, 21, 9], [96, 62, 41, 30, 42, 35, 6, 64, 28, 70], [82, 48, 51, 30, 10, 58, 56, 73, 57, 11], [63, 57, 40, 15, 95, 83, 2, 13, 51, 52], [4, 71, 70, 0, 11, 77, 79, 81, 38, 65], [71, 26, 83, 85, 58, 43, 46, 6, 28, 42], [71, 98, 79, 76, 60, 91, 84, 46, 57, 14], [12, 97, 84, 28, 0, 43, 45, 81, 37, 70], [66, 47, 89, 13, 5, 21, 63, 17, 91, 95], [92, 52, 13, 32, 78, 25, 24, 15, 50, 27], [58, 24, 91, 43, 13, 96, 53, 5, 81, 86], [75, 15, 98, 57, 58, 84, 64, 17, 63, 0], [67, 94, 6, 70, 55, 74, 61, 65, 99, 22], [88, 74, 22, 93, 16, 13, 49, 84, 76, 77], [41, 64, 49, 53, 77, 88, 97, 31, 35, 50], [43, 37, 57, 17, 52, 77, 89, 95, 65, 38], [89, 70, 40, 91, 94, 80, 27, 25, 26, 79], [91, 37, 87, 44, 86, 16, 90, 83, 22, 64], [69, 88, 81, 41, 90, 14, 97, 45, 71, 61], [74, 73, 85, 90, 53, 86, 95, 68, 36, 54], [1, 66, 13, 3, 48, 18, 6, 85, 7, 25], [34, 96, 20, 36, 87, 32, 18, 7, 88, 37], [26, 96, 69, 4, 45, 57, 13, 78, 86, 94], [71, 28, 95, 82, 49, 30, 65, 89, 36, 88], [85, 6, 48, 49, 53, 87, 40, 70, 1, 95], [32, 25, 79, 99, 56, 28, 81, 90, 82, 47], [96, 75, 69, 94, 77, 25, 99, 92, 24, 37], [57, 22, 87, 9, 23, 94, 65, 15, 48, 5], [54, 35, 71, 33, 16, 20, 74, 32, 0, 42], [59, 90, 19, 5, 41, 76, 6, 99, 80, 78], [84, 38, 62, 73, 70, 45, 9, 90, 40, 67], [28, 23, 66, 8, 64, 20, 53, 69, 68, 51], [11, 44, 28, 27, 84, 42, 46, 37, 96, 79], [67, 60, 71, 96, 83, 1, 14, 85, 44, 57], [30, 82, 78, 79, 31, 5, 87, 41, 48, 14], [49, 32, 69, 92, 36, 3, 66, 98, 47, 65], [65, 57, 62, 5, 37, 83, 24, 41, 10, 12], [21, 98, 68, 0, 80, 8, 26, 85, 99, 27], [87, 54, 13, 26, 68, 89, 94, 99, 55, 82], [9, 95, 87, 19, 3, 58, 91, 42, 4, 11], [9, 6, 22, 32, 71, 73, 29, 33, 52, 49], [57, 80, 51, 55, 40, 2, 49, 86, 15, 68], [0, 82, 92, 79, 8, 96, 95, 74, 5, 89], [9, 45, 64, 13, 37, 87, 38, 77, 11, 36], [97, 56, 48, 50, 84, 3, 60, 20, 68, 28], [17, 95, 50, 69, 37, 81, 18, 38, 86, 93], [82, 47, 1, 70, 71, 18, 15, 5, 0, 73], [76, 50, 68, 70, 10, 39, 26, 95, 94, 92], [44, 27, 52, 82, 65, 18, 20, 23, 28, 78], [32, 24, 14, 23, 85, 72, 6, 96, 70, 59], [82, 70, 9, 37, 85, 8, 32, 12, 25, 92], [74, 62, 42, 45, 16, 85, 83, 30, 12, 36], [79, 8, 24, 40, 62, 57, 42, 78, 84, 39], [88, 74, 19, 72, 46, 40, 54, 21, 0, 31], [28, 91, 94, 82, 55, 35, 46, 17, 86, 93], [42, 61, 59, 57, 45, 99, 39, 62, 70, 13], [22, 82, 77, 10, 35, 17, 68, 25, 33, 93], [80, 9, 93, 2, 64, 3, 74, 51, 98, 75], [27, 4, 70, 33, 69, 60, 83, 99, 19, 46], [50, 28, 98, 75, 93, 37, 17, 59, 64, 65], [11, 50, 46, 64, 1, 84, 30, 78, 87, 21], [96, 65, 17, 57, 20, 22, 72, 84, 18, 94], [93, 61, 45, 5, 28, 97, 62, 29, 8, 33], [47, 29, 5, 85, 26, 66, 46, 49, 60, 58...
        self.client_sample_stream_grad_similarity = [
            random.sample(
                self.train_clients,
                max(1, 5),
            )
            for _ in range(self.args.common.global_epoch)
        ]

        # if len(self.train_clients)>100:
        #     self.client_sample_Monitor_Clients = [0, 25, 50, 75, 99, 280, 460, 640, 820, 999]
        # elif len(self.train_clients)>50:
        #     self.client_sample_Monitor_Clients = [0, 25, 50, 75, 99]
        # else:
        #     self.client_sample_Monitor_Clients = [0, 25, 49]

        self.avg_sim_list = []
        self.sim_matrix = []
        self.Monitor_Clients_embedding_list = []
        self.Monitor_Clients_proto_list = []
        self.Monitor_Clients_net_embed_list = []
        self.routing_weights = None
        self.routing_client_ids = None
        self.routing_client_task_ids = None

        self.init_trainer(embed_net=self.embed_net, hyper_net=self.hyper_net, task_net=self.task_net)

    def _load_counterfactual_checkpoint(self, checkpoint_dir: str | Path):
        checkpoint_dir = Path(checkpoint_dir)
        required = ["embed_net.pt", "task_net.pt", "hyper_net.pt", "assign.pt"]
        missing = [name for name in required if not (checkpoint_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint files in {checkpoint_dir}: {', '.join(missing)}"
            )

        self.embed_net.load_state_dict(
            torch.load(checkpoint_dir / "embed_net.pt", map_location=self.device)
        )
        self.task_net.load_state_dict(
            torch.load(checkpoint_dir / "task_net.pt", map_location=self.device)
        )
        self.hyper_net.load_state_dict(
            torch.load(checkpoint_dir / "hyper_net.pt", map_location=self.device)
        )
        self.assign = torch.load(checkpoint_dir / "assign.pt", map_location=self.device)
        if torch.is_tensor(self.assign):
            self.assign = self.assign.to(self.device)
        else:
            self.assign = torch.as_tensor(self.assign, device=self.device)

    def _fresh_eval_template(self):
        return {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

    def _evaluate_current_intervention(self):
        self.testing = True
        if self.val_clients == self.train_clients == self.test_clients:
            results = {"all_clients": self._fresh_eval_template()}
            self.trainer.test(self.test_clients, results["all_clients"])
        else:
            results = {}
            if len(self.val_clients) > 0:
                results["val_clients"] = self._fresh_eval_template()
                self.trainer.test(self.val_clients, results["val_clients"])
            if len(self.test_clients) > 0:
                results["test_clients"] = self._fresh_eval_template()
                self.trainer.test(self.test_clients, results["test_clients"])
        self.testing = False
        return results

    def _run_counterfactual_routing_eval(self, checkpoint_dir: str):
        self.logger.log("Counterfactual routing eval checkpoint:", checkpoint_dir)
        self._load_counterfactual_checkpoint(checkpoint_dir)

        interventions = [
            item.strip()
            for item in str(self.args.trimoe_nofish_nofuzzy.routing_interventions).split(",")
            if item.strip()
        ]
        if not interventions:
            interventions = ["none"]

        eval_epoch = max(1, int(self.args.common.global_epoch)) - 1
        seed = int(self.args.trimoe_nofish_nofuzzy.routing_intervention_seed)
        rows = []
        baseline = {}
        self.test_results = {}

        for idx, intervention in enumerate(interventions, start=1):
            self.current_epoch = eval_epoch
            self.routing_intervention = intervention
            self.routing_intervention_seed = seed
            self.logger.log(
                f"Counterfactual routing intervention [{idx}/{len(interventions)}]: {intervention}"
            )
            results = self._evaluate_current_intervention()
            self.test_results[idx] = results

            for group_name, group_results in results.items():
                for split in ["train", "val", "test"]:
                    before_metrics = group_results["before"][split]
                    after_metrics = group_results["after"][split]
                    if before_metrics.size == 0 and after_metrics.size == 0:
                        continue
                    key = (group_name, split)
                    before_acc = float(before_metrics.accuracy)
                    if intervention in {"none", "learned"}:
                        baseline[key] = before_acc
                    rows.append(
                        {
                            "intervention": intervention,
                            "group": group_name,
                            "split": split,
                            "before_loss": float(before_metrics.loss),
                            "before_accuracy": before_acc,
                            "after_loss": float(after_metrics.loss),
                            "after_accuracy": float(after_metrics.accuracy),
                            "delta_vs_learned": "",
                        }
                    )

        for row in rows:
            key = (row["group"], row["split"])
            if key in baseline:
                row["delta_vs_learned"] = row["before_accuracy"] - baseline[key]

        out_csv = self.output_dir / "counterfactual_routing_results.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "intervention",
                    "group",
                    "split",
                    "before_loss",
                    "before_accuracy",
                    "after_loss",
                    "after_accuracy",
                    "delta_vs_learned",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        self.counterfactual_routing_results = rows
        self.logger.log("Counterfactual routing CSV:", str(out_csv))

    def train(self):
        eval_checkpoint_dir = str(
            getattr(self.args.trimoe_nofish_nofuzzy, "eval_checkpoint_dir", "") or ""
        )
        if eval_checkpoint_dir:
            self._run_counterfactual_routing_eval(eval_checkpoint_dir)
            return

        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.verbose = (self.current_epoch + 1) % self.args.common.verbose_gap == 0

            if self.verbose:
                self.logger.log("-" * 28, f"TRAINING EPOCH: {E + 1}", "-" * 28)

            if self.current_epoch == 0:
                if self.args.dataset.split == "user":
                    self.selected_clients = [
                        random.sample(
                            self.train_clients+self.test_clients,
                            max(1, int(self.client_num * 1)),
                            )
                    ][0]
                else:
                    self.selected_clients = [
                        random.sample(
                            self.train_clients,
                            max(1, int(self.client_num * 1)),
                        )
                    ][0]
            else:
                self.selected_clients = self.client_sample_stream[E]
                # self.selected_clients = self.client_sample_stream[E] + self.client_sample_stream_grad_similarity[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            avg_round_time = (avg_round_time * self.current_epoch + (end - begin)) / (
                    self.current_epoch + 1
            )

            if (
                    self.args.common.test.server.interval > 0
                    and (E + 1) % self.args.common.test.server.interval == 0
            ):
                self.test_global_model()
            if (
                    self.args.common.test.client.interval > 0
                    and (E + 1) % self.args.common.test.client.interval == 0
            ):
                # self.test_client_models()
                # test_client_packages_embedding, test_client_packages_z = self.test_client_models()
                # client_packages_embedding, client_packages_proto, client_packages_net_embed
                test_client_packages_embedding, test_client_packages_proto, test_client_packages_net_embed, test_client_packages_z = self.test_client_models()
                self.c_embedding = list(test_client_packages_embedding.values())
                # if self.args.dataset.split != "user":
                self.c_proto = list(test_client_packages_proto.values())
                self.c_net_embed = list(test_client_packages_net_embed.values())
                self._store_routing_diagnostics(test_client_packages_z)

                # if E == 999:
                # # if True:
                #     print()
                #     self.display_embedding_scatter(c_embedding, "embedding")
                #     self.display_embedding_scatter(c_proto, "proto")
                #     self.display_embedding_scatter(c_net_embed, "net_embed")
                # self.display_one_class(list(test_client_packages_z.values()), target_class=16)
                # self.display_z_scatter(list(test_client_packages_z.values()))

            self.display_metrics()

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    def _store_routing_diagnostics(self, client_packages_z: dict[int, torch.Tensor]):
        if not client_packages_z:
            return

        sorted_client_ids = sorted(client_packages_z.keys())
        routing_rows = []
        for client_id in sorted_client_ids:
            z = client_packages_z[client_id]
            if torch.is_tensor(z):
                routing_rows.append(z.detach().cpu())
            else:
                routing_rows.append(torch.as_tensor(z))

        self.routing_weights = torch.stack(routing_rows, dim=0)
        self.routing_client_ids = torch.tensor(sorted_client_ids, dtype=torch.long)

        if hasattr(self, "assign"):
            assign = self.assign.detach().cpu() if torch.is_tensor(self.assign) else torch.as_tensor(self.assign)
            self.routing_client_task_ids = torch.tensor(
                [int(assign[client_id].item()) for client_id in sorted_client_ids],
                dtype=torch.long,
            )

    def display_embedding_scatter(self, test_client_packages, named):
        # 1. 合并所有张量
        combined_tensors = torch.stack(test_client_packages, dim=0).to('cpu').numpy()

        # 3. 创建标签（用于区分来自不同原始张量的点）
        labels = []
        for i, tensor in enumerate(test_client_packages):
            labels.extend([i])
        labels = np.array(labels)

        # 可选：使用t-SNE进行降维（尤其适用于高维数据）
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(combined_tensors)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Original Tensor Index')

        # 1. 尝试获取 alpha，如果不存在则返回 None
        alpha_val = getattr(self.args.dataset, 'alpha', None)

        # 2. 判断逻辑：如果有 alpha 且不是 None，就用 alpha，否则用 classes_per_client
        if alpha_val is not None:
            distinct_param = str(alpha_val)
        else:
            # 假设如果没有 alpha，一定会有 classes_per_client
            distinct_param = str(self.args.dataset.classes_per_client)

        plt.title(f"t-SNE Visualization of trimoe Tensor Distribution of _"
                  f"{self.args.dataset.client_num}_{distinct_param}_{self.args.dataset.name}_{named}")
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(f"./model_saved/new_picture/t-SNE Visualization of trimoe Tensor Distribution of _"
                    f"{self.args.dataset.client_num}_{distinct_param}_{self.args.dataset.name}_{named}.svg", format='svg', bbox_inches='tight', transparent=True)
        plt.show()


    def select_representative_clients_kmeans(self, client_features, n_representative=5):
        """
        使用K-means聚类选择代表性客户端

        参数:
        client_features: 所有客户端的特征矩阵，形状为 [n_clients, n_features]
        n_representative: 要选择的代表性客户端数量

        返回:
        代表性客户端的索引列表
        """
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_representative, random_state=42)
        kmeans.fit(client_features)

        # 找到每个聚类中心最接近的客户端
        representative_indices = []
        for center in kmeans.cluster_centers_:
            # 计算每个客户端到聚类中心的距离
            distances = np.linalg.norm(client_features - center, axis=1)
            # 选择距离最小的客户端
            closest_idx = np.argmin(distances)
            representative_indices.append(closest_idx)

        return representative_indices

    def display_one_class(self, z, target_class):
        # 找到所有属于 target_class 的客户端索引
        idx = np.where(self.assign.cpu().numpy() == target_class)[0]
        # 只取这些客户端对应的 z
        z_filtered = [z[i] for i in idx]
        # 调用原始可视化函数
        self.display_z_scatter(z_filtered)

    def display_z_scatter(self, z):
        """
        使用热图快速可视化 [n, 1716, 3] 张量
        参数:
        tensor: 形状为 [n, 1716, 3] 的numpy数组，每行的3个值和为1
        """
        combined_tensors = torch.stack(z, dim=0).to('cpu')
        n = combined_tensors.shape[0]
        categories = combined_tensors.shape[1]

        # 创建图形
        fig, axes = plt.subplots(n, 1, figsize=(15, max(6, n * 0.5)))

        # 如果只有一个样本，确保axes是列表形式
        if n == 1:
            axes = [axes]

        # 对每个样本创建一个热图
        for i in range(n):
            # 将三个通道转换为RGB颜色
            # 注意：这里假设tensor的值在0-1范围内
            rgb_data = combined_tensors[i]  # 形状为 [1716, 3]

            # 重塑数据为适合imshow的形状 (1, 1716, 3)
            rgb_data_reshaped = rgb_data.reshape(1, categories, 3)

            # 显示热图
            axes[i].imshow(rgb_data_reshaped, aspect='auto', interpolation='nearest')
            axes[i].set_ylabel(f'{i+1}')
            axes[i].set_yticks([])  # 隐藏y轴刻度

            # 只在最后一个子图显示x轴标签
            if i == n - 1:
                axes[i].set_xlabel('Category Index')
            else:
                axes[i].set_xticks([])  # 隐藏x轴刻度

        plt.suptitle('Fast Visualization of Tensor with Shape [n, 1716, 3]')
        plt.tight_layout()
        plt.show()

    def test_client_models(self):
        """The function for testing FL method's output (a single global model
        or personalized client models)."""
        self.testing = True
        clients = list(set(self.val_clients + self.test_clients))
        client_packages_embedding = {}
        client_packages_proto = {}
        client_packages_net_embed = {}
        client_packages_z = {}
        template = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        if len(clients) > 0:
            if self.val_clients == self.train_clients == self.test_clients:
                results = {"all_clients": template}
                self.trainer.test(clients, results["all_clients"])
                client_packages_embedding = self.trainer.exec("get_client_embedding", clients)
                client_packages_proto = self.trainer.exec("get_client_proto", clients)
                client_packages_net_embed = self.trainer.exec("get_client_net_embed", clients)
                client_packages_z = self.trainer.exec("get_client_z", clients)
            else:
                results = {
                    "val_clients": deepcopy(template),
                    "test_clients": deepcopy(template),
                }
                if len(self.val_clients) > 0:
                    self.trainer.test(self.val_clients, results["val_clients"])
                if len(self.test_clients) > 0:
                    self.trainer.test(self.test_clients, results["test_clients"])

                client_packages_embedding = self.trainer.exec("get_client_embedding", clients)
                client_packages_proto = self.trainer.exec("get_client_proto", clients)
                client_packages_net_embed = self.trainer.exec("get_client_net_embed", clients)
                client_packages_z = self.trainer.exec("get_client_z", clients)

            if self.current_epoch + 1 not in self.test_results:
                self.test_results[self.current_epoch + 1] = results
            else:
                self.test_results[self.current_epoch + 1].update(results)
        self.testing = False

        return client_packages_embedding, client_packages_proto, client_packages_net_embed, client_packages_z

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""
        if self.current_epoch == 0:
            client_packages = self.trainer.train()
            client_distribution = np.concatenate([
                package["data_distribution"] for package in client_packages.values()
            ], axis=0)

            num_K = self.args.trimoe_nofish_nofuzzy.num_K
            self.assign = self.constrained_kmeans(client_distribution, num_K, int(1.2*self.client_num/num_K))
            self.logger.log('task:', list(self.assign))
            self.assign = torch.from_numpy(self.assign).to(self.device)

        else:
            client_packages = self.trainer.train()
            self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)

        if self.current_epoch == 0 and not self.testing:
            server_package["Count the number of tasks"] = True

        else:
            # 下发 embed / hyper / task 三网参数
            server_package["embed_net_params"] = self.embed_net.state_dict()
            server_package["hyper_net_params"] = self.hyper_net.state_dict()
            server_package["task_net_params"] = self.task_net.state_dict()
            # 下发 client_id 对应的簇 ID 与 epoch
            server_package["data_id"] = torch.tensor(client_id).to(self.device)
            server_package["task_id"] = self.assign[client_id]
            server_package["epoch"] = self.current_epoch
            server_package["routing_intervention"] = getattr(
                self,
                "routing_intervention",
                getattr(self.args.trimoe_nofish_nofuzzy, "routing_intervention", "none"),
            )
            server_package["routing_intervention_seed"] = getattr(
                self,
                "routing_intervention_seed",
                getattr(self.args.trimoe_nofish_nofuzzy, "routing_intervention_seed", 0),
            )

        return server_package

    def aggregate_client_updates(self, client_packages: OrderedDict[int, dict[str, Any]]):
        """
        1) 把客户端前一轮上传的 gradient 聚合
        2) 加上（可选）CL 正则后，更新 embed_net、task_net、hyper_net
        3) 同时在这里保存“上一轮输出”以备下一轮 CL 用
        """
        all_embed_net_grads = [pkg["embed_net_grads"] for pkg in client_packages.values()]
        all_task_net_grads = [pkg["task_net_grads"] for pkg in client_packages.values()]
        all_hyper_net_grads = [pkg["hyper_net_grads"] for pkg in client_packages.values()]

        # client_packages_embedding = self.trainer.exec("get_client_embedding", self.client_sample_Monitor_Clients)
        # self.Monitor_Clients_embedding_list.append(client_packages_embedding)
        # client_packages_proto = self.trainer.exec("get_client_proto", self.client_sample_Monitor_Clients)
        # self.Monitor_Clients_proto_list.append(client_packages_proto)
        # client_packages_net_embed = self.trainer.exec("get_client_net_embed", self.client_sample_Monitor_Clients)
        # self.Monitor_Clients_net_embed_list.append(client_packages_net_embed)

        self.embed_hyper_optimizer.zero_grad()

        self.embed_net = self.embed_net.to(self.device)
        self.task_net = self.task_net.to(self.device)
        self.hyper_net = self.hyper_net.to(self.device)

        # —— 聚合 embed_net 梯度 —— #
        for param, grads in zip(self.embed_net.parameters(), zip(*all_embed_net_grads)):
            param.grad = torch.stack(grads, dim=0).mean(dim=0)

        # # —— 聚合 task_net 梯度 —— #
        for param, grads in zip(self.task_net.parameters(), zip(*all_task_net_grads)):
            param.grad = torch.stack(grads, dim=0).mean(dim=0)

        # —— 聚合 hyper_net 梯度 —— #
        for param, grads in zip(self.hyper_net.parameters(), zip(*all_hyper_net_grads)):
            param.grad = torch.stack(grads, dim=0).mean(dim=0)

        # —— 最后一步：一次性更新 embed_net、task_net、hyper_net —— #
        self.embed_hyper_optimizer.step()

    def display_metrics(self):
        if self.current_epoch == 0:
            self.logger.log("-" * 28, "KMeans completed", "-" * 28)
        else:
            super().display_metrics()

    def constrained_kmeans(self, X, K, max_cluster_size, num_iters=100):
        N, D = X.shape
        centroids = KMeans(n_clusters=K).fit(X).cluster_centers_
        for _ in range(num_iters):
            distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
            cluster_assignment = np.argmin(distances, axis=1)
            for j in range(K):
                if (cluster_assignment == j).sum() == 0:
                    centroids[j] = X[np.random.choice(N)]
            for j in range(K):
                while (cluster_assignment == j).sum() > max_cluster_size:
                    idxs = np.where(cluster_assignment == j)[0]
                    far_idx = idxs[np.argmax(distances[idxs, j])]
                    cluster_assignment[far_idx] = -1
                    distances[far_idx, j] = np.inf
            for i in range(N):
                if cluster_assignment[i] == -1:
                    for j in np.argsort(distances[i]):
                        if (cluster_assignment == j).sum() < max_cluster_size:
                            cluster_assignment[i] = j
                            break
            for j in range(K):
                if (cluster_assignment == j).sum() > 0:
                    centroids[j] = X[cluster_assignment == j].mean(axis=0)
        return cluster_assignment
