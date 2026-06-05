import numpy as np

from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.client.fedavg import FedAvgClient
from src.utils.constants import INPUT_CHANNELS, NUM_CLASSES
import warnings
warnings.filterwarnings("ignore")

import math
import random

class trimoe_nofish_nofuzzyClient(FedAvgClient):
    def __init__(self, embed_net, hyper_net, task_net, **commons):
        super().__init__(**commons)
        self.embed_net: EmbedNetwork = deepcopy(embed_net).to(self.device)
        self.hyper_net: TriBranchHyperNet = deepcopy(hyper_net).to(self.device)
        self.task_net: TaskEmbedding = deepcopy(task_net).to(self.device)
        self.all_epoch = self.args.common.global_epoch

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

        # 载入服务器下发的 embed_net、hyper_net、task_net 参数
        self.embed_net.load_state_dict(package["embed_net_params"])
        self.hyper_net.load_state_dict(package["hyper_net_params"])
        self.task_net.load_state_dict(package["task_net_params"])
        if hasattr(self.hyper_net, "set_routing_intervention"):
            routing_intervention = package.get(
                "routing_intervention",
                getattr(self.args.trimoe_nofish_nofuzzy, "routing_intervention", "none"),
            )
            routing_intervention_seed = package.get(
                "routing_intervention_seed",
                getattr(self.args.trimoe_nofish_nofuzzy, "routing_intervention_seed", 0),
            )
            self.hyper_net.set_routing_intervention(
                routing_intervention,
                int(routing_intervention_seed),
            )

        # 获取消融配置 (示例)
        mode = getattr(
            getattr(self.args, "trimoe_nofish_nofuzzy", None),
            "ablation_mode",
            getattr(self.args, "ablation_mode", "full"),
        )

        use_g = (mode != 'no_global')
        use_t = (mode != 'no_task')
        use_c = (mode != 'no_client')

        # mode = getattr(self.args, 'emb_ablation_mode', 'full')
        # use_lam = True
        # use_proto = (mode != 'no_proto')
        # use_net_emb = (mode != 'no_net_emb')

        # 利用最新的 embed/task/hyper 三网和曲率生成本地 personalized 模型
        embedding = torch.zeros(self.args.trimoe_nofish_nofuzzy.embed_dim, device=self.device)
        size = 0
        data_loader = self.trainloader

        if self.testing or package["epoch"]==0:
            for i, (x, y) in enumerate(data_loader):
                embedding += self.embed_net(x.to(self.device), y.to(self.device)).sum(dim=0)
                size += len(x)

        else:
            for i, (x, y) in enumerate(data_loader):
                embedding += self.embed_net(x.to(self.device), y.to(self.device)).sum(dim=0)
                size += len(x)
                if i + 1 == self.args.trimoe_nofish_nofuzzy.embed_num_batches:
                    break

        embedding /= size
        embedding = (embedding - embedding.mean()) / embedding.std()
        # self.embedding = embedding

        # proto = self.embed_net.client_embedding(package["data_id"]) # shape [D]
        proto = self.embed_net.get_proto(package["data_id"]) # shape [D]
        proto = F.normalize(proto, p=2, dim=0)

        T = self.all_epoch
        self.proto = proto
        self.net_embed = embedding

        ###################
        lam = package["epoch"] / max(1, T)
        self.lam = lam
        # lam = self.embed_net.get_fusion_weight()
        # self.beta = package["epoch"] / max(1, T)

        # 现在的 lam 是 0.0 ~ 1.0 之间
        # 含义：从 Proto 平滑过渡到 Embedding，而不是把 Embedding 减掉
        final_emb = (1 - lam) * proto + lam * embedding

        # 加上 LayerNorm 还是必要的
        final_emb = F.layer_norm(final_emb, final_emb.shape)
        self.embedding = final_emb
        ###################

        self.current_epoch = package["epoch"]
        self.task_id = package["task_id"]
        if self.task_id == -1:
            self.task_embedding = torch.zeros_like(self.task_net(torch.tensor(0).to(self.device)))
        else:
            self.task_embedding = self.task_net(self.task_id)

        # --- 核心修改：动态生成参数列表 ---
        self.model_list_params = []

        # 1. 单独的 Global 分支 (作为 Regularization 或 Warmup)
        if use_g:
            self.global_model_params = self.hyper_net.extract_global_params()
            self.model_list_params.append(self.global_model_params)

        # 2. 单独的 Task 分支
        if use_t:
            self.task_model_params = self.hyper_net.extract_task_params(self.task_embedding)
            self.model_list_params.append(self.task_model_params)

        # 3. 单独的 Client (Data) 分支
        if use_c:
            self.data_model_params = self.hyper_net.extract_data_params(self.embedding)
            self.model_list_params.append(self.data_model_params)

        # 4. 融合后的参数 (Fusion)
        # 调用 forward 时传入开关
        self.regular_model_params = self.hyper_net(
            self.embedding, self.task_embedding,
            enable_global=use_g, enable_task=use_t, enable_client=use_c
        )

        # Perturbed fused view.
        # noisy = OrderedDict({k: (v + 1e-3 * torch.randn_like(v)) for k, v in self.regular_model_params.items()})
        noisy_embedding = self.embedding + 0.1 * torch.randn_like(self.embedding)
        noisy = self.hyper_net(
            noisy_embedding, self.task_embedding,
            enable_global=use_g, enable_task=use_t, enable_client=use_c
        )

        self.model_list_params.append(noisy)
        self.model_list_params.append(self.regular_model_params)

        # 加载最后一个用来做最终评估
        self.model.load_state_dict(self.regular_model_params)

    def set_data_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

    def train(self, server_package: dict[str, Any]) -> dict:
        if "Count the number of tasks" in server_package.keys():
            self.set_data_parameters(server_package)
            return self.count_task_package()
        else:
            self.set_parameters(server_package)
            self.train_with_eval()
        return self.package()

    def fit(self):
        self.model.train()
        self.dataset.train()

        # 初始化
        self.moe_grad = []
        self.ft_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Train five local views: Global, Task, Data, Perturbed fused, Fused.
        for idx, epoch_params in enumerate(self.model_list_params):

            # --- [Ablation Check] 如果是被消融的模块，直接填 None 并跳过 ---
            if epoch_params is None:
                self.moe_grad.append(None)
                continue

            # 1. 加载参数 (每个分支只加载一次，速度极快)
            self.model.load_state_dict(epoch_params)

            # 2. 训练一个完整的 Epoch (让这个分支充分学习)
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                logit = self.model(x)
                loss = self.ft_criterion(logit, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.trimoe_nofish_nofuzzy.clip_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 3. 计算“伪梯度” (Pseudo-Gradient)
            # 用 (Old_Params - New_Params) 近似梯度方向
            # 这种方法不需要 retain_graph，显存占用极小，速度极快！

            # 必须 detach，防止显存泄露
            grad_outputs = []
            for param_new, param_old in zip(self.model.parameters(), epoch_params.values()):
                grad_outputs.append((param_old - param_new).detach())

            # 4. 对超网络求导
            # 注意：这里我们允许 unused，因为 Global/Task 分支不涉及 EmbedNet
            grads = torch.autograd.grad(
                list(epoch_params.values()), # Outputs
                list(self.embed_net.parameters()) +
                list(self.task_net.parameters()) +
                list(self.hyper_net.parameters()), # Inputs
                grad_outputs=grad_outputs,
                allow_unused=True, # 允许部分参数无梯度 (返回 None)
                retain_graph=True  # 必须保留，因为超网络在下一个分支还要用
            )

            self.moe_grad.append(grads)

            # 清理
            del grad_outputs, grads
            torch.cuda.empty_cache()

        # 循环结束

    def compute_label_distribution(self, dataloader, num_classes):
        counts = torch.zeros(num_classes, dtype=torch.float64)
        total = 0
        for _, labels in dataloader:
            for label in labels.view(-1):
                counts[label.item()] += 1
            total += labels.numel()
        if total == 0:
            return np.zeros(num_classes, dtype=np.float64)
        return (counts / total).numpy()

    def count_task_package(self):
        client_package = super().package()
        data_loader = self.trainloader if len(self.trainset) > 0 else self.testloader
        data_distribution = self.compute_label_distribution(data_loader, len(self.dataset.classes))
        client_package["data_distribution"] = [data_distribution]

        # embedding = torch.zeros(self.args.trimoe_nofish_nofuzzy.embed_dim, device=self.device)
        # size = 0
        # for i, (x, y) in enumerate(self.trainloader):
        #     embedding += self.embed_net(x.to(self.device), y.to(self.device)).sum(dim=0)
        #     size += len(x)
        #     if i + 1 == self.args.trimoe_nofish_nofuzzy.embed_num_batches:
        #         break
        #
        # embedding /= size
        # embedding = (embedding - embedding.mean()) / embedding.std()
        # client_package["embedding"] = embedding

        return client_package

    def package(self):
        client_package = super().package()

        # 1. 准备参数引用
        all_params_refs = list(self.embed_net.parameters()) + \
                          list(self.task_net.parameters()) + \
                          list(self.hyper_net.parameters())

        num_embed_net_params = len(list(self.embed_net.parameters()))
        num_task_embed_net_params = len(list(self.task_net.parameters()))

        # 获取长度
        valid_grad_template = next((g for g in self.moe_grad if g is not None), None)
        if valid_grad_template is None:
            grad_len = len(all_params_refs)
        else:
            grad_len = len(valid_grad_template)

        merged = [None] * grad_len

        # --- [防爆修改 1] 聚合时强制 detach ---
        for k in range(grad_len):
            grads_k = [g[k] for g in self.moe_grad if g is not None and g[k] is not None]

            if len(grads_k) == 0:
                merged[k] = torch.zeros_like(all_params_refs[k]).to(self.device)
            else:
                # 这里的 .detach() 是关键！防止把 fit 阶段的计算图带过来
                s = grads_k[0].detach().clone()
                for t in grads_k[1:]:
                    s += t.detach() # 只加数值，不加图
                merged[k] = s / len(grads_k)

        joint_grads = merged

        # -----------------------------------------------------------------
        # Alignment Loss (内存消耗大户)
        # -----------------------------------------------------------------

        raw_embedding = torch.zeros(self.args.trimoe_nofish_nofuzzy.embed_dim, device=self.device)
        size = 0

        self.embed_net.train()
        self.embed_net.zero_grad()

        # 遍历少量 Batch
        for i, (x, y) in enumerate(self.trainloader):
            # 这里的计算图是必须的，但要小心
            batch_emb = self.embed_net(x.to(self.device), y.to(self.device)).sum(dim=0)
            raw_embedding += batch_emb
            size += len(x)

            # 显存清理：立即删除临时变量
            del batch_emb

            if i + 1 == self.args.trimoe_nofish_nofuzzy.embed_num_batches:
                break

        raw_embedding /= size
        curr_embedding = (raw_embedding - raw_embedding.mean()) / (raw_embedding.std() + 1e-6)

        # 计算 Loss 和 梯度
        target_proto = self.proto.detach()
        align_loss = 1 - F.cosine_similarity(curr_embedding, target_proto, dim=0)

        # create_graph=False (默认), retain_graph=False (默认)
        # 这样算完 align_grads 后，raw_embedding 构建的图就会被释放
        align_grads = torch.autograd.grad(
            align_loss,
            list(self.embed_net.parameters()),
            allow_unused=True,
            retain_graph=False
        )

        # -----------------------------------------------------------------
        # 梯度融合
        # -----------------------------------------------------------------
        beta = self.lam
        # beta = max((1 - self.beta), 0.1)

        # [防爆修改 3] 融合时确保 align_grads 也是 detach 的（通常 autograd.grad 出来就是 detached，但保险起见）
        if align_grads is not None:
            for i in range(num_embed_net_params):
                if joint_grads[i] is not None and align_grads[i] is not None:
                    # 原地操作 (+=) 比 joint_grads[i] = ... 更省内存，但这里为了逻辑清晰用加法
                    joint_grads[i] = joint_grads[i] + beta * align_grads[i].detach()

        # -----------------------------------------------------------------
        # 打包返回
        # -----------------------------------------------------------------
        client_package["embed_net_grads"] = [g for g in joint_grads[:num_embed_net_params]]
        client_package["task_net_grads"] = [g for g in joint_grads[num_embed_net_params:num_embed_net_params + num_task_embed_net_params]]
        client_package["hyper_net_grads"] = [g for g in joint_grads[num_embed_net_params + num_task_embed_net_params:]]
        client_package["embedding"] = self.embedding

        # [防爆修改 5] 手动清理大变量
        del raw_embedding, curr_embedding, align_loss, align_grads, joint_grads, merged, self.moe_grad
        # self.moe_grad 非常大，打包完如果不删，会留到下一轮
        self.moe_grad = []

        # 强制回收显存
        torch.cuda.empty_cache()

        return client_package

    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()

        # 初始化
        self.moe_grad = []
        self.ft_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Train five local views: Global, Task, Data, Perturbed fused, Fused.
        for idx, epoch_params in enumerate(self.model_list_params):

            # --- [Ablation Check] 如果是被消融的模块，直接填 None 并跳过 ---
            if epoch_params is None:
                self.moe_grad.append(None)
                continue

            # 1. 加载参数 (每个分支只加载一次，速度极快)
            self.model.load_state_dict(epoch_params)

            # 2. 训练一个完整的 Epoch (让这个分支充分学习)
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                logit = self.model(x)
                loss = self.ft_criterion(logit, y)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.trimoe_nofish_nofuzzy.clip_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 3. 计算“伪梯度” (Pseudo-Gradient)
            # 用 (Old_Params - New_Params) 近似梯度方向
            # 这种方法不需要 retain_graph，显存占用极小，速度极快！

            # 必须 detach，防止显存泄露
            grad_outputs = []
            for param_new, param_old in zip(self.model.parameters(), epoch_params.values()):
                grad_outputs.append((param_old - param_new).detach())

            # 4. 对超网络求导
            # 注意：这里我们允许 unused，因为 Global/Task 分支不涉及 EmbedNet
            grads = torch.autograd.grad(
                list(epoch_params.values()), # Outputs
                list(self.embed_net.parameters()) +
                list(self.task_net.parameters()) +
                list(self.hyper_net.parameters()), # Inputs
                grad_outputs=grad_outputs,
                allow_unused=True, # 允许部分参数无梯度 (返回 None)
                retain_graph=True  # 必须保留，因为超网络在下一个分支还要用
            )

            self.moe_grad.append(grads)

            # 清理
            del grad_outputs, grads
            torch.cuda.empty_cache()

        # 循环结束

    @torch.no_grad()
    def get_client_embedding(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        return self.embedding

    @torch.no_grad()
    def get_client_z(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        z = self.hyper_net.get_z(self.embedding, self.task_embedding)

        return z

    @torch.no_grad()
    def get_client_proto(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        return self.proto

    @torch.no_grad()
    def get_client_net_embed(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)

        return self.net_embed

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any

class TriBranchHyperNet(nn.Module):
    """
    【结构化增强版】TriBranchHyperNet
    核心创新：
    1. Structured Fading Generation (结构化渐变生成):
       强制 Visual Embedding 指导浅层，Semantic Embedding 指导深层。
    2. Gate Regularization & Smoothing (保留了你之前的平滑/正则逻辑)。
    """

    def __init__(
            self,
            backbone: nn.Module,
            args: Any,
            *,
            beta: float = 0.99,
            gamma: float = 0.15,
            lambda_reg: float = 1e-3,
            target_gate_std: float = 0.18,
            mu_global: float = 0.005
    ):
        super().__init__()

        self.embed_dim = args.trimoe_nofish_nofuzzy.embed_dim
        self.hidden_dim = args.trimoe_nofish_nofuzzy.hyper_hidden_dim
        self.total_epochs = getattr(args.common, 'global_epoch', 1)
        self.current_epoch = 0

        # -------- target parameters -------- #
        self.target_params = [(name, p.shape) for name, p in backbone.named_parameters()]
        self.total_params = sum(int(torch.tensor(s).prod()) for _, s in self.target_params)

        # -------- block-size logic -------- #
        effective_q = int(getattr(args.trimoe_nofish_nofuzzy, "effective_block_size_q", 0) or 0)
        if effective_q > 0:
            self.chunk_size = effective_q
        else:
            base_chunk = args.trimoe_nofish_nofuzzy.chunk_size
            self.chunk_size = max(1, int(base_chunk) // 2)
        self.num_chunks = (self.total_params + self.chunk_size - 1) // self.chunk_size
        self.routing_granularity = getattr(
            args.trimoe_nofish_nofuzzy,
            "routing_granularity",
            "block",
        )
        if self.routing_granularity not in {
            "block",
            "layer_tied",
            "model_tied",
            "adaptive_smooth",
        }:
            raise ValueError(f"Unknown routing_granularity: {self.routing_granularity}")
        self.chunk_metadata = self._build_chunk_metadata()
        layer_names = []
        layer_to_id = {}
        layer_ids = []
        for row in self.chunk_metadata:
            layer = row["primary_layer"]
            if layer not in layer_to_id:
                layer_to_id[layer] = len(layer_names)
                layer_names.append(layer)
            layer_ids.append(layer_to_id[layer])
        self.chunk_primary_layers = layer_names
        self.register_buffer(
            "chunk_layer_group_ids",
            torch.tensor(layer_ids, dtype=torch.long),
            persistent=False,
        )
        if self.routing_granularity == "adaptive_smooth":
            init_probs = torch.tensor([0.80, 0.15, 0.05], dtype=torch.float32)
            init_logits = init_probs.log()
            self.granularity_logits = nn.Parameter(
                init_logits.repeat(len(self.chunk_primary_layers), 1)
            )
        self.routing_intervention = getattr(
            args.trimoe_nofish_nofuzzy,
            "routing_intervention",
            "none",
        )
        self.routing_intervention_seed = int(
            getattr(args.trimoe_nofish_nofuzzy, "routing_intervention_seed", 0)
        )

        # -------- 三路生成网络 -------- #
        self.pers_mlp = self._make_mlp(
            self.embed_dim,
            self.hidden_dim,
            args.trimoe_nofish_nofuzzy.hyper_num_hidden_layers
        )
        self.pers_head = nn.Linear(self.hidden_dim, self.num_chunks * self.chunk_size)

        # gate_mlp
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.num_chunks * 3)
        )

        # -------- global / task transforms -------- #
        self.global_emb = nn.Parameter(torch.randn(self.embed_dim))

        # ============================================================
        # 【核心创新】层级引导向量 (Layer Guidance)
        # 形状: [num_chunks, 1]
        # 值从 0.0 (浅层) 线性增加到 1.0 (深层)
        # 用 register_buffer 保证它保存到模型里，但不作为参数更新
        # ============================================================
        guide = torch.linspace(0, 1, self.num_chunks).view(-1, 1)
        # self.register_buffer('layer_guide', guide)
        self.layer_guide_logits = nn.Parameter(guide)

    def _make_mlp(self, in_d, hid, layers):
        seq = [nn.Linear(in_d, hid), nn.ReLU(True)]
        for _ in range(layers):
            seq.append(nn.Linear(hid, hid))
            seq.append(nn.ReLU(True))
        return nn.Sequential(*seq)

    def _build_chunk_metadata(self):
        param_ranges = []
        cursor = 0
        for name, shape in self.target_params:
            size = int(torch.tensor(shape).prod().item())
            param_ranges.append((name, cursor, cursor + size))
            cursor += size

        rows = []
        for chunk_id in range(self.num_chunks):
            flat_start = chunk_id * self.chunk_size
            flat_end = min(flat_start + self.chunk_size, self.total_params)
            overlaps = []
            for name, param_start, param_end in param_ranges:
                overlap = max(0, min(flat_end, param_end) - max(flat_start, param_start))
                if overlap > 0:
                    overlaps.append((name, overlap))

            primary_param = max(overlaps, key=lambda item: item[1])[0] if overlaps else ""
            primary_layer = primary_param.rsplit(".", 1)[0] if "." in primary_param else primary_param
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "flat_start": flat_start,
                    "flat_end": flat_end,
                    "primary_param": primary_param,
                    "primary_layer": primary_layer,
                    "crosses_param_boundary": int(len(overlaps) > 1),
                }
            )
        return rows

    def _mean_logits_by_layer(self, logits: torch.Tensor) -> torch.Tensor:
        tied = torch.empty_like(logits)
        group_ids = self.chunk_layer_group_ids.to(logits.device)
        for group_id in torch.unique(group_ids):
            mask = group_ids == group_id
            tied[mask] = logits[mask].mean(dim=0, keepdim=True)
        return tied

    def _adaptive_smooth_routing_logits(self, logits: torch.Tensor) -> torch.Tensor:
        block_logits = logits
        layer_logits = self._mean_logits_by_layer(logits)
        model_logits = logits.mean(dim=0, keepdim=True).expand_as(logits)

        group_ids = self.chunk_layer_group_ids.to(logits.device)
        alphas_by_layer = torch.softmax(self.granularity_logits, dim=-1).to(logits.device)
        alpha = alphas_by_layer[group_ids]
        return (
            alpha[:, 0:1] * block_logits
            + alpha[:, 1:2] * layer_logits
            + alpha[:, 2:3] * model_logits
        )

    def _tie_routing_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.routing_granularity == "block":
            return logits
        if self.routing_granularity == "model_tied":
            return logits.mean(dim=0, keepdim=True).expand_as(logits)
        if self.routing_granularity == "layer_tied":
            return self._mean_logits_by_layer(logits)
        if self.routing_granularity == "adaptive_smooth":
            return self._adaptive_smooth_routing_logits(logits)
        raise ValueError(f"Unknown routing_granularity: {self.routing_granularity}")

    def set_routing_intervention(self, mode: str = "none", seed: int = 0):
        mode = "none" if mode is None or str(mode).strip() == "" else str(mode).strip()
        if mode == "learned":
            mode = "none"
        allowed = {
            "none",
            "layer_mean",
            "model_mean",
            "uniform",
            "permute_blocks",
            "reverse_blocks",
            "second_best",
            "hard_argmax",
        }
        if mode not in allowed:
            raise ValueError(f"Unknown routing_intervention: {mode}")
        self.routing_intervention = mode
        self.routing_intervention_seed = int(seed)

    def _apply_routing_intervention(self, logits: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "routing_intervention", "none")
        if mode in {"none", "learned"}:
            return self._tie_routing_logits(logits)
        if mode == "layer_mean":
            return self._mean_logits_by_layer(logits)
        if mode == "model_mean":
            return logits.mean(dim=0, keepdim=True).expand_as(logits)
        if mode == "uniform":
            return torch.zeros_like(logits)
        if mode == "reverse_blocks":
            return torch.flip(logits, dims=[0])
        if mode == "permute_blocks":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(getattr(self, "routing_intervention_seed", 0)))
            perm = torch.randperm(logits.shape[0], generator=generator).to(logits.device)
            return logits[perm]
        if mode == "hard_argmax":
            top = torch.argmax(logits, dim=-1, keepdim=True)
            forced = torch.full_like(logits, -1e9)
            return forced.scatter(1, top, 0.0)
        if mode == "second_best":
            top2 = torch.topk(logits, k=2, dim=-1).indices[:, 1:2]
            forced = torch.full_like(logits, -1e9)
            return forced.scatter(1, top2, 0.0)
        raise ValueError(f"Unknown routing_intervention: {mode}")

    def get_granularity_alphas(self):
        if not hasattr(self, "granularity_logits"):
            return None
        return torch.softmax(self.granularity_logits.detach().cpu(), dim=-1)

    # ============================================================
    # 【核心创新】结构化参数生成器
    # 实现了 "Implicit Alignment"
    # ============================================================
    def _generate_structured_params(self, emb):
        """
        输入: [embed_dim]
        输出: [num_chunks, chunk_size]
        逻辑:
          P_vis = Generate(Emb_Visual_Only)
          P_sem = Generate(Emb_Semantic_Only)
          P_final = (1 - guide) * P_vis + guide * P_sem
        """
        half = self.embed_dim // 2

        # 1. 制造两个视角的输入
        # Visual View: 屏蔽后半段语义信息
        emb_vis = emb.clone()
        emb_vis[half:] = 0

        # Semantic View: 屏蔽前半段视觉信息
        emb_sem = emb.clone()
        emb_sem[:half] = 0

        # 2. 分别生成参数
        # 视觉流生成的参数 (倾向于指导浅层)
        feat_vis = self.pers_mlp(emb_vis)
        P_vis = self.pers_head(feat_vis).view(self.num_chunks, self.chunk_size)

        # 语义流生成的参数 (倾向于指导深层)
        feat_sem = self.pers_mlp(emb_sem)
        P_sem = self.pers_head(feat_sem).view(self.num_chunks, self.chunk_size)

        # 3. 结构化渐变融合 (Linear Fading)
        # 浅层 (guide≈0): 主要取 P_vis
        # 深层 (guide≈1): 主要取 P_sem
        # 中间层: 平滑过渡
        # [num_chunks, 1] * [num_chunks, chunk_size]
        # P_final = (1.0 - self.layer_guide) * P_vis + self.layer_guide * P_sem
        # 动态计算 guide
        # guide = torch.sigmoid(self.layer_guide_logits)
        # P_final = (1.0 - guide) * P_vis + guide * P_sem

        # 增加一个由原始完整 emb 生成的 Base 参数
        feat_base = self.pers_mlp(emb) # 使用完整 emb
        P_base = self.pers_head(feat_base).view(self.num_chunks, self.chunk_size)

        # 融合：Base + 渐变部分
        # 这样即使 VIS/SEM 冲突，模型也可以回退到使用 Base 参数
        guide = torch.sigmoid(self.layer_guide_logits)
        P_final = P_base + 0.5 * ((1.0 - guide) * P_vis + guide * P_sem)

        return P_final

    def compute_routing_weights(self, client_emb: torch.Tensor, task_emb: torch.Tensor,
                                enable_global=True, enable_task=True, enable_client=True):
        """Compute the same per-chunk routing weights used by forward()."""
        raw_g = self.gate_mlp(self.global_emb)
        raw_t = self.gate_mlp(task_emb)
        raw_p = self.gate_mlp(client_emb)

        logits = (raw_g + raw_t + raw_p).view(self.num_chunks, 3)
        logits = self._apply_routing_intervention(logits)

        mask = torch.zeros_like(logits)
        if not enable_global:
            mask[:, 0] = -1e9
        if not enable_task:
            mask[:, 1] = -1e9
        if not enable_client:
            mask[:, 2] = -1e9

        logits = logits + mask
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        return F.softmax(logits, dim=-1)

        logits = logits + mask # 屏蔽掉未开启的分支
        logits = logits - logits.max(dim=-1, keepdim=True)[0] # 数值稳定性
        return F.softmax(logits, dim=-1)

    # exposed helper to retrieve routing weights
    def get_z(self, client_emb: torch.Tensor, task_emb: torch.Tensor):
        return self.compute_routing_weights(client_emb, task_emb)

    def extract_global_params(self):
        # 使用结构化生成
        P = self._generate_structured_params(self.global_emb)

        flat = P.reshape(-1)[:self.total_params]
        params = OrderedDict()
        idx = 0
        for name, shape in self.target_params:
            n = int(torch.tensor(shape).prod())
            params[name] = flat[idx:idx+n].view(*shape)
            idx += n
        return params

    def extract_task_params(self, task_emb):
        # 使用结构化生成
        P = self._generate_structured_params(task_emb)

        flat = P.reshape(-1)[:self.total_params]
        params = OrderedDict()
        idx = 0
        for name, shape in self.target_params:
            n = int(torch.tensor(shape).prod())
            params[name] = flat[idx:idx+n].view(*shape)
            idx += n
        return params

    def extract_data_params(self, data_emb):
        # 使用结构化生成
        P = self._generate_structured_params(data_emb)

        flat = P.reshape(-1)[:self.total_params]
        params = OrderedDict()
        idx = 0
        for name, shape in self.target_params:
            n = int(torch.tensor(shape).prod())
            params[name] = flat[idx:idx+n].view(*shape)
            idx += n
        return params

    def forward(self, client_emb, task_emb,
                enable_global=True, enable_task=True, enable_client=True):
        """
        增加了 enable_* 开关，用于消融实验
        """
        # 1. 生成参数 (只生成开启的部分，节省显存)
        P_g = self._generate_structured_params(self.global_emb) if enable_global else 0.
        P_t = self._generate_structured_params(task_emb) if enable_task else 0.
        P_p = self._generate_structured_params(client_emb) if enable_client else 0.

        z = self.compute_routing_weights(
            client_emb,
            task_emb,
            enable_global=enable_global,
            enable_task=enable_task,
            enable_client=enable_client,
        )
        """

        # 2. 计算 Gate (核心修改：Mask机制)
        # 我们不能只算 softmax，必须先把不开启的分支 logit 设为负无穷
        raw_g = self.gate_mlp(self.global_emb)
        raw_t = self.gate_mlp(task_emb)
        raw_p = self.gate_mlp(client_emb)

        logits = (raw_g + raw_t + raw_p).view(self.num_chunks, 3)

        # --- Ablation Masking Start ---
        mask = torch.zeros_like(logits)
        if not enable_global:
            mask[:, 0] = -1e9
        if not enable_task:
            mask[:, 1] = -1e9
        if not enable_client:
            mask[:, 2] = -1e9

        logits = logits + mask # 屏蔽掉未开启的分支
        # --- Ablation Masking End ---

        # 重新标准化 (注意：如果只开一个分支，softmax后该分支权重恒为1，符合逻辑)
        logits = logits - logits.max(dim=-1, keepdim=True)[0] # 数值稳定性
        z = F.softmax(logits, dim=-1)

        """
        z_g = z[:, 0:1]
        z_t = z[:, 1:2]
        z_p = z[:, 2:3]

        # 3. 融合
        fused = 0.
        if enable_global:
            fused += z_g * P_g
        if enable_task:
            fused += z_t * P_t
        if enable_client:
            fused += z_p * P_p

        # ----- flatten & split ----- #
        flat = fused.reshape(-1)[:self.total_params]
        params = OrderedDict()

        idx = 0
        for name, shape in self.target_params:
            n = int(torch.tensor(shape).prod())
            params[name] = flat[idx:idx+n].view(*shape)
            idx += n

        return params

class TaskEmbedding(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.embeddings = nn.Embedding(
            args.trimoe_nofish_nofuzzy.num_K,
            args.trimoe_nofish_nofuzzy.embed_dim
        )
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, task_id):
        return self.embeddings(task_id)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from argparse import Namespace
from src.utils.constants import INPUT_CHANNELS, NUM_CLASSES

class HierarchicalFusion(nn.Module):
    """
    层级解耦融合模块：
    不把特征完全混在一起，而是保留一部分视觉特性，保留一部分语义特性。
    """
    def __init__(self, dim):
        super().__init__()
        self.half_dim = dim // 2

        # 视觉流的门控：由语义(Label)来决定保留多少视觉信息
        self.vis_gate = nn.Sequential(
            nn.Linear(dim * 2, self.half_dim),
            nn.Sigmoid()
        )
        self.img_proj = nn.Linear(dim, self.half_dim)

        # 语义流的门控：由视觉(Image)来决定语义信息的偏移量
        self.sem_gate = nn.Sequential(
            nn.Linear(dim * 2, self.half_dim),
            nn.Sigmoid()
        )
        self.lbl_proj = nn.Linear(dim, self.half_dim)

    def forward(self, img_feat, lbl_feat):
        # img_feat, lbl_feat: [Batch, 84]

        combined = torch.cat([img_feat, lbl_feat], dim=1)

        # 1. 前半部分 (Visual Part)
        # 基础是图片特征，受标签门控调节
        g_v = self.vis_gate(combined)
        z_vis = self.img_proj(img_feat) * g_v

        # 2. 后半部分 (Semantic Part)
        # 基础是标签特征，受图片门控调节
        g_s = self.sem_gate(combined)
        z_sem = self.lbl_proj(lbl_feat) + g_s * (self.img_proj(img_feat)) # 残差修正

        # 3. 拼接 [Batch, 42+42] -> [Batch, 84]
        # 这样输出的向量，物理含义上就有了区分
        return torch.cat([z_vis, z_sem], dim=1)

class EmbedNetwork(nn.Module):
    def __init__(self, args: Namespace):
        super(EmbedNetwork, self).__init__()
        self.args = args

        # === 1. Image Stream (GroupNorm 防爆) ===
        raw_in_channels = INPUT_CHANNELS[self.args.dataset.name]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(raw_in_channels, self.args.trimoe_nofish_nofuzzy.embed_num_kernels, 5),
            nn.GroupNorm(4, self.args.trimoe_nofish_nofuzzy.embed_num_kernels),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.args.trimoe_nofish_nofuzzy.embed_num_kernels, 2 * self.args.trimoe_nofish_nofuzzy.embed_num_kernels, 5),
            nn.GroupNorm(4, 2 * self.args.trimoe_nofish_nofuzzy.embed_num_kernels),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * self.args.trimoe_nofish_nofuzzy.embed_num_kernels * 5 * 5, 120),
            nn.LayerNorm(120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.LayerNorm(84),
            nn.Tanh() # 限制在 -1~1
        )
        self.resize = transforms.Resize((32, 32))

        # === 2. Label Stream ===
        self.use_y = bool(self.args.trimoe_nofish_nofuzzy.embed_y)
        if self.use_y:
            self.label_encoder = nn.Sequential(
                nn.Embedding(NUM_CLASSES[self.args.dataset.name], 84),
                nn.LayerNorm(84)
            )
            # 使用层级融合
            self.fusion = HierarchicalFusion(dim=84)

        # === 3. Final Output ===
        # 这里不需要再混合了，只是映射维度
        self.final_proj = nn.Linear(84, self.args.trimoe_nofish_nofuzzy.embed_dim)

        # Client Embedding (Proto)
        self.client_embedding = nn.Embedding(
            args.trimoe_nofish_nofuzzy.client_num,
            args.trimoe_nofish_nofuzzy.embed_dim
        )
        nn.init.uniform_(self.client_embedding.weight, -0.1, 0.1)
        # self.alpha_logit = nn.Parameter(torch.tensor(-2.0))

    # 增加一个 helper 函数来获取受限的 Proto
    def get_proto(self, client_id):
        # 取出原始向量
        raw_proto = self.client_embedding(client_id)
        # 【关键修改 2】加上 Tanh
        # 你的 forward 结尾用了 Tanh，这里必须也用 Tanh
        # 这样 MSE Loss 计算的就是两个 (-1, 1) 区间向量的距离，非常科学
        return torch.tanh(raw_proto)

    def forward(self, x, y):
        # 1. Image
        if self.use_y:
            h, w = x.shape[2], x.shape[3]
            if h < 32 or w < 32:
                x = self.resize(x)
        img_feat = self.img_encoder(x)

        # 2. Fusion
        if self.use_y:
            lbl_feat = self.label_encoder(y)
            # 输出的前半截是 Visual，后半截是 Semantic
            combined = self.fusion(img_feat, lbl_feat)
        else:
            combined = img_feat

        # 3. Final Projection
        out = self.final_proj(combined)

        # 4. 终极防爆 + 结构保留
        # Tanh 会独立作用于向量的每个元素，不会破坏前半截和后半截的区别
        out = torch.tanh(out)

        return out
