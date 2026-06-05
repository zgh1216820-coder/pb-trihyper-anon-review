from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from data.utils.datasets import BaseDataset
from src.utils.functional import evaluate_model, get_optimal_cuda_device
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel

import torch.nn as nn

class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: DictConfig,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.eval_results = {}

        self.return_diff = return_diff

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation
        results.

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        eval_results["before"] = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            eval_results["after"] = self.evaluate()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.test.client.train, self.trainset],
            ["val", "green", self.args.common.test.client.val, self.valset],
            ["test", "cyan", self.args.common.test.client.test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    f"client [{self.client_id}]\t"
                    f"[{color}]({split}set)[/{color}]\t"
                    f"[red]loss: {eval_results['before'][split].loss:.4f} -> "
                    f"{eval_results['after'][split].loss:.4f}\t[/red]"
                    f"[blue]accuracy: {eval_results['before'][split].accuracy:.2f}% -> {eval_results['after'][split].accuracy:.2f}%[/blue]"
                )

        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if (
            package["optimizer_state"]
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

    def train(self, server_package: dict[str, Any]) -> dict:
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server. You can
        override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        """Evaluating client model.

        Args:
            model: Used model. Defaults to None, which will fallback to `self.model`.

        Returns:
            A evalution results dict: {
                `train`: results on client training set.
                `val`: results on client validation set.
                `test`: results on client test set.
            }
        """
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if (
            len(self.testset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.test
        ):
            test_metrics = evaluate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.valset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.val
        ):
            val_metrics = evaluate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.trainset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.train
        ):
            train_metrics = evaluate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        """Test client model. If `finetune_epoch > 0`, `finetune()` will be
        activated.

        Args:
            server_package: Parameter package.

        Returns:
            A model evaluation results dict : {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        self.testing = True
        self.set_parameters(server_package)

        results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        results["before"] = self.evaluate()

        if self.args.dataset.split == "user" and self.args.method not in ['trimoe_nofish_nofuzzy', "pefll", "pfedhn"]:
            finetune_epoch = self.args.common.test.client.finetune_epoch
            self.args.common.test.client.finetune_epoch = max(
                1, self.args.common.test.client.finetune_epoch
            )
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            self.model.load_state_dict(frz_params_dict)
            self.args.common.test.client.finetune_epoch = finetune_epoch

        elif self.args.common.test.client.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            self.model.load_state_dict(frz_params_dict)

        elif self.args.dataset.split == "user" and self.args.method in ["pfedhn"]:
            # --- pFedHN 特有的新客户端适配逻辑 ---
            if "hn_state_dict" in server_package:
                # 1. 在本地临时构建一个超网络副本
                # 注意：这里需要确保客户端有 HyperNetwork 的类定义
                from src.server.pfedhn import HyperNetwork
                local_hn = HyperNetwork(self.model, self.args.pfedhn, server_package["client_num"])
                local_hn.load_state_dict(server_package["hn_state_dict"])
                local_hn.to(self.device)

                embed_lr = (
                    self.args.pfedhn.embed_lr
                    if self.args.pfedhn.embed_lr is not None
                    else self.args.pfedhn.hn_lr
                )
                self.hn_optimizer = torch.optim.SGD(
                    [
                        {
                            "params": [
                                param
                                for name, param in local_hn.named_parameters()
                                if "embed" not in name
                            ]
                        },
                        {
                            "params": [
                                param
                                for name, param in local_hn.named_parameters()
                                if "embed" in name
                            ],
                            "lr": embed_lr,
                        },
                    ],
                    lr=self.args.pfedhn.hn_lr,
                    momentum=self.args.pfedhn.hn_momentum,
                    weight_decay=self.args.pfedhn.hn_weight_decay,
                )

                # 通过 v 生成权重 (注意：这里不能 detach，要保持 v 的梯度链)
                # 需要修改 HyperNetwork 的 forward 逻辑，使其支持直接传入 v
                # 或者在这里手动实现 MLP 生成逻辑
                weights = OrderedDict(
                    zip(
                        server_package["ublic_model_param_names"],
                        local_hn(torch.tensor(server_package["client_id"]).to(self.device)),
                    )
                )
                # 将生成的权重加载到本地模型
                self.model.load_state_dict(weights)
                self.model.to(self.device)
                self.model.train()
                self.dataset.train()

                # 3. 进行 k 步适配优化 v
                for _ in range(server_package["k_steps"]):
                    for x, y in self.trainloader:
                        if len(x) <= 1:
                            continue

                        x, y = x.to(self.device), y.to(self.device)
                        logit = self.model(x)
                        loss = self.criterion(logit, y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    hn_grads = torch.autograd.grad(
                        outputs=local_hn.outputs,
                        inputs=local_hn.parameters(),
                        grad_outputs=[
                            (param_old.to(self.device) - param_new.to(self.device)).detach()
                            for param_new, param_old in zip(
                                self.model.parameters(), weights.values()
                            )
                        ],
                        allow_unused=True,
                        retain_graph=True
                    )

                    self.hn_optimizer.zero_grad()

                    # 2. 遍历参数并进行过滤赋值
                    target_embedding_param = local_hn.embedding.weight

                    for param, grad in zip(local_hn.parameters(), hn_grads):
                        if param is target_embedding_param:
                            # 仅为 embedding 权重赋值梯度
                            param.grad = grad
                        else:
                            # 将 MLP 和生成器部分的梯度显式设为 None，确保它们不会被更新
                            param.grad = None

                    # 3. 梯度裁剪与更新（仅更新 embedding）
                    torch.nn.utils.clip_grad_norm_([target_embedding_param], self.args.pfedhn.norm_clip)
                    self.hn_optimizer.step()

                # 4. 适配完成后，使用最终的 v 生成测试权重
                final_weights = OrderedDict(
                    zip(
                        server_package["ublic_model_param_names"],
                        local_hn(torch.tensor(server_package["client_id"]).to(self.device)),
                    )
                )
                self.model.load_state_dict(final_weights, strict=False)
                results["after"] = self.evaluate()

        self.testing = False
        return results

    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
