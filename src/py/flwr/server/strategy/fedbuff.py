# Copyright 2020 Adap GmbH. All Rights Reserved.
# Copyright 2023 William Ashton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy


# flake8: noqa: E501
class FedBuff(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        concurrency: int = 3,
        buffer_size: int = 2,
        staleness_fn: Optional[Callable[int, float]] = None,
    ) -> None:
        """Federated Buffering asynchronous aggregation strategy.

        NOTE: requires server to be in asynchronous mode

        Implementation based on https://arxiv.org/abs/2106.06639

        Parameters
        ----------
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        concurrency : int
            Number of clients that should be training at any one time. Note, additional clients are only added after
            aggregation rounds, so there may be periods with fewer clients in use.
        buffer_size : int
            Number of client updates to collect before running aggregation.
        staleness_fn : Optional[Callable[int, float]]
            Function that takes the age of an update in aggregation rounds (where 0 is no missed rounds) and outputs
            a multiplier used to scale the gradients from that update. Defaults to returning 1.0 for all inputs.
        """
        super().__init__()

        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        self.concurrency = concurrency
        self.buffer_size = buffer_size

        if staleness_fn is None:
            self.staleness_fn = lambda x: 1.0
        else:
            self.staleness_fn = staleness_fn

        self.busy_clients = {}  # dict from cid to server round

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_additional_clients = self.concurrency - len(self.busy_clients)
        return num_additional_clients, num_additional_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> Tuple[int, List[Tuple[ClientProxy, FitIns]]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Save so can work out gradients
        self.current_params_ndarray = parameters_to_ndarrays(parameters)

        print("Working out which clients to instruct")

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        print(f"Want {sample_size} more clients, minimum {min_num_clients}")

        occupied_clients = self.busy_clients.keys()

        class NotBusyCriterion(Criterion):
            """Criterion to select only non busy clients."""

            def select(self, client: ClientProxy) -> bool:
                is_not_busy = True if client.cid not in occupied_clients else False
                return is_not_busy

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=NotBusyCriterion(),
        )
        print(f"Selected clients = {[c.cid for c in clients]}")

        self.busy_clients.update({c.cid: server_round for c in clients})

        # Return client/config pairs
        return self.buffer_size, [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # No federated evaluation
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, Tuple[ClientProxy, FitRes]]],
        failures: List[Tuple[int, Union[Tuple[ClientProxy, FitRes], BaseException]]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        ## Split off IDs from results and failures
        success_cids = [res[0] for res in results]
        results = [res[1] for res in results]
        # How many rounds off each result is
        staleness = [server_round - self.busy_clients[c] for c in success_cids]

        print(
            f"These clients sent updates: {[(id,age) for id,age in zip(success_cids,staleness)]}"
        )
        for c in success_cids:
            self.busy_clients.pop(c)

        fail_cids = [fail[0] for fail in failures]
        failures = [fail[1] for fail in failures]
        print(f"These clients failed: {fail_cids}")
        for c in fail_cids:
            self.busy_clients.pop(c)

        # Convert results to list of (delta,weight) tuples
        deltas_results = [
            (
                # Need to do each layer separately as may be different sizes
                [
                    new_layer_params - current_layer_params
                    for new_layer_params, current_layer_params in zip(
                        parameters_to_ndarrays(fit_res.parameters),
                        self.current_params_ndarray,
                    )
                ],
                # Weight by num_examples scaled according to staleness
                self.staleness_fn(age) * fit_res.num_examples,
            )
            for age, (_, fit_res) in zip(staleness, results)
        ]

        # Need to add each layer delta seperately
        self.current_params_ndarray = [
            layer_delta + current_layer_params
            for layer_delta, current_layer_params in zip(
                aggregate(deltas_results),
                self.current_params_ndarray,
            )
        ]

        parameters_aggregated = ndarrays_to_parameters(self.current_params_ndarray)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # No federated evaluation
        return None, {}
