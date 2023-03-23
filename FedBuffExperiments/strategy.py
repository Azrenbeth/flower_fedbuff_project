#!/usr/bin/env python
# -*-coding:utf-8 -*-

# @File    :   client.py
# @Time    :   2023/01/21 11:36:46
# @Author  :   Alexandru-Andrei Iacob
# @Contact :   aai30@cam.ac.uk
# @Author  :   Lorenzo Sani
# @Contact :   ls985@cam.ac.uk, lollonasi97@gmail.com
# @Version :   1.0
# @License :   (C)Copyright 2023, Alexandru-Andrei Iacob, Lorenzo Sani
# @Desc    :   None

from typing import Callable, Dict, List, Optional, Tuple, Union

from logging import INFO

import numpy as np

from flwr.common import (
    FitIns,
    EvaluateIns,
    FitRes,
    EvaluateRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import FedAvgM, FedBuff

from flwr.server.criterion import Criterion

from client_manager import CustomClientManager


# flake8: noqa: E501
class FedAvgTraces(FedAvgM):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            server_learning_rate=server_learning_rate,
            server_momentum=server_momentum,
        )
        self.current_virtual_clock = 0.0

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # Add virtual clock to config
        config["current_virtual_clock"] = self.current_virtual_clock
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            current_virtual_clock=self.current_virtual_clock,
            server_round=server_round,
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # Add virtual clock to config
        config["current_virtual_clock"] = self.current_virtual_clock
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            current_virtual_clock=self.current_virtual_clock,
            server_round=server_round,
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # TODO: add printing failures to the result metrics.
        # First, try to print `failures` to see its content.
        # Second, see whether adding `failures` to `results` makes sense.
        # Alternatively, modify the `fit_metrics_aggregation_fn` to receive `failures` as well.
        # The objective is to make the `failures` visible to the user in the outputs.
        self._increase_current_virtual_clock(results)  # type: ignore
        return super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        self._increase_current_virtual_clock(results)  # type: ignore
        return super().aggregate_evaluate(
            server_round=server_round,
            results=results,
            failures=failures,
        )

    def _increase_current_virtual_clock(
        self,
        results: List[Tuple[ClientProxy, Union[FitRes, EvaluateRes]]],
    ) -> None:
        client_completion_times = [
            res.metrics["client_completion_time"] for _, res in results
        ]
        log(INFO, f"Completion times of clients: {client_completion_times}")
        max_client_completion_time = np.max(client_completion_times)
        log(INFO, f"Maximum completion time of clients: {max_client_completion_time}")
        self.current_virtual_clock += max_client_completion_time

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f"Server round {server_round}")
        eval_res = super().evaluate(server_round=server_round, parameters=parameters)
        if eval_res is None:
            metrics = {"eval_time": self.current_virtual_clock}
            return 0.0, metrics
        else:
            loss, metrics = eval_res
            metrics["eval_time"] = self.current_virtual_clock
            return loss, metrics


# flake8: noqa: E501
class FedBuffTraces(FedBuff):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
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
        super().__init__(
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            concurrency=concurrency,
            buffer_size=buffer_size,
            staleness_fn=staleness_fn,
        )
        self.current_virtual_clock = 0.0
        # Results that arrived early (i.e. "after" current virtual clock time)
        self.unused_results = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # Add virtual clock to config
        config["current_virtual_clock"] = self.current_virtual_clock
        fit_ins = FitIns(parameters, config)

        # Save so can work out gradients
        self.current_params_ndarray = parameters_to_ndarrays(parameters)

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
            current_virtual_clock=self.current_virtual_clock,
            server_round=server_round,
            criterion=NotBusyCriterion(),
        )
        print(f"Selected clients = {[c.cid for c in clients]}")

        self.busy_clients.update({c.cid: server_round for c in clients})

        # Return client/config pairs and buffer_size
        # On first round want ALL responses (to ensure that strategy can select the ones
        # that finish with the earliest virtual clock times)
        if server_round == 1:
            buffer_size = self.concurrency
        else:
            buffer_size = self.buffer_size
        return buffer_size, [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
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
        # Store results from all clients (with no failures should always
        # have length of concurrency after this line)
        self.unused_results += [
            (res[1][1].metrics["client_completion_time"], res) for res in results
        ]
        # Order by completion time
        self.unused_results.sort(key=lambda x: x[0])

        # Only pass along the first K to return (or fewer if many failures)
        actual_results = []
        while (len(actual_results) < self.buffer_size) and (
            len(self.unused_results) > 0
        ):
            time, res = self.unused_results.pop(0)
            actual_results.append(res)

        self._increase_current_virtual_clock(actual_results)  # type: ignore

        # NOTE: No need to withold failures until the correct time. As these are either simulation
        # bugs (out of memory) in which case it is fine to start that client again (rare) OR this
        # is the client dropping out before it can finish fitting (in which case the ActivityCriterion)
        # used in the experiment should prevent it being selected until it is "available" again
        return super().aggregate_fit(
            server_round=server_round,
            results=actual_results,
            failures=failures,
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        return None, {}

    def _increase_current_virtual_clock(
        self, results: List[Tuple[int, Tuple[ClientProxy, FitRes]]]
    ) -> None:
        # print(results)
        client_completion_times = [
            res.metrics["client_completion_time"] for _, (_, res) in results
        ]
        log(INFO, f"Completion times of clients: {client_completion_times}")
        max_client_completion_time = np.max(client_completion_times)
        log(INFO, f"Maximum completion time of clients: {max_client_completion_time}")
        self.current_virtual_clock += max_client_completion_time

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f"Server round {server_round}")
        eval_res = super().evaluate(server_round=server_round, parameters=parameters)
        if eval_res is None:
            metrics = {"eval_time": self.current_virtual_clock}
            return 0.0, metrics
        else:
            loss, metrics = eval_res
            metrics["eval_time"] = self.current_virtual_clock
            return loss, metrics


class DeterministicSampleFedAvg(FedAvgM):
    """Configurable FedAvg strategy implementation."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        [print(f) for f in failures]
        return super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round,
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            server_round=server_round,
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
