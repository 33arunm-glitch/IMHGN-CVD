from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple
import math

import numpy as np


@dataclass
class BatState:
    position: np.ndarray
    velocity: np.ndarray
    frequency: float
    loudness: float
    pulse_rate: float
    fitness: float = -np.inf


class SearchSpace:
    def __init__(self, spec: Mapping[str, Mapping[str, Any]]):
        self.spec = dict(spec)
        self.names = list(self.spec.keys())
        if not self.names:
            raise ValueError("ABO search space is empty.")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=len(self.names))

    def decode(self, position: np.ndarray) -> Dict[str, Any]:
        pos = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
        out = {}
        for i, name in enumerate(self.names):
            s = self.spec[name]
            typ = s["type"]
            u = float(pos[i])
            if typ == "float":
                out[name] = float(s["low"] + u * (s["high"] - s["low"]))
            elif typ == "log_float":
                lo, hi = math.log(float(s["low"])), math.log(float(s["high"]))
                out[name] = float(math.exp(lo + u * (hi - lo)))
            elif typ == "int":
                lo, hi = int(s["low"]), int(s["high"])
                out[name] = int(round(lo + u * (hi - lo)))
            elif typ == "categorical":
                choices = list(s["choices"])
                idx = min(len(choices) - 1, int(math.floor(u * len(choices))))
                out[name] = choices[idx]
            else:
                raise ValueError(f"Unsupported search-space type '{typ}' for {name}.")
        return out


class AdaptiveBatOptimizer:
    """
    Bounded Adaptive Bat Optimization for mixed hyperparameters.

    Positions are normalized to [0, 1]^d and decoded by SearchSpace.
    """

    def __init__(
        self,
        search_space: Mapping[str, Mapping[str, Any]],
        *,
        population_size: int = 8,
        iterations: int = 8,
        loudness: float = 0.9,
        pulse_rate: float = 0.1,
        alpha: float = 0.95,
        gamma: float = 0.9,
        frequency_min: float = 0.0,
        frequency_max: float = 2.0,
        seed: int = 42,
    ):
        self.space = SearchSpace(search_space)
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.initial_loudness = float(loudness)
        self.initial_pulse = float(pulse_rate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.fmin = float(frequency_min)
        self.fmax = float(frequency_max)
        self.rng = np.random.default_rng(seed)

    def optimize(self, objective: Callable[[Dict[str, Any]], float]):
        bats = []
        global_best = None
        best_position = None
        best_fitness = -np.inf
        history = []

        for _ in range(self.population_size):
            pos = self.space.sample(self.rng)
            cfg = self.space.decode(pos)
            fit = float(objective(cfg))
            bat = BatState(
                position=pos,
                velocity=np.zeros_like(pos),
                frequency=0.0,
                loudness=self.initial_loudness,
                pulse_rate=self.initial_pulse,
                fitness=fit,
            )
            bats.append(bat)
            if fit > best_fitness:
                best_fitness = fit
                best_position = pos.copy()

        for t in range(1, self.iterations + 1):
            mean_loudness = float(np.mean([b.loudness for b in bats]))

            for bat in bats:
                beta = self.rng.random()
                bat.frequency = self.fmin + (self.fmax - self.fmin) * beta
                bat.velocity = bat.velocity + (bat.position - best_position) * bat.frequency
                candidate = np.clip(bat.position + bat.velocity, 0.0, 1.0)

                # Adaptive local search around current best.
                if self.rng.random() > bat.pulse_rate:
                    epsilon = self.rng.normal(0.0, 1.0, size=candidate.shape)
                    step = mean_loudness / math.sqrt(t + 1.0)
                    candidate = np.clip(best_position + step * epsilon, 0.0, 1.0)

                cfg = self.space.decode(candidate)
                fit = float(objective(cfg))

                # Accept improvements; loudness also controls occasional accepted local moves.
                if (fit >= bat.fitness) and (self.rng.random() < bat.loudness):
                    bat.position = candidate
                    bat.fitness = fit
                    bat.loudness *= self.alpha
                    bat.pulse_rate = self.initial_pulse * (1.0 - math.exp(-self.gamma * t))

                if fit > best_fitness:
                    best_fitness = fit
                    best_position = candidate.copy()

            history.append(
                {
                    "iteration": t,
                    "best_fitness": float(best_fitness),
                    "best_config": self.space.decode(best_position),
                }
            )

        return self.space.decode(best_position), float(best_fitness), history
