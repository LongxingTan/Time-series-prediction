"""Run a small STGCN forecast with an explicitly constructed graph."""

import numpy as np
import tensorflow as tf

from tfts import AutoConfig, AutoModelForForecasting, GraphStructure, TimeSeriesBatch, from_knn


def main():
    coordinates = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    graph = from_knn(coordinates, k=2)
    batch = TimeSeriesBatch(
        past_values=tf.random.normal([2, 24, 4, 1]),
        structure=GraphStructure(num_nodes=4, adjacency=graph.adjacency, node_ids=("nw", "ne", "sw", "se")),
    )
    model = AutoModelForForecasting.from_config(AutoConfig.for_model("stgcn"), prediction_length=6)
    forecast = model(batch)
    print(forecast.shape)


if __name__ == "__main__":
    main()
