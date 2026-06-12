# power-system-simulation

This is a student project for Power System Simulation.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

In the root of the repository, sync all dependencies using:

```shell
uv sync
```

After installation, run the test.

```shell
uv run pytest
```

## Code style and quality check

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

You can run the following command to check and auto-fix code issues:

```shell
uv run ruff check --fix .
```

You can run the following command to format your code:

```shell
uv run ruff format .
```

## Working with Jupyter Notebooks

Jupyter notebooks in the `example/` folder can be opened directly in VS Code. The project includes `ipykernel` in the development dependencies, which allows VS Code to run notebook cells using the `.venv` environment.

## Folder structure of the repository

The folder structure of the repository is explained as below.

* [`src/power_system_simulation`](./src/power_system_simulation) is the main folder of the package. You should put your new functionality code there.
* [`tests`](./tests) is the folder containing the test files. You should put your test code there.
* [`example`](./example) contains the example notebook. You should modify the notebook for your presentation.
* [`.vscode`](./.vscode) contains the setting file for the IDE VSCode.
* [`.github/workflows`](./.github/workflows) contains the continuous integration (CI) configurations.

## Graph processing

`GraphProcessor` uses a table with one row per edge. This avoids keeping edge
IDs, endpoints, and statuses aligned across separate lists.

```python
import pandas as pd

from power_system_simulation.graph_processing import GraphProcessor

edge_table = pd.DataFrame(
    {
        "edge_id": [1, 2, 3],
        "from_vertex": [0, 1, 0],
        "to_vertex": [1, 2, 2],
        "enabled": [True, True, False],
    }
)
graph = GraphProcessor(edge_table, source_vertex_id=0)
```
