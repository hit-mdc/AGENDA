## Installation

The system uses Python 3.11+, Streamlit, Plotly, NumPy, and pandas. Data collections, DSA methods, and D3S indexes are registered through one configuration file. The adapters prepare their artifacts from the registered data and expose them to the four interactive workspaces.

#### Install the system

```bash
git clone https://github.com/paradox0503/AGENDA.git
cd AGENDA
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the system and the dependencies needed by your adapters:

```bash
python -m pip install -r requirements.txt
```

Alternatively, run `conda env create -f environment.yml` and `conda activate AGENDA`.

#### Register datasets, methods, and indexes

Copy `config.example.json` to `config.json`, then configure the registrations. Paths in the configuration are relative to its directory. The `artifacts` setting specifies the output root. Each registration has a unique `id`; its `name` is displayed in the interface.

**Datasets.** The `data` entry points to a numeric `[number_of_series, series_length]` matrix. Use `format: "npy"` for a NumPy array or `format: "f32"` with `shape: [number_of_series, series_length]` for a row-major little-endian float32 file. Each collection may have its own length and number of series. Rows must contain finite values. Apply the desired normalization consistently to indexed series and queries before import.

The `queries` entry supports two forms:

```json
{"path": "data/query_ids.npy", "kind": "ids"}
```

This is a one-dimensional NumPy integer array of row IDs in the data matrix. Queries use the corresponding series, and search excludes that row from the answers.

```json
{"path": "data/queries.npy", "kind": "series", "format": "npy"}
```

This is a separate query matrix with the same series length as the collection. Separate queries are assigned internal negative handles and are evaluated against all indexed rows. Omitting `queries` makes collection rows available as query examples. Single Query also accepts a one-column CSV in the same input space.

**DSA methods.** Each method specifies `dimensions`, `factory`, and `options`. A factory such as `adapters.encoder:EncoderAdapter` names an importable Python class; `options` are passed to its constructor. The adapter can wrap an installed library, a trained model, or a separately managed service. Its interface is defined by `DSAPlugin` in `backend/plugins.py`:

| Method | Responsibility |
| --- | --- |
| `fit(data, artifact_dir)` | Train, adapt, or initialize the method on the supplied collection and save the state needed for subsequent inference. A pretrained integration may initialize from its configured checkpoint. |
| `load(artifact_dir)` | Restore that state. |
| `transform(rows)` | Return a finite numeric array of shape `[len(rows), dimensions]`. The pipeline calls this in batches. |
| `approximate(query, cardinality)` | Return `backend.contracts.Approximation` with one query's `coefficients`, optional `reconstructed` values, and optional integer `symbols`. Use empty arrays for optional outputs. Query and batch encodings must share the same representation space. |

Additional catalog fields include `group`, `description`, and optional `flops` (GFLOPs) and `params` (millions of parameters). These fields populate method cards and efficiency views. The sidebar reads all methods from the registry. Optional `presets` map preset names to lists of method IDs.

**D3S indexes.** Each index has a `factory` and constructor `options`. The `IndexPlugin` interface in `backend/plugins.py` defines:

| Method | Responsibility |
| --- | --- |
| `build(data, representations, artifact_dir)` | Build and persist the index for the supplied collection and DSA representation matrix. |
| `load(data, representations, artifact_dir)` | Open its persisted files for queries. |
| `describe()` | Return the nodes and physical statistics used by the index explorer. |
| `search_curve(request)` | Execute a budgeted query and return snapshots keyed by the requested budgets. |

`IndexRequest` contains the raw query, encoded query, indexed data, `top_k`, `exclude_id`, requested `budgets_ms`, and measured `encoding_ms`. Budgets include encoding time. The plugin returns a distinct `SearchResult` for each requested budget, capturing the answers and counters at that point in the same execution. An exhausted budget yields the answers obtained so far. Plugins may include index traversal, I/O, and candidate distance work according to the implementation's search procedure.

`SearchResult` contains:

- `ids`: unique integer row IDs into the indexed collection, excluding `exclude_id`.
- `distances`: the corresponding finite distances, in answer order.
- `nodes`: the index nodes used by the replay view.
- `trace`: ordered traversal events up to that snapshot.
- `metrics`: query timings and work counters measured by the adapter.

For timings, `metrics["estimated_ms"]` is the measured total elapsed query time at the snapshot, including encoding, rather than a predicted duration. Additional fields are `io_ms`, `candidates` (raw distance computations), `lb_checks`, `leaves_read`, `pruned_leaves`, `pruned_nodes`, `pruned_series`, and `series_excluded_by_lb`. The wrapper records measured query encoding as `encoding_ms`. Adapters should report the counters their implementation measures; these counters drive the tables and charts.

`describe()` returns `{"nodes": [...], "leaf_pages": integer}`. Every node has `id`, `parent`, `children`, `label`, `depth`, and `ids` (representative indexed row IDs). Node IDs are unique nonnegative integers, parents precede children, and roots have `parent: null`. Optional fields include `size`, `leaf_pages`, `cardinalities`, `split_dimensions`, and `index_type`. Supply a tree hierarchy for the structure visualization; an index with another topology can expose a tree projection for this view.

Each trace event contains `node`, `action`, and `label`. Supported action labels are `expand`, `scan`, `partial scan`, `prune`, `heuristic prune`, and `priority queue`. Events can include `lower_bound`, `bsf`, `comparison_bsf`, `distance_computations`, `pruned_series`, `cumulative_distance_computations`, and `cumulative_pruned_series`. These values appear in hover details and current/cumulative replay cards. The index plugin determines each event and its statistics from the query execution.

The six dataset collections, nine full DSA methods, and three index families described below can be registered through this same interface, together with additional collections or methods. Registrations determine which combinations appear in the system. Each registered implementation supplies its own model and index-specific behavior through the common contracts.

#### Add a DSA method or D3S index

Place local integration code in the `adapters/` package. For example, adding a DSA method called MyDSA and a D3S index called MyIndex can use this layout:

```text
AGENDA/
    app.py
    prepare.py
    config.json
    adapters/
        __init__.py
        my_dsa.py       # defines MyDSAAdapter
        my_index.py     # defines MyIndexAdapter
```

Implement the DSA lifecycle methods listed above in `MyDSAAdapter` and the index lifecycle methods in `MyIndexAdapter`. Then append the following objects to the `methods` and `indexes` arrays in `config.json`, respectively:

```json
{
  "id": "my_dsa",
  "name": "MyDSA",
  "dimensions": 16,
  "factory": "adapters.my_dsa:MyDSAAdapter",
  "options": {}
}
```

```json
{
  "id": "my_index",
  "name": "MyIndex",
  "factory": "adapters.my_index:MyIndexAdapter",
  "options": {}
}
```

The `factory` syntax is **`python.module.path:ClassName`**. For `adapters.my_dsa:MyDSAAdapter`, the loader imports `adapters/my_dsa.py`, locates `MyDSAAdapter`, and instantiates it with the keyword arguments in `options`. This is equivalent to:

```python
from adapters.my_dsa import MyDSAAdapter

adapter = MyDSAAdapter(**registration["options"])
```

| Name | Purpose | Naming rule |
| --- | --- | --- |
| `id` | Internal selection key and generated artifact directory name | Unique within its registry; letters, digits, underscores, and hyphens |
| `name` | Human-readable label in the frontend | Any descriptive display name |
| Module filename | Location of the adapter code | An importable Python module, conventionally `snake_case.py` |
| Class name | Adapter instantiated by the loader | An importable callable, conventionally a `PascalCase` class |
| `factory` | Connection from registration to implementation | Must match the actual module path and class name |

The registration ID, filename, and class name may differ. The `factory` field connects them explicitly. An installed package can also supply an adapter, for example `my_package.encoders:MyDSAAdapter`. Run commands from the repository root so local `adapters` modules are importable, or install the integration package in the same Python environment. Paths in adapter `options` are passed directly to its constructor; the adapter resolves any checkpoint or library-specific paths it accepts.

The preparation call path is:

```text
python prepare.py --config config.json
    -> backend.registry.read_config(): read registrations
    -> backend.pipeline.prepare(): load each selected dataset
    -> backend.registry.create(): import the DSA factory and instantiate it
    -> DSA.fit(data, artifact_dir)
    -> DSA.transform(batch): write representations.npy
    -> backend.registry.create(): import the index factory and instantiate it
    -> Index.build(data, representations, artifact_dir)
```

On subsequent preparations, matching artifacts are reused. At runtime, the call path is:

```text
python -m streamlit run app.py
    -> frontend reads the registered datasets, methods, and indexes
    -> user selects a dataset / DSA / index combination
    -> ExternalModelProvider: instantiate the DSA and call DSA.load(...)
    -> ExternalIndexProvider: instantiate the index and call Index.load(...)
    -> Index.describe(): supply the index structure for visualization
    -> user runs a query
    -> DSA.approximate(...): encode the query
    -> Index.search_curve(IndexRequest(...)): return actual search snapshots
    -> backend.evaluation: compute coverage, recall, and matching diagnostics
    -> frontend renders tables, representations, neighbors, and query replay
```

`backend/registry.py` implements the dynamic import, `backend/pipeline.py` calls the preparation lifecycle, and `backend/external.py` bridges the adapters to the frontend. Adding a registration makes the method available to the existing menus and comparison views.

#### Prepare artifacts

```bash
python prepare.py --config config.json
```

The pipeline reads each registered collection, initializes each selected DSA adapter, batch-encodes its series, and builds each selected index over those representations. The generated files have the following organization:

```text
artifacts/<dataset_id>/<method_id>/
    manifest.json
    representations.npy
    <adapter-managed model files>
    indexes/<index_id>/
        manifest.json
        <adapter-managed index files>
```

Existing artifacts are reused when their configuration and source-file metadata match. Input file paths, sizes, modification times, and registration options determine the signature. Use `--force` after changing adapter code or model checkpoints, or whenever artifacts need rebuilding. Select individual registrations with:

```bash
python prepare.py --config config.json --datasets collection --methods encoder --indexes search_index
python prepare.py --config config.json --force
```

Install or place each adapter module on Python's import path before preparing or launching. The `adapters/` directory is a convenient location for local integration modules. Constructor options, checkpoint locations, and method-specific hyperparameters are supplied by the adapter registration.

#### Launch the interface

```bash
# Linux / macOS: optional when using the default config.json
export AGENDA_CONFIG=config.json
# Windows PowerShell equivalent:
# $env:AGENDA_CONFIG="config.json"
python -m streamlit run app.py
```

Open <http://localhost:8501>. **Explore DSA methods** displays registered methods and comparison selections. **Explore D3S indexes** loads the selected artifacts and exposes structure, partitions, raw series, and reduced representations. **Query Set** evaluates a shared workload. **Single Query** provides representation views, neighborhood projections, matched and missed neighbors, distance diagnostics, and synchronized index traversal.

Coverage is computed from exact Euclidean neighbors in representation space against exact Euclidean neighbors in original series space. Recall is computed from the index plugin's returned IDs against that same original-space reference. Matching uses one-to-one original-space distance tolerance. Reference-neighbor computation is evaluation work outside the index-query timer. Query inference is timed by the system; the index adapter supplies measured search timings, counters, and event snapshots.

The callable interfaces are in `backend/contracts.py` and `backend/plugins.py`; configuration loading is in `backend/registry.py`; preparation is in `backend/pipeline.py`; runtime adapters are in `backend/external.py`; evaluation is in `backend/evaluation.py`.

In the following section, we describe the datasets and baseline methods considered in our benchmark and evaluation.

