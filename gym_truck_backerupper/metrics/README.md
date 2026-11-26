# training_metrics

A lightweight submodule for storing, loading and managing training episode metrics using CSV files.

Each training session is associated with a CSV file that is automatically created or loaded on first access. Records are appended to disk immediately when added. CSV files use `;` as a delimiter.

## Features

- Store metrics for multiple training sessions
- Automatically create a CSV file when accessing a new training
- Load existing training metrics from a directory on startup
- Append new episode records to disk immediately
- Validate CSV headers when loading existing files

## Data Structure

Each episode is represented by a `TrainingEpisodeRecord`:

| Field | Type | Description |
|---|---|---|
| `episode_id` | `int` | Episode identifier |
| `reward` | `float` | Episode reward |
| `loss` | `float` | Episode loss |
| `was_success` | `bool` | Whether the episode was successful |
| `precision` | `float` | Episode driving precision |
| `stability` | `float` | Episode driving stability |

## Usage

### Import

```python
from metrics import TrainingMetricsDirectory, TrainingEpisodeRecord
```

### Create a metrics directory

```python
metrics = TrainingMetricsDirectory("./training_metrics")
```

The directory is created if it does not exist. Existing CSV files are loaded automatically.

### Access a training document

```python
document = metrics.get_document("my_training") # for example, metrics.get_document("proximal_policy")
```

If no CSV file exists for that training name, one is created automatically.

### Add an episode record

```python
record = TrainingEpisodeRecord(
    episode_id=1,
    reward=100.0,
    loss=0.05,
    was_success=True,
    precision=0.95,
    stability=0.90
)

document.add_record(record)
```

The record is immediately appended to the CSV file.

### Retrieve all records

```python
for record in document.all_records:
    print(record)
```

## CSV Format

Each training run is stored in a separate `;`-delimited CSV file.

**Example — `my_training.csv`:**

```
episode_id;reward;loss;success;precision;stability
1;100.0;0.05;True;0.95;0.9
2;120.0;0.03;True;0.97;0.92
3;50.0;0.15;False;0.8;0.75
```

> ⚠️ Manually edited files must preserve the exact header order, or loading will raise a `ValueError`.

## Class Overview

| Class | Responsibility |
|---|---|
| `TrainingEpisodeRecord` | Immutable dataclass representing a single episode |
| `TrainingMetricsDocument` | In-memory collection of episode records |
| `TrainingMetricsFile` | Extends `TrainingMetricsDocument` with CSV read/write support |
| `TrainingMetricsRepertory` | Generic cache of named documents |
| `TrainingMetricsDirectory` | Extends `TrainingMetricsRepertory` to manage a folder of CSV files |

