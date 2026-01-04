# Streaming Data from Parallel Processes

The LiveView experiment supports pub/sub for streaming data from parallel processes (threads, async tasks, etc.) into a unified dashboard view.

## Quick Start

### From the Main Experiment Thread

```python
from contrib.liveview_experiment.src.monty_experiment_with_liveview import (
    MontyExperimentWithLiveView
)

experiment = MontyExperimentWithLiveView(config)

# Access the broadcaster
broadcaster = experiment.broadcaster

# Publish metrics
await broadcaster.publish_metric("loss", 0.5, epoch=1, step=100)

# Publish custom data streams
await broadcaster.publish_data("sensor_readings", {
    "sensor_1": 0.8,
    "sensor_2": 0.3,
    "timestamp": "2024-01-01T12:00:00"
})

# Publish logs
await broadcaster.publish_log("info", "Training step completed", step=100)
```

### From a Thread (Synchronous)

```python
import threading

def worker_thread(experiment):
    broadcaster = experiment.broadcaster
    
    # Use sync methods for threads
    broadcaster.publish_metric_sync("loss", 0.5, epoch=1, step=100)
    broadcaster.publish_data_sync("sensor_data", {"value": 123})

# Start thread
thread = threading.Thread(target=worker_thread, args=(experiment,))
thread.start()
```

### From an Async Task

```python
import asyncio

async def data_collector(experiment):
    broadcaster = experiment.broadcaster
    
    while True:
        # Collect data
        data = collect_sensor_data()
        
        # Publish to LiveView
        await broadcaster.publish_data("sensor_stream", data)
        
        await asyncio.sleep(0.1)  # 10 Hz

# Start async task
task = asyncio.create_task(data_collector(experiment))
```

### From a Separate Process

If you have a completely separate process, you can create a broadcaster with the same topic:

```python
from contrib.liveview_experiment.src.broadcaster import ExperimentBroadcaster

# Use the same base topic as the experiment
broadcaster = ExperimentBroadcaster(base_topic="experiment:updates:root")

# Publish data (requires async context)
import asyncio
asyncio.run(broadcaster.publish_metric("loss", 0.5))
```

## Data Types

### Metrics

Metrics are numeric values that can be tracked over time:

```python
await broadcaster.publish_metric(
    name="accuracy",
    value=0.95,
    epoch=10,
    step=1000,
    phase="validation"
)
```

### Data Streams

Data streams are named collections of arbitrary data:

```python
await broadcaster.publish_data(
    stream_name="activations",
    data={
        "layer_1": [0.1, 0.2, 0.3],
        "layer_2": [0.4, 0.5, 0.6],
        "timestamp": datetime.now().isoformat()
    }
)
```

### Logs

Log messages for real-time monitoring:

```python
await broadcaster.publish_log(
    level="warning",
    message="High memory usage detected",
    memory_mb=8192,
    threshold_mb=4096
)
```

## Integration Examples

### Monitoring a Training Loop

```python
async def training_loop(experiment):
    broadcaster = experiment.broadcaster
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            
            # Publish metrics every N steps
            if step % 100 == 0:
                await broadcaster.publish_metric("loss", loss, epoch=epoch, step=step)
                await broadcaster.publish_metric("learning_rate", lr, epoch=epoch, step=step)
            
            # Publish activations periodically
            if step % 1000 == 0:
                activations = get_layer_activations()
                await broadcaster.publish_data("activations", activations)
```

### Monitoring Sensor Data

```python
async def sensor_monitor(experiment):
    broadcaster = experiment.broadcaster
    
    while experiment.state_manager.experiment_state.status == "running":
        sensor_data = read_sensors()
        
        await broadcaster.publish_data("sensors", {
            "camera": sensor_data.camera,
            "lidar": sensor_data.lidar,
            "imu": sensor_data.imu,
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.1)  # 10 Hz
```

### Background Metrics Collection

```python
import threading
import time

def metrics_collector(experiment):
    broadcaster = experiment.broadcaster
    
    while experiment.state_manager.experiment_state.status == "running":
        # Collect system metrics
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()
        gpu_usage = get_gpu_usage()
        
        broadcaster.publish_data_sync("system_metrics", {
            "cpu_percent": cpu_usage,
            "memory_mb": memory_usage,
            "gpu_percent": gpu_usage,
            "timestamp": time.time()
        })
        
        time.sleep(1.0)  # 1 Hz

# Start in background thread
thread = threading.Thread(target=metrics_collector, args=(experiment,), daemon=True)
thread.start()
```

## Viewing in LiveView

All published data automatically appears in the LiveView dashboard at `http://127.0.0.1:8000`:

- **Metrics**: Displayed in the metrics section
- **Data Streams**: Available in the `data_streams` dictionary
- **Logs**: Shown in the recent logs section (last 20 entries)

The dashboard updates in real-time as data is published via pub/sub.

