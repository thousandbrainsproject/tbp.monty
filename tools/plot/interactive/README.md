
# Interactive Visualizations Wrapper

This provides a wrapper library to help build interactive visualization in a more structured manner.
The core idea is that every visualization consists of a **collection of `Widget` instances** that communicate with eachother through **Pub/Sub topic messages**.
The behavior of each `Widget` is customized by passing `WidgetOps` which holds widget-specific functions (e.g., the UI element, messages to publish or listen to, etc.)



## The `Widget` class

The `Widget` class abstracts away the shared behavior between different types of widgets.
For example, All widgets may choose to add/remove elements to the UI, or they may publish a state on a specific topic based on UI events (e.g., slider moved or button pressed).

The `Widget` performs a number of tasks:
1) **Scheduling of debounced messages.**
This uses a `VtkDebounceScheduler` to collapse repeated widget state messages within a user configurable amount of time (`debounce_sec`) and sends only the last message.
2) **Ability to deduplicate messages.**
Messages may not be published if the state remained the same, even after UI interaction.
For example, a button pressed few times and cycles back to the original state.
This is configurable using the `dedupe` boolean argument.
3) **Ability to customize widget behavior as composed functionality.**
The `Widget` class is initialized by passing a `WidgetOps` that defines it's behavior.
It looks for specific functions within the `WidgetOps` to perform different tasks.
4) **Type of `Widget[W, S]` can be defined by two Generic variables**.
We use the Generic type W for widget type and S for state.
For example, the widget type `Slider2D` would have a state `int`, or a `Button` widget would have a state `str`.


```python
Widget[Slider2D, int](
    widget_ops=StepSliderWidgetOps(
        plotter=plotter,
        data_parser=data_parser,
    ),
    bus=event_bus,
    scheduler=scheduler,
    debounce_sec=0.5,
    dedupe=True,
)
```

## The `WidgetOps`

The `WidgetOps` can fully customize a `Widget` behavior by providing custom functions that are called from the `Widget` class.
The `WidgetOps` can significantly vary across widgets.
For example, some widgets may not add a UI element, others may not publish any messages (e.g., mesh visualizer).
Therefore, the base `WidgetOpsProto` (protocol) is empty but other functions are defined as added "capabilities".
These capabilities are `runtime_checkable` and are used in the `Widget` class (i.e., through `is_instance`) to check
if the capability exists.

```python
# === Base Protocol === #
@runtime_checkable
class WidgetOpsProto(Protocol):
    pass

# === Capabilities === #
@runtime_checkable
class SupportsExtractState(Protocol[W, S]):
    def extract_state(self, widget: W | None) -> S | None: ...

@runtime_checkable
class SupportsSetState(Protocol[W, S]):
    def set_state(self, widget: W | None, value: S | None) -> None: ...

@runtime_checkable
class HasUpdaters(Protocol[W]):
    updaters: Iterable[WidgetUpdaterProto[W]]

# And more ...
```

### The `WidgetUpdater` class

The `WidgetOps` can have the `updaters` capability.
The `updaters` listen to different topic messages and trigger a callback function when it receives messages on those topics.
Every `WidgetOps` can have a list of these updaters with different callback functions.

```python
updaters = [
    WidgetUpdater(
        topics=[
            TopicSpec("episode_number", required=True),
            TopicSpec("step_number", required=True),
            TopicSpec("current_object", required=True),
            TopicSpec("age_threshold", required=True),
        ],
        callback=update_plot,
    ),
    WidgetUpdater(
        topics=[
            TopicSpec("click_location", required=True),
        ],
        callback=update_selection,
    ),
]
```

### Other Capabilities

The `WidgetOps` can include other capabilities to add or remove objects to the UI, define how to extract or set the state, and more.
One important capability is to define the payload function that converts a state to a message to be published on specific topics.
This can be defined using the `state_to_messages` function which receives a `state` and returns a list of `TopicMessages`.
Each topic message can be sent to a different topic.

```python
@runtime_checkable
class HasStateToMessages(Protocol[S]):
    def state_to_messages(self, state: S | None) -> Iterable[TopicMessage]: ...
```

### Accessing the Data

It is common for `WidgetOps` to require accessing data from experiment logs. 
A `WidgetOps` can define it's own `DataLocator` instances that describe how to access a dict structure to retrieve information.
The `DataLocator` describes the path (e.g., keys and indices) to access the required information.
The `DataLocator` can also be used to query the available options (i.e., keys or indices) at a level of the dictionary.
This is useful e.g., when setting the step slider range based on a specific episode.


```python
locator = DataLocator(
    path=[
        DataLocatorStep.key(name="episode"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(
            name="telemetry", value="hypotheses_updater_telemetry"
        ),
        DataLocatorStep.index(name="step"),
        DataLocatorStep.key(name="obj"),
        DataLocatorStep.key(name="channel"),
    ],
)
step_data = data_parser.extract(locator, episode=2, step=3, obj="mug", channel="patch")
available_objects = data_parser.query(locator, episode=2, step=3)
available_channels = data_parser.query(locator, episode=2, step=3, obj="banana")
```


## Toy Example

We can create a simple visualization with two slider widgets; one for episode and another for step.

#### Episode `WidgetOps`:

The episode slider defines a locator that queries the number of existing episodes, and publishes its state changes when the slider moves.

```python
class EpisodeSliderWidgetOps:
    """WidgetOps implementation for an Episode slider."""

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = dict(
            xmin=0, xmax=10, value=0, pos=[(0.1, 0.2), (0.7, 0.2)], title="Episode"
        )

        self._locators = {
            "episode": DataLocator(
                path=[
                    DataLocatorStep.key(name="episode"),
                ]
            )
        }

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        """Create the slider widget and set its range from the data."""
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["episode"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        widget = self.plotter.add_slider(callback, **kwargs)
        self.plotter.render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value from its VTK representation."""
        return round(widget.GetRepresentation().GetValue())

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages."""
        return [TopicMessage(name="episode_number", value=state)]

```


#### Step `WidgetOps`:

The step slider listens to the episode slider state and updates its range by querying the data to find the number of steps at this specific episode.

```python

class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider."""

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_slider_range,
            )
        ]

        self._add_kwargs = dict(
            xmin=0,
            xmax=10,
            value=0,
            pos=[(0.1, 0.1), (0.7, 0.1)],
            title="Step",
        )

        self._locators = {
            "step": DataLocator(
                path=[
                    DataLocatorStep.key(name="episode"),
                    DataLocatorStep.key(name="lm", value="LM_0"),
                    DataLocatorStep.key(
                        name="telemetry", value="hypotheses_updater_telemetry"
                    ),
                    DataLocatorStep.index(name="step"),
                ]
            )
        }

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget."""
        widget = self.plotter.add_slider(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value from its VTK representation."""
        return round(widget.GetRepresentation().GetValue())

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages."""
        return [TopicMessage(name="step_number", value=state)]

    def update_slider_range(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        """Adjust slider range based on the selected episode.

        Returns:
            A tuple `(widget, True)` indicating the updated widget and whether
            a publish should occur.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}

        # set widget range to the correct step number
        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(msgs_dict["episode_number"])
                )
            )
            - 1,
        ]

        self.plotter.render()

        return widget, True
```


#### More examples

More complex examples of more widgets can be found in the `interactive_hypothesis_space_correlation` plot.
