# Extending synbio-torch

synbio-torch is a set of `Protocol`s with interchangeable implementations. Adding a
capability means writing one implementation and registering it in the relevant
`build_*` factory; the training engine and pipeline stay untouched.

## Add a tokenizer

Implement the `Tokenizer` protocol (`synbiotorch.tokenize.base`):

```python
class MyTokenizer:
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def mask_token_id(self) -> int | None: ...
    @property
    def special_token_ids(self) -> frozenset[int]: ...
    @property
    def max_length(self) -> int: ...
    def tokenize_content(self, sequence: str) -> list[int]: ...   # no special wrapping
    def encode(self, sequence: str) -> Encoded: ...               # with <cls>/<sep>
```

Register it in `build_tokenizer` and add a `kind` to `TokenizerConfig`.

## Add an encoder (input modality)

A tensor encoder implements `Encoder` (`synbiotorch.encoders.base`):

```python
class MyEncoder:
    def encode(self, obj: Design) -> ModelInput: ...   # input_ids, attention_mask, label
    @property
    def output_spec(self) -> EncoderSpec: ...              # vocab_size, pad/mask ids, max_length
```

Register it in `build_encoder` and add a `kind` to `EncoderConfig`. Encoders that
produce a non-tensor batch (like the graph encoder, which returns a PyG `Data`)
are wired in the pipeline alongside a matching `BatchAdapter` and loader.

## Add an objective (task)

Implement the `Task` protocol (`synbiotorch.tasks.base`):

```python
class MyTask:
    label_dtype: str  # "float" or "long"
    def loss(self, logits, labels): ...
    def predict(self, logits): ...
    def epoch_metrics(self, preds, labels) -> dict[str, float]: ...
    @property
    def primary_metric(self) -> tuple[str, str]: ...   # (name, "min"|"max")
```

Register it in `build_task` and add a `kind` to `TaskConfig`. If the objective
needs a different head/model, extend `build_model` (as `mlm` and `causal` do, each
wrapping a different HuggingFace `AutoModelFor…`). A new objective may also need a
matching collator — `supervised` pads, `mlm` masks, `causal` shifts targets.

## Add a callback

Subclass `Callback` (`synbiotorch.engine.callbacks`) and override
`on_train_start` / `on_step_end` / `on_epoch_end` / `on_train_end`. The bundled
callbacks are `EarlyStopping`, `ModelCheckpoint`, `PeriodicCheckpoint`,
`MetricLogger`, and `WandbLogger`; the pipeline assembles them from config.
`on_train_end` is guaranteed to run (the trainer tears callbacks down in a
`finally`), so it is the place to release resources. `on_step_end(trainer, step,
logs)` fires after each optimizer step with `step_loss` and `lr`. Callbacks that
write should gate on rank (the bundled ones take `is_main`) so a distributed run
writes once.

## Add a batch modality

To train on a batch shape the engine doesn't yet understand, implement a
`BatchAdapter` (`synbiotorch.engine.batch`):

```python
class MyBatchAdapter:
    def to_device(self, batch, device): ...
    def forward(self, model, batch) -> torch.Tensor: ...   # logits
    def labels(self, batch) -> torch.Tensor: ...
```

Pass it to `Trainer(..., batch_adapter=...)`. `TensorBatchAdapter` handles
`dict[str, Tensor]` batches; `GraphBatchAdapter` handles PyG `Batch` objects.

## Add a data source

Implement the `Corpus` protocol (`synbiotorch.data.corpus`) — `__iter__` yielding
`Design`s and a `fingerprint()` for caching — in a module under
`synbiotorch.sources`, then register it in `build_corpus` with a new
`CorpusConfig.source`. The existing sources (`fasta`, `table`, `genbank`, `sbol`,
`sbol_db`, `synthetic`) are the templates. Downstream code (materialization,
splitting, encoding, training) works unchanged.
