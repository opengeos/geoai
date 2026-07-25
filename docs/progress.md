# progress module

Centralizes progress bar handling for GeoAI. Every progress bar in the package is created through the `tqdm` wrapper defined here, so all of them can be silenced at once with `geoai.disable_progress_bars()` or the `GEOAI_DISABLE_PROGRESS` environment variable. This is handy in notebooks (e.g., Google Colab) where long-running progress output can freeze the browser tab.

```python
import geoai

geoai.disable_progress_bars()  # silence every progress bar
geoai.enable_progress_bars()  # turn them back on
```

Individual functions still honor their own `quiet=True` / `verbose=False` arguments.

::: geoai.progress
