"""Tests for ``geoai.pipeline`` module."""

import json
import os
import tempfile
import unittest

from geoai.pipeline import (
    CheckpointManager,
    ErrorPolicy,
    FunctionStep,
    GlobStep,
    ItemStatus,
    Pipeline,
    PipelineResult,
    PipelineStep,
    StepResult,
    _step_from_dict,
    _step_to_dict,
    load_pipeline,
    register_step,
)

# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------


class TestFunctionStep(unittest.TestCase):
    """Tests for FunctionStep."""

    def test_basic_processing(self):
        """FunctionStep processes an item through the wrapped function."""

        def add_key(item):
            item["processed"] = True
            return item

        step = FunctionStep("test", add_key)
        result = step.process({"input_path": "a.tif"})
        self.assertTrue(result["processed"])
        self.assertEqual(result["input_path"], "a.tif")

    def test_setup_teardown_callbacks(self):
        """FunctionStep invokes optional setup/teardown callables."""
        state = {"setup": False, "teardown": False}

        step = FunctionStep(
            "test",
            fn=lambda item: item,
            setup_fn=lambda: state.update({"setup": True}),
            teardown_fn=lambda: state.update({"teardown": True}),
        )
        step.setup()
        self.assertTrue(state["setup"])

        step.teardown()
        self.assertTrue(state["teardown"])

    def test_setup_teardown_none(self):
        """FunctionStep with no callbacks does not raise on setup/teardown."""
        step = FunctionStep("test", fn=lambda item: item)
        step.setup()
        step.teardown()

    def test_repr(self):
        """FunctionStep repr includes the step name."""
        step = FunctionStep("my_step", lambda x: x)
        self.assertIn("my_step", repr(step))
        self.assertIn("FunctionStep", repr(step))


class TestGlobStep(unittest.TestCase):
    """Tests for GlobStep."""

    def test_expand_directory(self):
        """GlobStep expands a directory into per-file items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                open(os.path.join(tmpdir, f"img_{i}.tif"), "w").close()
            # Non-matching file
            open(os.path.join(tmpdir, "readme.txt"), "w").close()

            step = GlobStep()
            items = step.expand({"input_dir": tmpdir})
            self.assertEqual(len(items), 3)
            for item in items:
                self.assertIn("input_path", item)
                self.assertTrue(item["input_path"].endswith(".tif"))
                self.assertNotIn("input_dir", item)

    def test_expand_custom_extensions(self):
        """GlobStep respects custom extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.jp2"), "w").close()
            open(os.path.join(tmpdir, "b.tif"), "w").close()

            step = GlobStep(extensions=[".jp2"])
            items = step.expand({"input_dir": tmpdir})
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0]["input_path"].endswith(".jp2"))

    def test_expand_pattern(self):
        """GlobStep expands an input_pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                open(os.path.join(tmpdir, f"sat_{i}.tif"), "w").close()
            open(os.path.join(tmpdir, "other.tif"), "w").close()

            step = GlobStep()
            pattern = os.path.join(tmpdir, "sat_*.tif")
            items = step.expand({"input_pattern": pattern})
            self.assertEqual(len(items), 2)

    def test_expand_empty_directory(self):
        """GlobStep returns empty list for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            step = GlobStep()
            items = step.expand({"input_dir": tmpdir})
            self.assertEqual(len(items), 0)

    def test_expand_missing_keys_raises(self):
        """GlobStep raises ValueError when no input key is present."""
        step = GlobStep()
        with self.assertRaises(ValueError):
            step.expand({})

    def test_expand_preserves_extra_keys(self):
        """GlobStep preserves extra keys from the source item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.tif"), "w").close()

            step = GlobStep()
            items = step.expand(
                {"input_dir": tmpdir, "output_dir": "/out", "extra": 42}
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["output_dir"], "/out")
            self.assertEqual(items[0]["extra"], 42)

    def test_process_passthrough(self):
        """GlobStep.process is a pass-through."""
        step = GlobStep()
        item = {"key": "value"}
        self.assertEqual(step.process(item), item)

    def test_repr(self):
        """GlobStep repr includes the step name."""
        step = GlobStep()
        self.assertIn("glob", repr(step))


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager."""

    def test_mark_and_check_completed(self):
        """Marking an item as completed makes is_completed return True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.json")
            mgr = CheckpointManager(path)
            self.assertFalse(mgr.is_completed("item1"))

            mgr.mark_completed("item1", ["step_a", "step_b"])
            self.assertTrue(mgr.is_completed("item1"))

    def test_save_and_reload(self):
        """Checkpoint persists across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.json")
            mgr = CheckpointManager(path, config_hash="abc")
            mgr.mark_completed("item1", ["s1"])
            mgr.save()

            mgr2 = CheckpointManager(path, config_hash="abc")
            self.assertTrue(mgr2.is_completed("item1"))

    def test_config_hash_mismatch_resets(self):
        """Changed config hash resets checkpoint entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.json")
            mgr = CheckpointManager(path, config_hash="abc")
            mgr.mark_completed("item1", ["s1"])
            mgr.save()

            mgr2 = CheckpointManager(path, config_hash="xyz")
            self.assertFalse(mgr2.is_completed("item1"))

    def test_mark_failed(self):
        """Failed items are not reported as completed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.json")
            mgr = CheckpointManager(path)
            mgr.mark_failed("item1", "some error", ["s1"])
            self.assertFalse(mgr.is_completed("item1"))

    def test_stats(self):
        """Stats returns correct counts per status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.json")
            mgr = CheckpointManager(path)
            mgr.mark_completed("a", [])
            mgr.mark_completed("b", [])
            mgr.mark_failed("c", "err", [])

            stats = mgr.stats
            self.assertEqual(stats["completed"], 2)
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(stats["pending"], 0)

    def test_nonexistent_file(self):
        """CheckpointManager handles nonexistent checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.json")
            mgr = CheckpointManager(path)
            self.assertEqual(mgr.stats["completed"], 0)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline(unittest.TestCase):
    """Tests for Pipeline class."""

    def test_sequential_run(self):
        """Pipeline processes items sequentially through all steps."""
        step1 = FunctionStep("s1", lambda item: {**item, "s1": True})
        step2 = FunctionStep("s2", lambda item: {**item, "s2": True})

        pipe = Pipeline(steps=[step1, step2], max_workers=1, quiet=True)
        result = pipe.run(
            items=[
                {"input_path": "a.tif"},
                {"input_path": "b.tif"},
            ]
        )
        self.assertEqual(len(result.completed), 2)
        self.assertEqual(len(result.failed), 0)
        for item in result.completed:
            self.assertTrue(item["s1"])
            self.assertTrue(item["s2"])

    def test_parallel_run(self):
        """Pipeline processes items in parallel."""
        step = FunctionStep("s1", lambda item: {**item, "done": True})
        pipe = Pipeline(steps=[step], max_workers=2, quiet=True)
        result = pipe.run(items=[{"input_path": f"img_{i}.tif"} for i in range(10)])
        self.assertEqual(len(result.completed), 10)

    def test_error_skip(self):
        """Pipeline with on_error=skip continues past failures."""

        def failing_step(item):
            if "fail" in item.get("input_path", ""):
                raise ValueError("Intentional failure")
            return item

        pipe = Pipeline(
            steps=[FunctionStep("s1", failing_step)],
            max_workers=1,
            on_error="skip",
            quiet=True,
        )
        result = pipe.run(
            items=[
                {"input_path": "ok.tif"},
                {"input_path": "fail.tif"},
                {"input_path": "ok2.tif"},
            ]
        )
        self.assertEqual(len(result.completed), 2)
        self.assertEqual(len(result.failed), 1)

    def test_error_fail_raises(self):
        """Pipeline with on_error=fail raises on first failure."""

        def failing_step(item):
            raise ValueError("boom")

        pipe = Pipeline(
            steps=[FunctionStep("s1", failing_step)],
            on_error="fail",
            quiet=True,
        )
        with self.assertRaises(RuntimeError):
            pipe.run(items=[{"input_path": "a.tif"}])

    def test_checkpoint_resume(self):
        """Pipeline skips already-completed items on resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            counter = {"calls": 0}

            def counting_step(item):
                counter["calls"] += 1
                return item

            pipe = Pipeline(
                steps=[FunctionStep("s1", counting_step)],
                checkpoint_dir=tmpdir,
                name="test_pipe",
                quiet=True,
            )

            items = [{"input_path": f"img_{i}.tif"} for i in range(5)]

            result1 = pipe.run(items=list(items))
            self.assertEqual(counter["calls"], 5)
            self.assertEqual(len(result1.completed), 5)
            self.assertIsNotNone(result1.checkpoint_path)

            # Second run should skip all
            counter["calls"] = 0
            result2 = pipe.run(items=list(items))
            self.assertEqual(counter["calls"], 0)
            self.assertEqual(len(result2.skipped), 5)
            self.assertEqual(len(result2.completed), 0)

    def test_with_glob_step(self):
        """Pipeline integrates with GlobStep for directory input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                open(os.path.join(tmpdir, f"img_{i}.tif"), "w").close()

            step = FunctionStep("s1", lambda item: {**item, "processed": True})
            pipe = Pipeline(steps=[GlobStep(), step], quiet=True)
            result = pipe.run(input_dir=tmpdir)
            self.assertEqual(len(result.completed), 3)

    def test_auto_glob_without_glob_step(self):
        """Pipeline auto-globs when input_dir given but no GlobStep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                open(os.path.join(tmpdir, f"img_{i}.tif"), "w").close()

            step = FunctionStep("s1", lambda item: item)
            pipe = Pipeline(steps=[step], quiet=True)
            result = pipe.run(input_dir=tmpdir)
            self.assertEqual(len(result.completed), 2)

    def test_output_dir_injected(self):
        """Output dir is injected into items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            step = FunctionStep("s1", lambda item: item)
            pipe = Pipeline(steps=[step], quiet=True)
            result = pipe.run(
                items=[{"input_path": "a.tif"}],
                output_dir=tmpdir,
            )
            self.assertEqual(len(result.completed), 1)
            self.assertEqual(result.completed[0]["output_dir"], tmpdir)

    def test_empty_items(self):
        """Pipeline handles empty item list gracefully."""
        pipe = Pipeline(steps=[FunctionStep("s1", lambda x: x)], quiet=True)
        result = pipe.run(items=[])
        self.assertEqual(len(result.completed), 0)
        self.assertEqual(len(result.failed), 0)

    def test_no_items_or_input_dir_raises(self):
        """Pipeline.run raises ValueError when neither input is provided."""
        pipe = Pipeline(steps=[FunctionStep("s1", lambda x: x)])
        with self.assertRaises(ValueError):
            pipe.run()

    def test_setup_and_teardown_called(self):
        """Pipeline calls setup and teardown on all processing steps."""
        state = {"setup": 0, "teardown": 0}

        step = FunctionStep(
            "s1",
            fn=lambda item: item,
            setup_fn=lambda: state.update({"setup": state["setup"] + 1}),
            teardown_fn=lambda: state.update({"teardown": state["teardown"] + 1}),
        )
        pipe = Pipeline(steps=[step], quiet=True)
        pipe.run(items=[{"input_path": "a.tif"}])
        self.assertEqual(state["setup"], 1)
        self.assertEqual(state["teardown"], 1)

    def test_teardown_called_on_error(self):
        """Teardown runs even when a step raises and on_error=fail."""
        state = {"teardown": False}

        def failing_fn(item):
            raise ValueError("boom")

        step = FunctionStep(
            "s1",
            fn=failing_fn,
            teardown_fn=lambda: state.update({"teardown": True}),
        )
        pipe = Pipeline(steps=[step], on_error="fail", quiet=True)
        with self.assertRaises(RuntimeError):
            pipe.run(items=[{"input_path": "a.tif"}])
        self.assertTrue(state["teardown"])

    def test_repr(self):
        """Pipeline repr includes name and step count."""
        pipe = Pipeline(steps=[FunctionStep("s1", lambda x: x)], name="my_pipe")
        r = repr(pipe)
        self.assertIn("my_pipe", r)
        self.assertIn("1", r)

    def test_total_duration_populated(self):
        """PipelineResult has a positive total_duration."""
        pipe = Pipeline(steps=[FunctionStep("s1", lambda x: x)], quiet=True)
        result = pipe.run(items=[{"input_path": "a.tif"}])
        self.assertGreater(result.total_duration, 0)

    def test_invalid_executor_type_raises(self):
        """Pipeline raises ValueError for unsupported executor_type."""
        with self.assertRaises(ValueError):
            Pipeline(
                steps=[FunctionStep("s1", lambda x: x)],
                executor_type="process",
            )

    def test_failed_items_preserve_partial_results(self):
        """Failed items contain partial results from earlier steps."""

        def step1_fn(item):
            item["step1_done"] = True
            return item

        def step2_fn(item):
            raise ValueError("step2 error")

        pipe = Pipeline(
            steps=[
                FunctionStep("s1", step1_fn),
                FunctionStep("s2", step2_fn),
            ],
            on_error="skip",
            quiet=True,
        )
        result = pipe.run(items=[{"input_path": "a.tif"}])
        self.assertEqual(len(result.failed), 1)
        failed_item, error_msg = result.failed[0]
        self.assertTrue(failed_item.get("step1_done"))
        self.assertIn("step2", error_msg)


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult(unittest.TestCase):
    """Tests for PipelineResult."""

    def test_summary(self):
        """Summary returns correct counts."""
        result = PipelineResult(
            completed=[{}, {}],
            failed=[({}, "err")],
            skipped=[{}],
            total_duration=5.123,
        )
        summary = result.summary
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["total_duration"], 5.12)

    def test_empty_summary(self):
        """Summary works on empty result."""
        result = PipelineResult()
        summary = result.summary
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["total_duration"], 0)


class TestStepResult(unittest.TestCase):
    """Tests for StepResult."""

    def test_success_result(self):
        """StepResult with success=True has no error."""
        r = StepResult(item={"a": 1}, success=True, duration=0.5)
        self.assertTrue(r.success)
        self.assertIsNone(r.error)

    def test_failure_result(self):
        """StepResult with success=False carries error message."""
        r = StepResult(item={}, success=False, error="boom")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "boom")


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization(unittest.TestCase):
    """Tests for pipeline serialization (to_dict, to_json, to_yaml)."""

    def test_to_dict(self):
        """Pipeline.to_dict produces a valid config dict."""
        pipe = Pipeline(
            steps=[GlobStep(), FunctionStep("noop", lambda x: x)],
            name="test",
            max_workers=2,
            on_error="skip",
        )
        d = pipe.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["max_workers"], 2)
        self.assertEqual(d["on_error"], "skip")
        self.assertEqual(len(d["steps"]), 2)
        self.assertEqual(d["steps"][0]["type"], "GlobStep")

    def test_to_json_roundtrip(self):
        """Pipeline serialized to JSON can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = Pipeline(
                steps=[GlobStep(extensions=[".tif"])],
                name="json_test",
                max_workers=3,
            )
            json_path = os.path.join(tmpdir, "config.json")
            pipe.to_json(json_path)

            self.assertTrue(os.path.exists(json_path))
            loaded = load_pipeline(json_path)
            self.assertEqual(loaded.name, "json_test")
            self.assertEqual(loaded.max_workers, 3)
            self.assertEqual(len(loaded.steps), 1)
            self.assertIsInstance(loaded.steps[0], GlobStep)

    def test_to_yaml_roundtrip(self):
        """Pipeline serialized to YAML can be loaded back."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe = Pipeline(
                steps=[GlobStep()],
                name="yaml_test",
                max_workers=2,
            )
            yaml_path = os.path.join(tmpdir, "config.yaml")
            pipe.to_yaml(yaml_path)

            self.assertTrue(os.path.exists(yaml_path))
            loaded = load_pipeline(yaml_path)
            self.assertEqual(loaded.name, "yaml_test")

    def test_step_to_dict(self):
        """_step_to_dict captures public attributes."""
        step = GlobStep(name="my_glob", extensions=[".jp2"])
        d = _step_to_dict(step)
        self.assertEqual(d["type"], "GlobStep")
        self.assertEqual(d["name"], "my_glob")
        self.assertEqual(d["extensions"], [".jp2"])


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestLoadPipeline(unittest.TestCase):
    """Tests for load_pipeline function."""

    def test_load_json(self):
        """load_pipeline reads a JSON config correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "test",
                "max_workers": 2,
                "on_error": "skip",
                "steps": [
                    {"type": "GlobStep", "name": "glob"},
                ],
            }
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                json.dump(config, f)

            pipe = load_pipeline(path)
            self.assertEqual(pipe.name, "test")
            self.assertEqual(pipe.max_workers, 2)
            self.assertEqual(len(pipe.steps), 1)
            self.assertIsInstance(pipe.steps[0], GlobStep)

    def test_load_yaml(self):
        """load_pipeline reads a YAML config correctly."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "yaml_test",
                "max_workers": 1,
                "on_error": "skip",
                "steps": [
                    {"type": "GlobStep", "name": "glob"},
                ],
            }
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                yaml.dump(config, f)

            pipe = load_pipeline(path)
            self.assertEqual(pipe.name, "yaml_test")

    def test_load_nonexistent_raises(self):
        """load_pipeline raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_pipeline("/nonexistent/config.json")

    def test_load_unsupported_format_raises(self):
        """load_pipeline raises ValueError for unsupported extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.toml")
            open(path, "w").close()
            with self.assertRaises(ValueError):
                load_pipeline(path)

    def test_load_with_overrides(self):
        """Overrides take precedence over config file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "test",
                "max_workers": 1,
                "steps": [],
            }
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                json.dump(config, f)

            pipe = load_pipeline(path, max_workers=8)
            self.assertEqual(pipe.max_workers, 8)

    def test_load_multiple_steps(self):
        """load_pipeline deserializes multiple steps of different types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "multi",
                "steps": [
                    {"type": "GlobStep", "name": "glob"},
                    {
                        "type": "SemanticSegmentationStep",
                        "name": "seg",
                        "model_path": "/tmp/model.pth",
                        "num_classes": 3,
                    },
                    {
                        "type": "RasterToVectorStep",
                        "name": "vec",
                        "output_format": ".gpkg",
                    },
                ],
            }
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                json.dump(config, f)

            pipe = load_pipeline(path)
            self.assertEqual(len(pipe.steps), 3)
            self.assertIsInstance(pipe.steps[0], GlobStep)
            from geoai.pipeline import SemanticSegmentationStep

            self.assertIsInstance(pipe.steps[1], SemanticSegmentationStep)
            self.assertEqual(pipe.steps[1].num_classes, 3)
            from geoai.pipeline import RasterToVectorStep

            self.assertIsInstance(pipe.steps[2], RasterToVectorStep)
            self.assertEqual(pipe.steps[2].output_format, ".gpkg")


# ---------------------------------------------------------------------------
# Step registry tests
# ---------------------------------------------------------------------------


class TestStepRegistry(unittest.TestCase):
    """Tests for step registration and deserialization."""

    def test_register_custom_step(self):
        """Custom step classes can be registered and deserialized."""

        @register_step
        class _TestCustomStep(PipelineStep):
            def __init__(self, name="custom", value=42):
                super().__init__(name)
                self.value = value

            def process(self, item):
                item["value"] = self.value
                return item

        d = {"type": "_TestCustomStep", "name": "custom", "value": 99}
        step = _step_from_dict(d)
        self.assertEqual(step.value, 99)

        result = step.process({})
        self.assertEqual(result["value"], 99)

    def test_unknown_step_type_raises(self):
        """Deserializing an unknown step type raises ValueError."""
        with self.assertRaises(ValueError):
            _step_from_dict({"type": "NonexistentStep123", "name": "x"})


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums(unittest.TestCase):
    """Tests for ErrorPolicy and ItemStatus enums."""

    def test_error_policy_values(self):
        """ErrorPolicy has skip and fail values."""
        self.assertEqual(ErrorPolicy.SKIP.value, "skip")
        self.assertEqual(ErrorPolicy.FAIL.value, "fail")
        self.assertEqual(ErrorPolicy("skip"), ErrorPolicy.SKIP)

    def test_item_status_values(self):
        """ItemStatus has expected values."""
        self.assertEqual(ItemStatus.PENDING.value, "pending")
        self.assertEqual(ItemStatus.COMPLETED.value, "completed")
        self.assertEqual(ItemStatus.FAILED.value, "failed")
        self.assertEqual(ItemStatus.SKIPPED.value, "skipped")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIPipelineCommand(unittest.TestCase):
    """Tests for CLI pipeline commands."""

    def setUp(self):
        from click.testing import CliRunner

        self.runner = CliRunner()

    def test_pipeline_help(self):
        """``geoai pipeline --help`` shows pipeline help text."""
        from geoai.cli import main

        result = self.runner.invoke(main, ["pipeline", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("pipeline", result.output.lower())

    def test_pipeline_run_help(self):
        """``geoai pipeline run --help`` shows run command options."""
        from geoai.cli import main

        result = self.runner.invoke(main, ["pipeline", "run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CONFIG_PATH", result.output)

    def test_pipeline_show_help(self):
        """``geoai pipeline show --help`` shows show command options."""
        from geoai.cli import main

        result = self.runner.invoke(main, ["pipeline", "show", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CONFIG_PATH", result.output)

    def test_pipeline_run_with_json_config(self):
        """``geoai pipeline run`` processes a JSON config end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test input files
            input_dir = os.path.join(tmpdir, "input")
            os.makedirs(input_dir)
            for i in range(2):
                open(os.path.join(input_dir, f"img_{i}.tif"), "w").close()

            output_dir = os.path.join(tmpdir, "output")

            config = {
                "name": "cli_test",
                "max_workers": 1,
                "on_error": "skip",
                "steps": [
                    {"type": "GlobStep", "name": "glob"},
                ],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            from geoai.cli import main

            result = self.runner.invoke(
                main,
                [
                    "pipeline",
                    "run",
                    config_path,
                    "-i",
                    input_dir,
                    "-o",
                    output_dir,
                    "-q",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Completed", result.output)

    def test_pipeline_show_with_json_config(self):
        """``geoai pipeline show`` displays pipeline configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "name": "show_test",
                "max_workers": 4,
                "on_error": "skip",
                "steps": [
                    {"type": "GlobStep", "name": "find_files"},
                ],
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            from geoai.cli import main

            result = self.runner.invoke(main, ["pipeline", "show", config_path])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("show_test", result.output)
            self.assertIn("4", result.output)


if __name__ == "__main__":
    unittest.main()
