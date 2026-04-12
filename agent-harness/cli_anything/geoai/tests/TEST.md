# GeoAI CLI Test Plan and Results

## Test Inventory Plan

| File | Type | Estimated Tests |
|------|------|-----------------|
| `test_core.py` | Unit tests | ~40 tests |
| `test_full_e2e.py` | E2E + subprocess tests | ~20 tests |

**Total planned: ~60 tests**

---

## Unit Test Plan (`test_core.py`)

### core/project.py (10 tests)

- `test_create_project_defaults` -- Create project with default values
- `test_create_project_custom` -- Create project with custom name, CRS
- `test_open_project_valid` -- Open a valid JSON project file
- `test_open_project_missing_file` -- FileNotFoundError for missing file
- `test_open_project_invalid_json` -- ValueError for invalid JSON
- `test_save_project` -- Save and reload project round-trip
- `test_get_project_info` -- Verify info summary counts
- `test_add_file_raster` -- Add a raster file entry
- `test_add_file_auto_detect` -- Auto-detect file type from extension
- `test_remove_file` -- Remove file by index, verify IndexError on bad index

### core/session.py (10 tests)

- `test_session_empty` -- New session has no project
- `test_session_set_project` -- Load project into session
- `test_session_get_project_no_project` -- RuntimeError when no project
- `test_session_snapshot_and_undo` -- Snapshot, mutate, undo restores state
- `test_session_redo` -- Undo then redo restores mutation
- `test_session_undo_empty` -- RuntimeError when nothing to undo
- `test_session_redo_empty` -- RuntimeError when nothing to redo
- `test_session_history` -- Verify history entries after snapshots
- `test_session_max_undo_limit` -- Undo stack respects MAX_UNDO
- `test_session_save_and_load` -- Save session to disk, verify file

### core/raster.py (6 tests)

- `test_get_raster_info` -- Verify info dict keys from a real GeoTIFF
- `test_get_raster_info_missing_file` -- FileNotFoundError
- `test_get_raster_stats` -- Verify min/max/mean/std for a real raster
- `test_get_raster_stats_bad_band` -- ValueError for out-of-range band
- `test_tile_raster` -- Verify tiles created from a real raster
- `test_vectorize_raster` -- Verify vector output from raster

### core/vector.py (4 tests)

- `test_get_vector_info` -- Verify info dict from a real vector file
- `test_get_vector_info_missing_file` -- FileNotFoundError
- `test_rasterize_vector` -- Rasterize vector with template, verify output
- `test_rasterize_vector_missing_file` -- FileNotFoundError

### core/data.py (4 tests)

- `test_parse_bbox_valid` -- Parse "minx,miny,maxx,maxy"
- `test_parse_bbox_invalid` -- ValueError for bad formats
- `test_list_sources` -- Verify source list structure
- `test_list_sources_contains_naip` -- Verify naip is in the list

### core/segment.py (3 tests)

- `test_list_sam_models` -- Verify SAM model list structure
- `test_list_architectures` -- Verify architecture list
- `test_run_sam_missing_file` -- FileNotFoundError for missing raster

### core/detect.py (3 tests)

- `test_list_models` -- Verify detection model list structure
- `test_list_input_formats` -- Verify input format list
- `test_run_detection_missing_file` -- FileNotFoundError for missing raster

### core/change.py (2 tests)

- `test_list_methods` -- Verify change method list structure
- `test_detect_changes_missing_file` -- FileNotFoundError for missing images

---

## E2E Test Plan (`test_full_e2e.py`)

### Raster Operations E2E (4 tests)

- `test_raster_info_real_file` -- Get info on real GeoTIFF (knoxville_landsat)
- `test_raster_stats_all_bands` -- Stats for all bands, verify numeric values
- `test_raster_tile_and_count` -- Tile a raster, verify output files exist
- `test_raster_vectorize_real` -- Vectorize NDVI raster, verify GeoJSON output

### Project Workflow E2E (3 tests)

- `test_project_create_add_save_reload` -- Full create/add/save/open cycle
- `test_project_session_undo_redo` -- Undo and redo file additions
- `test_project_info_after_multiple_files` -- Verify counts after adding files

### CLI Subprocess Tests (8 tests)

- `test_cli_help` -- `--help` exits 0 with usage text
- `test_cli_version` -- `--version` shows version
- `test_cli_raster_info_json` -- `--json raster info <file>` returns valid JSON
- `test_cli_raster_stats_json` -- `--json raster stats <file>` returns valid JSON
- `test_cli_project_new_json` -- `--json project new -o <file>` creates project
- `test_cli_system_info_json` -- `--json system-info` returns valid JSON
- `test_cli_session_status_json` -- `--json session status` returns valid JSON
- `test_cli_segment_list_models` -- `--json segment list-models` returns list

### Realistic Workflow Scenarios (3 tests)

- **Raster Analysis Pipeline**
  - Simulates: Analyzing satellite imagery metadata and statistics
  - Operations: raster info -> raster stats all bands -> project add-file
  - Verified: All JSON responses valid, stats contain numeric values

- **Project Management Workflow**
  - Simulates: Setting up a geospatial AI workspace
  - Operations: project new -> add multiple files -> list files -> save -> reopen
  - Verified: File count matches, project reloads correctly

- **Data Discovery Workflow**
  - Simulates: Exploring available data sources
  - Operations: data sources -> segment list-models -> detect list-models
  - Verified: All list commands return non-empty results

---

## Test Results

**Run date:** 2026-04-11
**Python:** 3.12.12 | **pytest:** 9.0.2 | **Platform:** linux

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/qiusheng/Documents/GitHub/geoai/agent-harness
configfile: pytest.ini
collected 60 items

cli_anything/geoai/tests/test_core.py::TestProject::test_create_project_defaults PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_create_project_custom PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_open_project_valid PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_open_project_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_open_project_invalid_json PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_save_project PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_get_project_info PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_add_file_raster PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_add_file_auto_detect PASSED
cli_anything/geoai/tests/test_core.py::TestProject::test_remove_file PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_empty PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_set_project PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_get_project_no_project PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_snapshot_and_undo PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_redo PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_undo_empty PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_redo_empty PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_history PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_max_undo_limit PASSED
cli_anything/geoai/tests/test_core.py::TestSession::test_session_save_and_load PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_get_raster_info PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_get_raster_info_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_get_raster_stats PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_get_raster_stats_bad_band PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_tile_raster PASSED
cli_anything/geoai/tests/test_core.py::TestRaster::test_vectorize_raster PASSED
cli_anything/geoai/tests/test_core.py::TestVector::test_get_vector_info PASSED
cli_anything/geoai/tests/test_core.py::TestVector::test_get_vector_info_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestVector::test_rasterize_vector PASSED
cli_anything/geoai/tests/test_core.py::TestVector::test_rasterize_vector_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestData::test_parse_bbox_valid PASSED
cli_anything/geoai/tests/test_core.py::TestData::test_parse_bbox_invalid PASSED
cli_anything/geoai/tests/test_core.py::TestData::test_list_sources PASSED
cli_anything/geoai/tests/test_core.py::TestData::test_list_sources_contains_naip PASSED
cli_anything/geoai/tests/test_core.py::TestSegment::test_list_sam_models PASSED
cli_anything/geoai/tests/test_core.py::TestSegment::test_list_architectures PASSED
cli_anything/geoai/tests/test_core.py::TestSegment::test_run_sam_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestDetect::test_list_models PASSED
cli_anything/geoai/tests/test_core.py::TestDetect::test_list_input_formats PASSED
cli_anything/geoai/tests/test_core.py::TestDetect::test_run_detection_missing_file PASSED
cli_anything/geoai/tests/test_core.py::TestChange::test_list_methods PASSED
cli_anything/geoai/tests/test_core.py::TestChange::test_detect_changes_missing_file PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRasterE2E::test_raster_info_real_file PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRasterE2E::test_raster_stats_all_bands PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRasterE2E::test_raster_tile_and_count PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRasterE2E::test_raster_vectorize_real PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestProjectWorkflowE2E::test_project_create_add_save_reload PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestProjectWorkflowE2E::test_project_session_undo_redo PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestProjectWorkflowE2E::test_project_info_after_multiple_files PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_help PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_version PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_raster_info_json PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_raster_stats_json PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_project_new_json PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_system_info_json PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_session_status_json PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestCLISubprocess::test_cli_segment_list_models PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRealisticWorkflows::test_raster_analysis_pipeline PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRealisticWorkflows::test_project_management_workflow PASSED
cli_anything/geoai/tests/test_full_e2e.py::TestRealisticWorkflows::test_data_discovery_workflow PASSED

============================== 60 passed in 9.90s ==============================
```

### Summary

| Metric | Value |
|--------|-------|
| Total tests | 60 |
| Passed | 60 |
| Failed | 0 |
| Pass rate | 100% |
| Duration | 9.90s |

### Coverage Notes

- **Unit tests**: 42 tests covering all 8 core modules
- **E2E raster tests**: 4 tests with real GeoTIFF data (Knoxville Landsat + synthetic)
- **E2E project workflow tests**: 3 tests covering full create/save/load lifecycle
- **CLI subprocess tests**: 8 tests via installed `cli-anything-geoai` command
- **Workflow integration tests**: 3 multi-step scenarios simulating real agent usage
- **Not tested (require GPU/models)**: SAM inference, model training, change detection
  inference. These require model downloads and GPU availability. The CLI commands for
  these are tested for error handling (missing files) but not end-to-end inference.
