from geoai.dialogs.diagnostics import DiagnosticsDialog


class DeletedWorker:
    def isRunning(self):
        raise RuntimeError(
            "wrapped C/C++ object of type _DiagnosticsWorker has been deleted"
        )


def test_diagnostics_dialog_worker_check_tolerates_deleted_worker():
    dialog = DiagnosticsDialog.__new__(DiagnosticsDialog)
    dialog._worker = DeletedWorker()

    assert dialog._worker_is_running() is False
    assert dialog._worker is None
