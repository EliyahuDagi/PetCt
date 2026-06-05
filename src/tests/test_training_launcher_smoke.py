import os
import tempfile
import unittest
import importlib.util


class TestRunAllTrainingSmoke(unittest.TestCase):
    def _load_module(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        path = os.path.join(root, "scripts", "run_all_training.py")
        spec = importlib.util.spec_from_file_location("run_all_training", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resume_and_start_step(self):
        module = self._load_module()

        calls = []

        def step_fn(name, fail=False):
            def _inner(argv):
                calls.append(name)
                if fail:
                    raise RuntimeError("boom")
            return _inner

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            steps = [
                ("ae2d", step_fn("ae2d"), ["ae2d"]),
                ("diff2d", step_fn("diff2d", fail=True), ["diff2d"]),
                ("ft3d", step_fn("ft3d"), ["ft3d"]),
            ]

            with self.assertRaises(RuntimeError):
                module.run_all(steps, state_path, resume=False, start_step=None)

            self.assertEqual(calls, ["ae2d", "diff2d"])

            calls.clear()
            steps = [
                ("ae2d", step_fn("ae2d"), ["ae2d"]),
                ("diff2d", step_fn("diff2d"), ["diff2d"]),
                ("ft3d", step_fn("ft3d"), ["ft3d"]),
            ]
            module.run_all(steps, state_path, resume=True, start_step=None)
            self.assertEqual(calls, ["diff2d", "ft3d"])

            calls.clear()
            module.run_all(steps, state_path, resume=False, start_step="diff2d")
            self.assertEqual(calls, ["diff2d", "ft3d"])


if __name__ == "__main__":
    unittest.main()
