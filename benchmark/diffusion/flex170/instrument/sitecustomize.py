"""Attach the stage observer only in workers launched by the Flex benchmark."""

import importlib.abc
import importlib.machinery
import os
import sys


class ExecutorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if (
            fullname
            != "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor"
        ):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        execute = spec.loader.exec_module

        def attach(module):
            execute(module)
            from stage import install

            install(module.PipelineExecutor)
            if os.environ.get("FLEX_CHECK_RESIDENCY"):
                from residency_monitor import install as observe

                observe(module.PipelineExecutor)
            if os.environ.get("FLEX_CAPTURE_ATTENTION"):
                if os.environ["FLEX_BENCH_FAMILY"] == "moss":
                    from capture import install_moss as capture
                else:
                    from capture import install as capture

                capture(module.PipelineExecutor)

        spec.loader.exec_module = attach
        return spec


if os.environ.get("FLEX_BENCH_FAMILY"):
    sys.meta_path.insert(0, ExecutorFinder())
