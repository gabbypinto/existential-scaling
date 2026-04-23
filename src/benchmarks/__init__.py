from benchmarks.aime25 import AIME25Benchmark
from benchmarks.lcb import LCBBenchmark
from benchmarks.gpqa import GPQABenchmark
from benchmarks.piqa_global import GlobalPIQABenchmark
from benchmarks.scicode import SciCodeBenchmark

REGISTRY: dict[str, type] = {
    "aime25":      AIME25Benchmark,
    "lcb":         LCBBenchmark,
    "gpqa":        GPQABenchmark,
    "piqa_global": GlobalPIQABenchmark,
    "scicode":     SciCodeBenchmark,
}
