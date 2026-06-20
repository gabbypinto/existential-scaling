from benchmarks.aime24 import AIME24Benchmark
from benchmarks.aime25 import AIME25Benchmark
from benchmarks.aa_omniscience import AAOmniscienceBenchmark
from benchmarks.global_mmlu_lite import GlobalMMLULiteBenchmark
from benchmarks.gpqa import GPQABenchmark
from benchmarks.lcb import LCBBenchmark
from benchmarks.lcb_pro import LCBProBenchmark
from benchmarks.matharena_apex import MathArenaApexBenchmark
from benchmarks.mmlu import MMLUBenchmark
from benchmarks.piqa_global import GlobalPIQABenchmark
from benchmarks.scicode import SciCodeBenchmark

REGISTRY: dict[str, type] = {
    "aime24":          AIME24Benchmark,
    "aime25":          AIME25Benchmark,
    "aa_omniscience":  AAOmniscienceBenchmark,
    "global_mmlu_lite": GlobalMMLULiteBenchmark,
    "gpqa":            GPQABenchmark,
    "lcb":             LCBBenchmark,
    "lcb_pro":         LCBProBenchmark,
    "matharena_apex":  MathArenaApexBenchmark,
    "mmlu":            MMLUBenchmark,
    "piqa_global":     GlobalPIQABenchmark,
    "scicode":         SciCodeBenchmark,
}
