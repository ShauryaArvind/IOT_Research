"""
Comprehensive Edge Hardware Profiler & Benchmarking Suite
=========================================================
Part of Phase 11: Physical Edge Hardware Deployment & Benchmarking.

Measures inference latency, throughput scaling, memory footprint and
burst throughput of the ONNX pipeline ON THE MACHINE IT RUNS ON. The host
(CPU, OS, ONNX Runtime version) is recorded with the results. Numbers from
a laptop/desktop do not transfer to micro-controllers or gateways; run this
on the target device (e.g. Raspberry Pi) before making edge claims.

Fixes relative to the original version:
  * flows come from the isolated test set (not the SMOTE training file);
  * the PyTorch-native path standardises its input (it used to feed raw
    features to the model);
  * model footprint includes the external .onnx.data weight files;
  * RAM is reported as RSS before/after loading and the delta, not just the
    process total (which is dominated by the Python/ORT runtime);
  * the burst test reports throughput only. Its "100% interception" came from
    a hardcoded attacker context that triggers context-only DENY rules.

Metrics Profiled:
  1. Per-Flow Processing Latency: Mean, p50, p90, p95, p99, Min, Max (µs and ms).
  2. Throughput Scaling: Sustained flows/sec across batch sizes [1 to 1024].
  3. Memory Footprint: Model on-disk size (KB) and Runtime RAM (RSS in MB).
  4. Burst test: 10,000 flows processed in one batch (throughput only).
  5. Comparative Engine Profiling: PyTorch Native vs. ONNX Runtime.

Outputs:
  - Numerical CSV: `edge_benchmark_results.csv`
  - Multi-Panel Visualization: `edge_performance_benchmark.png`
"""

import os
import sys

# Force UTF-8 stdout encoding for Windows compatibility
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import platform
import time
import psutil
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from edge_inference_engine import EdgeDefensePipeline
from zero_trust_engine import ZeroTrustEngine

# ---------------------------------------------------------------------------
# PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

BASELINE_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier.onnx')
ADV_ONNX = os.path.join(MODELS_DIR, 'network_risk_classifier_adv.onnx')
DAE_ONNX = os.path.join(MODELS_DIR, 'denoising_autoencoder.onnx')

CSV_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'edge_benchmark_results.csv')
PLOT_PATH = os.path.join(os.path.dirname(__file__), 'edge_performance_benchmark.png')

N_LATENCY_TRIALS = 2000
WARMUP_TRIALS = 200
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
DDOS_BURST_FLOWS = 10000


def pprint(*args, **kwargs):
    """Print wrapper forcing immediate stdout flush for real-time logging."""
    print(*args, **kwargs)
    sys.stdout.flush()


def get_current_process_memory_mb():
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def host_info():
    import onnxruntime
    return {
        'host_platform': platform.platform(),
        'host_machine': platform.machine(),
        'host_processor': platform.processor() or platform.machine(),
        'host_cpu_count': os.cpu_count(),
        'host_ram_gb': round(psutil.virtual_memory().total / 1024 ** 3, 1),
        'onnxruntime_version': onnxruntime.__version__,
        'python_version': platform.python_version(),
    }


def load_test_flows(n_samples=2000, seed=42):
    """Raw (unscaled) flows sampled from the isolated test set."""
    from data import TEST_FILES, read_xy
    pprint("  Loading flows from the isolated test set...")
    X, _ = read_xy(TEST_FILES)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    return X.iloc[idx].to_numpy(np.float32)


def benchmark_single_flow_latency(edge_pipeline: EdgeDefensePipeline, test_flows: np.ndarray):
    """Measures single-flow latency distributions under sequential packet arrival."""
    n_flows = len(test_flows)

    # 1. Warmup
    for i in range(WARMUP_TRIALS):
        _ = edge_pipeline.process_flow(test_flows[i % n_flows])

    # 2. Benchmark ONNX Full Defense (DAE + AT + ZT)
    latencies_us_full = []
    for i in range(n_flows):
        start = time.perf_counter()
        _ = edge_pipeline.process_flow(test_flows[i])
        latencies_us_full.append((time.perf_counter() - start) * 1_000_000.0)

    # 3. Benchmark ONNX Classifier Only (No DAE)
    pipeline_no_dae = EdgeDefensePipeline(enable_dae_sanitization=False)
    for i in range(WARMUP_TRIALS):
        _ = pipeline_no_dae.process_flow(test_flows[i % n_flows])

    latencies_us_clf = []
    for i in range(n_flows):
        start = time.perf_counter()
        _ = pipeline_no_dae.process_flow(test_flows[i])
        latencies_us_clf.append((time.perf_counter() - start) * 1_000_000.0)

    return np.array(latencies_us_full), np.array(latencies_us_clf)


def benchmark_pytorch_native_latency(test_flows: np.ndarray, config: str, seed: int, robust_defense: str):
    """Native PyTorch (preprocess + DAE + robust classifier + ZT) on CPU, for comparison."""
    from denoising_autoencoder import load_dae
    from training import load_run, run_dir

    rd = run_dir(config, robust_defense, seed)
    if not (rd / 'classifier.pth').exists():
        pprint(f"  [-] no {robust_defense} run at {rd}; skipping PyTorch-native latency")
        return None
    adv_model, _, prep = load_run(rd)
    dae_model, _ = load_dae(config, seed)
    zt_engine = ZeroTrustEngine()
    n_flows = len(test_flows)

    def step(x):
        t = torch.from_numpy(prep.transform(x[None, :]))
        probs = torch.softmax(adv_model(dae_model(t)), dim=1).numpy()[0]
        return zt_engine.evaluate(ml_risk_score=float(1.0 - probs[0]), context={'device_trust': 0.8, 'geo_risk': 0.2})

    with torch.no_grad():
        for i in range(WARMUP_TRIALS):
            step(test_flows[i % n_flows])
        lat = []
        for i in range(n_flows):
            start = time.perf_counter()
            step(test_flows[i])
            lat.append((time.perf_counter() - start) * 1_000_000.0)
    return np.array(lat)


def benchmark_throughput_scaling(edge_pipeline: EdgeDefensePipeline, test_flows: np.ndarray):
    """Measures flows/second processed across varying batch sizes."""
    throughput_results = []
    n_available = len(test_flows)

    for bsz in BATCH_SIZES:
        # Build batch by tiling if needed
        tile_count = int(np.ceil(bsz / n_available))
        batch = np.tile(test_flows, (tile_count, 1))[:bsz]

        # Warmup
        for _ in range(5):
            _ = edge_pipeline.process_batch(batch)

        # Timed execution over multiple iterations
        n_iters = max(10, int(2000 / bsz))
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = edge_pipeline.process_batch(batch)
        total_time = time.perf_counter() - start

        total_flows = bsz * n_iters
        flows_per_sec = total_flows / total_time
        avg_latency_ms = (total_time / total_flows) * 1000.0

        throughput_results.append({
            'batch_size': bsz,
            'flows_per_sec': flows_per_sec,
            'avg_latency_ms': avg_latency_ms,
            'mpps_equivalent': flows_per_sec / 1_000_000.0
        })

    return pd.DataFrame(throughput_results)


def benchmark_ddos_burst_stress(edge_pipeline: EdgeDefensePipeline, test_flows: np.ndarray, burst_count=DDOS_BURST_FLOWS):
    """Time one batch of `burst_count` flows (tiled from the test sample)."""
    tile_count = int(np.ceil(burst_count / len(test_flows)))
    burst_data = np.tile(test_flows, (tile_count, 1))[:burst_count]

    mem_before = get_current_process_memory_mb()
    start_time = time.perf_counter()
    edge_pipeline.process_batch(burst_data)
    total_time = time.perf_counter() - start_time
    mem_after = get_current_process_memory_mb()

    return {
        'burst_flows': burst_count,
        'total_duration_sec': total_time,
        'sustained_throughput_fps': burst_count / total_time,
        'memory_delta_mb': mem_after - mem_before,
        'peak_memory_mb': mem_after,
    }


def execute_edge_benchmark(argv=None):
    from export_onnx import onnx_size_kb
    from training import DEFAULT_CONFIG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--robust-defense', default='pgd_at')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    pprint("=" * 80)
    pprint("PHASE 11: EDGE RUNTIME BENCHMARK (host machine, see host info below)")
    pprint("=" * 80)
    host = host_info()
    for k, v in host.items():
        pprint(f"  {k:<20}: {v}")

    rss_start = get_current_process_memory_mb()
    test_flows = load_test_flows(N_LATENCY_TRIALS, a.seed)

    onnx_clf_size_kb = onnx_size_kb(ADV_ONNX) if os.path.exists(ADV_ONNX) else 0.0
    onnx_dae_size_kb = onnx_size_kb(DAE_ONNX) if os.path.exists(DAE_ONNX) else 0.0
    total_edge_footprint_kb = onnx_clf_size_kb + onnx_dae_size_kb

    pprint("\n[1/5] Model footprint on disk (graph + external weights):")
    pprint(f"  ONNX classifier : {onnx_clf_size_kb:.1f} KB")
    pprint(f"  ONNX DAE        : {onnx_dae_size_kb:.1f} KB")
    pprint(f"  Total           : {total_edge_footprint_kb:.1f} KB")

    rss_before_models = get_current_process_memory_mb()
    edge_pipeline = EdgeDefensePipeline(
        classifier_onnx_path=ADV_ONNX,
        dae_onnx_path=DAE_ONNX,
        enable_dae_sanitization=True,
        num_threads=a.threads
    )
    rss_after_models = get_current_process_memory_mb()

    pprint(f"\n[2/5] Per-flow latency over {N_LATENCY_TRIALS} sequential flows...")
    lat_onnx_full, lat_onnx_clf = benchmark_single_flow_latency(edge_pipeline, test_flows)
    lat_pt_full = benchmark_pytorch_native_latency(test_flows, a.config, a.seed, a.robust_defense)

    def row(name, lat):
        pprint(f"{name:<38} | {np.mean(lat):>9.1f} | {np.percentile(lat, 50):>8.1f} | "
               f"{np.percentile(lat, 95):>8.1f} | {np.percentile(lat, 99):>8.1f}")

    pprint("\n" + "-" * 75)
    pprint(f"{'Runtime Configuration':<38} | {'Mean (µs)':<10} | {'p50 (µs)':<9} | {'p95 (µs)':<9} | {'p99 (µs)':<9}")
    pprint("-" * 75)
    if lat_pt_full is not None:
        row('1. PyTorch Native (DAE + AT + ZT)', lat_pt_full)
    row('2. ONNX Runtime (Classifier Only)', lat_onnx_clf)
    row('3. ONNX Full Defense (DAE + AT + ZT)', lat_onnx_full)
    pprint("-" * 75)

    pprint("\n[3/5] Batch throughput scaling...")
    df_throughput = benchmark_throughput_scaling(edge_pipeline, test_flows)
    pprint("-" * 75)
    pprint(f"{'Batch Size':<12} | {'Throughput (Flows/sec)':<25} | {'Avg Latency / Flow':<20}")
    pprint("-" * 75)
    for _, r in df_throughput.iterrows():
        pprint(f"{int(r['batch_size']):<12} | {r['flows_per_sec']:>23,.1f} | {r['avg_latency_ms']*1000:>15.1f} µs")
    pprint("-" * 75)

    pprint(f"\n[4/5] Burst test ({DDOS_BURST_FLOWS:,} flows in one batch)...")
    ddos_res = benchmark_ddos_burst_stress(edge_pipeline, test_flows, DDOS_BURST_FLOWS)
    pprint(f"  Duration   : {ddos_res['total_duration_sec']:.3f} s")
    pprint(f"  Throughput : {ddos_res['sustained_throughput_fps']:,.1f} flows / s")
    pprint(f"  RSS        : start {rss_start:.1f} MB | before models {rss_before_models:.1f} MB | "
           f"after models {rss_after_models:.1f} MB | after burst {ddos_res['peak_memory_mb']:.1f} MB")
    pprint("  (RSS is dominated by the Python + NumPy + ONNX Runtime process, not by the models.)")

    df_throughput['single_flow_p50_us'] = np.percentile(lat_onnx_full, 50)
    df_throughput['single_flow_p95_us'] = np.percentile(lat_onnx_full, 95)
    df_throughput['single_flow_p99_us'] = np.percentile(lat_onnx_full, 99)
    df_throughput['onnx_total_size_kb'] = total_edge_footprint_kb
    df_throughput['rss_before_models_mb'] = rss_before_models
    df_throughput['rss_models_delta_mb'] = rss_after_models - rss_before_models
    df_throughput['rss_after_burst_mb'] = ddos_res['peak_memory_mb']
    for k, v in host.items():
        df_throughput[k] = v
    df_throughput.to_csv(CSV_RESULTS_PATH, index=False)
    pprint(f"\n[5/5] Saved numerical benchmark results to: {CSV_RESULTS_PATH}")

    generate_edge_visualization(lat_pt_full, lat_onnx_clf, lat_onnx_full, df_throughput, host)


def generate_edge_visualization(lat_pt, lat_onnx_clf, lat_onnx_full, df_tp, host):
    """Two-panel plot: single-flow latency distribution and batch throughput."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    lat_data, labels = [lat_onnx_clf, lat_onnx_full], ['ONNX (Clf Only)', 'ONNX (Full Defense)']
    if lat_pt is not None:
        lat_data, labels = [lat_pt] + lat_data, ['PyTorch (Full)'] + labels
    ax1.boxplot(lat_data, tick_labels=labels, showfliers=False)
    ax1.set_title('Single-flow latency', fontsize=11)
    ax1.set_ylabel('Latency (µs)')
    for i, data in enumerate(lat_data):
        med = np.percentile(data, 50)
        ax1.text(i + 1, med, f' {med:.0f} µs', ha='left', va='center', fontsize=9)

    ax2.plot(df_tp['batch_size'], df_tp['flows_per_sec'], 'o-')
    ax2.set_title('Batch throughput', fontsize=11)
    ax2.set_xlabel('Batch size')
    ax2.set_ylabel('Flows / second')
    ax2.set_xscale('log', base=2)

    fig.suptitle(f"Host: {host['host_processor']} ({host['host_platform']}), "
                 f"ORT {host['onnxruntime_version']}. Not an edge-device measurement.", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    pprint(f"  [+] Saved plot to: {PLOT_PATH}")


if __name__ == '__main__':
    execute_edge_benchmark()
