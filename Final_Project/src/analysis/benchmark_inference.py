"""
Inference Benchmarking Script

Benchmarks inference latency, throughput, and memory usage for trained models
on CPU or GPU to quantify deployment efficiency trade-offs.

Usage:
    python -m src.analysis.benchmark_inference \
        --model roberta-large \
        --checkpoint artifacts/models/roberta-large \
        --device cpu \
        --batch-sizes 1,8,16,32 \
        --num-samples 1000 \
        --output artifacts/stats/efficiency/roberta-cpu.csv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint directory
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info(f"Target device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Device: {next(model.parameters()).device}")

    return model, tokenizer


def generate_synthetic_data(num_samples: int, max_length: int = 128) -> List[str]:
    """Generate synthetic text data for benchmarking.

    Args:
        num_samples: Number of samples to generate
        max_length: Maximum text length in tokens

    Returns:
        List of synthetic text strings
    """
    logger.info(f"Generating {num_samples} synthetic samples...")

    # Use realistic emotion classification text patterns
    templates = [
        "I am so {} about this amazing news!",
        "This is absolutely {} and I can't believe it happened.",
        "Feeling really {} right now after what just occurred.",
        "That moment when you're just completely {}.",
        "I'm {} that things turned out this way.",
    ]

    emotions = [
        "happy", "excited", "grateful", "proud", "joyful",
        "sad", "angry", "frustrated", "disappointed", "worried",
        "surprised", "confused", "curious", "hopeful", "relaxed"
    ]

    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        emotion = emotions[i % len(emotions)]
        samples.append(template.format(emotion))

    return samples


def benchmark_batch(model, tokenizer, texts: List[str], device: str,
                   batch_size: int, warmup_runs: int = 3) -> Dict:
    """Benchmark inference for a specific batch size.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        texts: List of input texts
        device: Device to run on
        batch_size: Batch size to test
        warmup_runs: Number of warmup iterations

    Returns:
        Dictionary with benchmark metrics
    """
    logger.info(f"  Benchmarking batch_size={batch_size}...")

    # Prepare batches
    num_batches = len(texts) // batch_size
    actual_samples = num_batches * batch_size

    # Warmup
    for _ in range(warmup_runs):
        batch_texts = texts[:batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

    # Benchmark
    latencies = []
    torch.cuda.synchronize() if device == 'cuda' else None

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]

        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.perf_counter()

        batch_latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(batch_latency)

    # Compute metrics
    latencies = np.array(latencies)
    mean_batch_latency = np.mean(latencies)
    std_batch_latency = np.std(latencies)
    p50_batch_latency = np.percentile(latencies, 50)
    p95_batch_latency = np.percentile(latencies, 95)
    p99_batch_latency = np.percentile(latencies, 99)

    # Per-sample latency
    mean_sample_latency = mean_batch_latency / batch_size

    # Throughput (samples per second)
    throughput = 1000.0 / mean_sample_latency  # 1000ms / latency_per_sample

    logger.info(f"    Mean latency: {mean_batch_latency:.2f} ms/batch "
               f"({mean_sample_latency:.2f} ms/sample)")
    logger.info(f"    Throughput: {throughput:.2f} samples/sec")

    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'total_samples': actual_samples,
        'mean_batch_latency_ms': mean_batch_latency,
        'std_batch_latency_ms': std_batch_latency,
        'p50_batch_latency_ms': p50_batch_latency,
        'p95_batch_latency_ms': p95_batch_latency,
        'p99_batch_latency_ms': p99_batch_latency,
        'mean_sample_latency_ms': mean_sample_latency,
        'throughput_samples_per_sec': throughput,
    }


def run_benchmarks(model, tokenizer, texts: List[str], device: str,
                   batch_sizes: List[int]) -> pd.DataFrame:
    """Run benchmarks for multiple batch sizes.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        texts: List of input texts
        device: Device to run on
        batch_sizes: List of batch sizes to test

    Returns:
        DataFrame with benchmark results
    """
    logger.info(f"\nRunning benchmarks for {len(batch_sizes)} batch sizes...")

    results = []
    for batch_size in batch_sizes:
        try:
            metrics = benchmark_batch(model, tokenizer, texts, device, batch_size)
            results.append(metrics)
        except Exception as e:
            logger.error(f"  Failed for batch_size={batch_size}: {e}")
            continue

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark inference performance for emotion classification models'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model type (e.g., roberta-large, distilbert-base)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        required=True,
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--batch-sizes',
        type=str,
        required=True,
        help='Comma-separated list of batch sizes to test (e.g., 1,8,16,32)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        required=True,
        help='Number of samples to use for benchmarking'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    logger.info("="*70)
    logger.info("INFERENCE BENCHMARKING")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info("="*70)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)

    # Generate synthetic data
    texts = generate_synthetic_data(args.num_samples)

    # Run benchmarks
    results_df = run_benchmarks(model, tokenizer, texts, args.device, batch_sizes)

    # Add metadata
    results_df['model'] = args.model
    results_df['device'] = args.device
    results_df['checkpoint'] = args.checkpoint

    # Reorder columns
    column_order = [
        'model', 'device', 'batch_size', 'num_batches', 'total_samples',
        'mean_batch_latency_ms', 'std_batch_latency_ms',
        'p50_batch_latency_ms', 'p95_batch_latency_ms', 'p99_batch_latency_ms',
        'mean_sample_latency_ms', 'throughput_samples_per_sec', 'checkpoint'
    ]
    results_df = results_df[column_order]

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False, float_format='%.4f')

    logger.info("\n" + "="*70)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*70)
    logger.info(f"\n{results_df.to_string(index=False)}\n")
    logger.info("="*70)
    logger.info(f"Results saved to: {args.output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
