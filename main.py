import tensorflow as tf


def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")

    cpu_devices = tf.config.list_physical_devices("CPU")
    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"CPUs: {cpu_devices}")
    print(f"GPUs: {gpu_devices}")

    if not gpu_devices:
        print("No GPU detected.")
        return

    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)

    print(f"Result device: {c.device}")
    print(f"Result shape: {c.shape}")
    print(f"Result sum: {float(tf.reduce_sum(c).numpy())}")


if __name__ == "__main__":
    main()
