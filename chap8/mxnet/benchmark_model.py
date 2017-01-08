import time
import mxnet as mx

benchmark_dataiter = mx.io.ImageRecordIter(
    path_imgrec="../data/test.rec",
    data_shape=(1, 28, 28),
    batch_size=64,
    mean_r=128,
    scale=0.00390625,
)

mod = mx.mod.Module.load('mnist_lenet', 35, context=mx.gpu(2))
mod.bind(
    data_shapes=benchmark_dataiter.provide_data, 
    label_shapes=benchmark_dataiter.provide_label, 
    for_training=False)

start = time.time()
for i, batch in enumerate(benchmark_dataiter):
    mod.forward(batch)
time_elapsed = time.time() - start
msg = '{} batches iterated!\nAverage forward time per batch: {:.6f} ms'
print(msg.format(i+1, 1000*time_elapsed/float(i)))
