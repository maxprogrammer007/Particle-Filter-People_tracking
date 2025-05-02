import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver

# -----------------------------------------------------------------------------
# Helpers to build / load a TRT Engine
# -----------------------------------------------------------------------------

def load_engine(runtime: trt.Runtime, engine_path: str) -> trt.ICudaEngine:
    """Load a serialized TensorRT engine from disk."""
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")
    with open(engine_path, "rb") as f:
        serialized = f.read()
    return runtime.deserialize_cuda_engine(serialized)


# -----------------------------------------------------------------------------
# Helpers to allocate I/O buffers for a given engine
# -----------------------------------------------------------------------------

def allocate_buffers(engine):
    """
    For each binding (input + output), allocate host & device buffers
    Returns:
      inputs  : list of dict { 'host': np.ndarray, 'device': cuda.DeviceAllocation }
      outputs : same for outputs
      bindings: list of device ptrs, in binding order
      stream  : a single CUDA stream
    """
    import numpy as np

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    # figure out how many bindings by just trying get_binding_dtype() until it fails
    nb = 0
    while True:
        try:
            engine.get_binding_dtype(nb)
        except:
            break
        nb += 1

    for idx in range(nb):
        dtype = engine.get_binding_dtype(idx)
        shape = engine.get_binding_shape(idx)
        size  = trt.volume(shape)

        # host array
        host_mem = np.empty(size, dtype=trt.nptype(dtype))
        # device buffer
        dev_mem  = cuda.mem_alloc(host_mem.nbytes)

        if engine.binding_is_input(idx):
            inputs.append({'host': host_mem, 'device': dev_mem})
        else:
            outputs.append({'host': host_mem, 'device': dev_mem})

        bindings.append(int(dev_mem))

    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """
    Perform inference given:
      context  - the execution context
      bindings - list of device pointers in TRT binding order
      inputs   - list of dicts for each input binding
      outputs  - list of dicts for each output binding
      stream   - CUDA stream
    Returns:
      list of numpy arrays, one per output
    """
    # 1) copy input host -> device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # 2) execute
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 3) copy device -> host
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # wait for everything to finish
    stream.synchronize()
    return [out['host'] for out in outputs]
