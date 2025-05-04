# tensorrt_utils.py

import pycuda.autoinit            # creates CUDA context
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2

def load_engine(runtime: trt.Runtime, engine_path: str) -> trt.ICudaEngine:
    """Load a serialized TRT engine from disk (GPU)."""
    with open(engine_path, 'rb') as f:
        buf = f.read()
    return runtime.deserialize_cuda_engine(buf)

def allocate_buffers(engine: trt.ICudaEngine):
    """
    Allocate host & device buffers for each binding.
    Returns: inputs, outputs, bindings, stream
    """
    # determine number of bindings
    try:
        num_bindings = engine.num_bindings
    except AttributeError:
        num_bindings = 0
        while True:
            try:
                engine.get_binding_shape(num_bindings)
                num_bindings += 1
            except:
                break

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for idx in range(num_bindings):
        # clamp dynamic dims to 1
        shp  = engine.get_binding_shape(idx)
        shape = tuple(1 if d < 0 else d for d in shp)
        size  = int(np.prod(shape))
        dtype = trt.nptype(engine.get_binding_dtype(idx))

        host_mem = np.empty(size, dtype=dtype)
        dev_mem  = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))

        if engine.binding_is_input(idx):
            inputs.append({'host': host_mem, 'device': dev_mem})
        else:
            outputs.append({'host': host_mem, 'device': dev_mem})

    return inputs, outputs, bindings, stream

def do_inference(
    context: trt.IExecutionContext,
    bindings: list[int],
    inputs: list[dict],
    outputs: list[dict],
    stream: cuda.Stream,
    batch_size: int = 1
) -> list[np.ndarray]:
    """
    Asynchronously:
      1) H2D input copy
      2) engine execution
      3) D2H output copy
      4) sync stream
    """
    # host->device
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

    # inference (on GPU)
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)

    # device->host
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

    stream.synchronize()
    return [out['host'] for out in outputs]

def preprocess_for_trt(patch: np.ndarray, host_buffer: np.ndarray):
    """
    Copy H×W×3 uint8 BGR patch → host_buffer as
    NCHW float32 [0..1] on CPU, ready for a GPU memcpy.
    """
    N, C, H, W = host_buffer.shape
    img = cv2.resize(patch, (W, H))
    img = img[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB + normalize
    img = img.transpose(2, 0, 1)                     # HWC->CHW
    host_buffer[0] = img
