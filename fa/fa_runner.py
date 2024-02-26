#!/usr/bin/env python
import os
import subprocess
import argparse
import numpy as np
import re
import logging

logging.basicConfig()
logger = logging.getLogger('[fa]')
logger.setLevel(logging.INFO)

def get_rocm_flags(args):
    return [
      f'--iree-hal-target-backends={args.backend}',\
      f'--iree-rocm-target-chip={args.chip}',
      '--iree-rocm-link-bc=true',
      '--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode',
      '--iree-rocm-waves-per-eu=2',
      '--iree-codegen-gpu-native-math-precision=true',
    ]

class Hyperparameters:
    SEED = 7
    IREE_BENCHMARK_REPS = 100
    VALIDATION_TOL = 1e-1
    VMFB_NAME = 'attn.vmfb'
    FUNC_NAME = 'attention'
    def get_backend_flags(args):
        if args.backend == 'rocm':
            return get_rocm_flags(args)

def get_vmfb_file(args):
    return f'{args.artifact_dir}/{Hyperparameters.VMFB_NAME}'

def compute_tflops(batch, num_heads, seq_len, head_dim, time_in_ms):
    """Computes the TFLOPS / sec for FA (2 matmuls)"""
    time_in_s = (time_in_ms) * 1e-3
    flops = ( (4 * (seq_len**2) * head_dim * batch * num_heads ) / time_in_s )
    return (flops / 1e12)

def execute_command(command, output_file=''):
    """Executes the given command and logs the output to the given file."""
    logger.info('Executing command: ' + ' '.join(command))
    out = None
    err = None
    if output_file != '':
        with open(output_file, 'w') as f:
            process = subprocess.Popen(command, stderr=f)
            process.wait()
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        out, err = process.communicate()
        process.wait()
    return out, err

def get_transposed_shape(args):
    if args.transpose_v is None:
        return args.shape
    split_values = args.shape.split('x')
    split_values[2], split_values[1] = split_values[1], split_values[2]
    return 'x'.join(split_values)

def create_mlir(args):
    ir = ""
    value_shape = get_transposed_shape(args)
    if args.transpose_v is None:
        transpose_v = 'false'
    else:
        transpose_v = 'true'

    if args.inline:
        ir += f"func.func @{Hyperparameters.FUNC_NAME}() -> tensor<{args.shape}> {{\n"
        ir += f"%query = util.unfoldable_constant dense<1.0> : tensor<{args.shape}>\n"
        ir += f"%key = util.unfoldable_constant dense<2.0> : tensor<{args.shape}>\n"
        ir += f"%value = util.unfoldable_constant dense<0.5> : tensor<{value_shape}>\n"
    else:
        ir += f"func.func @{Hyperparameters.FUNC_NAME}(%query: tensor<{args.shape}>, %key: tensor<{args.shape}>, %value: tensor<{value_shape}>) -> tensor<{args.shape}> {{\n"
    ir += f"  %0 = tensor.empty() : tensor<{args.shape}>\n"
    ir += f"  %1 = iree_linalg_ext.attention {{transpose_v = {transpose_v}}} ins(%query, %key, %value : tensor<{args.shape}>, tensor<{args.shape}>, tensor<{value_shape}>) outs(%0 : tensor<{args.shape}>) -> tensor<{args.shape}>\n"
    ir += f"  return %1 : tensor<{args.shape}>\n"
    ir += "}\n"
    filename = "attention_" + args.shape + ".mlir"
    with open(filename, 'w') as f:
        f.write(ir)
    return filename

def get_td_flags(args):
    return [
      '--iree-codegen-llvmgpu-enable-transform-dialect-jit=false',\
      f'--iree-codegen-transform-dialect-library={args.spec_file}'
    ]

def get_debug_flags(args):
    return [
      '--mlir-disable-threading',\
      '--mlir-print-ir-after-all',\
      f'--iree-hal-dump-executable-binaries-to={args.artifact_dir}',\
      f'--iree-hal-dump-executable-intermediates-to={args.artifact_dir}'
    ]

def compile(args):
    flags = [
      f'{args.iree_build_dir}/tools/iree-compile',\
      '--iree-vm-bytecode-module-output-format=flatbuffer-binary',\
    ]
    # TD specific flags
    flags += get_td_flags(args)
    # Backend specific flags
    flags += Hyperparameters.get_backend_flags(args)
    if args.dump:
        flags += get_debug_flags(args)
    flags += [f'-iree-hal-benchmark-dispatch-repeat-count={Hyperparameters.IREE_BENCHMARK_REPS}']
    flags += [
      f'{args.input_file}',
      '-o',
      get_vmfb_file(args)
    ]
    execute_command(flags, 'fa_dump.txt')
    if not os.path.exists(get_vmfb_file(args)):
        logger.warning("Compilation failed!")


def compute_reference_inputs_and_outputs(args):
    # TODO: Load binary numpy file when available
    import torch
    shape = args.shape
    B, N, d = [int(x) for x in shape.split('x')[:-1]]
    torch.manual_seed(Hyperparameters.SEED)

    def compute_attention_reference(q, k, v):
        kT = torch.permute(k, (0, 2, 1))
        s = torch.matmul(q, kT)
        p = torch.nn.Softmax(dim=2)(s)
        return torch.matmul(p, v)

    def construct_inputs(B, N, d):
        q = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
        k = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
        v = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
        return q, k, v

    q, k, v = construct_inputs(B, N, d)
    output = compute_attention_reference(q, k, v)

    # Write matrices
    with open(f'query_{shape}.npy', 'wb') as f:
        np.save(f, q.detach().cpu().numpy())
    with open(f'key_{shape}.npy', 'wb') as f:
        np.save(f, k.detach().cpu().numpy())
    if args.transpose_v:
        v = torch.permute(v, (0, 2, 1))
    with open(f'value_{shape}.npy', 'wb') as f:
        np.save(f, v.detach().cpu().numpy())
    with open(f'output_{shape}.npy', 'wb') as f:
        np.save(f, output.detach().cpu().numpy())

def check_result(args):
  golden = np.load(f'output_{args.shape}.npy')
  computed = np.load(f'computed_{args.shape}.npy')
  error = np.max(np.abs(golden - computed))
  # TODO: This tolerance might be too high
  if error < Hyperparameters.VALIDATION_TOL:
      logger.info(f"[Success] With error = {error} < {Hyperparameters.VALIDATION_TOL}")
  else:
      logger.info(f"[Failure] Got {error} > {Hyperparameters.VALIDATION_TOL}")

def validate(args):
    compute_reference_inputs_and_outputs(args)
    flags = [
      f'{args.iree_build_dir}/tools/iree-run-module',\
      '--module=' + get_vmfb_file(args),\
      f'--function={Hyperparameters.FUNC_NAME}',\
      f'--input=@query_{args.shape}.npy',\
      f'--input=@key_{args.shape}.npy',\
      f'--input=@value_{args.shape}.npy',\
      f'--device={args.backend}',\
      f'--output=@computed_{args.shape}.npy'
    ]
    execute_command(flags)
    check_result(args)

def extract_time(out):
    output = out.decode('utf-8')
    logger.info("\n" + output)
    time_in_ms = float(re.findall(r"[-+]?(?:\d*\.*\d+)", output.split('\n')[3])[0])
    return time_in_ms

def split_shape(args):
    shape = args.shape.split('x')
    batch = 1
    num_heads = int(shape[0])
    seq_len = int(shape[1])
    head_dim = int(shape[2])
    return batch, num_heads, seq_len, head_dim

def benchmark(args):
    output_file = f'attention_{args.shape}'
    command = [
      f'{args.iree_build_dir}/tools/iree-benchmark-module',\
      '--module=' + get_vmfb_file(args),\
      f'--function={Hyperparameters.FUNC_NAME}',\
      f'--device={args.backend}',
      f'--batch_size={Hyperparameters.IREE_BENCHMARK_REPS}'
    ]
    if not args.inline:
        value_shape = get_transposed_shape(args)
        command += [
            f'--input="{args.shape}"',\
            f'--input="{args.shape}"',\
            f'--input="{value_shape}"'
        ]
    out, _ = execute_command(command)
    if out is None:
        logger.warning("Failed to extract metrics!")
        return
    time_in_ms = extract_time(out)
    tflops = compute_tflops(*split_shape(args), time_in_ms)
    logger.info("Throughput (TFLOPS/s) = " + str(tflops))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-compile_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("-benchmark_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("-dump", action=argparse.BooleanOptionalAction)
    parser.add_argument("-shape", type=str, default='1x1024x128xf16')
    parser.add_argument("-spec_file", type=str, default='attention_transform_spec.mlir')
    parser.add_argument("-inline", action=argparse.BooleanOptionalAction)
    parser.add_argument("-iree_build_dir", type=str, default='/home/harsh/iree-build')
    parser.add_argument("-chip", type=str, default='gfx90a')
    parser.add_argument("-artifact_dir", type=str, default='/home/harsh/iree/tmp')
    parser.add_argument("-backend", type=str, default='rocm')
    parser.add_argument("-transpose_v", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    class State:
        COMPILE = 0
        VALIDATE = 1
        BENCHMARK = 2
        def __init__(self, start, end):
            self.start = start
            self.state = start
            self.end = end
            self.increment = 1
        def evaluate(self, args):
            if self.state == State.COMPILE: compile(args)
            if self.state == State.VALIDATE: validate(args)
            if self.state == State.BENCHMARK: benchmark(args)
        def print(self):
            if self.state == State.COMPILE: logger.info("Compiling ...")
            if self.state == State.VALIDATE: logger.info("Validating ...")
            if self.state == State.BENCHMARK: logger.info("Benchmarking ...")
        def run(self, args):
            done = False
            while not done:
                self.print()
                self.evaluate(args)
                self.state += self.increment
                if self.state > self.end:
                    done = True


    s = State(State.COMPILE, State.BENCHMARK)

    if args.compile_only:
        s.end = State.COMPILE
    if args.benchmark_only:
        s.state = State.BENCHMARK
    if args.inline:
        s.increment = State.BENCHMARK
    if args.dump is None:
        args.dump = False

    args.input_file = create_mlir(args)
    s.run(args)

