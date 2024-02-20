import argparse
import subprocess
import numpy as np
import re
import os

batch_size = 100


def get_form(form):
    matmul_transpose_a = False
    matmul_transpose_b = False
    if form == "mtm":
        matmul_transpose_a = True
    if form == "mmt":
        matmul_transpose_b = True
    return matmul_transpose_a, matmul_transpose_b


def generate_matmul_func(m, n, k, form, chip):
    """Generates the MxNxK matrix multiplication function in MLIR."""
    matmul_transpose_a, matmul_transpose_b = get_form(form)
    it = "f16"
    ot = "f32"
    if args.chip == "gfx1100":
        ot = "f16"
    if matmul_transpose_a:
        matmul_function = (
            f"func.func @matmul(%lhs: tensor<{k}x{m}x{it}>, %rhs: tensor<{k}x{n}x{it}>) -> tensor<{m}x{n}x{ot}> {{\n"
            f"  %c0 = arith.constant 0.0 : {ot}\n"
            f"  %init = tensor.empty() : tensor<{m}x{n}x{ot}>\n"
            f"  %inital_result = linalg.fill ins(%c0 : {ot}) outs(%init : tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}x{ot}>\n"
            f"  %result = linalg.matmul_transpose_a ins(%lhs, %rhs: tensor<{k}x{m}x{it}>, tensor<{k}x{n}x{it}>)\n"
            f"             outs(%inital_result: tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}xf16>\n"
            f"  return %result : tensor<{m}x{n}x{ot}>\n"
            f"}}\n"
        )
    elif matmul_transpose_b:
        matmul_function = (
            f"func.func @matmul(%lhs: tensor<{m}x{k}x{it}>, %rhs: tensor<{n}x{k}x{it}>) -> tensor<{m}x{n}x{ot}> {{\n"
            f"  %c0 = arith.constant 0.0 : {ot}\n"
            f"  %init = tensor.empty() : tensor<{m}x{n}x{ot}>\n"
            f"  %inital_result = linalg.fill ins(%c0 : {ot}) outs(%init : tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}x{ot}>\n"
            f"  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<{m}x{k}x{it}>, tensor<{n}x{k}x{it}>)\n"
            f"             outs(%inital_result: tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}x{ot}>\n"
            f"  return %result : tensor<{m}x{n}x{ot}>\n"
            f"}}\n"
        )
    else:
        matmul_function = (
            f"func.func @matmul(%lhs: tensor<{m}x{k}x{it}>, %rhs: tensor<{k}x{n}x{it}>) -> tensor<{m}x{n}x{ot}> {{\n"
            f"  %c0 = arith.constant 0.0 : {ot}\n"
            f"  %init = tensor.empty() : tensor<{m}x{n}x{ot}>\n"
            f"  %inital_result = linalg.fill ins(%c0 : {ot}) outs(%init : tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}x{ot}>\n"
            f"  %result = linalg.matmul ins(%lhs, %rhs: tensor<{m}x{k}x{it}>, tensor<{k}x{n}x{it}>)\n"
            f"             outs(%inital_result: tensor<{m}x{n}x{ot}>) -> tensor<{m}x{n}x{ot}>\n"
            f"  return %result : tensor<{m}x{n}x{ot}>\n"
            f"}}\n"
        )
    return matmul_function


def execute_command(command, output_file=""):
    """Executes the given command and logs the output to the given file."""

    print("Executing command: ", command)
    print(" ".join(command))
    print(command)
    out = None
    err = None
    if output_file != "":
        with open(output_file, "w") as f:
            process = subprocess.Popen(command, stderr=f)
            process.wait()
    else:
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        out, err = process.communicate()
        process.wait()
    return out, err


def generate_mlir(args):
    matmul_str = generate_matmul_func(args.m, args.n, args.k, args.mma_form, args.chip)
    fname = f"tmp/matmul_m{args.m}_n{args.n}_k{args.k}.mlir"
    with open(fname, "w") as f:
        f.write(matmul_str)
    return fname


def compile(args):
    global batch_size
    command = [
        f"{args.iree_build}/tools/iree-compile",
        f"--iree-hal-benchmark-dispatch-repeat-count={batch_size}",
        f"{args.fname}",
        "-o",
        "matmul.vmfb",
    ]
    rocm_flags = [
        "--iree-hal-target-backends=rocm",
        f"--iree-rocm-target-chip={args.chip}",
        "--iree-rocm-link-bc=true",
        "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
    ]
    vulkan_flags = [
        "--iree-hal-target-backends=vulkan",
        #'--iree-vulkan-target-triple=adreno-a740-linux',
        "--iree-vulkan-target-triple=rdna3-unknown-linux",
    ]
    if args.vulkan:
        command += vulkan_flags
    else:
        command += rocm_flags
    if args.transform_dialect:
        command += [
            "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
            "--iree-codegen-use-transform-dialect-strategy=codegen",
            f"--iree-codegen-transform-dialect-library={args.spec_file}",
        ]
    if args.vector_distribution:
        command += [
            "--iree-codegen-llvmgpu-use-vector-distribution",
        ]
    if args.exec_dump:
        command += [f'--iree-hal-dump-executable-binaries-to={os.getcwd()}/tmp']
    if args.dump:
        command += [
            "-mlir-print-ir-after-all",
            "-mlir-disable-threading",
            f"--iree-hal-dump-executable-binaries-to={os.getcwd()}/tmp",
            f"--iree-hal-dump-executable-intermediates-to={os.getcwd()}/tmp",
        ]
    execute_command(command, "tmp/mlirdump.txt")


def validate(args):
    matmul_transpose_a, matmul_transpose_b = get_form(args.mma_form)
    if matmul_transpose_a:
        lhs = 0.005 * np.random.rand(int(args.k), int(args.m)).astype("float16")
        rhs = 0.005 * np.random.rand(int(args.k), int(args.n)).astype("float16")
        output = np.matmul(np.transpose(lhs), rhs)
    elif matmul_transpose_b:
        lhs = 0.005 * np.random.rand(int(args.m), int(args.k)).astype("float16")
        rhs = 0.005 * np.random.rand(int(args.n), int(args.k)).astype("float16")
        output = np.matmul(lhs, np.transpose(rhs))
    else:
        lhs = 0.005 * np.random.rand(int(args.m), int(args.k)).astype("float16")
        rhs = 0.005 * np.random.rand(int(args.k), int(args.n)).astype("float16")
        output = np.matmul(lhs, rhs)
    lhs_filename = f"lhs_m{args.m}_n{args.n}_k{args.k}.npy"
    with open(lhs_filename, "wb") as f:
        np.save(f, lhs.astype("float16"))
    rhs_filename = f"rhs_m{args.m}_n{args.n}_k{args.k}.npy"
    with open(rhs_filename, "wb") as f:
        np.save(f, rhs.astype("float16"))
    output_filename = f"output_m{args.m}_n{args.n}_k{args.k}.npy"
    if args.chip == "gfx90a" or args.chip == "gfx940":
        output = output.astype("float32")
    with open(output_filename, "wb") as f:
        np.save(f, output)
    device = "vulkan" if args.vulkan else "rocm"
    command = [
        f"{args.iree_build}/tools/iree-run-module",
        f"--device={device}",
        "--module=matmul.vmfb",
        '--function="matmul"',
        f"--input=@{lhs_filename}",
        f"--input=@{rhs_filename}",
        f"--expected_output=@{output_filename}",
    ]
    out, err = execute_command(command)
    output = out.decode("utf-8")
    print(output)


def benchmark(args):
    global batch_size
    matmul_transpose_a, matmul_transpose_b = get_form(args.mma_form)
    device = "vulkan" if args.vulkan else "rocm"
    if matmul_transpose_a:
        command = [
            f"{args.iree_build}/tools/iree-benchmark-module",
            "--module=matmul.vmfb",
            "--function=matmul",
            f'--input="{args.k}x{args.m}xf16"',
            f'--input="{args.k}x{args.n}xf16"',
            f"--device={device}",
            f"--batch_size={batch_size}",
        ]
    elif matmul_transpose_b:
        command = [
            f"{args.iree_build}/tools/iree-benchmark-module",
            "--module=matmul.vmfb",
            "--function=matmul",
            f'--input="{args.m}x{args.k}xf16"',
            f'--input="{args.n}x{args.k}xf16"',
            f"--device={device}",
            f"--batch_size={batch_size}",
        ]
    else:
        command = [
            f"{args.iree_build}/tools/iree-benchmark-module",
            "--module=matmul.vmfb",
            "--function=matmul",
            f'--input="{args.m}x{args.k}xf16"',
            f'--input="{args.k}x{args.n}xf16"',
            f"--device={device}",
            f"--batch_size={batch_size}",
        ]
    out, err = execute_command(command)
    output = out.decode("utf-8")
    print(output)
    time = float(re.findall(r"[-+]?(?:\d*\.*\d+)", output.split("\n")[3])[0])
    flops = (2 * int(args.m) * int(args.n) * int(args.k) / (time * 1e-3)) / (1e12)
    print("Throughput (TFLOPS/s) = ", flops)


parser = argparse.ArgumentParser(description="Matmul ROCM runner")

parser.add_argument("-i", "--input_file", help="The input mlir file.")
parser.add_argument("-s", "--spec_file", help="The spec file.")
parser.add_argument("-m", help="matrix M shape.")
parser.add_argument("-n", help="matrix N shape.")
parser.add_argument("-k", help="matrix K shape.")
parser.add_argument("-d", "--dump", action="store_true", help="Dump the ir after all.")
parser.add_argument(
    "-e", "--exec_dump", action="store_true", help="Dump the executable binary."
)
parser.add_argument("-c", "--compile", action="store_true", help="Compile the program.")
parser.add_argument("-r", "--run", action="store_true", help="Run the program.")
parser.add_argument(
    "-b", "--benchmark", action="store_true", help="Benchmark the program."
)
parser.add_argument(
    "-t", "--transform_dialect", action="store_true", help="Use td script"
)
parser.add_argument(
    "-vd", "--vector-distribution", action="store_true", help="Use vector distribution pipeline"
)
parser.add_argument("-v", "--vulkan", action="store_true", help="Use vulkan backend")
parser.add_argument(
    "-f",
    "--mma_form",
    choices=["mm", "mmt", "mtm"],
    default="mtm",
    nargs="?",
    const="mmt",
    help="MMA Form = mm, mmt, mtm",
)
parser.add_argument(
    "-x",
    "--chip",
    choices=["gfx1100", "gfx90a", "gfx940"],
    default="gfx90a",
    nargs="?",
    const="mtm",
    help="Supported chips = gfx1100, gfx90a, gfx940",
)
parser.add_argument(
    "-ib", "--iree_build", default="../iree-build", help="Path to iree-built directory."
)

try:
    os.makedirs("tmp")
except FileExistsError:
    print("tmp dir already exist, re-using it.")

args = parser.parse_args()

fname = generate_mlir(args)
args.fname = fname
if args.mma_form == "mtm":
    print("MMA is of the form : transpose(A) * B")
elif args.mma_form == "mmt":
    print("MMA is of the form : A * transpose(B)")
else:
    print("MMA is of the form : A * B")
print(f"Compiling for {args.chip}")

if args.compile:
    print("Compiling the program...")
    compile(args)
if args.run:
    print("Running the program...")
    validate(args)
if args.benchmark:
    print("Benchmarking the program...")
    benchmark(args)
