CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -I. -I../../phi-core/c_core
NVCC = nvcc
# native arch: the exchange binary builds+runs on the same box (override
# NVFLAGS for cross-builds, e.g. -gencode arch=compute_80,code=sm_80).
NVFLAGS = -O2 -arch=native -I. -I../../phi-core/c_core
PHICORE = ../../phi-core/c_core
BIN = /tmp/holo_c_test
XCHG = /tmp/holo_conv_exchange
SXCHG = /tmp/holo_splat_exchange
MXCHG = /tmp/holo_matmul_exchange
CXCHG = /tmp/holo_matmul_cuda_exchange
UCXCHG = /tmp/holo_units_cuda_exchange

all: check

check: $(BIN) $(XCHG) $(SXCHG) $(MXCHG)
	$(BIN)

cuda: $(CXCHG) $(UCXCHG)

$(CXCHG): matmul_cuda.cu
	$(NVCC) $(NVFLAGS) -o $(CXCHG) matmul_cuda.cu

$(UCXCHG): holo_units.cu
	$(NVCC) $(NVFLAGS) -o $(UCXCHG) holo_units.cu

$(BIN): holo_ops.c test_holo_c.c holo_ops.h
	$(CC) $(CFLAGS) -o $(BIN) holo_ops.c test_holo_c.c $(PHICORE)/bridge.c

$(XCHG): holo_ops.c holo_conv.c conv_exchange.c holo_ops.h holo_conv.h
	$(CC) $(CFLAGS) -o $(XCHG) holo_ops.c holo_conv.c conv_exchange.c $(PHICORE)/bridge.c

$(SXCHG): holo_ops.c holo_conv.c splat_exchange.c holo_ops.h holo_conv.h
	$(CC) $(CFLAGS) -o $(SXCHG) holo_ops.c holo_conv.c splat_exchange.c $(PHICORE)/bridge.c

$(MXCHG): holo_matmul.c matmul_exchange.c holo_matmul.h
	$(CC) $(CFLAGS) -o $(MXCHG) holo_matmul.c matmul_exchange.c $(PHICORE)/bridge.c

trap:
	for f in *.c *.h *.cu; do sed -e 's|//.*||' "$$f"; done | grep -nE "math\.h|[^_a-zA-Z](float|double)[ \t*(]|(float|double)\(" && echo FLOAT-FOUND && exit 1 || echo "no float in c_chain C/CUDA sources"

clean:
	rm -f $(BIN) $(XCHG) $(SXCHG) $(MXCHG) $(CXCHG) $(UCXCHG)

.PHONY: all check trap clean cuda
