CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -I. -I../../phi-core/c_core
PHICORE = ../../phi-core/c_core
BIN = /tmp/holo_c_test
XCHG = /tmp/holo_conv_exchange
SXCHG = /tmp/holo_splat_exchange

all: check

check: $(BIN) $(XCHG) $(SXCHG)
	$(BIN)

$(BIN): holo_ops.c test_holo_c.c holo_ops.h
	$(CC) $(CFLAGS) -o $(BIN) holo_ops.c test_holo_c.c $(PHICORE)/bridge.c

$(XCHG): holo_ops.c holo_conv.c conv_exchange.c holo_ops.h holo_conv.h
	$(CC) $(CFLAGS) -o $(XCHG) holo_ops.c holo_conv.c conv_exchange.c $(PHICORE)/bridge.c

$(SXCHG): holo_ops.c holo_conv.c splat_exchange.c holo_ops.h holo_conv.h
	$(CC) $(CFLAGS) -o $(SXCHG) holo_ops.c holo_conv.c splat_exchange.c $(PHICORE)/bridge.c

trap:
	for f in *.c *.h; do sed -e 's|//.*||' "$$f"; done | grep -nE "math\.h|[^_a-zA-Z](float|double)[ \t*(]|(float|double)\(" && echo FLOAT-FOUND && exit 1 || echo "no float in c_chain sources"

clean:
	rm -f $(BIN) $(XCHG) $(SXCHG)

.PHONY: all check trap clean
