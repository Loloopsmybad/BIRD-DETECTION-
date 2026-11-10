# birdcpp-a - OPTION-A trial build (smaller model letterbox, see cpp/trial-a/)
#
#   make          build birdcpp-a
#
# Same board toolchain as the production cpp/Makefile. Produces a SEPARATE
# binary so the production /home/root/birdcpp build stays untouched.
#
CC       = gcc
CFLAGS   = -O2 -std=gnu99 -Wall -pthread -I/home/root/ort/include
LDLIBS   = -l:libturbojpeg.so.0 -lm -ldl

all: birdcpp-a

birdcpp-a: birdcpp.c
	$(CC) $(CFLAGS) birdcpp.c -o birdcpp-a $(LDLIBS)

clean:
	rm -f birdcpp-a

.PHONY: all clean
