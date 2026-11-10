# birdcpp-fv2 - YOLO-FastestV2 trial build (see cpp/trial-fv2/README.txt)
#
#   make          build birdcpp-fv2 (sandbox server, two-output fv2 head)
#   make yf2eval  build the offline C accuracy harness (files, no camera)
#
# Same board toolchain as the production cpp/Makefile. Produces SEPARATE
# binaries so `killall` can never touch the production /home/root/birdcpp.
#
CC       = gcc
CFLAGS   = -O2 -std=gnu99 -Wall -pthread -I/home/root/ort/include
LDLIBS   = -l:libturbojpeg.so.0 -lm -ldl

all: birdcpp-fv2

birdcpp-fv2: birdcpp.c
	$(CC) $(CFLAGS) birdcpp.c -o birdcpp-fv2 $(LDLIBS)

yf2eval: eval.c
	$(CC) $(CFLAGS) eval.c -o yf2eval $(LDLIBS)

clean:
	rm -f birdcpp-fv2 yf2eval

.PHONY: all yf2eval clean
