# birdcpp - the C birdboard for the TI SK-AM62 (see README-CPP.md)
#
#   make          build birdcpp
#   make tjtest   build + run the channel-order self-check
#                 (writes /tmp/tjtest_red.jpg + /tmp/tjtest_blue.jpg)
#
CC       = gcc
CFLAGS   = -O2 -std=gnu99 -Wall -pthread -I/home/root/ort/include
LDLIBS   = -l:libturbojpeg.so.0 -lm -ldl

all: birdcpp

birdcpp: birdcpp.c
	$(CC) $(CFLAGS) birdcpp.c -o birdcpp $(LDLIBS)

tjtest: birdcpp
	./birdcpp --tjtest

clean:
	rm -f birdcpp

.PHONY: all tjtest clean
