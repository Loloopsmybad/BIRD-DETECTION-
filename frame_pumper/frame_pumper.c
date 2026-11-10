/*
 * frame_pumper.c - camera frame copier for camkit (TI SK-AM62, IMX219).
 *
 * Streams RAW RG10 frames from /dev/video0 to stdout, one write() per
 * frame. The heavy part - copying 4 MB from uncached DMA mmap memory -
 * is split across N threads in parallel (uncached reads are latency
 * bound, so thread count converts directly into bandwidth). Frames leave
 * one after another, no gaps.
 *
 * usage: frame_pumper <device> <width> <height> [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <signal.h>
#include <pthread.h>

#define V4L2_BUF_TYPE_VIDEO_CAPTURE 1
#define V4L2_MEMORY_MMAP 1

/* litte-endian aarch64, kernel-5.10 vendor ABI (proven):
 * v4l2_requestbuffers = 20 bytes, v4l2_buffer = 88 bytes */
#define VIDIOC_REQBUFS   0xC0145608
#define VIDIOC_QUERYBUF  0xC0585609
#define VIDIOC_QBUF      0xC058560F
#define VIDIOC_DQBUF     0xC0585611
#define VIDIOC_STREAMON  0x40045612
#define VIDIOC_STREAMOFF 0x40045613

struct v4l2_buf_hdr_tail {
    unsigned int reserved2;
    int request_fd;
};

struct v4l2_buffer_88 {
    unsigned int index, type, bytesused, flags, field;
    long ts_sec, ts_usec;
    unsigned int tc_type, tc_flags;
    unsigned char tc_frames, tc_sec, tc_min, tc_hour, tc_user[4];
    unsigned int sequence, memory;
    union { unsigned int offset; unsigned long userptr; int fd; } m;
    unsigned int length, reserved2;
    int request_fd;
};

struct v4l2_requestbuffers_20 {
    unsigned int count, type, memory, capabilities;
    unsigned char reserved[4];
};

static int fd_v;
static unsigned char *OUT = NULL;

static unsigned char *out_buf(void) { return OUT; }

static unsigned int frame_bytes;

struct chunk {
    const unsigned char *src;
    unsigned char *dst;
    unsigned long len;
};

static void *copier(void *arg)
{
    struct chunk *c = (struct chunk *)arg;
    memcpy(c->dst, c->src, c->len);
    return NULL;
}

static int stream_frames(int nframes, int nthreads)
{
    struct v4l2_requestbuffers_20 rb;
    struct v4l2_buffer_88 b;
    unsigned char *maps[8];
    unsigned int lens[8];
    int nbuf, i, seq, dequeued;
    struct chunk chunks[16];
    pthread_t th[16];
    struct pollfd pfd;

    memset(&rb, 0, sizeof(rb));
    rb.count = 4; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_v, VIDIOC_REQBUFS, &rb) < 0) {
        perror("REQBUFS"); return -1;
    }
    nbuf = rb.count;
    if (nbuf > 8) nbuf = 8;
    for (i = 0; i < nbuf; i++) {
        memset(&b, 0, sizeof(b));
        b.index = i; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_v, VIDIOC_QUERYBUF, &b) < 0) { perror("QUERYBUF"); return -1; }
        lens[i] = b.length;
        maps[i] = mmap(NULL, b.length, PROT_READ, MAP_SHARED, fd_v, b.m.offset);
        if (maps[i] == MAP_FAILED) { perror("MMAP"); return -1; }
    }
    for (i = 0; i < nbuf; i++) {
        memset(&b, 0, sizeof(b));
        b.index = i; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_v, VIDIOC_QBUF, &b) < 0) { perror("QBUF"); return -1; }
    }
    i = 1;
    if (ioctl(fd_v, VIDIOC_STREAMON, &i) < 0) { perror("STREAMON"); return -1; }

    for (seq = 0; seq < nframes; ) {
        pfd.fd = fd_v; pfd.events = POLLIN; pfd.revents = 0;
        if (poll(&pfd, 1, 2000) < 0) { perror("POLL"); continue; }
        if (!(pfd.revents & POLLIN)) continue;

        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_v, VIDIOC_DQBUF, &b) < 0) {
            if (errno == EAGAIN) continue;
            perror("DQBUF"); break;
        }
        i = b.index;
        dequeued = b.bytesused;
        {
            unsigned long per = frame_bytes / nthreads;
            unsigned long done = 0;
            int t, used = nthreads;
            for (t = 0; t < nthreads - 1; t++) {
                chunks[t].src = maps[i] + done;
                chunks[t].dst = out_buf() + done;
                chunks[t].len = per;
                done += per;
            }
            chunks[nthreads - 1].len = frame_bytes - done;
            chunks[nthreads - 1].src = maps[i] + done;
            chunks[nthreads - 1].dst = out_buf() + done;
            for (t = 0; t < used; t++)
                pthread_create(&th[t], NULL, copier, &chunks[t]);
            for (t = 0; t < used; t++)
                pthread_join(th[t], NULL);
        }
        {
            /* write-all: pipes are finite; a 4 MB frame can come back
             * short or EINTR and that silently truncates the feed */
            const unsigned char *p = out_buf();
            size_t left = frame_bytes;
            int bad = 0;
            while (left > 0) {
                ssize_t n = write(1, p, left);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    perror("write");
                    bad = 1;
                    break;
                }
                p += n;
                left -= (size_t)n;
            }
            if (bad) break;
        }
        memset(&b, 0, sizeof(b));
        b.index = i; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
        ioctl(fd_v, VIDIOC_QBUF, &b);
        seq++;
        (void)dequeued;
    }
    ioctl(fd_v, VIDIOC_STREAMOFF, &i);
    rb.count = 0;
    ioctl(fd_v, VIDIOC_REQBUFS, &rb);
    return 0;
}

int main(int argc, char **argv)
{
    int nframes = 1000000, nthreads = 4;
    long w, h;
    unsigned int one = 1;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <device> <width> <height> [threads] [frames]\n", argv[0]);
        return 2;
    }
    /* die with the parent (camkit) - never leak the camera */
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    w = atol(argv[2]); h = atol(argv[3]);
    if (argc > 4) nthreads = atoi(argv[4]);
    if (argc > 5) nframes = atoi(argv[5]);
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;

    frame_bytes = (unsigned int)(w * h * 2);
    OUT = malloc(frame_bytes);
    if (!OUT) { perror("malloc"); return 1; }

    fd_v = open(argv[1], O_RDWR);
    if (fd_v < 0) { perror("open"); return 1; }

    /* hint the parser: immediate output, unbuffered */
    setvbuf(stdout, NULL, _IONBF, 0);

    stream_frames(nframes, nthreads);
    close(fd_v);
    return 0;
}
