/* ort_probe.c - onnxruntime aarch64 C-API feasibility + timing probe.
 *
 * Runtime-loaded (dlopen) instead of linked: the board's libstdc++.so.6
 * carries no symbol-version records, so the strict GNU ld refuses the
 * versioned refs of libonnxruntime.so at link time; ld.so binds them
 * by name at runtime the same way it does for cv2 (proven working).
 *
 * gcc -O2 -std=gnu99 ort_probe.c -o ort_probe \
 *     -I/home/root/ort/include -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>
#include "onnxruntime_c_api.h"

typedef struct {
    const OrtApi *(*GetApi)(uint32_t);
    const char *(*GetVersionString)(void);
} OrtApiBaseMirror;

static double ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    const char *model = argc > 1 ? argv[1] : "/home/root/birdcpp/best.onnx";
    int threads = argc > 2 ? atoi(argv[2]) : 4;
    int nrun = argc > 3 ? atoi(argv[3]) : 3;

    (void)model; (void)threads; (void)nrun;
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    void *lib = dlopen("/home/root/ort/lib/libonnxruntime.so",
                       RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(libonnxruntime.so): %s\n", dlerror());
        return 1;
    }
    fprintf(stderr, "step: dlopen ok\n");
    /* OrtGetApiBase is a FUNCTION returning the OrtApiBase struct ptr,
     * not the struct itself. */
    OrtApiBaseMirror *(*ort_get_base)(void) =
        (OrtApiBaseMirror *(*)(void))dlsym(lib, "OrtGetApiBase");
    if (!ort_get_base) {
        fprintf(stderr, "cast(OrtGetApiBase): %s\n", dlerror());
        return 1;
    }
    OrtApiBaseMirror *base = ort_get_base();
    if (!base) {
        fprintf(stderr, "OrtGetApiBase() returned NULL\n");
        return 1;
    }
    fprintf(stderr, "step: OrtGetApiBase() ok\n");
    const char *ver = base->GetVersionString();
    fprintf(stderr, "step: GetVersionString ok -> %s\n",
            ver ? ver : "(null)");
    const OrtApi *api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "GetApi(%d) returned NULL\n", ORT_API_VERSION);
        return 1;
    }
    fprintf(stderr, "step: GetApi(%d) ok\n", ORT_API_VERSION);
    printf("ORT version: %s\n", ver ? ver : "?");
    printf("ORT C-API version: %d\n", ORT_API_VERSION);

#define CHECK(x)                                                     \
    do {                                                             \
        OrtStatus *s_ = (x);                                         \
        if (s_) {                                                    \
            fprintf(stderr, "ORT ERROR [%d]: %s\n",                  \
                    (int)api->GetErrorCode(s_),                      \
                    api->GetErrorMessage(s_));                       \
            api->ReleaseStatus(s_);                                  \
            return 2;                                                \
        }                                                            \
    } while (0)

    OrtEnv *env = NULL;
    CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "probe", &env));

    OrtSessionOptions *so = NULL;
    CHECK(api->CreateSessionOptions(&so));
    CHECK(api->SetIntraOpNumThreads(so, threads));
    CHECK(api->SetInterOpNumThreads(so, 1));

    OrtSession *sess = NULL;
    double t0 = ms();
    CHECK(api->CreateSession(env, model, so, &sess));
    printf("session create: %.0f ms (threads=%d)\n", ms() - t0, threads);

    size_t nin = 0, nout = 0;
    CHECK(api->SessionGetInputCount(sess, &nin));
    CHECK(api->SessionGetOutputCount(sess, &nout));
    OrtAllocator *alloc = NULL;
    CHECK(api->GetAllocatorWithDefaultOptions(&alloc));
    char *in_name = NULL, *out_name = NULL;
    CHECK(api->SessionGetInputName(sess, 0, alloc, &in_name));
    CHECK(api->SessionGetOutputName(sess, 0, alloc, &out_name));
    printf("inputs=%zu outputs=%zu  in[0]='%s' out[0]='%s'\n",
           nin, nout, in_name, out_name);

    OrtTypeInfo *ti = NULL;
    CHECK(api->SessionGetInputTypeInfo(sess, 0, &ti));
    const OrtTensorTypeAndShapeInfo *ots = NULL;
    CHECK(api->CastTypeInfoToTensorInfo(ti, &ots));
    size_t nd = 0;
    CHECK(api->GetDimensionsCount(ots, &nd));
    int64_t dims[8] = {0};
    CHECK(api->GetDimensions(ots, dims, nd));
    printf("input dims: [");
    for (size_t i = 0; i < nd; i++)
        printf("%lld%s", (long long)(dims[i] > 0 ? dims[i] : -1),
               i + 1 < nd ? "," : "");
    printf("]\n");
    api->ReleaseTypeInfo(ti);

    /* dummy 640x640 CHW f32 (letterboxed gray ~0.45) */
    const long long N = 3LL * 640 * 640;
    float *buf = (float *)malloc((size_t)N * sizeof(float));
    if (!buf) return 1;
    for (long long i = 0; i < N; i++) buf[i] = 0.45f;

    OrtMemoryInfo *mi = NULL;
    CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                   &mi));
    int64_t indims[4] = {1, 3, 640, 640};
    OrtValue *inval = NULL;
    CHECK(api->CreateTensorWithDataAsOrtValue(
        mi, (void *)buf, (size_t)N * sizeof(float), indims, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inval));

    const char *in_names[1] = {in_name};
    const char *out_names[1] = {out_name};
    OrtValue *outs[1] = {NULL};

    double total = 0.0;
    for (int i = 0; i < nrun + 1; i++) {
        double a = ms();
        CHECK(api->Run(sess, NULL, in_names,
                       (const OrtValue *const *)&inval, 1,
                       out_names, 1, outs));
        double d = ms() - a;
        if (i == 0) {
            printf("first forward (incl. warm/jit): %.0f ms\n", d);
        } else {
            total += d;
            printf("forward %d: %.0f ms\n", i, d);
        }
    }
    printf("steady forward avg: %.0f ms over %d runs (threads=%d)\n",
           total / nrun, nrun, threads);

    /* report output dims (YOLOv8: 1x(4+nc)x8400 or transposed) */
    OrtTensorTypeAndShapeInfo *oi = NULL;
    CHECK(api->GetTensorTypeAndShape(outs[0], &oi));
    size_t ond = 0;
    CHECK(api->GetDimensionsCount(oi, &ond));
    int64_t odims[8] = {0};
    CHECK(api->GetDimensions(oi, odims, ond));
    printf("output dims: [");
    for (size_t i = 0; i < ond; i++)
        printf("%lld%s", (long long)(odims[i] > 0 ? odims[i] : -1),
               i + 1 < ond ? "," : "");
    printf("]\n");
    api->ReleaseTensorTypeAndShapeInfo(oi);

    api->ReleaseValue(outs[0]);
    api->ReleaseValue(inval);
    free(buf);
    api->ReleaseMemoryInfo(mi);
    api->ReleaseSession(sess);
    api->ReleaseSessionOptions(so);
    api->ReleaseEnv(env);
    printf("PROBE-OK\n");
    return 0;
}
