#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/*
 * 本文件是面向本项目的最小 SM3 CPython 扩展。
 * 接口和测试向量与用户提供的 GmSSL 工程保持兼容，但不引入完整 GmSSL
 * 的 SM9 对象模型，避免改变现有动态累加器环签名的实验语义。
 */

#define SM3_BLOCK_SIZE 64
#define SM3_DIGEST_SIZE 32

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    uint8_t buffer[SM3_BLOCK_SIZE];
    size_t buffered;
} sm3_ctx;

static uint32_t rol32(uint32_t value, unsigned int bits) {
    bits &= 31U;
    return bits ? (value << bits) | (value >> (32U - bits)) : value;
}

static uint32_t load_be32(const uint8_t *data) {
    return ((uint32_t)data[0] << 24)
        | ((uint32_t)data[1] << 16)
        | ((uint32_t)data[2] << 8)
        | (uint32_t)data[3];
}

static void store_be32(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)(value >> 24);
    out[1] = (uint8_t)(value >> 16);
    out[2] = (uint8_t)(value >> 8);
    out[3] = (uint8_t)value;
}

static void store_be64(uint8_t *out, uint64_t value) {
    for (int i = 7; i >= 0; --i) {
        out[i] = (uint8_t)value;
        value >>= 8;
    }
}

static uint32_t p0(uint32_t value) {
    return value ^ rol32(value, 9) ^ rol32(value, 17);
}

static uint32_t p1(uint32_t value) {
    return value ^ rol32(value, 15) ^ rol32(value, 23);
}

static void sm3_compress(uint32_t state[8], const uint8_t block[SM3_BLOCK_SIZE]) {
    uint32_t w[68];
    uint32_t wp[64];
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 16; ++i) {
        w[i] = load_be32(block + i * 4);
    }
    for (int i = 16; i < 68; ++i) {
        w[i] = p1(w[i - 16] ^ w[i - 9] ^ rol32(w[i - 3], 15))
            ^ rol32(w[i - 13], 7) ^ w[i - 6];
    }
    for (int i = 0; i < 64; ++i) {
        wp[i] = w[i] ^ w[i + 4];
    }

    for (int i = 0; i < 64; ++i) {
        uint32_t ff;
        uint32_t gg;
        uint32_t constant;
        if (i < 16) {
            ff = a ^ b ^ c;
            gg = e ^ f ^ g;
            constant = 0x79cc4519U;
        } else {
            ff = (a & b) | (a & c) | (b & c);
            gg = (e & f) | ((~e) & g);
            constant = 0x7a879d8aU;
        }
        uint32_t ss1 = rol32(rol32(a, 12) + e + rol32(constant, (unsigned int)i), 7);
        uint32_t ss2 = ss1 ^ rol32(a, 12);
        uint32_t tt1 = ff + d + ss2 + wp[i];
        uint32_t tt2 = gg + h + ss1 + w[i];
        d = c;
        c = rol32(b, 9);
        b = a;
        a = tt1;
        h = g;
        g = rol32(f, 19);
        f = e;
        e = p0(tt2);
    }

    state[0] ^= a;
    state[1] ^= b;
    state[2] ^= c;
    state[3] ^= d;
    state[4] ^= e;
    state[5] ^= f;
    state[6] ^= g;
    state[7] ^= h;
}

static void sm3_init(sm3_ctx *ctx) {
    static const uint32_t initial_state[8] = {
        0x7380166fU, 0x4914b2b9U, 0x172442d7U, 0xda8a0600U,
        0xa96f30bcU, 0x163138aaU, 0xe38dee4dU, 0xb0fb0e4eU
    };
    memcpy(ctx->state, initial_state, sizeof(initial_state));
    ctx->total_bytes = 0;
    ctx->buffered = 0;
}

static void sm3_update(sm3_ctx *ctx, const uint8_t *data, size_t length) {
    ctx->total_bytes += (uint64_t)length;
    if (ctx->buffered) {
        size_t needed = SM3_BLOCK_SIZE - ctx->buffered;
        size_t take = length < needed ? length : needed;
        memcpy(ctx->buffer + ctx->buffered, data, take);
        ctx->buffered += take;
        data += take;
        length -= take;
        if (ctx->buffered == SM3_BLOCK_SIZE) {
            sm3_compress(ctx->state, ctx->buffer);
            ctx->buffered = 0;
        }
    }
    while (length >= SM3_BLOCK_SIZE) {
        sm3_compress(ctx->state, data);
        data += SM3_BLOCK_SIZE;
        length -= SM3_BLOCK_SIZE;
    }
    if (length) {
        memcpy(ctx->buffer, data, length);
        ctx->buffered = length;
    }
}

static void sm3_finish(sm3_ctx *ctx, uint8_t digest[SM3_DIGEST_SIZE]) {
    uint64_t total_bits = ctx->total_bytes * 8U;
    ctx->buffer[ctx->buffered++] = 0x80U;
    if (ctx->buffered > 56) {
        memset(ctx->buffer + ctx->buffered, 0, SM3_BLOCK_SIZE - ctx->buffered);
        sm3_compress(ctx->state, ctx->buffer);
        ctx->buffered = 0;
    }
    memset(ctx->buffer + ctx->buffered, 0, 56 - ctx->buffered);
    store_be64(ctx->buffer + 56, total_bits);
    sm3_compress(ctx->state, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        store_be32(digest + i * 4, ctx->state[i]);
    }
}

static PyObject *py_sm3_hexdigest(PyObject *self, PyObject *object) {
    (void)self;
    Py_buffer view;
    uint8_t digest[SM3_DIGEST_SIZE];
    char hex[SM3_DIGEST_SIZE * 2 + 1];
    static const char digits[] = "0123456789abcdef";

    if (PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) != 0) {
        return NULL;
    }
    sm3_ctx ctx;
    Py_BEGIN_ALLOW_THREADS
    sm3_init(&ctx);
    sm3_update(&ctx, (const uint8_t *)view.buf, (size_t)view.len);
    sm3_finish(&ctx, digest);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);

    for (int i = 0; i < SM3_DIGEST_SIZE; ++i) {
        hex[i * 2] = digits[digest[i] >> 4];
        hex[i * 2 + 1] = digits[digest[i] & 0x0f];
    }
    hex[SM3_DIGEST_SIZE * 2] = '\0';
    return PyUnicode_FromStringAndSize(hex, SM3_DIGEST_SIZE * 2);
}

static PyMethodDef native_sm3_methods[] = {
    {"sm3_hexdigest", (PyCFunction)py_sm3_hexdigest, METH_O,
     "Return the SM3 hexadecimal digest of a bytes-like object."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_sm3_module = {
    PyModuleDef_HEAD_INIT,
    "_native_sm3",
    "Native SM3 accelerator for SM9RRSFL.",
    -1,
    native_sm3_methods
};

PyMODINIT_FUNC PyInit__native_sm3(void) {
    return PyModule_Create(&native_sm3_module);
}
