#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <gmssl/sm3.h>
#include <gmssl/sm9_z256.h>

/*
 * Minimal GmSSL SM9 group bridge for protocol v2.
 *
 * This module deliberately contains no ring-signature, tracing, D-KGC, or
 * federated-learning policy.  It exposes only canonical byte-oriented group
 * operations so that the Python protocol remains the single implementation of
 * the equations in the paper.
 *
 * Group orientation follows the SM9 standard and the Word scheme:
 *
 *   paper G1 / P1  <->  GmSSL SM9_Z256_POINT        (65-byte SEC1 point)
 *   paper G2 / P2  <->  GmSSL SM9_Z256_TWIST_POINT (129-byte point)
 *   paper GT       <->  GmSSL sm9_z256_fp12_t       (384-byte element)
 *
 * GmSSL's pairing function accepts (G2, G1), so py_sm9_pairing reverses the
 * two paper-order arguments internally.
 *
 * No point-at-infinity encoding is exposed.  Protocol values are required to
 * be non-identity points; a zero point-multiplication scalar or an addition
 * producing infinity is therefore rejected explicitly.  GT identity is a
 * valid element and is available through gt_one().
 *
 * The bundled GmSSL z256 scalar multiplication routines are not documented as
 * constant-time and contain secret-dependent branches/table access.  This
 * bridge is suitable for the project's local reproducible experiments, but it
 * is not a side-channel-hardened production cryptographic module.
 */

#define SM9_SCALAR_SIZE 32
#define SM9_G1_SIZE 65
#define SM9_G2_SIZE 129
#define SM9_GT_SIZE 384

typedef enum {
    ELEMENT_VALID = 1,
    ELEMENT_INVALID_ENCODING = 0,
    ELEMENT_INFINITY = -1,
    ELEMENT_NOT_ON_CURVE = -2,
    ELEMENT_NOT_IN_SUBGROUP = -3,
    ELEMENT_NON_CANONICAL = -4,
    ELEMENT_ZERO = -5
} element_status;

typedef enum {
    SCALAR_VALID = 1,
    SCALAR_OUT_OF_RANGE = 0,
    SCALAR_ZERO = -1
} scalar_status;

static void secure_clear(void *buffer, size_t length)
{
    volatile uint8_t *p = (volatile uint8_t *)buffer;
    while (length--) {
        *p++ = 0;
    }
}

static int constant_time_equal(
    const uint8_t *left,
    const uint8_t *right,
    size_t length
)
{
    uint8_t difference = 0;
    size_t i;
    for (i = 0; i < length; ++i) {
        difference |= (uint8_t)(left[i] ^ right[i]);
    }
    return difference == 0;
}

static int require_exact_bytes(
    PyObject *object,
    const char *name,
    Py_ssize_t expected_length,
    const uint8_t **output
)
{
    Py_ssize_t actual_length;

    if (!PyBytes_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be bytes", name);
        return 0;
    }
    actual_length = PyBytes_GET_SIZE(object);
    if (actual_length != expected_length) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be exactly %zd bytes (got %zd)",
            name,
            expected_length,
            actual_length
        );
        return 0;
    }
    *output = (const uint8_t *)PyBytes_AS_STRING(object);
    return 1;
}

static int require_bytes(
    PyObject *object,
    const char *name,
    const uint8_t **output,
    Py_ssize_t *output_length
)
{
    if (!PyBytes_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be bytes", name);
        return 0;
    }
    *output = (const uint8_t *)PyBytes_AS_STRING(object);
    *output_length = PyBytes_GET_SIZE(object);
    return 1;
}

static scalar_status scalar_from_canonical_bytes(
    sm9_z256_t scalar,
    const uint8_t input[SM9_SCALAR_SIZE],
    int allow_zero
)
{
    sm9_z256_from_bytes(scalar, input);
    if (sm9_z256_cmp(scalar, sm9_z256_order()) >= 0) {
        return SCALAR_OUT_OF_RANGE;
    }
    if (!allow_zero && sm9_z256_is_zero(scalar)) {
        return SCALAR_ZERO;
    }
    return SCALAR_VALID;
}

/*
 * GmSSL 3.x's public sm9_z256_fp12_pow() asserts that the exponent is less
 * than N-1.  Subgroup validation needs the exact exponent N, and a canonical
 * protocol scalar may also equal N-1.  Use the same square-and-multiply
 * operation without that implementation-specific precondition.
 */
static void fp12_pow_full_range(
    sm9_z256_fp12_t output,
    const sm9_z256_fp12_t input,
    const sm9_z256_t exponent
)
{
    sm9_z256_fp12_t accumulator;
    uint64_t word;
    int word_index;
    int bit_index;

    sm9_z256_fp12_set_one(accumulator);
    for (word_index = 3; word_index >= 0; --word_index) {
        word = exponent[word_index];
        for (bit_index = 0; bit_index < 64; ++bit_index) {
            sm9_z256_fp12_sqr(accumulator, accumulator);
            if (word & UINT64_C(0x8000000000000000)) {
                sm9_z256_fp12_mul(accumulator, accumulator, input);
            }
            word <<= 1;
        }
    }
    sm9_z256_fp12_copy(output, accumulator);
    secure_clear(accumulator, sizeof(accumulator));
}

static PyObject *raise_scalar_error(const char *name, scalar_status status)
{
    if (status == SCALAR_ZERO) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be a non-zero canonical scalar modulo the SM9 order",
            name
        );
    } else {
        PyErr_Format(
            PyExc_ValueError,
            "%s is not a canonical scalar below the SM9 order",
            name
        );
    }
    return NULL;
}

static PyObject *raise_element_error(const char *name, element_status status)
{
    const char *reason;

    switch (status) {
    case ELEMENT_INFINITY:
        reason = "is the point at infinity";
        break;
    case ELEMENT_NOT_ON_CURVE:
        reason = "is not on the required SM9 curve";
        break;
    case ELEMENT_NOT_IN_SUBGROUP:
        reason = "is not in the prime-order SM9 subgroup";
        break;
    case ELEMENT_NON_CANONICAL:
        reason = "does not use the canonical uncompressed encoding";
        break;
    case ELEMENT_ZERO:
        reason = "is the zero field element";
        break;
    case ELEMENT_INVALID_ENCODING:
    default:
        reason = "has an invalid SM9 encoding";
        break;
    }
    PyErr_Format(PyExc_ValueError, "%s %s", name, reason);
    return NULL;
}

static element_status decode_g1_checked(
    SM9_Z256_POINT *point,
    const uint8_t input[SM9_G1_SIZE]
)
{
    SM9_Z256_POINT subgroup_check;
    uint8_t canonical[SM9_G1_SIZE];
    element_status result = ELEMENT_VALID;

    if (sm9_z256_point_from_uncompressed_octets(point, input) != 1) {
        result = ELEMENT_INVALID_ENCODING;
        goto end;
    }
    if (sm9_z256_point_is_at_infinity(point)) {
        result = ELEMENT_INFINITY;
        goto end;
    }
    if (sm9_z256_point_is_on_curve(point) != 1) {
        result = ELEMENT_NOT_ON_CURVE;
        goto end;
    }
    if (sm9_z256_point_to_uncompressed_octets(point, canonical) != 1
        || !constant_time_equal(input, canonical, sizeof(canonical))) {
        result = ELEMENT_NON_CANONICAL;
        goto end;
    }

    /* Reject on-curve points outside the subgroup generated by P1. */
    sm9_z256_point_mul(&subgroup_check, sm9_z256_order(), point);
    if (!sm9_z256_point_is_at_infinity(&subgroup_check)) {
        result = ELEMENT_NOT_IN_SUBGROUP;
    }

end:
    secure_clear(&subgroup_check, sizeof(subgroup_check));
    secure_clear(canonical, sizeof(canonical));
    return result;
}

static element_status decode_g2_checked(
    SM9_Z256_TWIST_POINT *point,
    const uint8_t input[SM9_G2_SIZE]
)
{
    SM9_Z256_TWIST_POINT subgroup_check;
    uint8_t canonical[SM9_G2_SIZE];
    element_status result = ELEMENT_VALID;

    if (sm9_z256_twist_point_from_uncompressed_octets(point, input) != 1) {
        result = ELEMENT_INVALID_ENCODING;
        goto end;
    }
    if (sm9_z256_twist_point_is_at_infinity(point)) {
        result = ELEMENT_INFINITY;
        goto end;
    }
    if (sm9_z256_twist_point_is_on_curve(point) != 1) {
        result = ELEMENT_NOT_ON_CURVE;
        goto end;
    }
    if (sm9_z256_twist_point_to_uncompressed_octets(point, canonical) != 1
        || !constant_time_equal(input, canonical, sizeof(canonical))) {
        result = ELEMENT_NON_CANONICAL;
        goto end;
    }

    /* The twist has a cofactor, so this subgroup check is security-critical. */
    sm9_z256_twist_point_mul(&subgroup_check, sm9_z256_order(), point);
    if (!sm9_z256_twist_point_is_at_infinity(&subgroup_check)) {
        result = ELEMENT_NOT_IN_SUBGROUP;
    }

end:
    secure_clear(&subgroup_check, sizeof(subgroup_check));
    secure_clear(canonical, sizeof(canonical));
    return result;
}

static element_status decode_gt_checked(
    sm9_z256_fp12_t value,
    const uint8_t input[SM9_GT_SIZE]
)
{
    sm9_z256_fp12_t zero;
    sm9_z256_fp12_t one;
    sm9_z256_fp12_t subgroup_check;
    uint8_t canonical[SM9_GT_SIZE];
    element_status result = ELEMENT_VALID;

    if (sm9_z256_fp12_from_bytes(value, input) != 1) {
        result = ELEMENT_INVALID_ENCODING;
        goto end;
    }
    sm9_z256_fp12_to_bytes(value, canonical);
    if (!constant_time_equal(input, canonical, sizeof(canonical))) {
        result = ELEMENT_NON_CANONICAL;
        goto end;
    }

    sm9_z256_fp12_set_zero(zero);
    if (sm9_z256_fp12_equ(value, zero)) {
        result = ELEMENT_ZERO;
        goto end;
    }

    /* A protocol GT value must lie in the order-N pairing subgroup. */
    fp12_pow_full_range(subgroup_check, value, sm9_z256_order());
    sm9_z256_fp12_set_one(one);
    if (!sm9_z256_fp12_equ(subgroup_check, one)) {
        result = ELEMENT_NOT_IN_SUBGROUP;
        goto end;
    }

end:
    secure_clear(zero, sizeof(zero));
    secure_clear(one, sizeof(one));
    secure_clear(subgroup_check, sizeof(subgroup_check));
    secure_clear(canonical, sizeof(canonical));
    return result;
}

static void hash_v_to_scalar(
    sm9_z256_t output,
    uint8_t prefix,
    const uint8_t *transcript,
    size_t transcript_length
)
{
    static const uint8_t counter_1[4] = {0, 0, 0, 1};
    static const uint8_t counter_2[4] = {0, 0, 0, 2};
    uint8_t expanded[64];
    SM3_CTX first;
    SM3_CTX second;

    sm3_init(&first);
    sm3_update(&first, &prefix, 1);
    if (transcript_length != 0) {
        sm3_update(&first, transcript, transcript_length);
    }
    second = first;

    sm3_update(&first, counter_1, sizeof(counter_1));
    sm3_finish(&first, expanded);
    sm3_update(&second, counter_2, sizeof(counter_2));
    sm3_finish(&second, expanded + 32);

    /* SM9's 256-bit order requires the first 40 bytes of H_v. */
    sm9_z256_modn_from_hash(output, expanded);

    secure_clear(expanded, sizeof(expanded));
    secure_clear(&first, sizeof(first));
    secure_clear(&second, sizeof(second));
}

static PyObject *py_sm9_order(PyObject *self, PyObject *unused)
{
    uint8_t output[SM9_SCALAR_SIZE];
    PyObject *result;
    (void)self;
    (void)unused;

    sm9_z256_to_bytes(sm9_z256_order(), output);
    result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_hash_to_scalar(PyObject *self, PyObject *args)
{
    int prefix;
    PyObject *transcript_object;
    const uint8_t *transcript;
    Py_ssize_t transcript_length;
    sm9_z256_t scalar;
    uint8_t output[SM9_SCALAR_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(
            args,
            "iO:hash_to_scalar",
            &prefix,
            &transcript_object
        )) {
        return NULL;
    }
    if (prefix != 1 && prefix != 2) {
        PyErr_SetString(PyExc_ValueError, "SM9 H_v prefix must be 1 or 2");
        return NULL;
    }
    if (!require_bytes(
            transcript_object,
            "transcript",
            &transcript,
            &transcript_length
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    hash_v_to_scalar(
        scalar,
        (uint8_t)prefix,
        transcript,
        (size_t)transcript_length
    );
    sm9_z256_to_bytes(scalar, output);
    Py_END_ALLOW_THREADS

    result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    secure_clear(scalar, sizeof(scalar));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g1_generator(PyObject *self, PyObject *unused)
{
    uint8_t output[SM9_G1_SIZE];
    PyObject *result;
    (void)self;
    (void)unused;

    if (sm9_z256_point_to_uncompressed_octets(
            sm9_z256_generator(),
            output
        ) != 1) {
        PyErr_SetString(PyExc_RuntimeError, "GmSSL failed to encode the SM9 G1 generator");
        return NULL;
    }
    result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g2_generator(PyObject *self, PyObject *unused)
{
    uint8_t output[SM9_G2_SIZE];
    PyObject *result;
    (void)self;
    (void)unused;

    if (sm9_z256_twist_point_to_uncompressed_octets(
            sm9_z256_twist_generator(),
            output
        ) != 1) {
        PyErr_SetString(PyExc_RuntimeError, "GmSSL failed to encode the SM9 G2 generator");
        return NULL;
    }
    result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g1_validate(PyObject *self, PyObject *object)
{
    const uint8_t *input;
    SM9_Z256_POINT point;
    element_status status;
    (void)self;

    if (!require_exact_bytes(object, "g1", SM9_G1_SIZE, &input)) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    status = decode_g1_checked(&point, input);
    Py_END_ALLOW_THREADS
    secure_clear(&point, sizeof(point));

    if (status == ELEMENT_VALID) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *py_sm9_g2_validate(PyObject *self, PyObject *object)
{
    const uint8_t *input;
    SM9_Z256_TWIST_POINT point;
    element_status status;
    (void)self;

    if (!require_exact_bytes(object, "g2", SM9_G2_SIZE, &input)) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    status = decode_g2_checked(&point, input);
    Py_END_ALLOW_THREADS
    secure_clear(&point, sizeof(point));

    if (status == ELEMENT_VALID) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *py_sm9_gt_validate(PyObject *self, PyObject *object)
{
    const uint8_t *input;
    sm9_z256_fp12_t value;
    element_status status;
    (void)self;

    if (!require_exact_bytes(object, "gt", SM9_GT_SIZE, &input)) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    status = decode_gt_checked(value, input);
    Py_END_ALLOW_THREADS
    secure_clear(value, sizeof(value));

    if (status == ELEMENT_VALID) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *py_sm9_g1_mul(PyObject *self, PyObject *args)
{
    PyObject *point_object;
    PyObject *scalar_object;
    const uint8_t *point_bytes;
    const uint8_t *scalar_bytes;
    SM9_Z256_POINT point;
    SM9_Z256_POINT product;
    sm9_z256_t scalar;
    element_status point_status;
    scalar_status scalar_result;
    int output_ok = 0;
    uint8_t output[SM9_G1_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:g1_mul", &point_object, &scalar_object)) {
        return NULL;
    }
    if (!require_exact_bytes(point_object, "g1", SM9_G1_SIZE, &point_bytes)
        || !require_exact_bytes(
            scalar_object,
            "scalar",
            SM9_SCALAR_SIZE,
            &scalar_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    point_status = decode_g1_checked(&point, point_bytes);
    scalar_result = scalar_from_canonical_bytes(scalar, scalar_bytes, 0);
    if (point_status == ELEMENT_VALID && scalar_result == SCALAR_VALID) {
        sm9_z256_point_mul(&product, scalar, &point);
        if (!sm9_z256_point_is_at_infinity(&product)) {
            output_ok = sm9_z256_point_to_uncompressed_octets(
                &product,
                output
            ) == 1;
        }
    }
    Py_END_ALLOW_THREADS

    if (point_status != ELEMENT_VALID) {
        result = raise_element_error("g1", point_status);
    } else if (scalar_result != SCALAR_VALID) {
        result = raise_scalar_error("scalar", scalar_result);
    } else if (!output_ok) {
        PyErr_SetString(PyExc_RuntimeError, "G1 multiplication produced an invalid result");
        result = NULL;
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(&point, sizeof(point));
    secure_clear(&product, sizeof(product));
    secure_clear(scalar, sizeof(scalar));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g2_mul(PyObject *self, PyObject *args)
{
    PyObject *point_object;
    PyObject *scalar_object;
    const uint8_t *point_bytes;
    const uint8_t *scalar_bytes;
    SM9_Z256_TWIST_POINT point;
    SM9_Z256_TWIST_POINT product;
    sm9_z256_t scalar;
    element_status point_status;
    scalar_status scalar_result;
    int output_ok = 0;
    uint8_t output[SM9_G2_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:g2_mul", &point_object, &scalar_object)) {
        return NULL;
    }
    if (!require_exact_bytes(point_object, "g2", SM9_G2_SIZE, &point_bytes)
        || !require_exact_bytes(
            scalar_object,
            "scalar",
            SM9_SCALAR_SIZE,
            &scalar_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    point_status = decode_g2_checked(&point, point_bytes);
    scalar_result = scalar_from_canonical_bytes(scalar, scalar_bytes, 0);
    if (point_status == ELEMENT_VALID && scalar_result == SCALAR_VALID) {
        sm9_z256_twist_point_mul(&product, scalar, &point);
        if (!sm9_z256_twist_point_is_at_infinity(&product)) {
            output_ok = sm9_z256_twist_point_to_uncompressed_octets(
                &product,
                output
            ) == 1;
        }
    }
    Py_END_ALLOW_THREADS

    if (point_status != ELEMENT_VALID) {
        result = raise_element_error("g2", point_status);
    } else if (scalar_result != SCALAR_VALID) {
        result = raise_scalar_error("scalar", scalar_result);
    } else if (!output_ok) {
        PyErr_SetString(PyExc_RuntimeError, "G2 multiplication produced an invalid result");
        result = NULL;
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(&point, sizeof(point));
    secure_clear(&product, sizeof(product));
    secure_clear(scalar, sizeof(scalar));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g1_add(PyObject *self, PyObject *args)
{
    PyObject *left_object;
    PyObject *right_object;
    const uint8_t *left_bytes;
    const uint8_t *right_bytes;
    SM9_Z256_POINT left;
    SM9_Z256_POINT right;
    SM9_Z256_POINT sum;
    element_status left_status;
    element_status right_status = ELEMENT_INVALID_ENCODING;
    int result_infinity = 0;
    int output_ok = 0;
    uint8_t output[SM9_G1_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:g1_add", &left_object, &right_object)) {
        return NULL;
    }
    if (!require_exact_bytes(left_object, "left_g1", SM9_G1_SIZE, &left_bytes)
        || !require_exact_bytes(
            right_object,
            "right_g1",
            SM9_G1_SIZE,
            &right_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    left_status = decode_g1_checked(&left, left_bytes);
    if (left_status == ELEMENT_VALID) {
        right_status = decode_g1_checked(&right, right_bytes);
    }
    if (left_status == ELEMENT_VALID && right_status == ELEMENT_VALID) {
        sm9_z256_point_add(&sum, &left, &right);
        result_infinity = sm9_z256_point_is_at_infinity(&sum);
        if (!result_infinity) {
            output_ok = sm9_z256_point_to_uncompressed_octets(&sum, output) == 1;
        }
    }
    Py_END_ALLOW_THREADS

    if (left_status != ELEMENT_VALID) {
        result = raise_element_error("left_g1", left_status);
    } else if (right_status != ELEMENT_VALID) {
        result = raise_element_error("right_g1", right_status);
    } else if (result_infinity) {
        PyErr_SetString(PyExc_ValueError, "G1 addition produced the point at infinity");
        result = NULL;
    } else if (!output_ok) {
        PyErr_SetString(PyExc_RuntimeError, "GmSSL failed to encode the G1 sum");
        result = NULL;
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(&left, sizeof(left));
    secure_clear(&right, sizeof(right));
    secure_clear(&sum, sizeof(sum));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_g2_add(PyObject *self, PyObject *args)
{
    PyObject *left_object;
    PyObject *right_object;
    const uint8_t *left_bytes;
    const uint8_t *right_bytes;
    SM9_Z256_TWIST_POINT left;
    SM9_Z256_TWIST_POINT right;
    SM9_Z256_TWIST_POINT sum;
    element_status left_status;
    element_status right_status = ELEMENT_INVALID_ENCODING;
    int result_infinity = 0;
    int output_ok = 0;
    uint8_t output[SM9_G2_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:g2_add", &left_object, &right_object)) {
        return NULL;
    }
    if (!require_exact_bytes(left_object, "left_g2", SM9_G2_SIZE, &left_bytes)
        || !require_exact_bytes(
            right_object,
            "right_g2",
            SM9_G2_SIZE,
            &right_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    left_status = decode_g2_checked(&left, left_bytes);
    if (left_status == ELEMENT_VALID) {
        right_status = decode_g2_checked(&right, right_bytes);
    }
    if (left_status == ELEMENT_VALID && right_status == ELEMENT_VALID) {
        sm9_z256_twist_point_add_full(&sum, &left, &right);
        result_infinity = sm9_z256_twist_point_is_at_infinity(&sum);
        if (!result_infinity) {
            output_ok = sm9_z256_twist_point_to_uncompressed_octets(
                &sum,
                output
            ) == 1;
        }
    }
    Py_END_ALLOW_THREADS

    if (left_status != ELEMENT_VALID) {
        result = raise_element_error("left_g2", left_status);
    } else if (right_status != ELEMENT_VALID) {
        result = raise_element_error("right_g2", right_status);
    } else if (result_infinity) {
        PyErr_SetString(PyExc_ValueError, "G2 addition produced the point at infinity");
        result = NULL;
    } else if (!output_ok) {
        PyErr_SetString(PyExc_RuntimeError, "GmSSL failed to encode the G2 sum");
        result = NULL;
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(&left, sizeof(left));
    secure_clear(&right, sizeof(right));
    secure_clear(&sum, sizeof(sum));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_pairing(PyObject *self, PyObject *args)
{
    PyObject *g1_object;
    PyObject *g2_object;
    const uint8_t *g1_bytes;
    const uint8_t *g2_bytes;
    SM9_Z256_POINT g1;
    SM9_Z256_TWIST_POINT g2;
    sm9_z256_fp12_t pairing_value;
    element_status g1_status;
    element_status g2_status = ELEMENT_INVALID_ENCODING;
    uint8_t output[SM9_GT_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:pairing", &g1_object, &g2_object)) {
        return NULL;
    }
    if (!require_exact_bytes(g1_object, "g1", SM9_G1_SIZE, &g1_bytes)
        || !require_exact_bytes(g2_object, "g2", SM9_G2_SIZE, &g2_bytes)) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    g1_status = decode_g1_checked(&g1, g1_bytes);
    if (g1_status == ELEMENT_VALID) {
        g2_status = decode_g2_checked(&g2, g2_bytes);
    }
    if (g1_status == ELEMENT_VALID && g2_status == ELEMENT_VALID) {
        /* GmSSL takes (G2, G1); this API intentionally takes paper (G1, G2). */
        sm9_z256_pairing(pairing_value, &g2, &g1);
        sm9_z256_fp12_to_bytes(pairing_value, output);
    }
    Py_END_ALLOW_THREADS

    if (g1_status != ELEMENT_VALID) {
        result = raise_element_error("g1", g1_status);
    } else if (g2_status != ELEMENT_VALID) {
        result = raise_element_error("g2", g2_status);
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(&g1, sizeof(g1));
    secure_clear(&g2, sizeof(g2));
    secure_clear(pairing_value, sizeof(pairing_value));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_gt_one(PyObject *self, PyObject *unused)
{
    sm9_z256_fp12_t one;
    uint8_t output[SM9_GT_SIZE];
    PyObject *result;
    (void)self;
    (void)unused;

    sm9_z256_fp12_set_one(one);
    sm9_z256_fp12_to_bytes(one, output);
    result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    secure_clear(one, sizeof(one));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_gt_mul(PyObject *self, PyObject *args)
{
    PyObject *left_object;
    PyObject *right_object;
    const uint8_t *left_bytes;
    const uint8_t *right_bytes;
    sm9_z256_fp12_t left;
    sm9_z256_fp12_t right;
    sm9_z256_fp12_t product;
    element_status left_status;
    element_status right_status = ELEMENT_INVALID_ENCODING;
    uint8_t output[SM9_GT_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:gt_mul", &left_object, &right_object)) {
        return NULL;
    }
    if (!require_exact_bytes(left_object, "left_gt", SM9_GT_SIZE, &left_bytes)
        || !require_exact_bytes(
            right_object,
            "right_gt",
            SM9_GT_SIZE,
            &right_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    left_status = decode_gt_checked(left, left_bytes);
    if (left_status == ELEMENT_VALID) {
        right_status = decode_gt_checked(right, right_bytes);
    }
    if (left_status == ELEMENT_VALID && right_status == ELEMENT_VALID) {
        sm9_z256_fp12_mul(product, left, right);
        sm9_z256_fp12_to_bytes(product, output);
    }
    Py_END_ALLOW_THREADS

    if (left_status != ELEMENT_VALID) {
        result = raise_element_error("left_gt", left_status);
    } else if (right_status != ELEMENT_VALID) {
        result = raise_element_error("right_gt", right_status);
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(left, sizeof(left));
    secure_clear(right, sizeof(right));
    secure_clear(product, sizeof(product));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_gt_pow(PyObject *self, PyObject *args)
{
    PyObject *value_object;
    PyObject *scalar_object;
    const uint8_t *value_bytes;
    const uint8_t *scalar_bytes;
    sm9_z256_fp12_t value;
    sm9_z256_fp12_t power;
    sm9_z256_t scalar;
    element_status value_status;
    scalar_status scalar_result;
    uint8_t output[SM9_GT_SIZE];
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:gt_pow", &value_object, &scalar_object)) {
        return NULL;
    }
    if (!require_exact_bytes(value_object, "gt", SM9_GT_SIZE, &value_bytes)
        || !require_exact_bytes(
            scalar_object,
            "scalar",
            SM9_SCALAR_SIZE,
            &scalar_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    value_status = decode_gt_checked(value, value_bytes);
    scalar_result = scalar_from_canonical_bytes(scalar, scalar_bytes, 1);
    if (value_status == ELEMENT_VALID && scalar_result == SCALAR_VALID) {
        fp12_pow_full_range(power, value, scalar);
        sm9_z256_fp12_to_bytes(power, output);
    }
    Py_END_ALLOW_THREADS

    if (value_status != ELEMENT_VALID) {
        result = raise_element_error("gt", value_status);
    } else if (scalar_result != SCALAR_VALID) {
        result = raise_scalar_error("scalar", scalar_result);
    } else {
        result = PyBytes_FromStringAndSize((const char *)output, sizeof(output));
    }

    secure_clear(value, sizeof(value));
    secure_clear(power, sizeof(power));
    secure_clear(scalar, sizeof(scalar));
    secure_clear(output, sizeof(output));
    return result;
}

static PyObject *py_sm9_gt_equal(PyObject *self, PyObject *args)
{
    PyObject *left_object;
    PyObject *right_object;
    const uint8_t *left_bytes;
    const uint8_t *right_bytes;
    sm9_z256_fp12_t left;
    sm9_z256_fp12_t right;
    element_status left_status;
    element_status right_status = ELEMENT_INVALID_ENCODING;
    int equal = 0;
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO:gt_equal", &left_object, &right_object)) {
        return NULL;
    }
    if (!require_exact_bytes(left_object, "left_gt", SM9_GT_SIZE, &left_bytes)
        || !require_exact_bytes(
            right_object,
            "right_gt",
            SM9_GT_SIZE,
            &right_bytes
        )) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    left_status = decode_gt_checked(left, left_bytes);
    if (left_status == ELEMENT_VALID) {
        right_status = decode_gt_checked(right, right_bytes);
    }
    if (left_status == ELEMENT_VALID && right_status == ELEMENT_VALID) {
        equal = sm9_z256_fp12_equ(left, right);
    }
    Py_END_ALLOW_THREADS

    if (left_status != ELEMENT_VALID) {
        result = raise_element_error("left_gt", left_status);
    } else if (right_status != ELEMENT_VALID) {
        result = raise_element_error("right_gt", right_status);
    } else if (equal) {
        Py_INCREF(Py_True);
        result = Py_True;
    } else {
        Py_INCREF(Py_False);
        result = Py_False;
    }

    secure_clear(left, sizeof(left));
    secure_clear(right, sizeof(right));
    return result;
}

PyDoc_STRVAR(
    module_doc,
    "Canonical byte-oriented GmSSL SM9 group primitives for SM9-RRS-FL v2.\n"
    "\n"
    "G1 elements are 65-byte SM9 P1 points, G2 elements are 129-byte SM9\n"
    "P2 twist points, GT elements are 384 bytes, and scalars are canonical\n"
    "32-byte big-endian integers below the SM9 group order. This research\n"
    "bridge inherits GmSSL z256 timing behavior and is not side-channel hardened."
);

static PyMethodDef native_sm9_methods[] = {
    {
        "order",
        py_sm9_order,
        METH_NOARGS,
        "Return the SM9 group order as 32 canonical big-endian bytes."
    },
    {
        "hash_to_scalar",
        py_sm9_hash_to_scalar,
        METH_VARARGS,
        "hash_to_scalar(prefix, transcript) -> 32-byte SM9 H_v scalar."
    },
    {
        "g1_generator",
        py_sm9_g1_generator,
        METH_NOARGS,
        "Return the canonical 65-byte encoding of SM9 P1 in G1."
    },
    {
        "g2_generator",
        py_sm9_g2_generator,
        METH_NOARGS,
        "Return the canonical 129-byte encoding of SM9 P2 in G2."
    },
    {
        "g1_validate",
        py_sm9_g1_validate,
        METH_O,
        "Validate canonical encoding, curve membership, and subgroup membership in G1."
    },
    {
        "g2_validate",
        py_sm9_g2_validate,
        METH_O,
        "Validate canonical encoding, curve membership, and subgroup membership in G2."
    },
    {
        "gt_validate",
        py_sm9_gt_validate,
        METH_O,
        "Validate canonical encoding and order-N subgroup membership in GT."
    },
    {
        "g1_mul",
        py_sm9_g1_mul,
        METH_VARARGS,
        "g1_mul(g1, scalar) -> canonical non-identity G1 bytes."
    },
    {
        "g2_mul",
        py_sm9_g2_mul,
        METH_VARARGS,
        "g2_mul(g2, scalar) -> canonical non-identity G2 bytes."
    },
    {
        "g1_add",
        py_sm9_g1_add,
        METH_VARARGS,
        "g1_add(left, right) -> canonical non-identity G1 bytes."
    },
    {
        "g2_add",
        py_sm9_g2_add,
        METH_VARARGS,
        "g2_add(left, right) -> canonical non-identity G2 bytes."
    },
    {
        "pairing",
        py_sm9_pairing,
        METH_VARARGS,
        "pairing(g1, g2) -> canonical 384-byte SM9 target-group element."
    },
    {
        "gt_one",
        py_sm9_gt_one,
        METH_NOARGS,
        "Return the canonical 384-byte GT identity."
    },
    {
        "gt_mul",
        py_sm9_gt_mul,
        METH_VARARGS,
        "gt_mul(left, right) -> canonical GT product."
    },
    {
        "gt_pow",
        py_sm9_gt_pow,
        METH_VARARGS,
        "gt_pow(gt, scalar) -> canonical GT power; scalar zero is allowed."
    },
    {
        "gt_equal",
        py_sm9_gt_equal,
        METH_VARARGS,
        "Compare two fully validated canonical GT elements."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_sm9_module = {
    PyModuleDef_HEAD_INIT,
    "_native_sm9",
    module_doc,
    -1,
    native_sm9_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__native_sm9(void)
{
    PyObject *module = PyModule_Create(&native_sm9_module);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "ABI_VERSION", 1) < 0
        || PyModule_AddIntConstant(module, "SCALAR_SIZE", SM9_SCALAR_SIZE) < 0
        || PyModule_AddIntConstant(module, "G1_SIZE", SM9_G1_SIZE) < 0
        || PyModule_AddIntConstant(module, "G2_SIZE", SM9_G2_SIZE) < 0
        || PyModule_AddIntConstant(module, "GT_SIZE", SM9_GT_SIZE) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
