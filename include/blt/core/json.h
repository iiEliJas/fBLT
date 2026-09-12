#ifndef BLT_CORE_JSON_H
#define BLT_CORE_JSON_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "blt/core/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Minimal JSON parser (configs/*.json for the ablation sweeps).
//
// Supports: objects (nested), arrays of scalars, string / int / float /
// bool / null values. All allocation comes from the caller-passed arena;
// nothing is malloc'd here. Malformed input is fatal (BLT_FATAL) with the
// byte offset included in the message.
//
// Numbers containing '.', 'e' or 'E' parse as FLOAT, otherwise INT.
// Accessors treat INT and FLOAT interchangeably where a lossless conversion
// exists (get_float accepts an INT value, get_int truncates a FLOAT value).
// ==========================================================================

typedef enum {
    BLT_JSON_NULL = 0,
    BLT_JSON_BOOL,
    BLT_JSON_INT,
    BLT_JSON_FLOAT,
    BLT_JSON_STRING,
    BLT_JSON_ARRAY,
    BLT_JSON_OBJECT
} blt_json_type;

typedef struct blt_json_value {
    blt_json_type type;
    bool bool_val;    // BLT_JSON_BOOL
    int64_t int_val;  // BLT_JSON_INT
    double float_val; // BLT_JSON_FLOAT
    char *str_val;    // BLT_JSON_STRING (arena-owned, NUL-terminated)
    size_t str_len;   // BLT_JSON_STRING length in bytes
    // BLT_JSON_ARRAY / BLT_JSON_OBJECT children (arena-owned)
    struct blt_json_value **children;
    char **keys; // parallel to children for OBJECT, NULL for ARRAY
    size_t num_children;
} blt_json_value;

// Parse a NUL-terminated JSON document into the arena. Fatal on malformed input.
blt_json_value *blt_json_parse(blt_arena *arena, const char *text);

// Parse a JSON file (whole file is read, then parsed as above). Fatal on
// unreadable file or malformed input.
blt_json_value *blt_json_parse_file(blt_arena *arena, const char *path);

// Object lookup: returns NULL when the key is missing or value is not an object.
const blt_json_value *blt_json_get(const blt_json_value *obj, const char *key);

// Typed accessors with fallbacks; never fatal. They accept the corresponding
// compatible types (e.g. get_int on a FLOAT truncates toward zero).
bool blt_json_get_bool(const blt_json_value *obj, const char *key, bool fallback);
int64_t blt_json_get_int(const blt_json_value *obj, const char *key, int64_t fallback);
double blt_json_get_float(const blt_json_value *obj, const char *key, double fallback);
const char *blt_json_get_string(const blt_json_value *obj, const char *key, const char *fallback);

// Array helpers. blt_json_array_size returns 0 for non-array values.
size_t blt_json_array_size(const blt_json_value *arr);
const blt_json_value *blt_json_array_at(const blt_json_value *arr, size_t i);

// Scalar readers for a value already known to be present; compatible-type
// conversions as above. Fatal if the type cannot be converted.
bool blt_json_as_bool(const blt_json_value *v);
int64_t blt_json_as_int(const blt_json_value *v);
double blt_json_as_float(const blt_json_value *v);
const char *blt_json_as_string(const blt_json_value *v);

#ifdef __cplusplus
}
#endif

#endif // BLT_CORE_JSON_H
