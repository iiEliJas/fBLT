#include "blt/core/json.h"
#include "blt/core/backend.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    blt_arena *arena;
    const char *text;
    size_t pos;
    size_t len;
} json_parser;

static void skip_ws(json_parser *p) {
    while (p->pos < p->len) {
        char c = p->text[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static void parse_value(json_parser *p, blt_json_value *out);

static blt_json_value *alloc_value(blt_arena *arena) {
    blt_json_value *v = (blt_json_value *)blt_arena_alloc(arena, sizeof(blt_json_value), sizeof(void *));
    memset(v, 0, sizeof(*v));
    v->type = BLT_JSON_NULL;
    return v;
}

// Kept as a separate function so the fatal message format lives in one place.
static void fail(const json_parser *p, const char *msg) { BLT_FATAL("json: %s at byte offset %zu", msg, p->pos); }

static void expect(json_parser *p, char c) {
    skip_ws(p);
    if (p->pos >= p->len || p->text[p->pos] != c) {
        BLT_FATAL("json: expected '%c' at byte offset %zu", c, p->pos);
    }
    p->pos++;
}

// Strings — \uXXXX decoded to UTF-8
static void append_utf8(blt_arena *arena, char **buf, size_t *len, size_t *cap, uint32_t cp) {
    char tmp[4];
    size_t n = 0;
    if (cp < 0x80) {
        tmp[n++] = (char)cp;
    } else if (cp < 0x800) {
        tmp[n++] = (char)(0xC0 | (cp >> 6));
        tmp[n++] = (char)(0x80 | (cp & 0x3F));
    } else {
        tmp[n++] = (char)(0xE0 | (cp >> 12));
        tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[n++] = (char)(0x80 | (cp & 0x3F));
    }
    if (*len + n + 1 > *cap) {
        size_t new_cap = (*cap == 0) ? 16 : *cap * 2;
        char *nb = (char *)blt_arena_alloc(arena, new_cap, 1);
        if (*buf) memcpy(nb, *buf, *len);
        *buf = nb;
        *cap = new_cap;
    }
    memcpy(*buf + *len, tmp, n);
    *len += n;
}

static uint32_t parse_hex4(json_parser *p) {
    uint32_t val = 0;
    for (int i = 0; i < 4; i++) {
        if (p->pos >= p->len) {
            fail(p, "unterminated \\u escape");
        }
        char c = p->text[p->pos++];
        val <<= 4;
        if (c >= '0' && c <= '9') {
            val |= (uint32_t)(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            val |= (uint32_t)(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            val |= (uint32_t)(c - 'A' + 10);
        } else {
            fail(p, "bad hex digit in \\u escape");
        }
    }
    return val;
}

static char *parse_string_raw(json_parser *p, size_t *out_len) {
    skip_ws(p);
    if (p->pos >= p->len || p->text[p->pos] != '"') {
        fail(p, "expected string");
    }
    p->pos++;

    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;

    while (p->pos < p->len) {
        unsigned char c = (unsigned char)p->text[p->pos++];
        if (c == '"') {
            if (len + 1 > cap) {
                size_t new_cap = (cap == 0) ? 16 : cap * 2;
                char *nb = (char *)blt_arena_alloc(p->arena, new_cap, 1);
                if (buf) memcpy(nb, buf, len);
                buf = nb;
                cap = new_cap;
            }
            buf[len] = '\0';
            *out_len = len;
            return buf;
        } else if (c == '\\') {
            if (p->pos >= p->len) {
                fail(p, "unterminated escape");
            }
            char e = p->text[p->pos++];
            switch (e) {
            case '"':
                append_utf8(p->arena, &buf, &len, &cap, '"');
                break;
            case '\\':
                append_utf8(p->arena, &buf, &len, &cap, '\\');
                break;
            case '/':
                append_utf8(p->arena, &buf, &len, &cap, '/');
                break;
            case 'b':
                append_utf8(p->arena, &buf, &len, &cap, '\b');
                break;
            case 'f':
                append_utf8(p->arena, &buf, &len, &cap, '\f');
                break;
            case 'n':
                append_utf8(p->arena, &buf, &len, &cap, '\n');
                break;
            case 'r':
                append_utf8(p->arena, &buf, &len, &cap, '\r');
                break;
            case 't':
                append_utf8(p->arena, &buf, &len, &cap, '\t');
                break;
            case 'u': {
                uint32_t cp = parse_hex4(p);
                // Surrogate pairs: decode to a single codepoint.
                if (cp >= 0xD800 && cp <= 0xDBFF && p->pos + 1 < p->len && p->text[p->pos] == '\\' &&
                    p->text[p->pos + 1] == 'u') {
                    p->pos += 2;
                    uint32_t lo = parse_hex4(p);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else {
                        cp = 0xFFFD;
                    }
                } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                if (cp < 0x80) {
                    append_utf8(p->arena, &buf, &len, &cap, cp);
                } else if (cp < 0x800) {
                    append_utf8(p->arena, &buf, &len, &cap, cp);
                } else if (cp < 0x10000) {
                    append_utf8(p->arena, &buf, &len, &cap, cp);
                } else {
                    char tmp[4];
                    tmp[0] = (char)(0xF0 | (cp >> 18));
                    tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                    tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    tmp[3] = (char)(0x80 | (cp & 0x3F));
                    if (len + 4 + 1 > cap) {
                        size_t new_cap = (cap == 0) ? 16 : cap * 2;
                        char *nb = (char *)blt_arena_alloc(p->arena, new_cap, 1);
                        if (buf) memcpy(nb, buf, len);
                        buf = nb;
                        cap = new_cap;
                    }
                    memcpy(buf + len, tmp, 4);
                    len += 4;
                }
                break;
            }
            default:
                fail(p, "unknown escape");
            }
        } else {
            append_utf8(p->arena, &buf, &len, &cap, c);
        }
    }
    fail(p, "unterminated string");
    return NULL;
}

// Numbers / literals

static bool is_number_start(char c) { return c == '-' || (c >= '0' && c <= '9'); }

static void parse_number(json_parser *p, blt_json_value *out) {
    size_t start = p->pos;
    if (p->pos < p->len && p->text[p->pos] == '-') p->pos++;
    while (p->pos < p->len && isdigit((unsigned char)p->text[p->pos])) p->pos++;
    bool is_float = false;
    if (p->pos < p->len && p->text[p->pos] == '.') {
        is_float = true;
        p->pos++;
        while (p->pos < p->len && isdigit((unsigned char)p->text[p->pos])) p->pos++;
    }
    if (p->pos < p->len && (p->text[p->pos] == 'e' || p->text[p->pos] == 'E')) {
        is_float = true;
        p->pos++;
        if (p->pos < p->len && (p->text[p->pos] == '+' || p->text[p->pos] == '-')) p->pos++;
        while (p->pos < p->len && isdigit((unsigned char)p->text[p->pos])) p->pos++;
    }
    char tmp[64];
    size_t n = p->pos - start;
    if (n == 0 || n >= sizeof(tmp)) {
        fail(p, "invalid number");
    }
    memcpy(tmp, p->text + start, n);
    tmp[n] = '\0';
    if (is_float) {
        out->type = BLT_JSON_FLOAT;
        out->float_val = strtod(tmp, NULL);
    } else {
        out->type = BLT_JSON_INT;
        out->int_val = (int64_t)strtoll(tmp, NULL, 10);
    }
}

// Containers

static void grow_children(json_parser *p, blt_json_value *container, blt_json_value ***children, char ***keys,
                          size_t *cap) {
    if (*cap == 0) {
        *cap = 4;
        *children = (blt_json_value **)blt_arena_alloc(p->arena, *cap * sizeof(blt_json_value *), sizeof(void *));
        if (keys) {
            *keys = (char **)blt_arena_alloc(p->arena, *cap * sizeof(char *), sizeof(void *));
        }
    } else if (container->num_children == *cap) {
        size_t new_cap = *cap * 2;
        blt_json_value **nc =
            (blt_json_value **)blt_arena_alloc(p->arena, new_cap * sizeof(blt_json_value *), sizeof(void *));
        memcpy(nc, *children, *cap * sizeof(blt_json_value *));
        *children = nc;
        if (keys) {
            char **nk = (char **)blt_arena_alloc(p->arena, new_cap * sizeof(char *), sizeof(void *));
            memcpy(nk, *keys, *cap * sizeof(char *));
            *keys = nk;
        }
        *cap = new_cap;
    }
}

static void parse_object(json_parser *p, blt_json_value *out) {
    out->type = BLT_JSON_OBJECT;
    p->pos++;

    blt_json_value **children = NULL;
    char **keys = NULL;
    size_t cap = 0;
    out->num_children = 0;

    skip_ws(p);
    if (p->pos < p->len && p->text[p->pos] == '}') {
        p->pos++;
        out->children = NULL;
        out->keys = NULL;
        return;
    }

    for (;;) {
        skip_ws(p);
        size_t key_len = 0;
        char *key = parse_string_raw(p, &key_len);
        expect(p, ':');
        blt_json_value *child = alloc_value(p->arena);
        parse_value(p, child);

        grow_children(p, out, &children, &keys, &cap);
        children[out->num_children] = child;
        keys[out->num_children] = key;
        out->num_children++;

        skip_ws(p);
        if (p->pos < p->len && p->text[p->pos] == ',') {
            p->pos++;
            continue;
        }
        expect(p, '}');
        break;
    }
    out->children = children;
    out->keys = keys;
}

static void parse_array(json_parser *p, blt_json_value *out) {
    out->type = BLT_JSON_ARRAY;
    p->pos++;

    blt_json_value **children = NULL;
    size_t cap = 0;
    out->num_children = 0;

    skip_ws(p);
    if (p->pos < p->len && p->text[p->pos] == ']') {
        p->pos++;
        out->children = NULL;
        out->keys = NULL;
        return;
    }

    for (;;) {
        blt_json_value *child = alloc_value(p->arena);
        parse_value(p, child);
        grow_children(p, out, &children, NULL, &cap);
        children[out->num_children++] = child;

        skip_ws(p);
        if (p->pos < p->len && p->text[p->pos] == ',') {
            p->pos++;
            continue;
        }
        expect(p, ']');
        break;
    }
    out->children = children;
    out->keys = NULL;
}

static void parse_value(json_parser *p, blt_json_value *out) {
    skip_ws(p);
    if (p->pos >= p->len) {
        fail(p, "unexpected end of input");
    }
    char c = p->text[p->pos];
    if (c == '{') {
        parse_object(p, out);
    } else if (c == '[') {
        parse_array(p, out);
    } else if (c == '"') {
        out->type = BLT_JSON_STRING;
        out->str_val = parse_string_raw(p, &out->str_len);
    } else if (is_number_start(c)) {
        parse_number(p, out);
    } else if (p->pos + 4 <= p->len && memcmp(p->text + p->pos, "true", 4) == 0) {
        out->type = BLT_JSON_BOOL;
        out->bool_val = true;
        p->pos += 4;
    } else if (p->pos + 5 <= p->len && memcmp(p->text + p->pos, "false", 5) == 0) {
        out->type = BLT_JSON_BOOL;
        out->bool_val = false;
        p->pos += 5;
    } else if (p->pos + 4 <= p->len && memcmp(p->text + p->pos, "null", 4) == 0) {
        out->type = BLT_JSON_NULL;
        p->pos += 4;
    } else {
        fail(p, "unexpected token");
    }
}

// Public API

blt_json_value *blt_json_parse(blt_arena *arena, const char *text) {
    BLT_REQUIRE(arena != NULL && text != NULL, "blt_json_parse: arena and text cannot be NULL");

    json_parser p;
    p.arena = arena;
    p.text = text;
    p.len = strlen(text);
    p.pos = 0;

    blt_json_value *root = alloc_value(arena);
    parse_value(&p, root);
    skip_ws(&p);
    if (p.pos != p.len) {
        fail(&p, "trailing content after top-level value");
    }
    return root;
}

blt_json_value *blt_json_parse_file(blt_arena *arena, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        BLT_FATAL("blt_json_parse_file: cannot open %s", path);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        BLT_FATAL("blt_json_parse_file: ftell failed on %s", path);
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        BLT_FATAL("blt_json_parse_file: out of memory reading %s", path);
    }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';

    blt_json_value *root = blt_json_parse(arena, buf);
    free(buf);
    return root;
}

const blt_json_value *blt_json_get(const blt_json_value *obj, const char *key) {
    if (!obj || obj->type != BLT_JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->num_children; i++) {
        if (strcmp(obj->keys[i], key) == 0) {
            return obj->children[i];
        }
    }
    return NULL;
}

bool blt_json_as_bool(const blt_json_value *v) {
    if (!v || v->type != BLT_JSON_BOOL) {
        BLT_FATAL("blt_json_as_bool: value is not a bool");
    }
    return v->bool_val;
}

int64_t blt_json_as_int(const blt_json_value *v) {
    if (!v) {
        BLT_FATAL("blt_json_as_int: missing value");
    }
    if (v->type == BLT_JSON_INT) return v->int_val;
    if (v->type == BLT_JSON_FLOAT) return (int64_t)v->float_val;
    BLT_FATAL("blt_json_as_int: value is not numeric");
    return 0;
}

double blt_json_as_float(const blt_json_value *v) {
    if (!v) {
        BLT_FATAL("blt_json_as_float: missing value");
    }
    if (v->type == BLT_JSON_FLOAT) return v->float_val;
    if (v->type == BLT_JSON_INT) return (double)v->int_val;
    BLT_FATAL("blt_json_as_float: value is not numeric");
    return 0.0;
}

const char *blt_json_as_string(const blt_json_value *v) {
    if (!v || v->type != BLT_JSON_STRING) {
        BLT_FATAL("blt_json_as_string: value is not a string");
    }
    return v->str_val;
}

bool blt_json_get_bool(const blt_json_value *obj, const char *key, bool fallback) {
    const blt_json_value *v = blt_json_get(obj, key);
    return (v && v->type == BLT_JSON_BOOL) ? v->bool_val : fallback;
}

int64_t blt_json_get_int(const blt_json_value *obj, const char *key, int64_t fallback) {
    const blt_json_value *v = blt_json_get(obj, key);
    return (v && (v->type == BLT_JSON_INT || v->type == BLT_JSON_FLOAT)) ? blt_json_as_int(v) : fallback;
}

double blt_json_get_float(const blt_json_value *obj, const char *key, double fallback) {
    const blt_json_value *v = blt_json_get(obj, key);
    return (v && (v->type == BLT_JSON_INT || v->type == BLT_JSON_FLOAT)) ? blt_json_as_float(v) : fallback;
}

const char *blt_json_get_string(const blt_json_value *obj, const char *key, const char *fallback) {
    const blt_json_value *v = blt_json_get(obj, key);
    return (v && v->type == BLT_JSON_STRING) ? v->str_val : fallback;
}

size_t blt_json_array_size(const blt_json_value *arr) {
    if (!arr || arr->type != BLT_JSON_ARRAY) return 0;
    return arr->num_children;
}

const blt_json_value *blt_json_array_at(const blt_json_value *arr, size_t i) {
    if (!arr || arr->type != BLT_JSON_ARRAY || i >= arr->num_children) return NULL;
    return arr->children[i];
}
