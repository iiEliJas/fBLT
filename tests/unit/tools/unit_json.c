#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "core/json.h"

int run_json_parser_tests(void) {
    blt_arena *arena = blt_arena_create(64 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    const char *doc = "{"
                      "\"name\":\"baseline_p4\","
                      "\"tag\":\"baseline_p4\","
                      "\"steps\":1000,"
                      "\"lr\":0.01,"
                      "\"ratio\":2.5e-1,"
                      "\"enabled\":true,"
                      "\"disabled\":false,"
                      "\"nothing\":null,"
                      "\"sizes\":[3,4,5],"
                      "\"model\":{\"embed_dim\":128,\"path\":\"a\\\\b\\n\"}"
                      "}";

    blt_json_value *root = blt_json_parse(arena, doc);
    TEST_ASSERT(root != NULL && root->type == BLT_JSON_OBJECT);

    TEST_ASSERT(strcmp(blt_json_get_string(root, "name", "x"), "baseline_p4") == 0);
    TEST_ASSERT(strcmp(blt_json_get_string(root, "tag", "x"), "baseline_p4") == 0);
    TEST_ASSERT(blt_json_get_int(root, "steps", -1) == 1000);
    TEST_ASSERT(blt_json_get_int(root, "missing", 7) == 7);
    TEST_ASSERT(fabs(blt_json_get_float(root, "lr", 0.0) - 0.01) < 1e-9);
    TEST_ASSERT(fabs(blt_json_get_float(root, "ratio", 0.0) - 0.25) < 1e-9);
    // int field readable as float
    TEST_ASSERT(fabs(blt_json_get_float(root, "steps", 0.0) - 1000.0) < 1e-9);
    TEST_ASSERT(blt_json_get_bool(root, "enabled", false) == true);
    TEST_ASSERT(blt_json_get_bool(root, "disabled", true) == false);
    TEST_ASSERT(blt_json_get_bool(root, "nothing", true) == true); // null -> fallback

    const blt_json_value *sizes = blt_json_get(root, "sizes");
    TEST_ASSERT(sizes != NULL && sizes->type == BLT_JSON_ARRAY);
    TEST_ASSERT(blt_json_array_size(sizes) == 3);
    TEST_ASSERT(blt_json_array_at(sizes, 0)->type == BLT_JSON_INT);
    TEST_ASSERT(blt_json_as_int(blt_json_array_at(sizes, 2)) == 5);
    TEST_ASSERT(blt_json_array_at(sizes, 3) == NULL);
    TEST_ASSERT(blt_json_array_size(root) == 0); // not an array

    const blt_json_value *model = blt_json_get(root, "model");
    TEST_ASSERT(model != NULL && model->type == BLT_JSON_OBJECT);
    TEST_ASSERT(blt_json_get_int(model, "embed_dim", 0) == 128);

    // escape decoding
    const blt_json_value *path_v = blt_json_get(model, "path");
    TEST_ASSERT(path_v != NULL && path_v->type == BLT_JSON_STRING);
    TEST_ASSERT(path_v->str_len == 4);
    TEST_ASSERT(memcmp(path_v->str_val, "a\\b\n", 4) == 0);

    // unicode escape
    blt_json_value *u = blt_json_parse(arena, "\"\\u0041\\u00e9\"");
    TEST_ASSERT(u != NULL && u->type == BLT_JSON_STRING);
    TEST_ASSERT(u->str_len == 3); // 'A' + 2-byte UTF-8 e-acute
    TEST_ASSERT((unsigned char)u->str_val[0] == 'A');
    TEST_ASSERT((unsigned char)u->str_val[1] == 0xC3);
    TEST_ASSERT((unsigned char)u->str_val[2] == 0xA9);

    // arrays at top level + nested empty containers
    blt_json_value *arr = blt_json_parse(arena, "[1,-2,3.5,\"s\",true,null,[],{}]");
    TEST_ASSERT(arr != NULL && arr->type == BLT_JSON_ARRAY);
    TEST_ASSERT(blt_json_array_size(arr) == 8);
    TEST_ASSERT(blt_json_as_int(blt_json_array_at(arr, 1)) == -2);
    TEST_ASSERT(fabs(blt_json_as_float(blt_json_array_at(arr, 2)) - 3.5) < 1e-9);
    TEST_ASSERT(strcmp(blt_json_as_string(blt_json_array_at(arr, 3)), "s") == 0);
    TEST_ASSERT(blt_json_as_bool(blt_json_array_at(arr, 4)) == true);
    TEST_ASSERT(blt_json_array_at(arr, 5)->type == BLT_JSON_NULL);
    TEST_ASSERT(blt_json_array_size(blt_json_array_at(arr, 6)) == 0);
    TEST_ASSERT(blt_json_array_size(blt_json_array_at(arr, 7)) == 0);

    // float-formatted int ("1e3" -> FLOAT 1000)
    blt_json_value *sci = blt_json_parse(arena, "{\"v\":1e3}");
    TEST_ASSERT(sci != NULL);
    TEST_ASSERT(blt_json_get(sci, "v")->type == BLT_JSON_FLOAT);
    TEST_ASSERT(blt_json_get_int(sci, "v", 0) == 1000);

    blt_arena_destroy(arena);
    return 1;
}
