#pragma once
#include <stdlib.h>
#include <string.h>


typedef void FlexArray_t;

typedef enum FlexArrayData_t {
	FLX_FLOAT,
	FLX_DOUBLE,
	FLX_INT,
	FLX_UINT
} FlexArrayData_t;


size_t FlexArrayDataSize(FlexArrayData_t type);


typedef struct FlexArray {
	FlexArray_t* array;
	FlexArrayData_t type;
	size_t size;
	size_t alloc_size;
	size_t element_size;
} FlexArray;


FlexArray* FlexArray_new(FlexArrayData_t type, size_t size);

void FlexArray_free(FlexArray* arr);



int FlexArray_resize(FlexArray* arr, size_t req_size);

int FlexArray_append(FlexArray* arr, FlexArray_t* data, size_t count);

FlexArray_t* FlexArray_index_v(FlexArray* arr, size_t i);

#define FlexArray_index(type, arr, i) ((type*)FlexArray_index_v(arr, i))


FlexArray_t* FlexArray_index_append_v(FlexArray* arr, size_t i);

#define FlexArray_index_append(type, arr, i) ((type*)FlexArray_index_append_v(arr, i))
