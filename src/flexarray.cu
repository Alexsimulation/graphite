
#include <assert.h>
#include "flexarray.cuh"



size_t FlexArrayDataSize(FlexArrayData_t type) {
	size_t element_size = 1;
	switch (type) {
	case FLX_FLOAT:
		element_size = sizeof(float);
		break;
	case FLX_DOUBLE:
		element_size = sizeof(double);
		break;
	case FLX_INT:
		element_size = sizeof(int);
		break;
	case FLX_UINT:
		element_size = sizeof(unsigned int);
		break;
	}
	return element_size;
}


FlexArray* FlexArray_new(FlexArrayData_t type, size_t size) {
	FlexArray* arr = (FlexArray*)malloc(sizeof(FlexArray));

	if (arr == NULL) {
		return arr;
	}

	size_t element_size = FlexArrayDataSize(type);

	arr->array = (FlexArray_t*)malloc(element_size * size);
	arr->type = type;
	arr->size = size;
	arr->alloc_size = size;
	arr->element_size = element_size;

	return arr;
}

void FlexArray_free(FlexArray* arr) {
	free(arr->array);
	free(arr);
}



int FlexArray_resize(FlexArray* arr, size_t req_size) {
	size_t element_size = arr->element_size;

	// Check if the array is large enough
	if (req_size > arr->alloc_size) {
		// We need to grow the array
		size_t growth_a = (size_t)(((double)req_size) * 1.2);
		size_t growth_b = 32;
		size_t growth = growth_a > growth_b ? growth_a : growth_b;

		size_t new_size = req_size + growth;

		// Try to realloc
		void* new_ptr = realloc(arr->array, new_size * element_size);
		if (new_ptr == NULL) return 1;	// Error in realloc :(
		arr->array = new_ptr;

		arr->alloc_size = new_size;
	}
	// Resize array
	arr->size = req_size;

	return 0;
}

int FlexArray_append(FlexArray* arr, FlexArray_t* data, size_t count) {
	size_t element_size = arr->element_size;

	// Resize if needed
	if (FlexArray_resize(arr, arr->size + count)) {
		return 1;
	}

	// Copy data from send buffer to array
	FlexArray_t* arr_end = (FlexArray_t*)(((char*)arr->array) + arr->size * element_size);
	memcpy(arr_end, data, count * element_size);

	return 0;
}

FlexArray_t* FlexArray_index_v(FlexArray* arr, size_t i) {
	assert(i < arr->size);
	return (FlexArray_t*)(((char*)arr->array) + i * arr->element_size);
}



FlexArray_t* FlexArray_index_append_v(FlexArray* arr, size_t i) {
	if (i >= arr->size)
		assert(FlexArray_resize(arr, i + 1) == 0);

	return FlexArray_index_v(arr, i);
}



