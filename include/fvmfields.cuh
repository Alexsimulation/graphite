#pragma once

#include "datatypes.cuh"

typedef void FieldData_t;


typedef enum FvmFieldType {
	SCALAR,
	VECTOR,
	TENSOR,
	FIELD_UNKNOWN
} FvmFieldType;


typedef struct FvmFieldBase {
	size_t elem_size;
	size_t size;
	void* value;
	void* gradient;
	void* limiter;
} FvmFieldBase;


typedef struct FvmField {
	FvmFieldBase* _cpu;
	FvmFieldBase* _hgpu;
	FvmFieldBase* _gpu;
	char* name;
} FvmField;


FvmField* FvmFieldNew(size_t elem_size, size_t size, const char* name, int alloc_grads);

FvmFieldType FvmFieldGetType(const FvmField* v);

void* FvmFieldValueIndex(FvmField* v, const size_t& i);

void FvmFieldFree(FvmField* v);

void VarGpuToCpu(FvmField* var);

void VarAllGpuToCpu(FvmField* var);

