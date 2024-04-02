#include "fvmfields.cuh"





FvmField* FvmFieldNew(size_t elem_size, size_t size, const char* name, int alloc_grads) {
	FvmField* v = cpuAlloc(FvmField, 1);

	v->name = cpuAlloc(char, strlen(name));
	strcpy(v->name, name);

	// Allocate cpu var
	v->_cpu = cpuAlloc(FvmFieldBase, 1);
	v->_cpu->size = size;
	v->_cpu->elem_size = elem_size;

	v->_cpu->value = malloc(size * elem_size);
	if (alloc_grads) {
		v->_cpu->gradient = malloc(size * elem_size * 3);
		v->_cpu->limiter = malloc(size * elem_size);
	} else {
		v->_cpu->gradient = NULL;
		v->_cpu->limiter = NULL;
	}

	// Allocate gpu var
	v->_hgpu = cpuAlloc(FvmFieldBase, 1);
	v->_hgpu->size = size;
	v->_hgpu->elem_size = elem_size;

	cudaMalloc(&v->_hgpu->value, size * elem_size);
	if (alloc_grads) {
		cudaMalloc(&v->_hgpu->gradient, size * elem_size * 3);
		cudaMalloc(&v->_hgpu->limiter, size * elem_size);
	} else {
		v->_hgpu->gradient = NULL;
		v->_hgpu->limiter = NULL;
	}

	// Copy the hgpu pointer to gpu
	gpuAlloc(v->_gpu, FvmFieldBase, 1);
	cudaMemcpy(v->_gpu, v->_hgpu, sizeof(FvmFieldBase), cudaMemcpyHostToDevice);

	return v;
}


void FvmFieldFree(FvmField* v) {
	free(v->_cpu->value);
	if (v->_cpu->gradient != NULL) free(v->_cpu->gradient);
	if (v->_cpu->limiter != NULL) free(v->_cpu->limiter);
	free(v->_cpu);

	cudaFree(v->_hgpu->value);
	if (v->_hgpu->gradient != NULL) cudaFree(v->_hgpu->gradient);
	if (v->_hgpu->limiter != NULL) cudaFree(v->_hgpu->limiter);
	free(v->_hgpu);

	cudaFree(v->_gpu);

	free(v->name);
	free(v);
}


FvmFieldType FvmFieldGetType(const FvmField* v) {
	if (v->_cpu->elem_size == sizeof(scalar)) {
		return SCALAR;
	} else if (v->_cpu->elem_size == sizeof(vector)) {
		return VECTOR;
	} else if (v->_cpu->elem_size == sizeof(tensor)) {
		return TENSOR;
	} else {
		return FIELD_UNKNOWN;
	}
}


void* FvmFieldValueIndex(FvmField* var, const size_t& i) {
	return (void*)((char*)var->_cpu->value + var->_cpu->elem_size * i);
}


void VarGpuToCpu(FvmField* var) {
	cudaSync;

	cudaExec(cudaMemcpy(var->_cpu->value, var->_hgpu->value, var->_cpu->size * var->_cpu->elem_size, cudaMemcpyDeviceToHost));

	cudaSync;
}

void VarAllGpuToCpu(FvmField* var) {
	cudaSync;

	cudaExec(cudaMemcpy(var->_cpu->value, var->_hgpu->value, var->_cpu->size * var->_cpu->elem_size, cudaMemcpyDeviceToHost));
	cudaExec(cudaMemcpy(var->_cpu->gradient, var->_hgpu->gradient, var->_cpu->size * var->_cpu->elem_size * 3, cudaMemcpyDeviceToHost));
	cudaExec(cudaMemcpy(var->_cpu->limiter, var->_hgpu->limiter, var->_cpu->size * var->_cpu->elem_size, cudaMemcpyDeviceToHost));

	cudaSync;
}




