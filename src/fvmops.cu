#include "fvmops.cuh"


/*
	Field initiations
*/

__global__ void ScalarVarInit(
	cuMeshBase* mesh,
	FvmFieldBase* T,
	scalar value
) {
	int i = cudaGlobalId;
	if (i >= (mesh->n_cells + mesh->n_ghosts)) return;
	scalar* arr = (scalar*)T->value;
	vector* grad = (vector*)T->gradient;
	scalar* lim = (scalar*)T->limiter;

	arr[i] = value;
	grad[i] = make_vector(0.0, 0.0, 0.0);
	lim[i] = 1.0;
}


__global__ void VectorVarInit(
	cuMeshBase* mesh,
	FvmFieldBase* T,
	vector value
) {
	int i = cudaGlobalId;
	if (i >= (mesh->n_cells + mesh->n_ghosts)) return;
	vector* arr = (vector*)T->value;
	tensor* grad = (tensor*)T->gradient;
	vector* lim = (vector*)T->limiter;

	arr[i] = value;
	grad[i] = make_tensor(
		0.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.0f, 
		0.0f, 0.0f, 0.0f
	);
	lim[i] = make_vector(1.0, 1.0, 1.0);
}




/*
	Kernel to compute gradients on a scalar field
*/
__global__ void ScalarVarComputeGradient(
	cuMeshBase* mesh,
	FvmFieldBase* field
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_cells) return;

	scalar* value = (scalar*)field->value;
	vector* grad = (vector*)field->gradient;

	vector g;
	reset(g);

	// Loop on all connected cells
	uint* c = cuMeshGetCellConnect(mesh, i);
	uint size = cuMeshGetCellSize(mesh, i);
	for (uint _ignore = 0; _ignore < size; ++_ignore) {
		
		vector d = mesh->cell_centers[*c] - mesh->cell_centers[i];
		scalar w2 = 1.0 / dot(d, d);

		g += d * (value[*c] - value[i]) * w2;

		c++;
	}

	grad[i] = mesh->cell_matrices[i] * g;
}

__global__ void VectorVarComputeGradient(
	cuMeshBase* mesh,
	FvmFieldBase* field
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_cells) return;

	vector* value = (vector*)field->value;
	tensor* grad = (tensor*)field->gradient;

	tensor g;
	reset(g);

	// Loop on all connected cells
	uint* c = cuMeshGetCellConnect(mesh, i);
	uint size = cuMeshGetCellSize(mesh, i);
	for (uint _ignore = 0; _ignore < size; ++_ignore) {
		
		vector d = mesh->cell_centers[*c] - mesh->cell_centers[i];
		scalar w2 = 1.0 / dot(d, d);

		g.u += d * (value[*c].x - value[i].x) * w2;
		g.v += d * (value[*c].y - value[i].y) * w2;
		g.w += d * (value[*c].z - value[i].z) * w2;

		c++;
	}

	grad[i].u = mesh->cell_matrices[i] * g.u;
	grad[i].v = mesh->cell_matrices[i] * g.v;
	grad[i].w = mesh->cell_matrices[i] * g.w;
}



/*
	fvm limiter, venkatakrishnan
*/
// Barth and Jespersen
// #define LIMITER_FUNC(y) max(0.0, min(1.0, y))

// Venkatakrishnan
// #define LIMITER_FUNC(y) (y*y + 2.0*y)/(y*y + y + 2.0)

// Michalak
#define LIMITER_FUNC(y) ( ((y) < 2.0) ? ((y) - 0.25*(y)*(y)) : 1.0 )

#define LIMITER_K 1.0

__device__ inline scalar limiter(
	const scalar& u,
	const vector& u_grad,
	const scalar& u_min,
	const scalar& u_max,
	const vector& d,
	const scalar& v
) {
	scalar delta = dot(u_grad, d);
	scalar d_u = u_max - u;
	scalar d_d = u_min - u;

	scalar dmm2 = (d_u - d_d) * (d_u - d_d);

	scalar dx = norm(d);
	scalar k3v = LIMITER_K * dx;
	k3v *= (k3v * k3v);

	scalar sig = 1.0;

	if (dmm2 <= k3v) {
		sig = 1.0;
	} else if (dmm2 <= 2*k3v) {
		scalar y = dmm2 / k3v - 1.0;
		sig = 2.0 * y*y*y - 3.0 * y*y + 1.0;
	} else {
		sig = 0.0;
	}
	
	scalar lim = 1.0;

	if (delta > FVM_TOL) {
		lim = LIMITER_FUNC(d_u / delta);
	}
	else if (delta < -FVM_TOL) {
		lim = LIMITER_FUNC(d_d / delta);
	}
	lim *= 0.8;

	return sig + (1.0 - sig) * lim;
}


/*
	Kernel to compute limiters on a scalar field
*/
__global__ void ScalarVarComputeLimiters(
	cuMeshBase* mesh,
	FvmFieldBase* var
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_cells) return;

	scalar* field = (scalar*)var->value;
	vector* gradField = (vector*)var->gradient;
	scalar* limiterField = (scalar*)var->limiter;

	//const int size = cuMeshGetCellSize(mesh, i);

	// Compute min and max
	scalar field_min = field[i];
	scalar field_max = field[i];

	uint* f_start = cuMeshGetCellStart(mesh, i);
	uint* f_end = cuMeshGetCellEnd(mesh, i);
	uint* other_cell = cuMeshGetCellConnect(mesh, i);
	for (uint* fp = f_start; fp != f_end; ++fp) {
		//uint f = *fp;
		uint c1 = *other_cell;
		scalar field_val = field[c1];
		// Branchless min max operators
		field_min = fmin(field_min, field_val);
		field_max = fmax(field_max, field_val);

		other_cell++;
	}

	// Compute limiter
	//other_cell = cuMeshGetCellConnect(mesh, i);
	scalar lim = 2.0;
	for (uint* fp = f_start; fp != f_end; ++fp) {
		uint f = *fp;
		vector d = mesh->face_centers[f] - mesh->cell_centers[i];
		scalar lim_f = limiter(field[i], gradField[i], field_min, field_max, d, mesh->cell_volumes[i]);
		lim = fmin(lim, lim_f);
	}

	limiterField[i] = lim;
}


/*
	Kernel to compute limiters on a vector field
*/
__global__ void VectorVarComputeLimiters(
	cuMeshBase* mesh,
	FvmFieldBase* var
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_cells) return;

	vector* field = (vector*)var->value;
	tensor* gradField = (tensor*)var->gradient;
	vector* limiterField = (vector*)var->limiter;

	// Compute min and max
	vector field_min = field[i];
	vector field_max = field[i];

	uint* f_start = cuMeshGetCellStart(mesh, i);
	uint* f_end = cuMeshGetCellEnd(mesh, i);
	uint* other_cell = cuMeshGetCellConnect(mesh, i);
	for (uint* fp = f_start; fp != f_end; ++fp) {
		//uint f = *fp;
		uint c1 = *other_cell;
		vector field_val = field[c1];
		// Branchless min max operators
		field_min = minv(field_min, field_val);
		field_max = maxv(field_max, field_val);

		other_cell++;
	}

	// Compute limiter
	//other_cell = cuMeshGetCellConnect(mesh, i);
	vector lim = make_vector(2.0, 2.0, 2.0);
	for (uint* fp = f_start; fp != f_end; ++fp) {
		uint f = *fp;
		vector d = mesh->face_centers[f] - mesh->cell_centers[i];
		vector lim_f;
		lim_f.x = limiter(field[i].x, gradField[i].u, field_min.x, field_max.x, d, mesh->cell_volumes[i]);
		lim_f.y = limiter(field[i].y, gradField[i].v, field_min.y, field_max.y, d, mesh->cell_volumes[i]);
		lim_f.z = limiter(field[i].z, gradField[i].w, field_min.z, field_max.z, d, mesh->cell_volumes[i]);
		lim = minv(lim, lim_f);
	}

	limiterField[i] = lim;
}


/*
	Kernels to handle gradients and limiters on boundaries
*/
__global__ void fvmScalGradLimBc(
	cuMeshBase* mesh,
	FvmFieldBase* v
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_ghosts) return;

	uint cell = mesh->ghost_cell_ids[i];
	uint other_cell = *cuMeshGetCellConnect(mesh, cell);

	vector* grad = (vector*)v->gradient;
	scalar* lim = (scalar*)v->limiter;

	grad[cell] = grad[other_cell];
	lim[cell] = lim[other_cell];
}

__global__ void fvmVecGradLimBc(
	cuMeshBase* mesh,
	FvmFieldBase* v
) {
	int i = cudaGlobalId;
	if (i >= mesh->n_ghosts) return;

	uint cell = mesh->ghost_cell_ids[i];
	uint other_cell = *cuMeshGetCellConnect(mesh, cell);

	tensor* grad = (tensor*)v->gradient;
	vector* lim = (vector*)v->limiter;

	grad[cell] = grad[other_cell];
	lim[cell] = lim[other_cell];
}



__global__ void fvmScalarBcFixed(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	scalar bc_value,
	uint region_id
) {
	int i = cudaGlobalId;
	uint region_size = cuMeshGetRegionSize(mesh, region_id);
	if (i >= region_size) return;

	uint cell = mesh->ghost_cell_ids[mesh->ghost_cell_starts[region_id] + i];

	scalar* arr = (scalar*)v->value;

	arr[cell] = bc_value;
}


__global__ void fvmScalarBcZeroGrad(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	uint region_id
) {
	int i = cudaGlobalId;
	uint region_size = cuMeshGetRegionSize(mesh, region_id);
	if (i >= region_size) return;

	uint cell = mesh->ghost_cell_ids[mesh->ghost_cell_starts[region_id] + i];

	// Get this cell's connected cell
	uint other_cell = *cuMeshGetCellConnect(mesh, cell);

	scalar* arr = (scalar*)v->value;

	// Make zero gradient condition order 1 for now
	arr[cell] = arr[other_cell];
}


__global__ void fvmVectorBcZeroGrad(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	uint region_id
) {
	int i = cudaGlobalId;
	uint region_size = cuMeshGetRegionSize(mesh, region_id);
	if (i >= region_size) return;

	uint cell = mesh->ghost_cell_ids[mesh->ghost_cell_starts[region_id] + i];

	// Get this cell's connected cell
	uint other_cell = *cuMeshGetCellConnect(mesh, cell);

	vector* arr = (vector*)v->value;

	// Make zero gradient condition order 1 for now
	arr[cell] = arr[other_cell];
}



scalar FvmFieldReduceSum(
	cuMesh* mesh,
	FvmField* var
) {
	VarGpuToCpu(var);
	
	cudaSync;

	scalar sum = 0;
	for (size_t i=0; i<mesh->_cpu->n_cells; ++i) {
		sum += ((scalar*)var->_cpu->value)[i];
	}

	return sum;
}


