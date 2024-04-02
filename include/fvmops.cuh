#pragma once

#include "mesh.cuh"
#include "fvmfields.cuh"


/*
	Helper macros to define finite volume operators
		Usage:
		fvmOpC(out type [scalar/vector], name, in_type [scalar/vector])
			whatever you want as the integrand
		fvmOpEnd
		See examples below
*/


#define fvmOpEnd \
		other_cell++; \
	} \
	return x; \
}


#define fvmOpMakeInputVar(value_type, grad_type, var_name, input_name) \
	struct {value_type* val; grad_type* grad; value_type* lim;} var_name; \
	var_name.val = (value_type*)input_name->value; \
	var_name.grad = (grad_type*)input_name->gradient; \
	var_name.lim = (value_type*)input_name->limiter; \


#define fvmOp(op_out_type, op_name) \
__device__ inline op_out_type op_name (\
	cuMeshBase* mesh, \
	int c0, \


#define fvmOpSetup(op_out_type) \
	uint* f_start = cuMeshGetCellStart(mesh, c0); \
	uint* f_end = cuMeshGetCellEnd(mesh, c0); \
	uint* other_cell = cuMeshGetCellConnect(mesh, c0); \
	scalar v = mesh->cell_volumes[c0]; \
	op_out_type x; \
	reset(x); \
	for (uint* fp = f_start; fp != f_end; ++fp) { \
		uint f = *fp; \
		uint c1 = *other_cell; \
		vector d0 = mesh->face_centers[f] - mesh->cell_centers[c0]; \
		vector d1 = mesh->face_centers[f] - mesh->cell_centers[c1]; \
		vector n = outer_normal(mesh->face_normals[f], d0); \
		scalar ds = mesh->face_areas[f] / v; \



#define fvmOpEnd \
		other_cell++; \
	} \
	return x; \
} \


#define fvmOpVoidSetup \
	uint* f_start = cuMeshGetCellStart(mesh, c0); \
	uint* f_end = cuMeshGetCellEnd(mesh, c0); \
	uint* other_cell = cuMeshGetCellConnect(mesh, c0); \
	scalar v = mesh->cell_volumes[c0]; \
	for (uint* fp = f_start; fp != f_end; ++fp) { \
		uint f = *fp; \
		uint c1 = *other_cell; \
		vector d0 = mesh->face_centers[f] - mesh->cell_centers[c0]; \
		vector d1 = mesh->face_centers[f] - mesh->cell_centers[c1]; \
		vector n = outer_normal(mesh->face_normals[f], d0); \
		scalar ds = mesh->face_areas[f] / v; \

#define fvmOpVoidEnd \
		other_cell++; \
	} \
} \


/*
	fvm limiter, venkatakrishnan
*/
__device__ inline scalar limiter(
	const scalar& u,
	const vector& u_grad,
	const scalar& u_min,
	const scalar& u_max,
	const vector& d
);


/*
	Finite volume operators
*/

fvmOp(scalar, fvmLaplacianC)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(scalar)
	x += dot(u.grad[c1] + u.grad[c0], n) * 0.5 * ds;		
fvmOpEnd


fvmOp(vector, fvmScalGradC)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(vector)
	scalar phi = norm(d0) / (norm(d0- d1));
	x += n * (u.val[c1] * phi + u.val[c0] * (1.0 - phi)) * ds;
fvmOpEnd

fvmOp(vector, fvmScalGradCCorrect)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(vector)
	scalar phi = norm(d0) / (norm(d0 - d1));
	vector eval_position = (d0 - d1) * phi;
	vector d_err = mesh->face_centers[f] - eval_position;
	scalar u_corr = dot(u.grad[c0], d_err);
	x += n * u_corr * ds;
fvmOpEnd

fvmOp(tensor, fvmVecGradC)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(vector, tensor, u, u_in);
	fvmOpSetup(tensor)
	scalar phi = norm(d0) / (norm(d0) + norm(d1));
	x += outer(n, u.val[c1] * phi + u.val[c0] * (1.0 - phi)) * ds;
fvmOpEnd


fvmOp(vector, fvmBoundScalGradC)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(vector)
	x += n * (u.val[c1] - u.val[c0]) * ds;
fvmOpEnd

fvmOp(tensor, fvmBoundVecGradC)
	FvmFieldBase* u_in
) {
	fvmOpMakeInputVar(vector, tensor, u, u_in);
	fvmOpSetup(tensor)
	x += outer(u.val[c1] - u.val[c0], n) * ds;
fvmOpEnd


fvmOp(scalar, fvmDivC)			
	FvmFieldBase* u_in
	) {
	fvmOpMakeInputVar(vector, tensor, u, u_in);
	fvmOpSetup(scalar)
	x += dot(u.val[c0] + u.val[c1], n) * 0.5 * ds;
fvmOpEnd


fvmOp(scalar, fvmConvConstScalU)
	FvmFieldBase* u_in,
	vector phi
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(scalar)
	scalar u0 = u.val[c0];
	scalar u1 = u.val[c1];
	scalar a0 = dot(n, phi);
	scalar a1 = dot(n, phi);
	scalar flux0 = a0 * u0;	
	scalar flux1 = a1 * u1;
	x += ((flux0 + flux1) - fmax(fabs(a0), fabs(a1)) * (u1 - u0)) * 0.5 * ds;
fvmOpEnd


fvmOp(scalar, fvmConvConstScalU2)
	FvmFieldBase* u_in,
	vector phi
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(scalar)
	scalar u0 = u.val[c0] + dot(u.grad[c0], d0) * u.lim[c0];
	scalar u1 = u.val[c1] + dot(u.grad[c1], d1) * u.lim[c1];
	scalar a0 = dot(n, phi);
	scalar a1 = dot(n, phi);
	scalar flux0 = a0 * u0;	
	scalar flux1 = a1 * u1;
	x += ((flux0 + flux1) - fmax(fabs(a0), fabs(a1)) * (u1 - u0)) * 0.5 * ds;
fvmOpEnd


fvmOp(scalar, fvmConvScalU)
	FvmFieldBase* u_in,
	vector* phi
) {
	fvmOpMakeInputVar(scalar, vector, u, u_in);
	fvmOpSetup(scalar)
	scalar a0 = dot(n, phi[c0]);
	scalar a1 = dot(n, phi[c1]);
	scalar flux0 = a0 * u.val[c0];	
	scalar flux1 = a1 * u.val[c1];
	x += ((flux0 + flux1) - fmax(fabs(a0), fabs(a1)) * (u.val[c1] - u.val[c0])) * 0.5 * ds;
fvmOpEnd


fvmOp(vector, fvmConvVecU)
	FvmFieldBase* u_in,
	vector* phi
) {
	fvmOpMakeInputVar(vector, tensor, u, u_in);
	fvmOpSetup(vector)
	scalar a0 = dot(n, phi[c0]);
	scalar a1 = dot(n, phi[c1]);
	vector flux0 = u.val[c0] * a0;
	vector flux1 = u.val[c1] * a1;
	x += ((flux0 + flux1) -  (u.val[c1] - u.val[c0]) * fmax(a0, a1)) * 0.5 * ds;
fvmOpEnd



/*
	Field initiations
*/

__global__ void ScalarVarInit(
	cuMeshBase* mesh,
	FvmFieldBase* T,
	scalar value
);

__global__ void VectorVarInit(
	cuMeshBase* mesh,
	FvmFieldBase* T,
	vector value
);



/*
	Kernel to compute gradients on a scalar field
*/
__global__ void ScalarVarComputeGradient(
	cuMeshBase* mesh,
	FvmFieldBase* field
);

/*
	Kernel to compute limiters on a scalar field
*/
__global__ void ScalarVarComputeLimiters(
	cuMeshBase* mesh,
	FvmFieldBase* var
);

/*
	Kernels to handle gradients and limiters on boundaries
*/
__global__ void fvmScalGradLimBc(
	cuMeshBase* mesh,
	FvmFieldBase* v
);
__global__ void fvmVecGradLimBc(
	cuMeshBase* mesh,
	FvmFieldBase* v
);

/*
	Kernel to compute gradients on a vector field
*/
__global__ void VectorVarComputeGradient(
	cuMeshBase* mesh,
	FvmFieldBase* field
);

/*
	Kernel to compute limiters on a vector field
*/
__global__ void VectorVarComputeLimiters(
	cuMeshBase* mesh,
	FvmFieldBase* var
);



__global__ void fvmScalarBcFixed(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	scalar bc_value,
	uint region_id
);

__global__ void fvmScalarBcZeroGrad(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	uint region_id
);

__global__ void fvmVectorBcZeroGrad(
	cuMeshBase* mesh,
	FvmFieldBase* v,
	uint region_id
);



scalar FvmFieldReduceSum(
	cuMesh* mesh,
	FvmField* var
);
