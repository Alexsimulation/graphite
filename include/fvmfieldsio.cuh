#pragma once


#include "fvmfields.cuh"
#include "mesh.cuh"



/*
	Write a vtu file with the mesh and field variables
*/
int FieldsWriteVtu(
	const char* filename,
	cuMesh* mesh_in,
	const FvmField** vars,
	const size_t nvars
);

