#pragma once

#include "datatypes.cuh"


/*
    cuMeshBase structure
        Supports any kind of polyhedral cell unstructured mesh
        with the bare minimum amount of info necessary to not
        have to recompute information at each compute step
*/
typedef enum CUMESH_t {
    _CPU,
    _HGPU,
    _GPU
} CUMESH_t;

typedef struct cuMeshBase {
    // Type information
    CUMESH_t type;
    int allocated;

    // Size information
    size_t n_cells;
    size_t n_faces;
    size_t n_nodes;
    size_t n_ghosts;
    size_t n_regions;

    // Cell information
    uint* cell_faces;
    uint* cell_connects;
    uint* cell_faces_starts;
    scalar* cell_volumes;
    vector* cell_centers;

    tensor* cell_matrices;

    // Faces information
    uint* face_nodes;
    uint* face_nodes_starts;
    scalar* face_areas;
    vector* face_centers;
    vector* face_normals;

    // Nodes information
    vector* nodes;

    // Boundary information
    uint* ghost_cell_ids;
    uint* ghost_cell_starts;
} cuMeshBase;


typedef struct cuMesh {
    cuMeshBase* _cpu;
    cuMeshBase* _hgpu;
    cuMeshBase* _gpu;

    char** region_names;
    size_t region_names_alloc;
    size_t region_names_size;
} cuMesh;


size_t cuMeshGetVarSize(cuMesh* mesh);

size_t cuMeshGetnCells(cuMesh* mesh);

size_t cuMeshGetnRegionCells(cuMesh* mesh, const uint& region_id);

uint cuMeshGetRegionId(cuMesh* mesh, const char* name);

size_t cuMeshGetnGhosts(cuMesh* mesh);

size_t cuMeshGetnCellsAndGhosts(cuMesh* mesh);

cuMeshBase* cuMeshGetGpu(cuMesh* mesh);

/*
    Helper functions to avoid errors
*/
__host__ __device__ inline uint* cuMeshGetFaceStart(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->face_nodes + mesh->face_nodes_starts[i];
}
__host__ __device__ inline uint* cuMeshGetFaceEnd(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->face_nodes + mesh->face_nodes_starts[i + 1];
}
__host__ __device__ inline uint cuMeshGetFaceSize(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->face_nodes_starts[i + 1] - mesh->face_nodes_starts[i];
}

__host__ __device__ inline uint* cuMeshGetCellStart(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->cell_faces + mesh->cell_faces_starts[i];
}
__host__ __device__ inline uint* cuMeshGetCellEnd(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->cell_faces + mesh->cell_faces_starts[i + 1];
}
__host__ __device__ inline uint cuMeshGetCellSize(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->cell_faces_starts[i + 1] - mesh->cell_faces_starts[i];
}
__host__ __device__ inline uint* cuMeshGetCellConnect(
    cuMeshBase* mesh,
    const int& i
) {
    // Cell connectivity as the same pattern as cell to face connectivity
    // since all cells are connected through one face
    return mesh->cell_connects + mesh->cell_faces_starts[i];
}
__host__ __device__ inline uint cuMeshGetGhostCellFace(
    cuMeshBase* mesh,
    const int& i
) {
    return mesh->cell_faces[mesh->cell_faces_starts[i]];
}


__host__ __device__ inline uint cuMeshGetRegionSize(
    cuMeshBase* mesh,
    const uint& region_id
) {
    return mesh->ghost_cell_starts[region_id + 1] - mesh->ghost_cell_starts[region_id];
}


/*
    Kernel to compute mesh faces informations
*/
__global__ void cuMeshComputeFaces(
    cuMeshBase* mesh
);

/*
    Kernel to compute cuMeshBase cell information
*/
__global__ void cuMeshComputeCells(
    cuMeshBase* mesh
);

/*
    Kernel to compute cuMeshBase ghost cell information
*/
__global__ void cuMeshComputeGhostCells(
    cuMeshBase* mesh
);

/*
    Kernel to compute gradient matrices
*/
__global__ void cuMeshComputeCellMatrices(
    cuMeshBase* mesh
);


/*
    Allocate a cuMesh object
*/
void cuMeshAllocateCalcsGpu(
    cuMesh* mesh
);


/*
    Allocate a cuMesh object
*/
void cuMeshAllocate(
    cuMesh* mesh,
    size_t n_cells,
    size_t n_cells_faces,
    size_t n_faces,
    size_t n_faces_nodes,
    size_t n_nodes,
    size_t n_ghosts,
    size_t n_regions
);


void cuMeshBaseFreeCpu(
    cuMeshBase* mesh
);

void cuMeshBaseFreeGpu(
    cuMeshBase* mesh
);


cuMesh* cuMeshNew();

void cuMeshFree(cuMesh* mesh);


void cuMeshPassCpuToGpu(
    cuMesh* mesh
);


void cuMeshPassGpuToCpu(
    cuMesh* mesh
);


int cuMeshAddRegionName(
    cuMesh* mesh,
    const char* name
);


void cuMeshMake1D(
    cuMesh* mesh,
    size_t n,
    scalar x0,
    scalar x1
);


