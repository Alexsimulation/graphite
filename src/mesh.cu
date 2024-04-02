
#include "mesh.cuh"


size_t cuMeshGetVarSize(cuMesh* mesh) {
    return mesh->_cpu->n_cells + mesh->_cpu->n_ghosts;
}

size_t cuMeshGetnCells(cuMesh* mesh) {
    return mesh->_cpu->n_cells;
}

size_t cuMeshGetnRegionCells(cuMesh* mesh, const uint& region_id) {
    return mesh->_cpu->ghost_cell_starts[region_id + 1] - mesh->_cpu->ghost_cell_starts[region_id];
}

uint cuMeshGetRegionId(cuMesh* mesh, const char* name) {
    for (uint i = 0; i < mesh->region_names_size; ++i) {
        if (strcmp(mesh->region_names[i], name) == 0) return i;
    }
    printf("Error, region with name %s not found in mesh.", name);
    return 0;
}

size_t cuMeshGetnGhosts(cuMesh* mesh) {
    return mesh->_cpu->n_ghosts;
}

size_t cuMeshGetnCellsAndGhosts(cuMesh* mesh) {
    return mesh->_cpu->n_cells + mesh->_cpu->n_ghosts;
}

cuMeshBase* cuMeshGetGpu(cuMesh* mesh) {
    return mesh->_gpu;
}





/*
    Kernel to compute mesh faces informations
*/
__global__ void cuMeshComputeFaces(
    cuMeshBase* mesh
) {
    // i is the face id in this case
    int i = cudaGlobalId;
    if (i >= mesh->n_faces) return;

    // Get face start and end node reference
    uint* face_start = cuMeshGetFaceStart(mesh, i);
    uint* face_end = cuMeshGetFaceEnd(mesh, i);
    uint face_size = cuMeshGetFaceSize(mesh, i);

    // Compute faces center
    vector center = make_vector((scalar)0, 0, 0);
    for (uint* n = face_start; n != face_end; ++n) {
        center += mesh->nodes[*n];
    }
    center /= (float)face_size;

    // Compute face normal vector and area
    vector new_center = make_vector((scalar)0, 0, 0);
    vector normal = make_vector((scalar)0, 0, 0);
    scalar area = 0;
    for (uint* n0 = face_start; n0 != (face_end - 1); ++n0) {
        uint* n1 = n0 + 1;
        vector d0 = center - mesh->nodes[*n0];
        vector d1 = center - mesh->nodes[*n1];
        vector new_cn = (mesh->nodes[*n0] + mesh->nodes[*n1] + center) / 3.0;

        vector d0xd1 = cross(d0, d1);

        scalar norm_d0xd1 = norm(d0xd1);
        area += norm_d0xd1 * 0.5;
        normal += d0xd1;
        new_center += new_cn * norm_d0xd1 * 0.5;
    }
    {   // Last edge has different logic, avoid if with repetition
        uint* n0 = face_end - 1;
        uint* n1 = face_start;
        vector d0 = center - mesh->nodes[*n0];
        vector d1 = center - mesh->nodes[*n1];
        vector new_cn = (mesh->nodes[*n0] + mesh->nodes[*n1] + center) / 3.0;

        vector d0xd1 = cross(d0, d1);

        scalar norm_d0xd1 = norm(d0xd1);
        area += norm_d0xd1 * 0.5;
        normal += d0xd1;
        new_center += new_cn * norm_d0xd1 * 0.5;
    }
    mesh->face_centers[i] = new_center / area;
    mesh->face_normals[i] = normal / norm(normal);
    mesh->face_areas[i] = area;
}

/*
    Kernel to compute cuMeshBase cell information
*/
__global__ void cuMeshComputeCells(
    cuMeshBase* mesh
) {
    // i is the cell id in this case
    int i = cudaGlobalId;
    if (i >= mesh->n_cells) return;

    // Get cell start and end node reference
    uint* cell_start = cuMeshGetCellStart(mesh, i);
    uint* cell_end = cuMeshGetCellEnd(mesh, i);
    uint cell_size = cuMeshGetCellSize(mesh, i);

    // Compute cell center
    vector center = make_vector((scalar)0, 0, 0);
    scalar weight = 0;
    for (uint* f = cell_start; f != cell_end; ++f) {
        center += mesh->face_centers[*f] * mesh->face_areas[*f];
        weight += mesh->face_areas[*f];
    }
    center /= weight;

    vector new_center = make_vector((scalar)0, 0, 0);
    weight = 0;
    for (uint* f = cell_start; f != cell_end; ++f) {
        vector d = mesh->face_centers[*f] - center;
        scalar w = norm(d) * mesh->face_areas[*f];
        vector pc = center + d * 0.75;
        new_center += pc * w;
        weight += w;
    }
    new_center /= weight;
    mesh->cell_centers[i] = new_center;

    // Compute cell volume
    scalar volume = 0;
    for (uint* f = cell_start; f != cell_end; ++f) {
        vector center_to_face = mesh->face_centers[*f] - new_center;
        vector normal = outer_normal(mesh->face_normals[*f], center_to_face);
        volume += dot(center_to_face, normal) * mesh->face_areas[*f];
    }
    mesh->cell_volumes[i] = volume / 3.0;
}

/*
    Kernel to compute cuMeshBase ghost cell information
*/
__global__ void cuMeshComputeGhostCells(
    cuMeshBase* mesh
) {
    int thread_id = cudaGlobalId;
    if (thread_id >= mesh->n_ghosts) return;

    int i = mesh->ghost_cell_ids[thread_id];

    // Get the other cell
    uint other_cell = *cuMeshGetCellConnect(mesh, i);
    vector other_center = mesh->cell_centers[other_cell];

    // Get the one connecting face
    uint f = cuMeshGetGhostCellFace(mesh, i);
    vector fc = mesh->face_centers[f];
    vector n = mesh->face_normals[f];
    scalar a = mesh->face_areas[f];

    // Compute the cell center
    scalar dx = norm(fc - other_center);
    vector center = fc + outer_normal(n, fc - other_center) * dx;
    mesh->cell_centers[i] = center;

    // Compute the cell volume
    mesh->cell_volumes[i] = a * 2 * dx;
}

/*
    Kernel to compute gradient matrices
*/
__global__ void cuMeshComputeCellMatrices(
    cuMeshBase* mesh
) {
    int i = cudaGlobalId;
    if (i >= mesh->n_cells) return;


    // Compute ATA matrix
    tensor A;
    reset(A);

    // loop over connected cells
    
    uint* j = cuMeshGetCellConnect(mesh, i);
    uint size = cuMeshGetCellSize(mesh, i);
    for (uint _ignore = 0; _ignore < size; ++_ignore) {

        vector d = mesh->cell_centers[*j] - mesh->cell_centers[i];
        scalar w = 1.0 / norm(d);
        A += outer(d, d) * w * w;

        j++;
    }

    // We need to store its inverse
    tensor M;
    M.u.x = A.v.y * A.w.z - A.v.z * A.w.y;
    M.u.y = A.w.y * A.u.z - A.w.z * A.u.y;
    M.u.z = A.u.y * A.v.z - A.u.z * A.v.y;

    M.v.x = A.w.x * A.v.z - A.w.z * A.v.x;
    M.v.y = A.u.x * A.w.z - A.u.z * A.w.x;
    M.v.z = A.v.x * A.u.z - A.v.z * A.u.x;

    M.w.x = A.v.x * A.w.y - A.v.y * A.w.x;
    M.w.y = A.w.x * A.u.y - A.w.y * A.u.x;
    M.w.z = A.u.x * A.v.y - A.u.y * A.v.x;

    scalar det = A.u.x * M.u.x + A.v.x * M.u.y + A.w.x * M.u.z;

    mesh->cell_matrices[i] = M / det;
}



/*
    Allocate a cuMesh object
*/
void cuMeshAllocateCalcsGpu(
    cuMesh* mesh
) {
    // Allocate extra fields for metrics calculations
    uint n_nodes = mesh->_cpu->n_nodes;
    uint n_faces = mesh->_cpu->n_faces;
    uint n_cells = mesh->_cpu->n_cells;
    uint n_ghosts = mesh->_cpu->n_ghosts;
    uint n_regions = mesh->_cpu->n_regions;
    uint n_faces_nodes = mesh->_cpu->face_nodes_starts[n_faces];
    uint n_cells_faces = mesh->_cpu->cell_faces_starts[n_cells + n_ghosts];

    // Allocate cpu
    mesh->_cpu->cell_volumes = cpuAlloc(scalar, n_cells + n_ghosts);
    mesh->_cpu->cell_centers = cpuAlloc(vector, n_cells + n_ghosts);
    mesh->_cpu->cell_matrices = cpuAlloc(tensor, n_cells);

    mesh->_cpu->face_areas = cpuAlloc(scalar, n_faces);
    mesh->_cpu->face_normals = cpuAlloc(vector, n_faces);
    mesh->_cpu->face_centers = cpuAlloc(vector, n_faces);

    mesh->_cpu->allocated = 1;

    // Allocate host gpu
    gpuAlloc(mesh->_hgpu->cell_faces, uint, n_cells_faces + n_ghosts);
    gpuAlloc(mesh->_hgpu->cell_connects, uint, n_cells_faces + n_ghosts);
    gpuAlloc(mesh->_hgpu->cell_faces_starts, uint, n_cells + n_ghosts + 1);
    gpuAlloc(mesh->_hgpu->cell_volumes, scalar, n_cells + n_ghosts);
    gpuAlloc(mesh->_hgpu->cell_centers, vector, n_cells + n_ghosts);
    gpuAlloc(mesh->_hgpu->cell_matrices, tensor, n_cells);

    gpuAlloc(mesh->_hgpu->face_nodes, uint, n_faces_nodes);
    gpuAlloc(mesh->_hgpu->face_nodes_starts, uint, n_faces + 1);
    gpuAlloc(mesh->_hgpu->face_areas, scalar, n_faces);
    gpuAlloc(mesh->_hgpu->face_normals, vector, n_faces);
    gpuAlloc(mesh->_hgpu->face_centers, vector, n_faces);

    gpuAlloc(mesh->_hgpu->nodes, vector, n_nodes);

    gpuAlloc(mesh->_hgpu->ghost_cell_ids, uint, n_ghosts);
    gpuAlloc(mesh->_hgpu->ghost_cell_starts, uint, n_regions + 1);
    mesh->_hgpu->allocated = 1;

    mesh->_hgpu->n_cells = n_cells;
    mesh->_hgpu->n_faces = n_faces;
    mesh->_hgpu->n_nodes = n_nodes;
    mesh->_hgpu->n_ghosts = n_ghosts;
    mesh->_hgpu->n_regions = n_regions;

    // Pass the new values to the gpu
    cudaMemcpy(mesh->_gpu, mesh->_hgpu, sizeof(cuMeshBase), cudaMemcpyHostToDevice);
}


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
) {
    // Allocate cpu

    mesh->_cpu->cell_faces = cpuAlloc(uint, n_cells_faces + n_ghosts);
    mesh->_cpu->cell_connects = cpuAlloc(uint, n_cells_faces + n_ghosts);
    mesh->_cpu->cell_faces_starts = cpuAlloc(uint, n_cells + n_ghosts + 1);

    mesh->_cpu->face_nodes = cpuAlloc(uint, n_faces_nodes);
    mesh->_cpu->face_nodes_starts = cpuAlloc(uint, n_faces + 1);

    mesh->_cpu->nodes = cpuAlloc(vector, n_nodes);

    mesh->_cpu->ghost_cell_ids = cpuAlloc(uint, n_ghosts);
    mesh->_cpu->ghost_cell_starts = cpuAlloc(uint, n_regions + 1);

    mesh->_cpu->n_cells = n_cells;
    mesh->_cpu->n_faces = n_faces;
    mesh->_cpu->n_nodes = n_nodes;
    mesh->_cpu->n_ghosts = n_ghosts;
    mesh->_cpu->n_regions = n_regions;
    mesh->_cpu->face_nodes_starts[n_faces] = n_faces_nodes;
    mesh->_cpu->cell_faces_starts[n_cells + n_ghosts] = n_cells_faces;

    mesh->region_names = cpuAlloc(char*, n_regions);
    mesh->region_names_alloc = n_regions;
    mesh->region_names_size = 0;

    // Allocate calculations and gpu
    cuMeshAllocateCalcsGpu(mesh);
}


void cuMeshBaseFreeCpu(
    cuMeshBase* mesh
) {
    assert(mesh->type == _CPU);
#ifdef DEBUG_BUILD
    printf("Free cpu mesh, allocated = %d\n", mesh->allocated);
#endif
    if (mesh->allocated) {
        free(mesh->cell_faces);
        free(mesh->cell_connects);
        free(mesh->cell_faces_starts);
        free(mesh->cell_volumes);
        free(mesh->cell_centers);
        free(mesh->cell_matrices);

        free(mesh->face_nodes);
        free(mesh->face_nodes_starts);
        free(mesh->face_areas);
        free(mesh->face_normals);
        free(mesh->face_centers);

        free(mesh->nodes);

        free(mesh->ghost_cell_ids);
        free(mesh->ghost_cell_starts);
    }
    free(mesh);
}

void cuMeshBaseFreeGpu(
    cuMeshBase* mesh
) {
    assert(mesh->type == _HGPU);
#ifdef DEBUG_BUILD
    printf("Free hgpu mesh, allocated = %d\n", mesh->allocated);
#endif
    if (mesh->allocated) {
        cudaExec(cudaFree(mesh->cell_faces));
        cudaExec(cudaFree(mesh->cell_connects));
        cudaExec(cudaFree(mesh->cell_faces_starts));
        cudaExec(cudaFree(mesh->cell_volumes));
        cudaExec(cudaFree(mesh->cell_centers));
        cudaExec(cudaFree(mesh->cell_matrices));

        cudaExec(cudaFree(mesh->face_nodes));
        cudaExec(cudaFree(mesh->face_nodes_starts));
        cudaExec(cudaFree(mesh->face_areas));
        cudaExec(cudaFree(mesh->face_normals));
        cudaExec(cudaFree(mesh->face_centers));

        cudaExec(cudaFree(mesh->nodes));

        cudaExec(cudaFree(mesh->ghost_cell_ids));
        cudaExec(cudaFree(mesh->ghost_cell_starts));
    }
    free(mesh);
}


cuMesh* cuMeshNew() {
    cuMesh* mesh = (cuMesh*)malloc(sizeof(cuMesh));
    mesh->_cpu = (cuMeshBase*)malloc(sizeof(cuMeshBase));
    mesh->_cpu->type = _CPU;
    mesh->_cpu->allocated = 0;
    mesh->_hgpu = (cuMeshBase*)malloc(sizeof(cuMeshBase));
    mesh->_hgpu->type = _HGPU;
    mesh->_hgpu->allocated = 0;
    gpuAlloc(mesh->_gpu, cuMeshBase, 1);
#ifdef DEBUG_BUILD
    printf("Created new cuMesh at adress %p\n", mesh);
#endif
    mesh->region_names = (char**)malloc(sizeof(char*) * 2);
    mesh->region_names_alloc = 2;
    mesh->region_names_size = 0;
    return mesh;
}

void cuMeshFree(cuMesh* mesh) {
#ifdef DEBUG_BUILD
    printf("Freed cuMesh at adress % p\n", mesh);
#endif
    cuMeshBaseFreeCpu(mesh->_cpu);
    cuMeshBaseFreeGpu(mesh->_hgpu);
    cudaExec(cudaFree(mesh->_gpu));
    for (int i = 0; i < mesh->region_names_size; ++i) free(mesh->region_names[i]);
    free(mesh->region_names);
    free(mesh);
}


void cuMeshPassCpuToGpu(
    cuMesh* mesh
) {
    cuMeshBase* cpuMesh = mesh->_cpu;
    cuMeshBase* gpuMesh = mesh->_hgpu;
    cudaMemcpyKind memdir = cudaMemcpyHostToDevice;

    uint n_cells = cpuMesh->n_cells;
    uint n_cells_and_ghosts = cpuMesh->n_cells + cpuMesh->n_ghosts;
    uint n_cells_faces = cpuMesh->cell_faces_starts[n_cells_and_ghosts];

    uint n_face_nodes = cpuMesh->face_nodes_starts[cpuMesh->n_faces];
    uint n_faces = cpuMesh->n_faces;
    uint n_nodes = cpuMesh->n_nodes;
    uint n_ghosts = cpuMesh->n_ghosts;
    uint n_regions = cpuMesh->n_regions;

#ifdef DEBUG_BUILD
    printf("Passing cuMesh data from cpu to gpu\n");
    printf("- n_cells_faces = %d\n", n_cells_faces);
    printf("- n_cells_and_ghosts = %d\n", n_cells_and_ghosts);
    printf("- n_face_nodes = %d\n", n_face_nodes);
    printf("- n_faces = %d\n", n_faces);
    printf("- n_nodes = %d\n", n_nodes);
    printf("- n_ghosts = %d\n", n_ghosts);
    printf("- n_regions = %d\n", n_regions);
#endif

    cudaSync;

    cudaExec(cudaMemcpy(gpuMesh->cell_faces, cpuMesh->cell_faces, n_cells_faces * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->cell_connects, cpuMesh->cell_connects, n_cells_faces * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->cell_faces_starts, cpuMesh->cell_faces_starts, (n_cells_and_ghosts + 1) * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->cell_volumes, cpuMesh->cell_volumes, n_cells_and_ghosts * sizeof(scalar), memdir));
    cudaExec(cudaMemcpy(gpuMesh->cell_centers, cpuMesh->cell_centers, n_cells_and_ghosts * sizeof(vector), memdir));
    cudaExec(cudaMemcpy(gpuMesh->cell_matrices, cpuMesh->cell_matrices, n_cells * sizeof(tensor), memdir));

    cudaExec(cudaMemcpy(gpuMesh->face_nodes, cpuMesh->face_nodes, n_face_nodes * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->face_nodes_starts, cpuMesh->face_nodes_starts, (n_faces + 1) * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->face_areas, cpuMesh->face_areas, n_faces * sizeof(scalar), memdir));
    cudaExec(cudaMemcpy(gpuMesh->face_normals, cpuMesh->face_normals, n_faces * sizeof(vector), memdir));
    cudaExec(cudaMemcpy(gpuMesh->face_centers, cpuMesh->face_centers, n_faces * sizeof(vector), memdir));

    cudaExec(cudaMemcpy(gpuMesh->nodes, cpuMesh->nodes, n_nodes * sizeof(vector), memdir));

    cudaExec(cudaMemcpy(gpuMesh->ghost_cell_ids, cpuMesh->ghost_cell_ids, n_ghosts * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(gpuMesh->ghost_cell_starts, cpuMesh->ghost_cell_starts, (n_regions + 1) * sizeof(uint), memdir));

    cudaSync;
}



void cuMeshPassGpuToCpu(
    cuMesh* mesh
) {
    cuMeshBase* cpuMesh = mesh->_cpu;
    cuMeshBase* gpuMesh = mesh->_hgpu;
    cudaMemcpyKind memdir = cudaMemcpyDeviceToHost;

    uint n_cells = cpuMesh->n_cells;
    uint n_cells_and_ghosts = cpuMesh->n_cells + cpuMesh->n_ghosts;
    uint n_cells_faces = cpuMesh->cell_faces_starts[n_cells_and_ghosts];

    uint n_face_nodes = cpuMesh->face_nodes_starts[cpuMesh->n_faces];
    uint n_faces = cpuMesh->n_faces;
    uint n_nodes = cpuMesh->n_nodes;
    uint n_ghosts = cpuMesh->n_ghosts;
    uint n_regions = cpuMesh->n_regions;

#ifdef DEBUG_BUILD
    printf("Passing cuMesh data from gpu to cpu\n");
    printf("- n_cells_faces = %d\n", n_cells_faces);
    printf("- n_cells_and_ghosts = %d\n", n_cells_and_ghosts);
    printf("- n_face_nodes = %d\n", n_face_nodes);
    printf("- n_faces = %d\n", n_faces);
    printf("- n_nodes = %d\n", n_nodes);
    printf("- n_ghosts = %d\n", n_ghosts);
    printf("- n_regions = %d\n", n_regions);
#endif

    cudaSync;

    cudaExec(cudaMemcpy(cpuMesh->cell_faces, gpuMesh->cell_faces, n_cells_faces * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->cell_connects, gpuMesh->cell_connects, n_cells_faces * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->cell_faces_starts, gpuMesh->cell_faces_starts, (n_cells_and_ghosts + 1) * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->cell_volumes, gpuMesh->cell_volumes, n_cells_and_ghosts * sizeof(scalar), memdir));
    cudaExec(cudaMemcpy(cpuMesh->cell_centers, gpuMesh->cell_centers, n_cells_and_ghosts * sizeof(vector), memdir));
    cudaExec(cudaMemcpy(cpuMesh->cell_matrices, gpuMesh->cell_matrices, n_cells * sizeof(tensor), memdir));

    cudaExec(cudaMemcpy(cpuMesh->face_nodes, gpuMesh->face_nodes, n_face_nodes * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->face_nodes_starts, gpuMesh->face_nodes_starts, (n_faces + 1) * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->face_areas, gpuMesh->face_areas, n_faces * sizeof(scalar), memdir));
    cudaExec(cudaMemcpy(cpuMesh->face_normals, gpuMesh->face_normals, n_faces * sizeof(vector), memdir));
    cudaExec(cudaMemcpy(cpuMesh->face_centers, gpuMesh->face_centers, n_faces * sizeof(vector), memdir));

    cudaExec(cudaMemcpy(cpuMesh->nodes, gpuMesh->nodes, n_nodes * sizeof(vector), memdir));

    cudaExec(cudaMemcpy(cpuMesh->ghost_cell_ids, gpuMesh->ghost_cell_ids, n_ghosts * sizeof(uint), memdir));
    cudaExec(cudaMemcpy(cpuMesh->ghost_cell_starts, gpuMesh->ghost_cell_starts, (n_regions + 1) * sizeof(uint), memdir));

    cudaSync;

}


int cuMeshAddRegionName(
    cuMesh* mesh,
    const char* name
) {
    if (mesh->region_names_size >= mesh->region_names_alloc) {
        size_t new_size = mesh->region_names_size * 2;
        mesh->region_names = (char**)realloc(mesh->region_names, sizeof(char*) * new_size);
        if (mesh->region_names == NULL) return 1;
        mesh->region_names_alloc = new_size;
    }

    size_t i = mesh->region_names_size;
    mesh->region_names[i] = cpuAlloc(char, strlen(name) + 1);
    strcpy(mesh->region_names[i], name);
    mesh->region_names_size++;

    return 0;
}


void cuMeshMake1D(
    cuMesh* mesh,
    size_t n,
    scalar x0,
    scalar x1
) {
    // Allocate the memory
    cuMeshAllocate(
        mesh,        // mesh pointer
        n,              // n cells
        2 * n,          // n cells faces
        n + 1,          // n faces
        4 * (n + 1),    // n faces nodes
        4 * (n + 1),    // n nodes
        2,              // n ghosts
        2               // n regions
    );

    cuMeshBase* cpuMesh = mesh->_cpu;
    //cuMeshBase* gpuMesh = mesh->_hgpu;

    // Fill the cpu mesh
    // Make ends
    cpuMesh->cell_faces_starts[cpuMesh->n_cells + cpuMesh->n_ghosts] = 2 * n + 2;
    cpuMesh->face_nodes_starts[cpuMesh->n_faces] = 4 * (n + 1);

    // Fill the nodes
    for (size_t i = 0; i < (n + 1); ++i) {
        scalar x = ((scalar)i) / ((scalar)n);
        cpuMesh->nodes[4 * i + 0] = make_vector(x, 0, 0);
        cpuMesh->nodes[4 * i + 1] = make_vector(x, 1, 0);
        cpuMesh->nodes[4 * i + 2] = make_vector(x, 1, 1);
        cpuMesh->nodes[4 * i + 3] = make_vector(x, 0, 1);
    }

    // Fill the faces
    for (size_t i = 0; i < (n + 1); ++i) {
        cpuMesh->face_nodes[4 * i + 0] = 4 * i;
        cpuMesh->face_nodes[4 * i + 1] = 4 * i + 1;
        cpuMesh->face_nodes[4 * i + 2] = 4 * i + 2;
        cpuMesh->face_nodes[4 * i + 3] = 4 * i + 3;
        cpuMesh->face_nodes_starts[i] = 4 * i;
    }

    // Fill the cells faces
    for (size_t i = 0; i < n; ++i) {
        cpuMesh->cell_faces[2 * i] = i;
        cpuMesh->cell_faces[2 * i + 1] = i + 1;
        cpuMesh->cell_faces_starts[i] = 2 * i;
    }

    // Fill the cells connectivity
    for (size_t i = 1; i < (n - 1); ++i) {
        cpuMesh->cell_connects[2 * i] = i - 1;
        cpuMesh->cell_connects[2 * i + 1] = i + 1;
    }

    // First cell
    cpuMesh->cell_connects[2 * 0] = n;
    cpuMesh->cell_connects[2 * 0 + 1] = 1;
    // Last cell
    cpuMesh->cell_connects[2 * (n - 1)] = n - 2;
    cpuMesh->cell_connects[2 * (n - 1) + 1] = n + 1;


    // Fill the ghost cells connectivity
    cpuMesh->cell_faces[2 * n] = 0; // first face
    cpuMesh->cell_faces[2 * n + 1] = n; // last face
    cpuMesh->cell_faces_starts[n] = 2 * n;
    cpuMesh->cell_faces_starts[n + 1] = 2 * n + 1;
    cpuMesh->cell_connects[2 * n] = 0;
    cpuMesh->cell_connects[2 * n + 1] = n - 1;

    // Ghost cells and regions
    cpuMesh->ghost_cell_ids[0] = n;
    cpuMesh->ghost_cell_ids[1] = n + 1;
    cpuMesh->ghost_cell_starts[0] = 0;  // region 0
    cpuMesh->ghost_cell_starts[1] = 1;  // region 1
    cpuMesh->ghost_cell_starts[2] = 2;  // end of array


    // Pass data from the cpu mesh to the gpu mesh
    cuMeshPassCpuToGpu(mesh);

    // Compute faces and cells data
    cuMeshComputeFaces << <cuda_size(cpuMesh->n_faces, 32), 32 >> > (mesh->_gpu);
    cuMeshComputeCells << < cuda_size(cpuMesh->n_cells, 32), 32 >> > (mesh->_gpu);
    cuMeshComputeGhostCells << < cuda_size(cpuMesh->n_ghosts, 32), 32 >> > (mesh->_gpu);

    // Pass back data from gpu to cpu
    cuMeshPassGpuToCpu(mesh);


    // Completed mesh generation!
}




