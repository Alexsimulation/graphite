#include "meshio.cuh"






void cuMeshPrintInfo(cuMesh* mesh) {
    cuMeshBase* cpuMesh = mesh->_cpu;

    printf("Printing info on cuMesh object\n");
    printf("- n_cells = %zu\n", cpuMesh->n_cells);
    printf("- n_faces = %zu\n", cpuMesh->n_faces);
    printf("- n_nodes = %zu\n", cpuMesh->n_nodes);
    printf("- n_ghosts = %zu\n", cpuMesh->n_ghosts);
    printf("- n_regions = %zu\n", cpuMesh->n_regions);
    printf("- n_cells_faces = %u\n", cpuMesh->cell_faces_starts[cpuMesh->n_cells + cpuMesh->n_ghosts]);
    printf("- n_face_nodes = %u\n", cpuMesh->face_nodes_starts[cpuMesh->n_faces]);
    printf("- cpu mesh allocated = %s\n", cpuMesh->allocated ? "True" : "False");
    printf("- gpu mesh allocated = %s\n", mesh->_hgpu->allocated ? "True" : "False");
}


/*
    Print a mesh for debug purposes
*/
void cuMeshPrintDebug(cuMesh* mesh) {
    cuMeshBase* cpuMesh = mesh->_cpu;

    printf("Printing info on cuMesh object\n");
    printf("- n_cells = %zu\n", cpuMesh->n_cells);
    printf("- n_faces = %zu\n", cpuMesh->n_faces);
    printf("- n_nodes = %zu\n", cpuMesh->n_nodes);
    printf("- n_ghosts = %zu\n", cpuMesh->n_ghosts);
    printf("- n_regions = %zu\n", cpuMesh->n_regions);
    printf("- n_cells_faces = %u\n", cpuMesh->cell_faces_starts[cpuMesh->n_cells + cpuMesh->n_ghosts]);
    printf("- n_face_nodes = %u\n", cpuMesh->face_nodes_starts[cpuMesh->n_faces]);


    printf("Printing connectivity info\n");
    printf("  Printing cell faces\n");

    for (uint i = 0; i < (cpuMesh->n_cells + cpuMesh->n_ghosts); ++i) {
        printf("    Cell %d has faces range (%6d %6d) : ", i, cpuMesh->cell_faces_starts[i], cpuMesh->cell_faces_starts[i + 1]);

        for (uint j = cpuMesh->cell_faces_starts[i]; j < cpuMesh->cell_faces_starts[i + 1]; ++j) {
            printf("%6u ", cpuMesh->cell_faces[j]);
        }

        printf("\n");
    }

    printf("  Printing cell connects\n");

    for (uint i = 0; i < (cpuMesh->n_cells + cpuMesh->n_ghosts); ++i) {
        printf("    Cell %d has connect range (%6d %6d) : ", i, cpuMesh->cell_faces_starts[i], cpuMesh->cell_faces_starts[i + 1]);

        for (uint j = cpuMesh->cell_faces_starts[i]; j < cpuMesh->cell_faces_starts[i + 1]; ++j) {
            printf("%6u ", cpuMesh->cell_connects[j]);
        }

        printf("\n");
    }

    printf("  Printing faces nodes\n");

    for (uint i = 0; i < (cpuMesh->n_faces); ++i) {
        printf("    Face %d has nodes range (%6d %6d) : ", i, cpuMesh->face_nodes_starts[i], cpuMesh->face_nodes_starts[i + 1]);

        for (uint j = cpuMesh->face_nodes_starts[i]; j < cpuMesh->face_nodes_starts[i + 1]; ++j) {
            printf("%6u ", cpuMesh->face_nodes[j]);
        }

        printf("\n");
    }

    printf("Printing nodes data\n");

    for (uint i = 0; i < (cpuMesh->n_nodes); ++i) {
        printf("    %6u -> %.6f %.6f %.6f\n", i, cpuMesh->nodes[i].x, cpuMesh->nodes[i].y, cpuMesh->nodes[i].z);
    }

    printf("Printing cell data\n");

    printf("  Printing cell volumes\n");

    for (uint i = 0; i < (cpuMesh->n_cells + cpuMesh->n_ghosts); ++i) {
        printf("    %6u -> %.6f\n", i, cpuMesh->cell_volumes[i]);
    }

    printf("  Printing cell centers\n");

    for (uint i = 0; i < (cpuMesh->n_cells + cpuMesh->n_ghosts); ++i) {
        printf("    %6u -> %.6f %.6f %.6f\n", i, cpuMesh->cell_centers[i].x, cpuMesh->cell_centers[i].y, cpuMesh->cell_centers[i].z);
    }

    printf("Printing face data\n");

    printf("  Printing face areas\n");

    for (uint i = 0; i < (cpuMesh->n_faces); ++i) {
        printf("    %6u -> %.6f\n", i, cpuMesh->face_areas[i]);
    }

    printf("  Printing face centers\n");

    for (uint i = 0; i < (cpuMesh->n_faces); ++i) {
        printf("    %6u -> %.6f %.6f %.6f\n", i, cpuMesh->face_centers[i].x, cpuMesh->face_centers[i].y, cpuMesh->face_centers[i].z);
    }

    printf("  Printing face normals\n");

    for (uint i = 0; i < (cpuMesh->n_faces); ++i) {
        printf("    %6u -> %.6f %.6f %.6f\n", i, cpuMesh->face_normals[i].x, cpuMesh->face_normals[i].y, cpuMesh->face_normals[i].z);
    }

    printf("  Printing region names\n");

    for (uint i = 0; i < mesh->region_names_size; ++i) {
        printf("    %6u -> %s\n", i, mesh->region_names[i]);
    }

}




/*
    Read a cuMesh file

    File is defined as:

    nodes <# of nodes>
        x_0 y_0 z_0
        x_1 y_1 z_1
        ...
        x_n y_n z_n

    faces <# of faces> <# of faces nodes>
        face_size_0 n_00 n_01 n_02 ... n_0a
        face_size_1 n_10 n_11 n_12 ... n_1b
        ...
        face_size_v n_v0 n_v1 n_v2 ... n_vc

    cells <# of cells> <# of cells faces>
        // cell size, then cell's faces, then cell's connected cells
        cell_size_0   f_00 f_01 ... f_0a   c_00 c_01 ... c_0a
        cell_size_1   f_10 f_11 ... f_1b   c_10 c_11 ... c_1b
        ...
        cell_size_v   f_v0 f_v1 ... f_vc   c_v0 c_v1 ... c_vc

    regions <# of regions> <# of total region cells>
    region <name_0> <# of cells>
        cell_id_00
        cell_id_01
        ...
        cell_id_0n
    region <name_1> <# of cells>
        cell_id_10
        cell_id_11
        ...
        cell_id_1n
    // repeat for all regions

*/
cuMesh* cuMeshFromFile(const char* filename) {
    FILE* file = fopen(filename, "rb");

    if (file == NULL) {
        printf("Error reading mesh file %s\n", filename);
        exit(1);
    }

    cuMesh* mesh = cuMeshNew();
    cuMeshBase* m = mesh->_cpu;


    //char buff[200];

    // Read nodes
    uint n_nodes;
    fread(&n_nodes, sizeof(uint), 1, file);
    m->n_nodes = n_nodes;
    m->nodes = cpuAlloc(vector, m->n_nodes);
    for (int i = 0; i < m->n_nodes; ++i) {
        scalar x[3];
        fread(x, sizeof(x), 1, file);
        m->nodes[i] = make_vector(x[0], x[1], x[2]);
    }

    // Read faces
    uint n_faces, n_faces_tags;
    fread(&n_faces, sizeof(uint), 1, file);
    fread(&n_faces_tags, sizeof(uint), 1, file);
    m->n_faces = n_faces;
    m->face_nodes = cpuAlloc(uint, n_faces_tags);
    m->face_nodes_starts = cpuAlloc(uint, m->n_faces + 1);
    m->face_nodes_starts[m->n_faces] = n_faces_tags;

    uint offset = 0;
    for (int i = 0; i < m->n_faces; ++i) {
        uint f_size;
        fread(&f_size, sizeof(uint), 1, file);
        m->face_nodes_starts[i] = offset;

        for (int j = 0; j < f_size; ++j) {
            uint node;
            fread(&node, sizeof(uint), 1, file);
            m->face_nodes[offset + j] = node;
        }

        offset += f_size;
    }

    // Read all cells and ghosts
    uint n_cells_and_ghosts, n_total_cells_tags;
    fread(&n_cells_and_ghosts, sizeof(uint), 1, file);
    fread(&n_total_cells_tags, sizeof(uint), 1, file);

    m->cell_faces = cpuAlloc(uint, n_total_cells_tags);
    m->cell_connects = cpuAlloc(uint, n_total_cells_tags);
    m->cell_faces_starts = cpuAlloc(uint, n_cells_and_ghosts + 1);
    m->cell_faces_starts[n_cells_and_ghosts] = n_total_cells_tags;

    offset = 0;
    for (int i = 0; i < n_cells_and_ghosts; ++i) {
        uint c_size;
        fread(&c_size, sizeof(uint), 1, file);
        m->cell_faces_starts[i] = offset;

        for (int j = 0; j < c_size; ++j) {
            uint face;
            fread(&face, sizeof(uint), 1, file);
            m->cell_faces[offset + j] = face;
        }
        for (int j = 0; j < c_size; ++j) {
            uint connect_cell;
            fread(&connect_cell, sizeof(uint), 1, file);
            m->cell_connects[offset + j] = connect_cell;
        }
        offset += c_size;
    }

    // Read regions and ghost cells
    uint n_regions, n_ghosts;
    fread(&n_regions, sizeof(uint), 1, file);
    fread(&n_ghosts, sizeof(uint), 1, file);

    m->n_regions = n_regions;
    m->n_ghosts = n_ghosts;
    m->ghost_cell_ids = cpuAlloc(uint, m->n_ghosts);
    m->ghost_cell_starts = cpuAlloc(uint, m->n_regions + 1);
    m->n_cells = n_cells_and_ghosts - m->n_ghosts;
    offset = 0;
    for (int region = 0; region < m->n_regions; ++region) {
        char name[128];
        int n_ghosts;
        fread(name, sizeof(char), 128, file);
        fread(&n_ghosts, sizeof(uint), 1, file);

        m->ghost_cell_starts[region] = offset;
        (void)cuMeshAddRegionName(mesh, name);

        for (int i = 0; i < n_ghosts; ++i) {
            uint ghost_cell;
            fread(&ghost_cell, sizeof(uint), 1, file);
            m->ghost_cell_ids[offset] = ghost_cell;
            offset++;
        }
    }
    m->ghost_cell_starts[m->n_regions] = offset;

    // File reader finished
    fclose(file);

    // Allocate extra info
    cuMeshAllocateCalcsGpu(mesh);

    // Pass data from the cpu mesh to the gpu mesh
    cuMeshPassCpuToGpu(mesh);

    // Compute faces and cells data
    cuMeshComputeFaces <<< cuda_size(mesh->_cpu->n_faces, 32), 32 >>> (mesh->_gpu);
    cuMeshComputeCells <<< cuda_size(mesh->_cpu->n_cells, 32), 32 >>> (mesh->_gpu);
    cuMeshComputeGhostCells <<< cuda_size(mesh->_cpu->n_ghosts, 32), 32 >>> (mesh->_gpu);
    cuMeshComputeCellMatrices <<< cuda_size(mesh->_cpu->n_cells, 32), 32 >>> (mesh->_gpu);

    // Pass back data from gpu to cpu
    cuMeshPassGpuToCpu(mesh);

    return mesh;
}



// Size in number of nodes
uint su2KindToSize(uint kind) {
    switch (kind) {
    case 5: // triangle
        return 3;
    case 9: // quad
        return 4;
    case 10:    // tetra
        return 4;
    case 12:    // hexa
        return 8;
    case 13:    // wedge
        return 6;
    case 14:    // pyramid
        return 5;
    }
    return 1;
}

int su2KindToFaces(uint kind, int* face_nodes, int* n_faces) {
    // face_nodes has size at least 64
    switch (kind) {
    case 5:     // triangle
        *n_faces = 1;
        // face 1, 0 2 1
        face_nodes[0] = 0;
        face_nodes[1] = 1;
        face_nodes[2] = 2;
        face_nodes[3] = -1;

        return 0;
    case 9:     // quad
        *n_faces = 1;

        // face 1, 0 1 2 3
        face_nodes[0] = 0;
        face_nodes[1] = 1;
        face_nodes[2] = 2;
        face_nodes[3] = 3;

        return 0;
    case 10:    // tetra
        *n_faces = 4;

        // face 1, 0 2 1
        face_nodes[0] = 0;
        face_nodes[1] = 2;
        face_nodes[2] = 1;
        face_nodes[3] = -1;

        // face 2, 0 1 3
        face_nodes[4] = 0;
        face_nodes[5] = 1;
        face_nodes[6] = 3;
        face_nodes[7] = -1;

        // face 3, 1 2 3
        face_nodes[8] = 1;
        face_nodes[9] = 2;
        face_nodes[10] = 3;
        face_nodes[11] = -1;

        // face 4, 0 3 2
        face_nodes[12] = 0;
        face_nodes[13] = 3;
        face_nodes[14] = 2;
        face_nodes[15] = -1;

        return 0;
    case 12:
        *n_faces = 6;

        // face 1, 0 1 5 4
        face_nodes[0] = 0;
        face_nodes[1] = 1;
        face_nodes[2] = 5;
        face_nodes[3] = 4;

        // face 2, 1 2 6 5
        face_nodes[4] = 1;
        face_nodes[5] = 2;
        face_nodes[6] = 6;
        face_nodes[7] = 5;

        // face 3, 2 3 7 6
        face_nodes[8] = 2;
        face_nodes[9] = 3;
        face_nodes[10] = 7;
        face_nodes[11] = 6;

        // face 4, 3 0 4 7
        face_nodes[12] = 3;
        face_nodes[13] = 0;
        face_nodes[14] = 4;
        face_nodes[15] = 7;

        // face 5, 3 2 1 0
        face_nodes[16] = 3;
        face_nodes[17] = 2;
        face_nodes[18] = 1;
        face_nodes[19] = 0;

        // face 6, 4 5 6 7
        face_nodes[20] = 4;
        face_nodes[21] = 5;
        face_nodes[22] = 6;
        face_nodes[23] = 7;

        return 0;
    case 13:    // wedge
        *n_faces = 5;

        // face 1, 0 1 2
        face_nodes[0] = 0;
        face_nodes[1] = 1;
        face_nodes[2] = 2;
        face_nodes[3] = -1;

        // face 2, 3 5 4
        face_nodes[4] = 3;
        face_nodes[5] = 5;
        face_nodes[6] = 4;
        face_nodes[7] = -1;

        // face 3, 0 2 5 3
        face_nodes[8] = 0;
        face_nodes[9] = 2;
        face_nodes[10] = 5;
        face_nodes[11] = 3;

        // face 4, 1 4 5 2
        face_nodes[12] = 1;
        face_nodes[13] = 4;
        face_nodes[14] = 5;
        face_nodes[15] = 2;

        // face 5, 1 0 3 4
        face_nodes[16] = 1;
        face_nodes[17] = 0;
        face_nodes[18] = 3;
        face_nodes[19] = 4;

        return 0;
    case 14:    // pyramid
        *n_faces = 5;

        // face 1, 0 1 2 3
        face_nodes[0] = 0;
        face_nodes[1] = 1;
        face_nodes[2] = 2;
        face_nodes[3] = 3;

        // face 2, 0 4 3
        face_nodes[4] = 0;
        face_nodes[5] = 4;
        face_nodes[6] = 3;
        face_nodes[7] = -1;

        // face 3, 0 1 4
        face_nodes[8] = 0;
        face_nodes[9] = 1;
        face_nodes[10] = 4;
        face_nodes[11] = -1;

        // face 4, 1 2 4
        face_nodes[12] = 1;
        face_nodes[13] = 2;
        face_nodes[14] = 4;
        face_nodes[15] = -1;

        // face 5, 2 3 4
        face_nodes[16] = 2;
        face_nodes[17] = 3;
        face_nodes[18] = 4;
        face_nodes[19] = -1;

        return 0;
    }
    return 1;
}




void su2Elements_free(su2Elements* elem) {
    FlexArray_free(elem->nodes);
    FlexArray_free(elem->nodes_starts);
    FlexArray_free(elem->elem_types);
    free(elem);
}




FaceHashMap* FaceHashMap_new(uint size, uint hash_value) {
    FaceHashMap* h = (FaceHashMap*)malloc(sizeof(FaceHashMap));
    h->list = (FaceHashItem**)malloc(sizeof(FaceHashItem*) * size);
    for (int i = 0; i < size; ++i) h->list[i] = NULL;
    h->size = size;
    h->hash_value = hash_value;

    return h;
}


void FaceHashItem_recursive_free(FaceHashItem* item) {
    free(item->key);
    free(item->value);

    if (item->next_item == NULL) {
        free(item);
        return;
    }

    FaceHashItem_recursive_free(item->next_item);
    free(item);
}

void FaceHashMap_free(FaceHashMap* h) {
    for (size_t i = 0; i < h->size; ++i) {
        if (h->list[i] != NULL)
            FaceHashItem_recursive_free(h->list[i]);
    }
    free(h->list);
    free(h);
}

void sorter_swap(uint* xp, uint* yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
  
// Function to perform Selection Sort
void selectionSort(uint* arr, uint n)
{
    uint i, j, min_idx;
  
    // One by one move boundary of
    // unsorted subarray
    for (i = 0; i < n - 1; i++) {
        // Find the minimum element in
        // unsorted array
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;
  
        // Swap the found minimum element
        // with the first element
        sorter_swap(&arr[min_idx], &arr[i]);
    }
}

uint facehashfunc(FaceHashMap* h, uint* key, uint size) {
    // hash func in sorted order
    uint sorted[32];
    memcpy(sorted, key, size * sizeof(uint));
    selectionSort(sorted, size);

    // Compute hash
    uint v = 1;
    for (int i = 0; i < size; ++i) {
        v *= h->hash_value + 2 * sorted[i];
    }
    return (v / 2) % h->size;
}

int FaceHashItem_compare(FaceHashItem* a, FaceHashItem* b) {

    // If size not equal, not same, return 0
    if (a->key_size != b->key_size) {
        return 0;
    }

    uint as[32];
    uint bs[32];
    assert(a->key_size < 32);

    memcpy(as, a->key, a->key_size * sizeof(uint));
    memcpy(bs, b->key, b->key_size * sizeof(uint));

    selectionSort(as, a->key_size);
    selectionSort(bs, b->key_size);

    // Only check if contains same key, order not important
    for (int i = 0; i < a->key_size; ++i) {
        if (as[i] != bs[i]) return 0;   // Not equal
    }

    return 1;
}

int FaceHashItem_add_value(FaceHashItem* item, uint value) {
    if (item->value_size == item->value_added) {
        item->value_size *= 2;
        uint* new_array = (uint*)realloc(item->value, item->value_size * sizeof(uint));
        if (new_array == NULL) {
            printf("Error, realloc failed on hashmap item add value\n");
            return 1;
        }
        item->value = new_array;
    }

    item->value[item->value_added] = value;
    item->value_added += 1;
    return 0;
}

int FaceHashMap_add(
    FaceHashMap* h,
    uint* key,
    uint key_size,
    uint value
) {
    uint id = facehashfunc(h, key, key_size);

    // Create new item
    FaceHashItem* new_item = cpuAlloc(FaceHashItem, 1);
    new_item->next_item = NULL;
    new_item->key = (uint*)malloc(sizeof(uint) * key_size);
    memcpy(new_item->key, key, sizeof(uint) * key_size);
    new_item->value = (uint*)malloc(sizeof(uint) * 3);
    new_item->value[0] = value;
    new_item->key_size = key_size;
    new_item->value_size = 3;
    new_item->value_added = 1;

    // Place new item
    if (h->list[id] == NULL) {
        // no collision, yay!
        h->list[id] = new_item;
        return 0;
    }

    // Collision, go to the end of linked list
    int add_to_current = 0;
    FaceHashItem* current_item = h->list[id];
    if (FaceHashItem_compare(current_item, new_item)) {
        add_to_current = 1;
    }
    else {
        while (current_item->next_item != NULL) {
            // Check if this item is the item to add
            if (FaceHashItem_compare(current_item, new_item)) {
                add_to_current = 1;
                break;
            }
            current_item = current_item->next_item;
        }
        if (FaceHashItem_compare(current_item, new_item)) {
            add_to_current = 1;
        }
        if (!add_to_current) {
            // We're at the end and it hasn't been found, add to end
            current_item->next_item = new_item;
            return 0;
        }
    }

    // Theyre the same, add value and return without error
    FaceHashItem_recursive_free(new_item);

    // Add item to map value
    if (FaceHashItem_add_value(current_item, value)) return 1;

    return 0;
}



FaceHashItem* FaceHashMap_index(FaceHashMap* h, uint* key, uint size) {
    uint id = facehashfunc(h, key, size);
    if (h->list[id] == NULL) {
        printf("FaceHashMap_index found null in h->list[id]\n");
        printf("hash(%u", key[0]);
        for (int i=1; i<size; ++i) printf(", %u", key[i]);
        printf(") = %u, size = %u\n", id, size);
        return NULL;
    }

    // Create new item
    uint new_key[64];

    FaceHashItem new_item;
    new_item.next_item = NULL;
    memcpy(new_key, key, sizeof(uint) * size);
    new_item.key = new_key;
    new_item.key_size = size;

    // Go to end of linked list
    FaceHashItem* current_item = h->list[id];

    while (current_item != NULL) {
        if (FaceHashItem_compare(current_item, &new_item)) {
            // Found the item
            return current_item;
        }

        current_item = current_item->next_item;
    }

    printf("FaceHashMap_index found non-null in h->list[id], but key not in linked list\n");
    printf("hash(%u", key[0]);
    for (int i=1; i<size; ++i) printf(", %u", key[i]);
    printf(") = %u, size = %u\n", id, size);
    return NULL;
}

int FaceHashMap_is_hashed(FaceHashMap* h, uint* key, uint size) {
    return FaceHashMap_index(h, key, size) != NULL;
}




FaceHashMap* su2ComputeFaces(
    su2Elements* elements
) {
    // Compute the unique faces in a mesh from only the cell to node connectivity
    // Approach the problem using a serial approach
    FaceHashMap* faces = FaceHashMap_new(elements->nodes_starts->size, DEFAULT_HASH);

    for (size_t ei = 0; ei < elements->n_elements; ++ei) {
        // add this element's faces to the face_nodes array
        uint elem_start = *FlexArray_index(uint, elements->nodes_starts, ei);
        uint elem_end = *FlexArray_index(uint, elements->nodes_starts, ei + 1);

        // Get this element's nodes in an array
        uint elem_nodes[32];    // max number of nodes in an element is 32
        for (uint ni = elem_start; ni < elem_end; ++ni) {
            if ((ni - elem_start) >= 32) {
                printf("Error, number of nodes in element %zu bigger than MAX_ELEM_NODES=32, exit\n", ei);
                return NULL;
            }
            elem_nodes[ni - elem_start] = *FlexArray_index(uint, elements->nodes, ni);
        }

        // Get this element type and faces
        uint elem_type = *FlexArray_index(uint, elements->elem_types, ei);
        int elem_faces[64];
        int n_faces;
        if (su2KindToFaces(elem_type, elem_faces, &n_faces)) {
            printf("Error, unknown element type %u\n", elem_type);
            return NULL;
        }

        // Loop over this element's faces
        for (size_t fi = 0; fi < n_faces; ++fi) {

            // Get this face nodes
            uint face_nodes[64];
            size_t fn_size = 0;
            for (size_t fni = 0; fni < 4; ++fni) {
                if (elem_faces[fi * 4 + fni] != -1) {
                    face_nodes[fn_size] = elem_nodes[elem_faces[fi * 4 + fni]];
                    fn_size++;
                }
            }

            // Returns 1 if already in list, ignore that
            int error = FaceHashMap_add(faces, face_nodes, fn_size, ei);
            if (error) printf("Failed to add value to element %zu\n", ei);
        }
    }

    return faces;
}



/*
    Read elements into a su2Elements struct
*/
su2Elements* su2ReadElements(FILE* file, size_t count, int read_extra) {
    su2Elements* elements = cpuAlloc(su2Elements, 1);

    uint n_elements = (uint)count;
    elements->n_elements = n_elements;
    // Guess n_elements_tags
    uint n_elements_tags = 6 * n_elements;

    elements->nodes = FlexArray_new(FLX_UINT, n_elements_tags);
    elements->nodes_starts = FlexArray_new(FLX_UINT, n_elements + 1);
    elements->elem_types = FlexArray_new(FLX_UINT, n_elements);

    uint elements_added_size = 0;
    for (int i = 0; i < n_elements; ++i) {
        *FlexArray_index_append(uint, elements->nodes_starts, i) = elements_added_size;

        uint kind;
        fscanf(file, "%u", &kind);
        *FlexArray_index(uint, elements->elem_types, i) = kind;

        uint elem_size = su2KindToSize(kind);

        for (int j = 0; j < elem_size; ++j) {
            uint node;
            fscanf(file, "%u", &node);
            *FlexArray_index_append(uint, elements->nodes, elements_added_size + j)
                = node;
        }
        elements_added_size += elem_size;
        

        if (read_extra) {
            uint elem_id;
            fscanf(file, "%u", &elem_id);
            assert(elem_id == i);
        }
    }
    *FlexArray_index_append(uint, elements->nodes_starts, n_elements) = elements_added_size;

    return elements;
}


/*
    Read a su2 file and convert it to a fvm mesh file
*/
int cuMeshFromSu2(
    const char* su2_filename,
    const char* out_filename
) {
    // Read the su2 mesh object

    FILE* file = fopen(su2_filename, "rt");

    if (file == NULL) {
        printf("Error reading mesh file %s\n", su2_filename);
        exit(1);
    }

    uint dim;
    uint n_nodes;
    uint n_elements;
    //uint n_elements_tags;
    vector* nodes;
    su2Elements* elements;
    uint n_marks;
    char** marker_names;
    su2Elements** marker_elements;

    char buff[100];
    int tag;

    do {
        fscanf(file, "%s %d", buff, &tag);
        if (strcmp(buff, "NDIME=") == 0) {
            dim = (uint)tag;
            if (dim != 3) {
                printf("Error, mesh must be 3D.\n");
                return 1;
            }
        }
        else if (strcmp(buff, "NPOIN=") == 0) {
            n_nodes = (uint)tag;
            nodes = cpuAlloc(vector, n_nodes);

            for (int i = 0; i < n_nodes; ++i) {
                scalar x, y, z;
                if (dim == 2) {
                    fscanf(file, "%" READ_FORMAT " %" READ_FORMAT, &x, &y);
                    z = 0.0f;
                }
                else if (dim == 3) {
                    fscanf(file, "%" READ_FORMAT " %" READ_FORMAT " %" READ_FORMAT, &x, &y, &z);
                }
                nodes[i] = make_vector(x, y, z);
                // Read last ignore tag
                uint _ignore;
                fscanf(file, "%u", &_ignore);
            }
        }
        else if (strcmp(buff, "NELEM=") == 0) {
            n_elements = (uint)tag;

            elements = su2ReadElements(file, n_elements, 1);

        }
        else if (strcmp(buff, "NMARK=") == 0) {
            n_marks = (uint)tag;
            marker_elements = cpuAlloc(su2Elements*, n_marks);
            marker_names = cpuAlloc(char*, n_marks);
            for (int mark = 0; mark < n_marks; ++mark) {
                // Read marker name
                char name[100];
                fscanf(file, "%s %s", buff, name);
                assert(strcmp(buff, "MARKER_TAG=") == 0);
                marker_names[mark] = cpuAlloc(char, strlen(name) + 1);
                strcpy(marker_names[mark], name);
                // Read marker number of elements
                int mark_nelem;
                fscanf(file, "%s %d", buff, &mark_nelem);
                assert(strcmp(buff, "MARKER_ELEMS=") == 0);

                // Read this marker's elements
                marker_elements[mark] = su2ReadElements(file, mark_nelem, 0);
            }
        }
    } while (!feof(file));

    // Su2 file is read, close the file
    fclose(file);

    size_t n_elems = elements->n_elements;


    // Convert to fvm mesh, biggest part is computing connectivity
    // Compute cell - face connectivity in parallel on gpu
    // turns the O(n^2) algorithm into O(n)

    // Get all the internal faces into hash map
    FaceHashMap* faces = su2ComputeFaces(elements);

    if (faces == NULL) {
        printf("Faces is null\n");
        return 1;
    }

    // Now that we have a hash map of unique faces

    // Add all faces (all contained in the internal faces)
    size_t n_faces = 0;
    size_t n_faces_nodes = 0;
    for (size_t fi = 0; fi < faces->size; ++fi) {
        FaceHashItem* item = faces->list[fi];

        // Go through linked list
        while (item != NULL) {
            // Add this item's faces
            n_faces_nodes += item->key_size;
            n_faces++;

            item = item->next_item;
        }
    }
    // Now create the arrays and fill them
    uint* face_nodes = cpuAlloc(uint, n_faces_nodes);
    uint* face_nodes_starts = cpuAlloc(uint, n_faces + 1);

    uint face_id = 0;
    uint offset = 0;
    for (size_t fi = 0; fi < faces->size; ++fi) {
        FaceHashItem* item = faces->list[fi];

        // Go through linked list
        while (item != NULL) {
            // Add this item's faces
            face_nodes_starts[face_id] = offset;
            for (size_t i = 0; i < item->key_size; ++i) {
                face_nodes[offset] = item->key[i];
                offset++;
            }
            face_id++;

            item = item->next_item;
        }
    }
    face_nodes_starts[n_faces] = offset;

    // We have the face nodes, now fill the cells faces

    // for ghost cells, get the regions informations from the mark arrays
    size_t n_regions = n_marks;
    uint n_ghost_cells = 0;
    for (size_t region_id = 0; region_id < n_regions; ++region_id) {
        n_ghost_cells += marker_elements[region_id]->n_elements;
    }
    uint* ghost_cells_marks = cpuAlloc(uint, n_ghost_cells);
    uint* region_cells_start = cpuAlloc(uint, n_regions);
    offset = 0;
    for (size_t region_id = 0; region_id < n_regions; ++region_id) {
        region_cells_start[region_id] = offset;
        for (size_t i = 0; i < marker_elements[region_id]->n_elements; ++i) {
            ghost_cells_marks[offset] = region_id;
            offset++;
        }
    }

    // We must also add the ghost cell ids to the faces array before adding info
    for (size_t region_id = 0; region_id < n_regions; ++region_id) {
        su2Elements* region_elems = marker_elements[region_id];
        for (size_t i = 0; i < region_elems->n_elements; ++i) {
            // Global element id for this ghost element
            uint gid = region_cells_start[region_id] + i + n_elems;
            // Add it its face in the faces array
            uint elem_nodes_start = *FlexArray_index(uint, region_elems->nodes_starts, i);
            uint elem_nodes_size = *FlexArray_index(uint, region_elems->nodes_starts, i + 1) - elem_nodes_start;
            uint* elem_nodes = FlexArray_index(uint, region_elems->nodes, elem_nodes_start);

            FaceHashItem* item = FaceHashMap_index(faces, elem_nodes, elem_nodes_size);

            if (item == NULL) {
                printf("Error finding ghost cell in faces, faces_values is null\n");
                printf(" - face (%u", elem_nodes[0]);
                for (int i=1; i<elem_nodes_size; ++i) {
                    printf(", %u", elem_nodes[i]);
                }
                printf(") hash = %u, size = %u\n", facehashfunc(faces, elem_nodes, elem_nodes_size), elem_nodes_size);
                printf("hash info: %u %u\n", faces->hash_value, faces->size);
                return 1;
            }

            // Add the global id of this boundary element to the item in face hash map
            if (FaceHashItem_add_value(item, gid)) {
                printf("Error adding boundary region face ghost cell id to faces hashmap.\n");
                return 1;
            }
        }
    }


    // Fill the internal faces information
    uint* cell_faces_sizes = cpuAlloc(uint, n_elems + n_ghost_cells);
    for (size_t i = 0; i < (n_elems + n_ghost_cells); ++i)
        cell_faces_sizes[i] = 0;
    face_id = 0;
    for (size_t fi = 0; fi < faces->size; ++fi) {
        FaceHashItem* item = faces->list[fi];

        // Go through linked list
        while (item != NULL) {
            // Add this item's faces
            for (size_t i = 0; i < item->value_added; ++i) {
                cell_faces_sizes[item->value[i]] ++;
            }
            // Add the face id to this face
            //printf("%u\n", item->value_added);
            (void)FaceHashItem_add_value(item, face_id);
            face_id++;

            item = item->next_item;
        }
    }
    size_t n_cells_faces = cell_faces_sizes[0];
    uint* cell_faces_starts = cpuAlloc(uint, n_elems + n_ghost_cells + 1);
    cell_faces_starts[0] = 0;
    for (size_t ei = 1; ei < (n_elems + n_ghost_cells); ++ei) {
        cell_faces_starts[ei] = n_cells_faces;
        n_cells_faces += cell_faces_sizes[ei];
    }
    cell_faces_starts[n_elems + n_ghost_cells] = n_cells_faces;

    uint* cell_faces = cpuAlloc(uint, n_cells_faces);
    uint* cell_connect = cpuAlloc(uint, n_cells_faces);
    offset = 0;
    for (size_t ei = 0; ei < (n_elems + n_ghost_cells); ++ei) {

        uint elem_nodes[32];    // max number of nodes in an element is 32
        uint elem_type;

        if (ei < n_elems) {
            // Internal element
            // add this element's faces to the face_nodes array
            uint elem_start = *FlexArray_index(uint, elements->nodes_starts, ei);
            uint elem_end = *FlexArray_index(uint, elements->nodes_starts, ei + 1);

            // Get this element's nodes in an array
            for (uint ni = elem_start; ni < elem_end; ++ni) {
                if ((ni - elem_start) >= 32) {
                    printf("Error, number of nodes in element %zu bigger than MAX_ELEM_NODES=32, exit\n", ei);
                    return 1;
                }
                elem_nodes[ni - elem_start] = *FlexArray_index(uint, elements->nodes, ni);
            }

            // get element type and faces
            elem_type = *FlexArray_index(uint, elements->elem_types, ei);
        }
        else {
            // Boundary element
            su2Elements* elem_region = marker_elements[ghost_cells_marks[ei - n_elems]];
            uint ei_in_region = ei - n_elems - region_cells_start[ghost_cells_marks[ei - n_elems]];
            uint elem_start = *FlexArray_index(uint, elem_region->nodes_starts, ei_in_region);
            uint elem_end = *FlexArray_index(uint, elem_region->nodes_starts, ei_in_region + 1);

            for (uint ni = elem_start; ni < elem_end; ++ni) {
                elem_nodes[ni - elem_start] = *FlexArray_index(uint, elem_region->nodes, ni);
            }
            elem_type = *FlexArray_index(uint, elem_region->elem_types, ei_in_region);
        }

        int elem_faces[64];
        int n_faces;
        if (su2KindToFaces(elem_type, elem_faces, &n_faces)) {
            printf("Error, unknown element type %u\n", elem_type);
            return 1;
        }

        // Loop over this element's faces
        for (size_t fi = 0; fi < n_faces; ++fi) {

            // Get this face nodes
            uint face_nodes[64];
            size_t fn_size = 0;
            for (size_t fni = 0; fni < 4; ++fni) {
                if (elem_faces[fi * 4 + fni] != -1) {
                    face_nodes[fn_size] = elem_nodes[elem_faces[fi * 4 + fni]];
                    fn_size++;
                }
            }

            // Add this face id (position in arrays) to this cell's faces connectivity
            // Also add the other cell to the faces ids
            //uint face_value_size = 0;
            //uint* face_values = FaceHashMap_index(faces, face_nodes, fn_size, &face_value_size);
            FaceHashItem* item = FaceHashMap_index(faces, face_nodes, fn_size);

            if (item == NULL) {
                printf("Error, face values is null for element %zu, face: (%u", ei, face_nodes[0]);
                for (int i = 1; i < fn_size; ++i) printf(", %u", face_nodes[i]);
                printf("), hash value: %u\n", facehashfunc(faces, face_nodes, fn_size));
                return 1;
            }

            // Face id is the last face value
            uint global_face_id = item->value[item->value_added - 1];
            if (item->value_added != 3) {
                printf("Error, elements added to face(%u) = %u, must be 2\n", global_face_id, item->value_added - 1);
                printf("Face nodes: (%u", face_nodes[0]);
                for (int i=1; i< fn_size; ++i) {
                    printf(", %u", face_nodes[i]);
                }
                printf(")\n");
                printf("item values: (%u", item->value[0]);
                for (int i=1; i<item->value_added; ++i) {
                    printf(", %u", item->value[i]);
                }
                printf(")\n");
                return 1;
            }
            //assert(item->value_added == 3);
            cell_faces[offset] = global_face_id;
            uint other_cell;
            other_cell = item->value[0] == ei ? item->value[1] : item->value[0];
            cell_connect[offset] = other_cell;
            offset++;
        }
    }


    size_t n_cells = n_elems;

    // We now have the:
    // - faces_nodes
    // - faces_nodes_starts
    // - cell_faces
    // - cell_faces_starts
    // - cell_connect
    // arrays that define the mesh


    // Open the outfile
    FILE* outfile = fopen(out_filename, "wb");

    // Write the nodes
    uint n_nodes_write = (uint)n_nodes;
    fwrite(&n_nodes_write, sizeof(uint), 1, outfile);

    for (size_t i = 0; i < n_nodes; ++i) {
        fwrite(&nodes[i].x, sizeof(scalar), 1, outfile);
        fwrite(&nodes[i].y, sizeof(scalar), 1, outfile);
        fwrite(&nodes[i].z, sizeof(scalar), 1, outfile);
    }

    // Write the faces
    uint n_faces_write = (uint)n_faces;
    uint n_faces_nodes_write = (uint)n_faces_nodes;
    fwrite(&n_faces_write, sizeof(uint), 1, outfile);
    fwrite(&n_faces_nodes_write, sizeof(uint), 1, outfile);

    for (size_t i = 0; i < n_faces; ++i) {
        uint face_size = face_nodes_starts[i + 1] - face_nodes_starts[i];
        fwrite(&face_size, sizeof(uint), 1, outfile);
        for (uint j = face_nodes_starts[i]; j < face_nodes_starts[i + 1]; ++j) {
            fwrite(&face_nodes[j], sizeof(uint), 1, outfile);
        }
    }

    // Write the cells
    uint n_cells_and_ghosts_write = (uint) (n_cells + n_ghost_cells);
    uint n_cells_faces_write = (uint) n_cells_faces;
    fwrite(&n_cells_and_ghosts_write, sizeof(uint), 1, outfile);
    fwrite(&n_cells_faces_write, sizeof(uint), 1, outfile);

    for (size_t i = 0; i < (n_cells + n_ghost_cells); ++i) {
        uint cell_size = cell_faces_starts[i + 1] - cell_faces_starts[i];
        fwrite(&cell_size, sizeof(uint), 1, outfile);
        for (uint j = cell_faces_starts[i]; j < cell_faces_starts[i + 1]; ++j) {
            fwrite(&cell_faces[j], sizeof(uint), 1, outfile);
        }
        for (uint j = cell_faces_starts[i]; j < cell_faces_starts[i + 1]; ++j) {
            fwrite(&cell_connect[j], sizeof(uint), 1, outfile);
        }
    }

    // Now print the regions
    uint n_regions_write = (uint) n_regions;
    uint n_ghost_cells_write = (uint) n_ghost_cells;
    fwrite(&n_regions_write, sizeof(uint), 1, outfile);
    fwrite(&n_ghost_cells_write, sizeof(uint), 1, outfile);
    offset = n_elems;
    for (size_t region_id = 0; region_id < n_regions; ++region_id) {
        
        char name[128];
        strcpy(name, marker_names[region_id]);
        fwrite(name, sizeof(char), 128, outfile);

        uint mark_n_elems_write = (uint) (marker_elements[region_id]->n_elements);
        fwrite(&mark_n_elems_write, sizeof(uint), 1, outfile);

        for (size_t i = 0; i < marker_elements[region_id]->n_elements; ++i) {
            fwrite(&offset, sizeof(uint), 1, outfile);
            offset++;
        }
    }

    // close the outfile
    fclose(outfile);

    // Free memory

    // Free the elements arrays
    for (size_t i = 0; i < n_marks; ++i)
        su2Elements_free(marker_elements[i]);
    su2Elements_free(elements);

    // Free the faces hashmap
    FaceHashMap_free(faces);

    free(nodes);

    for (size_t i = 0; i < n_marks; ++i)
        free(marker_names[i]);
    free(marker_names);

    free(face_nodes);
    free(face_nodes_starts);

    free(ghost_cells_marks);

    free(region_cells_start);

    free(cell_faces_sizes);

    free(cell_faces_starts);

    free(cell_faces);

    free(cell_connect);

    return 0;
}



