#pragma once
#include "mesh.cuh"
#include "flexarray.cuh"


void cuMeshPrintInfo(cuMesh* mesh);


/*
    Print a mesh for debug purposes
*/
void cuMeshPrintDebug(cuMesh* mesh);




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
cuMesh* cuMeshFromFile(const char* filename);



// Size in number of nodes
uint su2KindToSize(uint kind);

int su2KindToFaces(uint kind, int* face_nodes, int* n_faces);




typedef struct su2Elements  {
    FlexArray* nodes;           // uint array
    FlexArray* nodes_starts;    // uint array
    FlexArray* elem_types;      // uint array
    size_t n_elements;
} su2Elements;

void su2Elements_free(su2Elements* elem);


// Hash map for faces construction
typedef struct FaceHashItem {
    uint* key;
    uint* value;
    uint key_size;
    uint value_size;
    uint value_added;
    struct FaceHashItem* next_item;
} FaceHashItem;

typedef struct FaceHashMap {
    FaceHashItem** list;
    uint size;
    uint hash_value;
} FaceHashMap;

FaceHashMap* FaceHashMap_new(uint size, uint hash_value);


void FaceHashItem_recursive_free(FaceHashItem* item);

void FaceHashMap_free(FaceHashMap* h);

uint facehashfunc(FaceHashMap* h, uint* key, uint size);

int FaceHashItem_compare(FaceHashItem* a, FaceHashItem* b);

int FaceHashItem_add_value(FaceHashItem* item, uint value);

int FaceHashMap_add(
    FaceHashMap* h,
    uint* key,
    uint key_size,
    uint value
);


FaceHashItem* FaceHashMap_index(FaceHashMap* h, uint* key, uint size);

int FaceHashMap_is_hashed(FaceHashMap* h, uint* key, uint size);




FaceHashMap* su2ComputeFaces(
    su2Elements* elements
);



/*
    Read elements into a su2Elements struct
*/
su2Elements* su2ReadElements(FILE* file, size_t count, int read_extra);


/*
    Read a su2 file and convert it to a fvm mesh file
*/
int cuMeshFromSu2(
    const char* su2_filename,
    const char* out_filename
);

