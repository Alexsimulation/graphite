#pragma once


#define GLEW_STATIC

//#include <windef.h> // According to comments above
#include <windows.h>
#include <GL/glew.h> // Before any gl headers
#include <GL/gl.h>

//#include <GL/glext.h> // Linux headers
//#include <GL/wglext.h> // Windows headers - Not sure which ones cygwin needs. Just try it

#include <GL/glu.h> // Always after gl.h

#define GLFW_STATIC
#include <GLFW/glfw3.h> // When all gl-headers have been included

#include <cuda_gl_interop.h>

#include <stdio.h>
#include <stdlib.h>


#include "datatypes.cuh"
#include "mesh.cuh"
#include "fvmfields.cuh"



typedef struct {
    scalar zoom;

    vector up_direction;
    vector camera_pos;
    vector camera_front;
    vector camera_up;
    vector camera_right;

    scalar pitch;
    scalar yaw;

    scalar last_position_x;
    scalar last_position_y;

    int position_start;
    int move;
    int compute;

    float matrix[16];
} RenderControls;



extern RenderControls RGC;


GLFWwindow* RenderStart();

int RenderEnd();

GLuint RenderSetupShaders();


void RenderUpdateCamera(GLFWwindow* window, GLuint shader_program);

void RenderSetupCamera(GLFWwindow* window);

void RenderCameraEvents(GLFWwindow* window, double elapsed_time);


void Render_update_fps_counter(GLFWwindow* window);

void RenderSetWireframe();

void RenderSetFill();





typedef struct {
    float* elements;
    uint* adresses;
    uint n_elements;
} _RMesh;

typedef struct {
    _RMesh* cpu;
    _RMesh* hgpu;
    _RMesh* gpu;

    GLuint VAO;
    GLuint VBO;
} RenderMesh;


RenderMesh* RenderMesh_new(
    const double* nodes,
    const uint* elements,
    uint n_nodes,
    uint n_elements
);

RenderMesh* RenderMesh_from_allocs(
    float* elements,
    uint* adresses,
    uint n_elements
);


void RenderMesh_free(RenderMesh* mesh);


RenderMesh* RenderMesh_from_file(const char* filename);

RenderMesh* RenderMesh_from_region(cuMesh* mesh, const char* region);

// Display a mesh
int RenderMesh_to_gl(RenderMesh* mesh);



void RenderMesh_map(RenderMesh* mesh);

void RenderMesh_unmap(RenderMesh* mesh);


void RenderMesh_draw(RenderMesh* mesh);



void RenderScalarField(
    RenderMesh* mesh, FvmField* field, scalar u_min, scalar u_max
);



void RenderCameraPositionToMesh(RenderMesh* mesh);




int printOglError(const char *file, int line);

#define printOpenGLError() printOglError(__FILE__, __LINE__)



