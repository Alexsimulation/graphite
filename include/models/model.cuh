#pragma once

#include <vector>

#include "mesh.cuh"
#include "render/window.cuh"



void print_time(double time);


class PhysicsBase {

protected:
    GLFWwindow* window;
    GLuint shader_program;

public:

    std::vector<RenderMesh*> rmeshs;
    cuMesh* mesh;

    virtual void setup() = 0;
    virtual void setup_graphics(GLFWwindow* window, GLuint shader_program) = 0;
    virtual int step(int iter) = 0;
    virtual void render() = 0;
    virtual void finish() = 0;

};



typedef struct {
    bool with_gui;
} ModelBaseOptions;


class ModelBase {

protected:
    GLFWwindow* window;
    GLuint shader_program;

    double _lastTime;
    double _print_time;
    double _elapsed_time;
    uint _print_iter;

    uint iter;

    ModelBaseOptions options;

    std::vector<PhysicsBase*> physics;

public:

    ModelBase(int argc, char** argv);

    void add_physics(PhysicsBase* p);

    int do_compute();

    void setup();
    int stepBefore();
    int stepAfter();
    void finish();

    virtual void run() = 0;

};



