#include "mesh.cuh"
#include "fvmfields.cuh"
#include "models/model.cuh"
#include "json.cuh"



class PhysicsNSI : public PhysicsBase {

protected:
    json::json iodata;

    FvmField* velocity;
    FvmField* pressure;

    vector init_velocity;
    scalar init_pressure;

    scalar density;
    scalar viscosity;

    scalar err0;
    scalar reduced_error;

public:
    PhysicsNSI(
        const std::string& filename
    );

    void setup();
    void setup_graphics(GLFWwindow* window, GLuint shader_program);
    int step(int iter);
    void render();
    void finish();

    scalar residuals();

    ~PhysicsNSI();

};


