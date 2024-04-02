
#include "mesh.cuh"
#include "fvmfields.cuh"
#include "models/model.cuh"
#include "json.cuh"



class PhysicsEuler : public PhysicsBase {

protected:
    json::json iodata;

    FvmField* rho;
    FvmField* rhou;
    FvmField* rhoe;
    FvmField* error;

    vector mach;
    scalar pressure;
    scalar temperature;

    scalar err0;
    scalar reduced_error;

public:
    PhysicsEuler(
        const std::string& filename
    );

    void setup();
    void setup_graphics(GLFWwindow* window, GLuint shader_program);
    int step(int iter);
    void render();
    void finish();

    scalar residuals();

    ~PhysicsEuler();

};



int FvmEulerInit(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const vector& mach,
    const scalar& pressure,
    const scalar& temperature
);


int FvmEulerSlipBc(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const char* region
);


int FvmEulerInletOutletBc(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const vector& mach,
    const scalar& pressure,
    const scalar& temperature,
    const char* region
);


int FvmEulerStepPrepare(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe
);


int FvmEulerStep(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const scalar& cfl,
    const char* time_scheme,
    FvmField* dt,
    FvmField* error
);


