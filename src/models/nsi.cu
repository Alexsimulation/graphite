
#include "fvmops.cuh"
#include "fvmfields.cuh"
#include "meshio.cuh"
#include "fvmfieldsio.cuh"

#include "models/nsi.cuh"
#include <fstream>



PhysicsNSI::PhysicsNSI(
    const std::string& filename
) {
    // read the file for parameters
    printf("Reading io data from file %s\n", filename.c_str());

    std::ifstream f(filename);
    iodata = json::json::parse(f);

    // Read mesh
    std::string meshfile = iodata["mesh"].get<std::string>();
    std::string meshfile_ext = meshfile.substr(meshfile.size()-3, 3);

    if (meshfile_ext == "su2") {
        // Convert mesh
        std::string new_meshfile = meshfile.substr(0, meshfile.size()-3) + "fvm";
        printf("Converting mesh to file %s...\n", new_meshfile.c_str());
        int error = cuMeshFromSu2(
            meshfile.c_str(), 
            new_meshfile.c_str()
        );
        if (error) {
            printf("Error during mesh conversion, exiting\n");
            exit(1);
        }
        meshfile = new_meshfile;
    }

    clock_t time_mesh_start = clock();
    printf("Reading mesh from file %s\n", meshfile.c_str());

    this->mesh = cuMeshFromFile(meshfile.c_str());
    cuMeshPrintInfo(mesh);

    clock_t time_mesh_end = clock();
    double mesh_time = (double)(time_mesh_end - time_mesh_start) / CLOCKS_PER_SEC;
    printf("Read mesh in "); print_time(mesh_time); printf("\n");

    // Create variables
    size_t var_size = cuMeshGetVarSize(mesh);
    velocity = FvmFieldNew(sizeof(vector), var_size, "velocity", 1);
    pressure = FvmFieldNew(sizeof(scalar), var_size, "pressure", 1);
    error = FvmFieldNew(sizeof(scalar), var_size, "error", 1);
}


void PhysicsNSI::setup() {
    printf("Setuping Navier-Stokes Incompressible physics model\n");

    printf("Initiating fields\n");

    init_velocity.x = iodata["init"]["velocity"]["x"].get<scalar>();
    init_velocity.y = iodata["init"]["velocity"]["y"].get<scalar>();
    init_velocity.z = iodata["init"]["velocity"]["z"].get<scalar>();
    init_pressure = iodata["init"]["pressure"].get<scalar>();
    
    density = iodata["constants"]["density"].get<scalar>();
    viscosity = iodata["constants"]["viscosity"].get<scalar>();

    FvmNSIInit(
        this->mesh,
        velocity,
        pressure,
        init_velocity,
        init_pressure
    );
}


void PhysicsNSI::setup_graphics(
    GLFWwindow* window,
    GLuint shader_program
) {
    printf("Setuping Navier Stokes Incompressible rendering objects\n");

    std::vector<std::string> render_mesh_regions = iodata["render-patches"].get<std::vector<std::string>>();

    for (const auto& render_region : render_mesh_regions) {
        printf("Adding render region %s\n", render_region.c_str());
        this->rmeshs.push_back(
            RenderMesh_from_region(this->mesh, render_region.c_str())
        );
    }
}





int PhysicsEuler::step(int iter) {
    // Step the physics

    // Apply bcs
    for (auto& elem : iodata["bcs"].items()) {
        auto& key = elem.key();
        auto& value = elem.value();
        
        std::string patch = key;

        std::string type = value["type"].get<std::string>();

        if (type == "slip") {
            FvmNSISlipBc(
                this->mesh, 
                this->velocity, 
                this->pressure,
                patch.c_str()
            );
        } else if (type == "inlet-outlet") {
            FvmNSIInletOutletBc(
                this->mesh, 
                velocity,
                pressure,
                make_vector(
                    value["velocity"]["x"].get<scalar>(),
                    value["velocity"]["y"].get<scalar>(),
                    value["velocity"]["z"].get<scalar>()
                ),
                value["pressure"].get<scalar>(),
                patch.c_str()
            );
        }
    }

    // Compute gradients and limiters
    FvmNSIStepPrepare(
        this->mesh, 
        this->velocity, 
        this->pressure
    );

    // Solve the momentum predictor
    FvmNSIStep(
        this->mesh, 
        this->velocity,
        this->pressure,
        iodata["controls"]["cfl"].get<scalar>(), 
        "steady", 
        NULL, 
        error
    );


    if (iodata["controls"]["time"].get<std::string>() == "steady") {
        scalar err = FvmFieldReduceSum(this->mesh, this->error);
        if (iter == 0) {
            err0 = err;
            printf("- err0: %lf\n", err0);
        }
        err /= err0;

        this->reduced_error = err;
        if (err < iodata["controls"]["tolerance"].get<scalar>()) {
            return 1;
        }
    }

    return 0;
}



scalar PhysicsNSI::residuals() {
    return this->reduced_error;
}




void PhysicsNSI::render() {
    std::string var_name = iodata["render-var"]["var"].get<std::string>();
    scalar umin = iodata["render-var"]["min"].get<scalar>();
    scalar umax = iodata["render-var"]["max"].get<scalar>();
    FvmField* var;
    if (var_name == "pressure") {
        var = pressure;
    } else {
        printf("Error, scalar variable %s unknown.\n", var_name.c_str());
        return;
    }
    for (RenderMesh* rmesh : this->rmeshs) {
        // Map this mesh
        RenderMesh_map(rmesh);
        RenderScalarField(rmesh, var, umin, umax);
        RenderMesh_unmap(rmesh);
    }
}



void PhysicsNSI::finish() {
    VarAllGpuToCpu(this->velocity);
    VarAllGpuToCpu(this->pressure);

    if (iodata.contains("save")) {
        std::string savefile = iodata["save"]["file"].get<std::string>();
        printf("Writing Navier Stokes Incompressible solution to file %s\n", savefile.c_str());

        const FvmField* vars[] = {this->rho, this->rhou, this->rhoe};
        FieldsWriteVtu(
            savefile.c_str(),
            this->mesh,
            vars,
            sizeof(vars)/sizeof(vars[0])
        );
    }
}



PhysicsNSI::~PhysicsNSI() {
    cuMeshFree(this->mesh);
    FvmFieldFree(velocity);
    FvmFieldFree(pressure);
    FvmFieldFree(error);
}

