
#include "fvmops.cuh"
#include "fvmfields.cuh"
#include "meshio.cuh"
#include "fvmfieldsio.cuh"

#include "models/euler.cuh"
#include <fstream>



PhysicsEuler::PhysicsEuler(
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
    rho = FvmFieldNew(sizeof(scalar), var_size, "rho", 1);
    rhou = FvmFieldNew(sizeof(vector), var_size, "rhou", 1);
    rhoe = FvmFieldNew(sizeof(scalar), var_size, "rhoe", 1);
    error = FvmFieldNew(sizeof(scalar), var_size, "error", 1);
}


void PhysicsEuler::setup() {
    printf("Setuping Euler physics model\n");

    printf("Initiating fields\n");

    mach.x = iodata["init"]["mach"]["x"].get<scalar>();
    mach.y = iodata["init"]["mach"]["y"].get<scalar>();
    mach.z = iodata["init"]["mach"]["z"].get<scalar>();
    pressure = iodata["init"]["pressure"].get<scalar>();
    temperature = iodata["init"]["temperature"].get<scalar>();

    FvmEulerInit(
        this->mesh,
        rho,
        rhou,
        rhoe,
        mach,
        pressure,
        temperature
    );
}


void PhysicsEuler::setup_graphics(
    GLFWwindow* window,
    GLuint shader_program
) {
    printf("Setuping Euler rendering objects\n");

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
            FvmEulerSlipBc(
                this->mesh, 
                this->rho, 
                this->rhou, 
                this->rhoe,
                patch.c_str()
            );
        } else if (type == "farfield") {
            FvmEulerInletOutletBc(
                this->mesh, 
                this->rho, 
                this->rhou, 
                this->rhoe,
                make_vector(
                    value["mach"]["x"].get<scalar>(),
                    value["mach"]["y"].get<scalar>(),
                    value["mach"]["z"].get<scalar>()
                ),
                value["pressure"].get<scalar>(),
                value["temperature"].get<scalar>(),
                patch.c_str()
            );
        }
    }

    // Compute gradients and limites
    FvmEulerStepPrepare(
        this->mesh, 
        this->rho, 
        this->rhou, 
        this->rhoe
    );

    // Solve the equation
    FvmEulerStep(
        this->mesh, 
        this->rho, 
        this->rhou, 
        this->rhoe,
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

scalar PhysicsEuler::residuals() {
    return this->reduced_error;
}


void PhysicsEuler::render() {
    std::string var_name = iodata["render-var"]["var"].get<std::string>();
    scalar umin = iodata["render-var"]["min"].get<scalar>();
    scalar umax = iodata["render-var"]["max"].get<scalar>();
    FvmField* var;
    if (var_name == "rho") {
        var = rho;
    } else if (var_name == "rhoe") {
        var = rhoe;
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


void PhysicsEuler::finish() {
    VarAllGpuToCpu(this->rho);
    VarAllGpuToCpu(this->rhou);
    VarAllGpuToCpu(this->rhoe);

    if (iodata.contains("save")) {
        std::string savefile = iodata["save"]["file"].get<std::string>();
        printf("Writing Euler solution to file %s\n", savefile.c_str());

        const FvmField* vars[] = {this->rho, this->rhou, this->rhoe};
        FieldsWriteVtu(
            savefile.c_str(),
            this->mesh,
            vars,
            sizeof(vars)/sizeof(vars[0])
        );
    }
}




PhysicsEuler::~PhysicsEuler() {
    cuMeshFree(this->mesh);
    FvmFieldFree(rho);
    FvmFieldFree(rhou);
    FvmFieldFree(rhoe);
    FvmFieldFree(error);
}



fvmOp(void, fvmEulerSteadyOp)
	FvmFieldBase* rho_in,
    FvmFieldBase* rhou_in,
    FvmFieldBase* rhoe_in,
    scalar& rho_flux,
    vector& rhou_flux,
    scalar& rhoe_flux,
    scalar& dt
) {
    fvmOpMakeInputVar(scalar, vector, rho, rho_in);
	fvmOpMakeInputVar(vector, tensor, rhou, rhou_in);
    fvmOpMakeInputVar(scalar, vector, rhoe, rhoe_in);
	fvmOpVoidSetup

    const scalar rho0 = rho.val[c0] + dot(rho.grad[c0], d0) * rho.lim[c0];
	const scalar rho1 = rho.val[c1] + dot(rho.grad[c1], d1) * rho.lim[c1];
	const vector rhou0 = rhou.val[c0] + dot(rhou.grad[c0], d0) * rhou.lim[c0];
	const vector rhou1 = rhou.val[c1] + dot(rhou.grad[c1], d1) * rhou.lim[c1];
    const scalar rhoe0 = rhoe.val[c0] + dot(rhoe.grad[c0], d0) * rhoe.lim[c0];
    const scalar rhoe1 = rhoe.val[c1] + dot(rhoe.grad[c1], d1) * rhoe.lim[c1];

    const scalar p0 = 0.4 * (rhoe0 - 0.5/rho0 * dot(rhou0, rhou0));
    const scalar p1 = 0.4 * (rhoe1 - 0.5/rho1 * dot(rhou1, rhou1));

    const scalar V0 = dot(rhou0, n) / rho0;
    const scalar V1 = dot(rhou1, n) / rho1;

    // Fluxes
    const scalar rho_flux0 = V0 * rho0;
	const scalar rho_flux1 = V1 * rho1;
    const vector rhou_flux0 = V0 * rhou0 + n * p0;	
	const vector rhou_flux1 = V1 * rhou1 + n * p1;
    const scalar rhoe_flux0 = V0 * (rhoe0 + p0);	
	const scalar rhoe_flux1 = V1 * (rhoe1 + p1);

    // Central flux
    rho_flux += (rho_flux0 + rho_flux1) * ds * 0.5;
    rhou_flux += (rhou_flux0 + rhou_flux1) * ds * 0.5;
    rhoe_flux += (rhoe_flux0 + rhoe_flux1) * ds * 0.5;

    // Artificial dissipation

    // Scalar dissipation
    
    // Max eigenvalue
    
    const scalar eig = fmax(
        norm(rhou0/rho0) + lsqrt(1.4 * p0 / rho0),
        norm(rhou1/rho1) + lsqrt(1.4 * p1 / rho1)
    );

    rho_flux  -= eig * (rho1  - rho0 ) * 0.5 * ds;
    rhou_flux -= eig * (rhou1 - rhou0) * 0.5 * ds;
    rhoe_flux -= eig * (rhoe1 - rhoe0) * 0.5 * ds;
    
    
    

    // CUSP scheme
    /*
    const scalar V = (V0 + V1) * 0.5;
    const scalar pn = 0.4 * ((rhoe0 + rhoe1)*0.5 - 0.5 / (0.5*(rho0 + rho1)) * dot(
                (rhou0 + rhou1) * 0.5,
                (rhou0 + rhou1) * 0.5
            ));
    const scalar Cn = lsqrt(1.4 * pn / ((rho0 + rho1) * 0.5));
    const scalar Mn = V / Cn;

    const scalar delta = 0.01;
    const scalar alpha = 
        (fabs(Mn) >= delta) * fabs(Mn) +
        (fabs(Mn) < delta) * (Mn*Mn + delta*delta)/(2.0*delta)
    ;
    
    scalar beta = 
        ((0 <= Mn)&(Mn < 1))    * fmax(0.0, 2.0*Mn - 1.0) + 
        ((-1 < Mn)&(Mn < 0))    * min(0.0, 2.0*Mn + 1.0) +
        (Mn >= 1) + (Mn <= -1) * -1
    ;
    const scalar alphac = alpha * Cn - beta * V;

    rho_flux -= (
            alphac * (rho1 - rho0) +
            beta * (rho_flux1 - rho_flux0)
        ) * 0.5 * ds;
    rhou_flux -= (
            alphac * (rhou1 - rhou0) +
            beta * (rhou_flux1 - rhou_flux0)
        ) * 0.5 * ds;
    rhoe_flux -= (
            alphac * (rhoe1 - rhoe0) +
            beta * (rhoe_flux1 - rhoe_flux0)
        ) * 0.5 * ds;

    const scalar eig = fmax(
        norm(rhou0/rho0) + lsqrt(1.4 * p0 / rho0),
        norm(rhou1/rho1) + lsqrt(1.4 * p1 / rho1)
    );
    */
    
    
    dt += eig * ds;

fvmOpVoidEnd





/*
    Example function to solve euler equation using euler explicit scheme
*/
__global__ void euler_steady_step(
    cuMeshBase* mesh,
    FvmFieldBase* rho,
    FvmFieldBase* rhou,
    FvmFieldBase* rhoe,
    FvmFieldBase* error,
    scalar cfl
) {
    int i = cudaGlobalId;
    if (i >= mesh->n_cells) return;
    scalar* rho_arr = (scalar*)rho->value;
    vector* rhou_arr = (vector*)rhou->value;
    scalar* rhoe_arr = (scalar*)rhoe->value;
    scalar* error_arr = (scalar*)error->value;


    // Use Euler explicit time integrator, centered space scheme
    scalar dt = 0.0;

    scalar rho_flux = 0;
    vector rhou_flux = make_vector((scalar)0.0, 0.0, 0.0);
    scalar rhoe_flux = 0;

    fvmEulerSteadyOp(mesh, i, rho, rhou, rhoe, rho_flux, rhou_flux, rhoe_flux, dt);

    dt = cfl / dt;

    // Update
    rho_arr[i] += -rho_flux * dt;
    rhou_arr[i] += -rhou_flux * dt;
    rhoe_arr[i] += -rhoe_flux * dt;

    error_arr[i] = (rho_flux*rho_flux + dot(rhou_flux, rhou_flux) + rhoe_flux * rhoe_flux) * mesh->cell_volumes[i];
}



__global__ void euler_init(
    cuMeshBase* mesh,
    FvmFieldBase* rho,
    FvmFieldBase* rhou,
    FvmFieldBase* rhoe,
    vector mach,
    scalar pressure,
    scalar temperature
) {
    int i = cudaGlobalId;
    if (i >= (mesh->n_cells + mesh->n_ghosts)) return;
    scalar* rho_arr = (scalar*)rho->value;
    vector* rhou_arr = (vector*)rhou->value;
    scalar* rhoe_arr = (scalar*)rhoe->value;


    scalar rho_v = pressure / temperature;
    vector velocity = mach * sqrt(1.4 * pressure / rho_v);
    vector rhou_v = velocity * rho_v;
    scalar rhoe_v = pressure / 0.4 + 0.5*rho_v * dot(velocity, velocity);

    rho_arr[i] = rho_v;
    rhou_arr[i] = rhou_v;
    rhoe_arr[i] = rhoe_v;
}


__global__ void euler_slip_bc(
    cuMeshBase* mesh,
    FvmFieldBase* rho_field,
    FvmFieldBase* rhou_field,
    FvmFieldBase* rhoe_field,
    uint region_id
) {
    int i = cudaGlobalId;
	uint region_size = cuMeshGetRegionSize(mesh, region_id);
	if (i >= region_size) return;
	uint c0 = mesh->ghost_cell_ids[mesh->ghost_cell_starts[region_id] + i];
    uint c1 = cuMeshGetCellConnect(mesh, c0)[0];

    scalar* rho = (scalar*)rho_field->value;
    vector* rhou = (vector*)rhou_field->value;
    scalar* rhoe = (scalar*)rhoe_field->value;

    // Get the face normal
    uint face = cuMeshGetCellStart(mesh, c0)[0];
    vector d0 = mesh->face_centers[face] - mesh->cell_centers[c0];
    vector d1 = mesh->face_centers[face] - mesh->cell_centers[c1];

    vector n = outer_normal(mesh->face_normals[face], d1);

    // Flip condition for vector rhou
    rhou[c0] = rhou[c1] - 2.0 * n * dot(n, rhou[c1]);

    // Zero gradient for rho and rhoe
    rho[c0] = rho[c1];
    rhoe[c0] = rhoe[c1];
}



__global__ void euler_inletoutlet_bc(
    cuMeshBase* mesh,
    FvmFieldBase* rho_field,
    FvmFieldBase* rhou_field,
    FvmFieldBase* rhoe_field,
    vector inletMach,
    scalar inletPressure,
    scalar inletTemperature,
    uint region_id
) {
    int i = cudaGlobalId;
	uint region_size = cuMeshGetRegionSize(mesh, region_id);
	if (i >= region_size) return;
	uint c0 = mesh->ghost_cell_ids[mesh->ghost_cell_starts[region_id] + i];
    uint c1 = cuMeshGetCellConnect(mesh, c0)[0];

    scalar* rho = (scalar*)rho_field->value;
    vector* rhou = (vector*)rhou_field->value;
    scalar* rhoe = (scalar*)rhoe_field->value;

    // Get the face normal
    uint face = cuMeshGetCellStart(mesh, c0)[0];
    vector d0 = mesh->face_centers[face] - mesh->cell_centers[c0];
    vector d1 = mesh->face_centers[face] - mesh->cell_centers[c1];

    vector n = outer_normal(mesh->face_normals[face], d1);

    
    vector u = rhou[c1] / rho[c1];
    scalar p = 0.4 * (rhoe[c1] - 0.5/rho[c1] * dot(rhou[c1], rhou[c1]));
    scalar t = p / (1.0 * rho[c1]);
    scalar mach = sqrt(dot(u, u) / (1.4 * p / rho[c1]));

    scalar inletDensity = inletPressure / (1.0 * inletTemperature);
    vector inletVelocity = inletMach * sqrt(1.4 * inletPressure / inletDensity);
    scalar inletEnergy = inletPressure / 0.4 + 0.5 * inletDensity * dot(inletVelocity, inletVelocity);

    if (dot(u, n) < 0) {
        // Flow enters the domain, inlet
        if (mach > 1.0f) {
            // Supersonic inlet
            // fix all variables
            scalar inletC = sqrt(1.4 * inletPressure / inletDensity);
            rho[c0] = inletDensity;
            rhou[c0] = inletDensity * inletMach * inletC;
            rhoe[c0] = inletEnergy;
        } else {
            // Subsonic inlet
            // free pressure
            rho[c0] = p / (1.0 * inletTemperature);
            scalar inletC = sqrt(1.4 * p / rho[c0]);
            inletVelocity = inletMach * inletC;
            rhou[c0] = rho[c0] * inletVelocity;
            rhoe[c0] = p / 0.4 + 0.5 * rho[c0] * dot(inletVelocity, inletVelocity);
        }
    } 
    else {
        // Flow exits the domain, outlet
        if (mach > 1.0f) {
            // Supersonic outlet
            // All zero gradient
            rho[c0] = rho[c1];
            rhou[c0] = rhou[c1];
            rhoe[c0] = rhoe[c1];
        }
        else {
            // Subsonic outlet
            // All zero gradient excepts pressure
            rho[c0] = inletPressure / (1.0 * t);
            rhou[c0] = rho[c0] * u;
            rhoe[c0] = inletPressure / 0.4 + 0.5 * rho[c0] * dot(u, u);
        }
    }
    
}




/*
    Public functions
*/

int FvmEulerInit(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const vector& mach,
    const scalar& pressure,
    const scalar& temperature
) {
    int n_cg = cuda_size(cuMeshGetnCellsAndGhosts(mesh), CKW);
    euler_init <<< n_cg, CKW >>> (
        kernelInput(mesh),
        kernelInput(rho),
        kernelInput(rhou),
        kernelInput(rhoe),
        mach,
        pressure,
        temperature
    );
    return 0;
}


int FvmEulerSlipBc(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const char* region
) {
    uint id = cuMeshGetRegionId(mesh, region);
    euler_slip_bc <<< cuda_size(cuMeshGetnRegionCells(mesh, id), CKW), CKW >>> (
        kernelInput(mesh),
        kernelInput(rho),
        kernelInput(rhou),
        kernelInput(rhoe),
        id
    );
    return 0;
}


int FvmEulerInletOutletBc(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const vector& mach,
    const scalar& pressure,
    const scalar& temperature,
    const char* region
) {
    uint id = cuMeshGetRegionId(mesh, region);
    euler_inletoutlet_bc <<< cuda_size(cuMeshGetnRegionCells(mesh, id), CKW), CKW >>> (
        kernelInput(mesh),
        kernelInput(rho),
        kernelInput(rhou),
        kernelInput(rhoe),
        mach,
        pressure,
        temperature,
        id
    );
    return 0;
}



int FvmEulerStepPrepare(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe
) {
    // Prepare the step by computing the gradients and limiters
    // To do after boundary conditions
    int n_c = cuda_size(cuMeshGetnCells(mesh), CKW);
    int n_g = cuda_size(cuMeshGetnGhosts(mesh), CKW);

    ScalarVarComputeGradient <<< n_c, CKW >>> (
        kernelInput(mesh),
        kernelInput(rho)
    );
    ScalarVarComputeLimiters <<< n_c, CKW >>> (
        kernelInput(mesh),
        kernelInput(rho)
    );
    fvmScalGradLimBc <<< n_g, CKW >>> (
        kernelInput(mesh),
        kernelInput(rho)
    );

    VectorVarComputeGradient <<< n_c, CKW >>> (
        kernelInput(mesh),
        kernelInput(rhou)
    );
    VectorVarComputeLimiters <<< n_c, CKW >>> (
        kernelInput(mesh),
        kernelInput(rhou)
    );
    fvmVecGradLimBc <<< n_g, CKW >>> (
        kernelInput(mesh),
        kernelInput(rhou)
    );

    ScalarVarComputeGradient <<< n_c,CKW >>> (
        kernelInput(mesh),
        kernelInput(rhoe)
    );
    ScalarVarComputeLimiters <<< n_c, CKW >>> (
        kernelInput(mesh),
        kernelInput(rhoe)
    );
    fvmScalGradLimBc <<< n_g, CKW >>> (
        kernelInput(mesh),
        kernelInput(rhoe)
    );
    return 0;
}

int FvmEulerStep(
    cuMesh* mesh, 
    FvmField* rho, 
    FvmField* rhou, 
    FvmField* rhoe,
    const scalar& cfl,
    const char* time_scheme,
    FvmField* dt,
    FvmField* error
) {
    int n_c = cuda_size(cuMeshGetnCells(mesh), CKW);

    if (strcmp(time_scheme, "steady") == 0) {
        euler_steady_step <<< n_c, CKW >>> (
            kernelInput(mesh),
            kernelInput(rho),
            kernelInput(rhou),
            kernelInput(rhoe),
            kernelInput(error),
            cfl
        );
    }

    return 0;
}

