#include "render/window.cuh"

#include "fvmops.cuh"
#include "fvmfields.cuh"
#include "meshio.cuh"
#include "fvmfieldsio.cuh"

#include "physics/euler.cuh"

#include <math.h>
#include <time.h>



void euler_solve(
    cuMesh* mesh,
    FvmField* rho,
    FvmField* rhou,
    FvmField* rhoe,
    scalar mach,
    scalar angle,
    scalar cfl,
    int n_steps,
    scalar tolerance
) {
    FvmField* error = FvmFieldNew(sizeof(scalar), cuMeshGetVarSize(mesh), "error", 0);

    // Init the field
    FvmEulerInit(
        mesh,
        rho,
        rhou,
        rhoe,
        make_vector(
            (scalar)0,
            mach * sin(angle * 3.1415927/180.0),
            -mach * cos(angle * 3.1415927/180.0)
        ),
        1.0f,
        0.71428571428f
    );

    uint wing = cuMeshGetRegionId(mesh, "wing");
    uint farfield = cuMeshGetRegionId(mesh, "farfield");
    uint sides = cuMeshGetRegionId(mesh, "sides");

    // Start the renderer
    GLFWwindow* window = RenderStart();
    if (window == NULL) {
        printf("ERROR: rendered failed to start\n");
    }

    // Make the render mesh
    RenderMesh* rmesh0 = RenderMesh_from_region(mesh, "sides");
    RenderMesh* rmesh1 = RenderMesh_from_region(mesh, "farfield");
    RenderMesh* rmesh2 = RenderMesh_from_region(mesh, "wing");
    RenderMesh* rmeshes[] = {rmesh0, rmesh1, rmesh2};
    size_t n_rmeshes = 3;
    //RenderMesh* rmesh = RenderMesh_from_file("data/circle.su2");
    //printf("Render nelems = %u\n", rmesh->cpu->n_elements);

    // Setup the shaders
    GLuint shader_program = RenderSetupShaders();

    RenderSetupCamera(window);

    //RenderCameraPositionToMesh(rmesh);

    double lastTime = glfwGetTime();

    //RenderSetFill();

    // Perform iterations of the solver
    uint print_interval = n_steps > 40 ? (n_steps/40) : 1;
    scalar err0;
    //for (int iter = 0; iter < n_steps; ++iter) {
    int iter = 0;
    double print_time = 0;
    int print_iter = 0;
    while (1) {
        if (glfwWindowShouldClose(window)) break;
        double nowTime = glfwGetTime();
        double elapsed_time = nowTime - lastTime;
        lastTime = nowTime;
        print_time += elapsed_time;

        if (print_time > 0.5) {
            printf("fps: %6.2f\n", ((double)(iter - print_iter)) / print_time );
            print_time = 0;
            print_iter = iter;
        }
        
        Render_update_fps_counter(window);
        //printf("Camera position: %lf %lf %lf\n", RGC.camera_pos.x, RGC.camera_pos.y, RGC.camera_pos.z);

        // Compute stuff
        if (RGC.compute) {
            // Set the boundary conditions
            FvmEulerSlipBc(
                mesh, rho, rhou, rhoe, "wing"
            );

            FvmEulerSlipBc(
                mesh, rho, rhou, rhoe, "sides"
            );

            FvmEulerInletOutletBc(
                mesh, rho, rhou, rhoe,
                make_vector(
                    (scalar)0,
                    mach * sin(angle * 3.1415927/180.0),
                    -mach * cos(angle * 3.1415927/180.0)
                ),
                1.0f,
                0.71428571428f,
                "farfield"
            );

            // Compute gradients and limites
            FvmEulerStepPrepare(mesh, rho, rhou, rhoe);

            // Solve the equation
            FvmEulerStep(mesh, rho, rhou, rhoe, cfl, "steady", NULL, error);

            // Get the error
            /*
            scalar err = FvmFieldReduceSum(mesh, error);
            if (iter == 0) {err0 = err;}
            err /= err0;

            if ((iter+1) % print_interval == 0) printf("- iter: %d, error: %.2e\n", iter+1, err);

            if (err < tolerance) break;
            if (isnan(err)|isinf(err)) {printf("Error, nan in error at iteration %d\n", iter+1); break;}
            */

            // Sync at the end of iteration
            cudaSync;

            for (int i=0; i<n_rmeshes; ++i) {
                RenderMesh* rmesh = rmeshes[i];
                RenderMesh_map(rmesh);
                RenderScalarField <<< cuda_size(rmesh->cpu->n_elements, CKW), CKW >>>(
                    rmesh->gpu, rho->_gpu,
                    0.0, 2.0
                );
                RenderMesh_unmap(rmesh);
            }

            iter++;
        }
        // End computations

        // Display

        // Clear window and bind shader program
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shader_program);

        RenderUpdateCamera(window, shader_program);


        // Draw objects
        for (int i=0; i<n_rmeshes; ++i) RenderMesh_draw(rmeshes[i]);

        // update other events like input handling 
        glfwPollEvents();
        // put the stuff we've been drawing onto the display
        glfwSwapBuffers(window);

        // Update the camera
        RenderCameraEvents(window, elapsed_time);

        printOpenGLError();

    }

    // close GL context and any other GLFW resources
    for (int i=0; i<n_rmeshes; ++i) RenderMesh_free(rmeshes[i]);
    if (RenderEnd()) {
        printf("ERROR: Renderer end failed\n");
    }

    // Get temperature data back from the gpu
    VarAllGpuToCpu(rho);
    VarAllGpuToCpu(rhou);
    VarAllGpuToCpu(rhoe);

    FvmFieldFree(error);

    cudaSync;
}



int main() {

    clock_t time_start = clock();

    // Convert su2 mesh to fvm mesh
    
    
    printf("Converting mesh...\n");
    int error = cuMeshFromSu2(
        "data/wing.su2", 
        "data/wing.fvm"
    );
    if (error) {
        printf("Error during mesh conversion, exiting\n");
        return error;
    }
    

    clock_t time_convert = clock();
    

    printf("Reading mesh...\n");
    cuMesh* mesh = cuMeshFromFile("data/wing.fvm");
    
    cuMeshPrintInfo(mesh);

    clock_t time_read = clock();


    printf("Starting simulation...\n");
    
    size_t var_size = cuMeshGetVarSize(mesh);

    // Create the temperature finite volume variable
    FvmField* rho = FvmFieldNew(sizeof(scalar), var_size, "rho", 1);
    FvmField* rhou = FvmFieldNew(sizeof(vector), var_size, "rhou", 1);
    FvmField* rhoe = FvmFieldNew(sizeof(scalar), var_size, "rhoe", 1);

    // Solve the problem
    euler_solve(mesh, rho, rhou, rhoe, 
        0.8,    // mach
        0.0,    // angle (degrees)
        0.6,    // cfl
        40000,    // n_iters,
        1e-6    // tolerance
    );

    clock_t time_solve = clock();

    /*

    // Save the solution
    printf("Writing solution...\n");

    const FvmField* vars[] = {rho, rhou, rhoe};
    FieldsWriteVtu(
        "data/swept-wing.vtu",
        mesh,
        vars,
        sizeof(vars)/sizeof(vars[0])
    );
    */

    // Free the mesh and fields
    cuMeshFree(mesh);
    FvmFieldFree(rho);
    FvmFieldFree(rhou);
    FvmFieldFree(rhoe);

    printf("Done.\n");

    clock_t time_end = clock();
    
    double mesh_time = (double)(time_read - time_start) / CLOCKS_PER_SEC;
    double solve_time = (double)(time_solve - time_read) / CLOCKS_PER_SEC;
    double total_time = (double)(time_end - time_start) / CLOCKS_PER_SEC;

    printf("Mesh reading time   = "); print_time(mesh_time); printf("\n");
    printf("Solver compute time = "); print_time(solve_time); printf("\n");
    printf("Total run time      = "); print_time(total_time); printf("\n");
    

    return 0;
}


