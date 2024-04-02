#include "models/model.cuh"



void print_time(double time) {
    if (time < 60) printf("%.3lf s", time);
    else if (time < 3600) printf("%.3lf min", time / 60.0);
    else printf("%lf h", time / 3600.0);
}




ModelBase::ModelBase(int argc, char** argv) {

    // Default options
    options.with_gui = true;

    for (int i=0; i<argc; ++i) {
        if (strcmp(argv[i], "--nogui") == 0) {
            options.with_gui = false;
        }
    }
}



void ModelBase::setup() {
    printf("Setuping physics\n");
    for (PhysicsBase* ps : physics) ps->setup();

    // Start the renderer
    if (options.with_gui) {
        printf("Starting renderer\n");
        window = RenderStart();
        if (window == NULL) {
            printf("ERROR: rendered failed to start\n");
        }

        // Setup the shaders
        shader_program = RenderSetupShaders();

        RenderSetupCamera(window);

        printf("Rendered started\n");

        for (PhysicsBase* ps : physics) ps->setup_graphics(window, shader_program);

        _lastTime = glfwGetTime();
        _print_time = 0;
        _print_iter = 0;
    }
}


void ModelBase::add_physics(PhysicsBase* ps) {
    this->physics.push_back(ps);
}


int ModelBase::stepBefore() {
    if (options.with_gui) {
        if (glfwWindowShouldClose(window)) return 1;
    
        
        double nowTime = glfwGetTime();
        _elapsed_time = nowTime - _lastTime;
        _lastTime = nowTime;
        _print_time += _elapsed_time;

        if (_print_time > 0.5) {
            printf("fps: %6.2f\n", ((double)(iter - _print_iter)) / _print_time );
            _print_time = 0;
            _print_iter = iter;
        }

        Render_update_fps_counter(window);
    }

    return 0;
}


int ModelBase::do_compute() {
    if (options.with_gui) {
        return RGC.compute;
    } else {
        return 1;
    }
}


int ModelBase::stepAfter() {

    cudaSync;


    // Draw objects
    if (options.with_gui) {
        for (PhysicsBase* ps : physics) {
            ps->render();
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shader_program);

        RenderUpdateCamera(window, shader_program);

        for (PhysicsBase* ps : physics) {
            for (RenderMesh* rmesh : ps->rmeshs) {
                RenderMesh_draw(rmesh);
            }
        }

        // update other events like input handling 
        glfwPollEvents();
        // put the stuff we've been drawing onto the display
        glfwSwapBuffers(window);

        // Update the camera
        RenderCameraEvents(window, _elapsed_time);

        printOpenGLError();
    }

    return 0;
}

void ModelBase::finish() {
    for (PhysicsBase* ps : physics) ps->finish();

    if (options.with_gui) {
        if (RenderEnd()) {
            printf("ERROR: Renderer end failed\n");
        }
    }
}

