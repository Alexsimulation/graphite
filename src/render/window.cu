
#include "render/window.cuh"




RenderControls RGC = {
    1,
    make_vector(0.0, 1.0, 0.0),
    make_vector(0.0, 0.0, 2.0),
    make_vector(1.0, 0.0, 0.0),
    make_vector(0.0, 1.0, 0.0),
    make_vector(0.0, 0.0, -1.0),
    0,
    0,
    0,
    0,
    1,
    1,
    0
};





void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    RGC.zoom *= (1.0 - yoffset * 0.1);
    RGC.zoom = max(1e-12, min(1e12, RGC.zoom));
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    if (RGC.move) {
        if (RGC.position_start) {
            RGC.position_start = 0;
            RGC.last_position_x = xpos;
            RGC.last_position_y = ypos;
        } else {
            /*
            double dx = (xpos - last_position_x)  / width * 2 / zoom;
            double dy = -(ypos - last_position_y)  / height * 2 / zoom;

            // Get "up" vector

            camera_pos.x += dx;
            camera_pos.y += dy;

            */

            double dx = -(xpos - RGC.last_position_x); // * 2 / zoom;
            double dy = -(ypos - RGC.last_position_y); // * 2 / zoom;

            RGC.yaw += dx * 1e-3;
            RGC.pitch += dy * 1e-3;

            RGC.camera_front.x = cos(RGC.yaw) * cos(RGC.pitch);
            RGC.camera_front.y = sin(RGC.pitch);
            RGC.camera_front.z = sin(RGC.yaw) * cos(RGC.pitch);
            RGC.camera_front /= sqrt(dot(RGC.camera_front, RGC.camera_front));

            RGC.camera_right = cross(RGC.up_direction, RGC.camera_front);
            RGC.camera_right /= sqrt(dot(RGC.camera_right, RGC.camera_right));

            RGC.camera_up = cross(RGC.camera_front, RGC.camera_right);
            RGC.camera_up /= sqrt(dot(RGC.camera_up, RGC.camera_up));

            RGC.last_position_x = xpos;
            RGC.last_position_y = ypos;
        }
    }
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        //move = 1;
        //position_start = 1;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        //move = 0;
    }

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
        RGC.compute = !RGC.compute;
}

void RenderCameraEvents(GLFWwindow* window, double elapsed_time) {
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(window, 1);
    }

    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_W)) {
        RGC.camera_pos += RGC.camera_front * elapsed_time;
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_S)) {
        RGC.camera_pos -= RGC.camera_front * elapsed_time;
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_A)) {
        RGC.camera_pos -= RGC.camera_right * elapsed_time;
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_D)) {
        RGC.camera_pos += RGC.camera_right * elapsed_time;
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_SPACE)) {
        RGC.camera_pos += RGC.camera_up * elapsed_time;
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
        RGC.camera_pos -= RGC.camera_up * elapsed_time;
    }
}



GLFWwindow* RenderStart() {
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        return NULL;
    } 


    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vmode = glfwGetVideoMode(mon);
    GLFWwindow* window = glfwCreateWindow(vmode->width, vmode->height, "Hello Triangle", mon, NULL);
    if (!window) {
        fprintf(stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
                                    
    // start GLEW extension handler
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        printf("ERROR: glew initialization failed\n");
        return NULL;
    }

    
    // get version info
    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION); // version as a string
    printf("Renderer: %s\n", renderer);
    printf("OpenGL version supported %s\n", version);

    // tell GL to only draw onto a pixel if the shape is closer to the viewer
    glEnable(GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"

    return window;
}

int RenderEnd() {
    // close GL context and any other GLFW resources
    glfwTerminate();
    return 0;
}



GLuint RenderSetupShaders() {
    
    /*
    const char* vertex_shader =
    "#version 400\n"
    "layout (location = 0) in vec3 vp;"
    "layout (location = 1) in vec3 col;"
    "uniform vec2 window;"
    "uniform vec3 camera_pos;"
    "uniform vec3 camera_front;"
    "uniform vec3 camera_up;"
    "uniform vec3 camera_right;"
    "uniform float zoom;"
    "out vec3 acol;"
    ""
    "void main() {"
    "   vec4 p = vec4(vp, 1.0);" //"   p.x *= window.y;"   "   p.y *= window.x;"
    "" // Transform to world coordinates
    "   mat4 rot = mat4(mat3(camera_right, camera_up, camera_front));"
    "   mat4 tran = mat4(1.0);"
    "   tran[3] = vec4(-camera_pos, 1.0);"
    "   mat4 proj = mat4(1.0);"
    "   proj[0][0] = 0.5 / zoom;"
    "   proj[1][1] = 0.5 / zoom;"
    "   proj[2][2] = 0.2 / zoom;"
    "   proj[2][3] = 0.5;"
    "   gl_Position = proj * (transpose(rot) * tran *  p);"
    "   gl_Position.x *= window.y / window.x;"
    "   acol = col;"
    "}";
    */

    const char* vertex_shader =
    "#version 400\n"
    "layout (location = 0) in vec3 vp;"
    "layout (location = 1) in vec3 col;"
    "uniform mat4 camera_matrix;"
    "out vec3 acol;"
    "void main() {"
    "   vec4 p = vec4(vp, 1.0);"
    "   gl_Position = camera_matrix * p;"
    "   acol = col;"
    "}";

    const char* fragment_shader =
    "#version 400\n"
    "out vec4 frag_colour;"
    "in vec3 acol;"
    "void main() {"
    "  frag_colour = vec4(acol, 1.0);"
    "}";

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, NULL);
    glCompileShader(vs);
    GLint success = 0;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        printf("Failed to compile vertex shader\n");
        GLint logSize = 0;
        glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logSize);
        char* log = (char*)malloc(sizeof(char) * logSize);
        GLsizei length;
        glGetShaderInfoLog(vs, logSize, &length, log);
        printf("%s\n", log);
    }
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        printf("Failed to compile fragment shader\n");
        GLint logSize = 0;
        glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logSize);
        char* log = (char*)malloc(sizeof(char) * logSize);
        GLsizei length;
        glGetShaderInfoLog(fs, logSize, &length, log);
        printf("%s\n", log);
    }

    GLuint shader_program = glCreateProgram();
    glAttachShader(shader_program, fs);
    glAttachShader(shader_program, vs);
    glLinkProgram(shader_program);

    GLint isLinked = 0;
    glGetProgramiv(shader_program, GL_LINK_STATUS, (int *)&isLinked);
    if (isLinked == GL_FALSE) {
        printf("Failed to link shader program\n");
        GLint maxLength = 0;
        glGetProgramiv(shader_program, GL_INFO_LOG_LENGTH, &maxLength);

        // The maxLength includes the NULL character
        GLchar* infoLog = (GLchar*)malloc(sizeof(GLchar)*maxLength);
        glGetProgramInfoLog(shader_program, maxLength, &maxLength, infoLog);
        
        // We don't need the program anymore.
        glDeleteProgram(shader_program);
        // Don't leak shaders either.
        glDeleteShader(vs);
        glDeleteShader(fs);

        // Use the infoLog as you see fit.
        printf("%s\n", infoLog);
    }

    return shader_program;
}


void RenderSetupCamera(GLFWwindow* window) {
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);

    glDepthRange(0,1000);
    glFrustum(-5,5,-5,5,1,1000);
}


void matrix4_mult(
    float* X, float* A, float* B
) {

    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            X[4*i + j] = 0.0;
            for (int k=0; k<4; ++k) X[4*i + j] += 
                A[4*i + k]*B[4*k + j]; 
        }
    }
}


void RenderUpdateCamera(GLFWwindow* window, GLuint shader_program) {
    /*
    GLuint window_loc = glGetUniformLocation(shader_program, "window");
    GLuint camera_pos_loc = glGetUniformLocation(shader_program, "camera_pos");
    GLuint camera_front_loc = glGetUniformLocation(shader_program, "camera_front");
    GLuint camera_up_loc = glGetUniformLocation(shader_program, "camera_up");
    GLuint camera_right_loc = glGetUniformLocation(shader_program, "camera_right");
    GLuint zoom_loc = glGetUniformLocation(shader_program, "zoom");
    */

    int width, height;

    glfwGetWindowSize(window, &width, &height);
    double aratio = ((double) height) / ((double) width);

    /*
    glUniform2f(window_loc, (GLfloat)(width) / 1000.0, (GLfloat)(height) / 1000.0);
    glUniform3f(camera_pos_loc, RGC.camera_pos.x, RGC.camera_pos.y, RGC.camera_pos.z);
    glUniform3f(camera_front_loc, RGC.camera_front.x, RGC.camera_front.y, RGC.camera_front.z);
    glUniform3f(camera_up_loc, RGC.camera_up.x, RGC.camera_up.y, RGC.camera_up.z);
    glUniform3f(camera_right_loc, RGC.camera_right.x, RGC.camera_right.y, RGC.camera_right.z);
    glUniform1f(zoom_loc, RGC.zoom);
    */


    // Update the matrix

    float rot[16] = {
        (float)RGC.camera_right.x, (float)RGC.camera_right.y, (float)RGC.camera_right.z, 0.0,
        (float)RGC.camera_up.x,  (float)RGC.camera_up.y, (float)RGC.camera_up.z, 0.0,
        (float)RGC.camera_front.x, (float)RGC.camera_front.y, (float)RGC.camera_front.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    float trans[16] = {
        1.0, 0.0, 0.0, -(float)RGC.camera_pos.x,
        0.0, 1.0, 0.0, -(float)RGC.camera_pos.y,
        0.0, 0.0, 1.0, -(float)RGC.camera_pos.z,
        0.0, 0.0, 0.0, 1.0
    };
    float proj[16] = {
        0.5f / (float)RGC.zoom * (float)aratio, 0.0, 0.0, 0.0,
        0.0, 0.5f / (float)RGC.zoom, 0.0, 0.0,
        0.0, 0.0, 0.2f / (float)RGC.zoom, 0.5f,
        0.0, 0.0, 0.0, 1.0f
    };
    float tmp[16];
    matrix4_mult(tmp, proj, rot);
    matrix4_mult(RGC.matrix, tmp, trans);

    printOpenGLError();


    GLuint matrix_loc = glGetUniformLocation(shader_program, "camera_matrix");

    printOpenGLError();


    glUniformMatrix4fv(
        matrix_loc, 
        1, 
        GL_TRUE, 
        RGC.matrix
    );
    
    printOpenGLError();


}



void Render_update_fps_counter(GLFWwindow* window) {
    static double previous_seconds = -1.0;
    if (previous_seconds == -1.0) previous_seconds = glfwGetTime();
    static int frame_count;
    double current_seconds = glfwGetTime();
    double elapsed_seconds = current_seconds - previous_seconds;
    if (elapsed_seconds > 0.25) {
        previous_seconds = current_seconds;
        double fps = (double)frame_count / elapsed_seconds;
        char tmp[128];
        sprintf(tmp, "opengl @ fps: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        frame_count = 0;
        //printf("fps: %5.1lf\n", fps);
    }
    frame_count++;
}



void RenderSetWireframe() {
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
}

void RenderSetFill() {
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}






RenderMesh* RenderMesh_new(
    const double* nodes,
    const uint* elements,
    uint n_nodes,
    uint n_elements
) {

    // Alloc cpu mesh
    float* r_elements = (float*)malloc(sizeof(float) * 18 * n_elements);
    uint* adresses = (uint*)malloc(sizeof(uint) * 2 * n_elements);

    for (size_t e = 0; e < n_elements; ++e) {
        // Loop over each node
        //if (e < 100) printf("element: %u %u %u\n", elements[3*e + 0], elements[3*e + 1], elements[3*e + 2]);
        for (size_t n = 0; n < 3; ++n) {
            // Position
            r_elements[18*e + 6*n + 0] = (float)nodes[3*elements[3*e + n] + 0];
            r_elements[18*e + 6*n + 1] = (float)nodes[3*elements[3*e + n] + 1];
            r_elements[18*e + 6*n + 2] = (float)nodes[3*elements[3*e + n] + 2];

            // Color
            r_elements[18*e + 6*n + 3] = 1.0f;
            r_elements[18*e + 6*n + 4] = 1.0f;
            r_elements[18*e + 6*n + 5] = 1.0f;
        }
    }

    RenderMesh* mesh = RenderMesh_from_allocs(
        r_elements,
        adresses,
        n_elements
    );

    return mesh;
}

RenderMesh* RenderMesh_from_allocs(
    float* elements,
    uint* adresses,
    uint n_elements
) {
    RenderMesh* mesh = (RenderMesh*)malloc(sizeof(RenderMesh));
    mesh->cpu = (_RMesh*)malloc(sizeof(_RMesh));
    mesh->hgpu = (_RMesh*)malloc(sizeof(_RMesh));

    // Alloc cpu mesh
    mesh->cpu->elements = elements;
    mesh->cpu->adresses = adresses;

    mesh->cpu->n_elements = n_elements;

    // Alloc host gpu mesh
    mesh->hgpu->n_elements = n_elements;
    cudaMalloc((void**)&mesh->hgpu->adresses, sizeof(uint) * 2 * n_elements);
    cudaMemcpy(mesh->hgpu->adresses, adresses, sizeof(uint) * 2 * n_elements, cudaMemcpyHostToDevice);

    // Alloc gpu mesh
    cudaMalloc((void**)&mesh->gpu, sizeof(_RMesh));

    cudaMemcpy(mesh->gpu, mesh->hgpu, sizeof(_RMesh), cudaMemcpyHostToDevice);


    if (RenderMesh_to_gl(mesh)) {
        printf("ERROR: Mapping render mesh to OpenGL failed.\n");
    }

    return mesh;
}


void RenderMesh_free(RenderMesh* mesh) {
    free(mesh->cpu->elements);
    free(mesh->cpu->adresses);
    free(mesh->cpu);

    cudaFree(mesh->hgpu->adresses);
    free(mesh->hgpu);

    cudaFree(mesh->gpu);

    cudaGLUnregisterBufferObject( mesh->VBO );

    free(mesh);
}


RenderMesh* RenderMesh_from_file(const char* filename) {

    FILE* file = fopen(filename, "rt");

    char buff[128];
    fscanf(file, "%s", buff);

    // Read elements
    while (strcmp(buff, "NELEM=") != 0) fscanf(file, "%s", buff);
    
    uint n_elements;
    fscanf(file, "%u", &n_elements);

    uint* elements = (uint*)malloc(sizeof(uint) * n_elements * 3);

    for (uint i=0; i<n_elements; ++i) {
        uint _ignore;
        fscanf(file, "%u", &_ignore);
        
        for (int j=0; j<3; ++j) {
            fscanf(file, "%u", elements + 3*i + j);
        }

        //fscanf(file, "%u", &_ignore);
    }

    // Read nodes
    while (strcmp(buff, "NPOIN=") != 0) fscanf(file, "%s", buff);
    
    uint n_nodes;
    fscanf(file, "%u", &n_nodes);

    double* nodes = (double*)malloc(sizeof(double) * n_nodes * 3);

    for (uint i=0; i<n_nodes; ++i) {
        uint _ignore;

        for (int j=0; j<3; ++j) {
            fscanf(file, "%lf", nodes + 3*i + j);
        }
        //nodes[3*i + 2] = 0;

        fscanf(file, "%u", &_ignore);
    }

    fclose(file);

    RenderMesh* mesh = RenderMesh_new(
        nodes,
        elements,
        n_nodes,
        n_elements
    );

    free(nodes);
    free(elements);

    return mesh;
}



RenderMesh* RenderMesh_from_region(cuMesh* mesh_in, const char* region_name) {

    cuMeshBase* mesh = mesh_in->_cpu;

    // Get region size info
    uint region = cuMeshGetRegionId(mesh_in, region_name);

    uint* ghosts_ids = mesh->ghost_cell_ids + mesh->ghost_cell_starts[region];
    uint size = cuMeshGetRegionSize(mesh, region);

    // get the number of triangles
    uint n_triangles = 0;
    for (uint i=0; i<size; ++i) {
        uint cell = ghosts_ids[i];
        uint face = mesh->cell_faces[mesh->cell_faces_starts[cell]];

        uint fsize = cuMeshGetFaceSize(mesh, face);
        uint n_ftris = fsize - 2;

        n_triangles += n_ftris;
    }

    // Alloc the elements and adresses arrays
    float* elements = (float*)malloc(sizeof(float) * 18 * n_triangles);
    uint* adresses = (uint*)malloc(sizeof(uint) * 2 * n_triangles);
    uint tri = 0;
    for (uint i=0; i<size; ++i) {
        uint cell = ghosts_ids[i];
        uint face = mesh->cell_faces[mesh->cell_faces_starts[cell]];

        uint fsize = cuMeshGetFaceSize(mesh, face);
        uint n_ftris = fsize - 2;

        uint* face_nodes = cuMeshGetFaceStart(mesh, face);

        for (uint j=0; j<n_ftris; ++j) {
            // Get this triangles nodes
            uint n0 = 0;
            uint n1 = j + 1;
            uint n2 = j + 2;
            uint tri_node[3];
            tri_node[0] = face_nodes[n0];
            tri_node[1] = face_nodes[n1];
            tri_node[2] = face_nodes[n2];

            // Add this triangle
            for (uint n = 0; n < 3; ++n) {
                vector node = mesh->nodes[tri_node[n]];
                // Position
                elements[18*tri + 6*n + 0] = node.x;
                elements[18*tri + 6*n + 1] = node.y;
                elements[18*tri + 6*n + 2] = node.z;

                // Color
                elements[18*tri + 6*n + 3] = 1.0f;
                elements[18*tri + 6*n + 4] = 1.0f;
                elements[18*tri + 6*n + 5] = 1.0f;
            }
            adresses[2*tri] = cell;
            adresses[2*tri + 1] = cuMeshGetCellConnect(mesh, cell)[0];

            tri++;
        }
    }

    // Create the render mesh
    RenderMesh* rmesh = RenderMesh_from_allocs(
        elements,
        adresses,
        n_triangles
    );

    return rmesh;
}



// Display a mesh
int RenderMesh_to_gl(RenderMesh* mesh) {

    glGenVertexArrays(1, &mesh->VAO);
    glGenBuffers(1, &mesh->VBO);

    glBindVertexArray(mesh->VAO); //Bind the VAO

    //Bind the buffers
    glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);
    
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mesh->cpu->n_elements * 18, mesh->cpu->elements, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);


    //Supply Index Buffer information
    glBindVertexArray(0); //Unbind the VAO

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Register buffer
    cudaGLRegisterBufferObject( mesh->VBO );

    return 0;
}


void RenderMesh_map(RenderMesh* mesh) {
    cudaGLMapBufferObject((void**)&mesh->hgpu->elements, mesh->VBO);
    cudaMemcpy(mesh->gpu, mesh->hgpu, sizeof(_RMesh), cudaMemcpyHostToDevice);
}

void RenderMesh_unmap(RenderMesh* mesh) {
    cudaGLUnmapBufferObject(mesh->VBO);
}


void RenderMesh_draw(RenderMesh* mesh) {
    glBindVertexArray(mesh->VAO);
    glDrawArrays(GL_TRIANGLES, 0, mesh->cpu->n_elements * 3);
}


__global__ void KernelRenderScalarField(
    _RMesh* mesh, FvmFieldBase* field,
    scalar u_min, scalar u_max
) {
    int i = cudaGlobalId;
    if (i >= mesh->n_elements) return;

    scalar* u_field = ((scalar*)field->value);
    vector* ug_field = ((vector*)field->gradient);
    scalar* ul_field = ((scalar*)field->limiter);

    // Set color attributes for this element
    uint cell0 = mesh->adresses[2*i];
    uint cell1 = mesh->adresses[2*i+1];

    scalar u =  (u_field[cell0] + u_field[cell1])*0.5;
    vector ug = (ug_field[cell0] + ug_field[cell1])*0.5;
    scalar ul = (ul_field[cell0] + ul_field[cell1])*0.5;

    vector c;
    reset(c);

    for (uint n = 0; n < 3; ++n) {
        uint ni = 18*i + 6*n;
        c.x += mesh->elements[ni + 0] / 3.0;
        c.y += mesh->elements[ni + 1] / 3.0;
        c.z += mesh->elements[ni + 2] / 3.0;
    }

    for (uint n = 0; n < 3; ++n) {
        uint ni = 18*i + 6*n;

        // Reconstructed value
        vector np = make_vector(
            (scalar)mesh->elements[ni + 0],
            (scalar)mesh->elements[ni + 1],
            (scalar)mesh->elements[ni + 2]
        );
        vector d = np - c;
        scalar u_f = u + dot(ug, d) * ul;

        u_f = (u_f - u_min) / (u_max - u_min);
        u_f = max(0.0, min(1.0, u_f));

        vector color;
        if (u_f <= 0.2) {
            color.x = 0;
            color.y = 0;
            color.z = 0.5 + u_f * 0.5 / 0.2;
        } else
        if (u_f <= 0.4) {
            color.x = 0;
            color.y = (u_f - 0.2) * 1.0 / 0.2;
            color.z = 1.0;
        } else
        if (u_f <= 0.6) {
            color.x = (u_f - 0.4) * 1.0 / 0.2;
            color.y = 1.0;
            color.z = 1.0 - (u_f - 0.4) * 1.0 / 0.2;
        } else
        if (u_f <= 0.8) {
            color.x = 1.0;
            color.y = 1.0 - (u_f - 0.6) * 1.0 / 0.2;
            color.z = 0;
        } else
        if (u_f <= 1.0) {
            color.x = 1.0 - (u_f - 0.8) * 0.5 / 0.2;
            color.y = 0;
            color.z = 0;
        }

        // Color attribute
        mesh->elements[ni + 3] = (float)color.x;
        mesh->elements[ni + 4] = (float)color.y;
        mesh->elements[ni + 5] = (float)color.z;
    }
}



void RenderScalarField(
    RenderMesh* mesh, FvmField* field, scalar u_min, scalar u_max
) {
    KernelRenderScalarField <<< cuda_size(mesh->cpu->n_elements, CKW), CKW >>>(
        mesh->gpu, field->_gpu,
        u_min, u_max
    );
}



void RenderCameraPositionToMesh(RenderMesh* mesh) {
    // Get average position
    float zmin = mesh->cpu->elements[0];
    float zmax = mesh->cpu->elements[0];
    for (uint i=0; i<mesh->cpu->n_elements; ++i) {
        for (uint j = 0; j < 3; ++j) {
            zmin = fmin(zmin, mesh->cpu->elements[18*i + 6*j + 2]);
            zmax = fmax(zmax, mesh->cpu->elements[18*i + 6*j + 2]);
        }
    }
    printf("z min/max = %lf / %lf \n", zmin, zmax);

    float dz = zmax - zmin;

    float cz = (zmax + zmin) * 0.5 - 2.0*dz;

    RGC.camera_pos.z = cz;
}


int printOglError(const char *file, int line) {
    /* Returns 1 if an OpenGL error occurred, 0 otherwise. */
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR) {
        printf("glError in file %s @ line %d: %s\n", file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}





