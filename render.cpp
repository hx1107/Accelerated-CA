//#define GLEW_STATIC
#define GL_GLEXT_PROTOTYPES
#include "render.h"
#include "config.h"
//#include <GL/gl.h>
#include <GL/glew.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <SDL2/SDL.h>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <unistd.h>
using namespace std;
GLuint load_and_compile_shader(const char* source);
GLuint load_shader_program(char* vtx_shader_file, char* frag_shader_file);

int do_stop_render = 0;

pthread_t win_thread_id;
SDL_Window* window;
SDL_GLContext context;
//GLfloat vertices[] = {
////  Position      Color             Texcoords
//-0.5f, 0.5f, 0.0f, 0.0f, // Top-left
//0.5f, 0.5f, 1.0f, 0.0f,  // Top-right
//0.5f, -0.5f, 1.0f, 1.0f, // Bottom-right
//-0.5f, -0.5f, 0.0f, 1.0f // Bottom-left
//};
GLfloat vertices[] = {
    -0.5f, 0.5f,  // Top-left
    0.5f, 0.5f,   // Top-right
    0.5f, -0.5f,  // Bottom-right
    -0.5f, -0.5f, // Bottom-left
};

void* window_thread(void* args)
{
    debug_print("Thread for window is running\n");
    SDL_Init(SDL_INIT_EVERYTHING);
    debug_print("OpenGL Render Initialized!\n");

    window = SDL_CreateWindow("Accelerated CA", 100, 100, 800, 600, SDL_WINDOW_OPENGL);
    context = SDL_GL_CreateContext(window);
    glewExperimental = GL_TRUE;
    glewInit();

    // Default to blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);
    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Compile shaders and use them
    GLuint shaderProgram = load_shader_program("./shaders/vtxShader.vs", "./shaders/fragShader.fs");
    glBindFragDataLocation(shaderProgram, 0, "color");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Draw a triangle from the vertices
    // TODO only one attr usable
    //GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    // Specify the layout of the vertex data
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    //             (index, size, type, stride, pointer);
    //glDisableVertexAttribArray(0);

    // Create texture
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    while (!host_buffer)
        ; // wait until host buffer is initialized
    debug_print("Host Buffer is ready!\n");
    sleep(1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, CANVAS_SIZE_X, CANVAS_SIZE_Y, 0, GL_RGB, GL_BYTE, host_buffer);
    //           target, level, internal_format, w, h, boarder,  format, pixels
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    SDL_Event windowEvent;
    while (!do_stop_render) {
        if (SDL_PollEvent(&windowEvent)) {
            if (windowEvent.type == SDL_QUIT) {
                debug_print("Exit\n");
                abort();
                break;
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glDrawArrays(GL_TRIANGLES, 0, COUNT_OF(vertices) / 2); // Draw the rectangle

        // draw texture
        //glBindTexture(GL_TEXTURE_2D, tex);
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(-RENDER_WIDTH, -RENDER_WIDTH);
        glTexCoord2f(0, RENDER_WIDTH);
        glVertex2f(-RENDER_WIDTH, RENDER_WIDTH);
        glTexCoord2f(RENDER_WIDTH, RENDER_WIDTH);
        glVertex2f(RENDER_WIDTH, RENDER_WIDTH);
        glTexCoord2f(RENDER_WIDTH, 0);
        glVertex2f(RENDER_WIDTH, -RENDER_WIDTH);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        // Swap buffers
        SDL_GL_SwapWindow(window);
    }

    return NULL;
}

void render_init()
{
    pthread_create(&win_thread_id, NULL, window_thread, NULL);
}

void render_disinit()
{
    do_stop_render = 1;
    sleep(1); // Race condition here, but doesn't really matter for now
    SDL_GL_DeleteContext(context);
    SDL_Quit();
}

/** Read file into string. */
inline string slurp(const string& path)
{
    ostringstream buf;
    ifstream input(path.c_str());
    buf << input.rdbuf();
    return buf.str();
}

GLuint load_and_compile_shader(const char* source)
{
    // Compile and upload shaders
    debug_print("Compiling Shaders: ");
    GLuint shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint status; // Retrieve status and log
    char log_buffer[1024];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    glGetShaderInfoLog(shader, sizeof(log_buffer), NULL, log_buffer);
    debug_print(" %s\n\t%s\n", status == GL_TRUE ? "Success" : "Failed", log_buffer);
    if (status != GL_TRUE) {
        fprintf(stderr, "Shader compilation failed, aborting...\n");
        abort();
    }

    return shader;
}
GLuint load_shader_program(const char* vtx_shader_file, const char* frag_shader_file)
{
    GLuint shaderProgram = glCreateProgram();

    string vtx_src = slurp(vtx_shader_file);
    GLuint vtx = load_and_compile_shader(vtx_src.c_str());
    glAttachShader(shaderProgram, vtx);
    glDeleteShader(vtx);

    string frag_src = slurp(frag_shader_file);
    GLuint frag = load_and_compile_shader(frag_src.c_str());
    glAttachShader(shaderProgram, frag);
    glDeleteShader(frag);

    return shaderProgram;
}
