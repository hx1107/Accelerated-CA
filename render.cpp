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
GLuint load_shader_program(const char* vtx_shader_file, const char* frag_shader_file);

int do_stop_render = 0;
int display_ready = 0;

pthread_t win_thread_id;
SDL_Window* window;
SDL_GLContext context;

float test_img[] = { 0, 1., 1, 0. };
//unsigned int test_img[] = { 0, 9999, 9999, 0 };

void check_GL_error()
{
    const GLubyte* errString = NULL;
    int errCode;
    if ((errCode = glGetError()) != GL_NO_ERROR) {
        errString = gluErrorString(errCode);
        debug_print("GL Error: %s\n", errString);
    } else {
        debug_print("No error\n");
    }
}

void* window_thread(void* args)
{
    debug_print("Thread for window is running\n");
    SDL_Init(SDL_INIT_EVERYTHING);
    debug_print("OpenGL Render Initialized!\n");

    window = SDL_CreateWindow("Accelerated CA", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT, SDL_WINDOW_OPENGL);
    context = SDL_GL_CreateContext(window);
    glewExperimental = GL_TRUE;
    glewInit();

    // Default to blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Compile shaders and use them
    //GLuint shaderProgram = load_shader_program("./shaders/vtxShader.vs", "./shaders/fragShader.fs");
    //glBindFragDataLocation(shaderProgram, 0, "color");
    //check_GL_error();
    //glLinkProgram(shaderProgram);
    //char log_buffer[1024];
    //glGetProgramInfoLog(shaderProgram, sizeof(log_buffer), NULL, log_buffer);
    //debug_print("Linking log: \n\t %s", log_buffer);
    //glUseProgram(shaderProgram);

    // Create texture
    glEnable(GL_TEXTURE_2D);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    while (host_buffer == nullptr)
        ; // wait until host buffer is initialized
    debug_print("Host Buffer is ready: %p\n", host_buffer);
    sleep(1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 2, 2, 0, GL_LUMINANCE, GL_BYTE, test_img);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, CANVAS_SIZE_X, CANVAS_SIZE_Y, 0, GL_LUMINANCE, GL_FLOAT, host_buffer);
    //           target, level, internal_format, w, h, boarder,  format, pixels
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glDisable(GL_TEXTURE_2D);
    check_GL_error();

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
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // draw texture
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 2, 2, 0, GL_RED, GL_UNSIGNED_INT, test_img);
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 2, 2, 0, GL_LUMINANCE, GL_FLOAT, test_img);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, CANVAS_SIZE_X, CANVAS_SIZE_Y, 0, GL_LUMINANCE, GL_FLOAT, host_buffer);

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

        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        // Swap buffers
        glFlush();
        SDL_GL_SwapWindow(window);
        display_ready = 1;
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

GLuint load_and_compile_shader(const char* source, GLenum type)
{
    // Compile and upload shaders
    debug_print("Compiling Shaders: ");
    GLuint shader = glCreateShader(type);
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
    GLuint vtx = load_and_compile_shader(vtx_src.c_str(), GL_VERTEX_SHADER);
    glAttachShader(shaderProgram, vtx);
    glDeleteShader(vtx);

    string frag_src = slurp(frag_shader_file);
    GLuint frag = load_and_compile_shader(frag_src.c_str(), GL_FRAGMENT_SHADER);
    glAttachShader(shaderProgram, frag);
    glDeleteShader(frag);

    return shaderProgram;
}
