#version 330 core
layout(location = 0) in vec2 position;
//in vec2 uv;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0); // A vec4 built-in variable
    //gl_UV = uv
}

