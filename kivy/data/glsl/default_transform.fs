#version 320 es

#ifdef GL_ES
precision mediump float;
#endif

out vec4 fragColor;

void main(){
    fragColor=vec4(1.,1.,1.,1.);
}
