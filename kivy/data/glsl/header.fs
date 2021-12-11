#ifdef GL_ES
precision highp float;
#endif

/* Outputs from the vertex shader */
in vec4 frag_color;
in vec2 tex_coord0;

/* uniform texture samplers */
uniform sampler2D texture0;

uniform mat4 frag_modelview_mat;

out vec4 fragColor;
