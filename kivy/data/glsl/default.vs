#version 320 es

$HEADER$
void main(void){
  frag_color=color*vec4(1.,1.,1.,opacity);
  tex_coord0=vTexCoords0;
  gl_Position=projection_mat*modelview_mat*vec4(vPosition.xy,0.,1.);
}
