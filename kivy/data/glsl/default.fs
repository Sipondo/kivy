#version 320 es

$HEADER$
void main(void){
    fragColor=frag_color*texture(texture0,tex_coord0);
}
