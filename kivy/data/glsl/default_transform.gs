#version 320 es

#ifdef GL_ES
precision mediump float;
#endif

layout(points)in;
layout(triangle_strip,max_vertices=3)out;

in VS_OUT{
    float geoValue;
}gs_in[];

out float outValue;

void main()
{
    for(int i=0;i<3;i++){
        outValue=gs_in[0].geoValue+0.1*float(i);
        EmitVertex();
    }
    
    EndPrimitive();
}
