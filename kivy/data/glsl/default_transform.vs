#version 320 es

#ifdef GL_ES
precision mediump float;
#endif

in float inValue;

out VS_OUT{
    float geoValue;
}vs_out;

void main()
{
    vs_out.geoValue=sqrt(inValue);
}
