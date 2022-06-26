"""
CGL/SDL: GL backend implementation using SDL2
"""

include "../common.pxi"
include "../../include/config.pxi"

from kivy.graphics.cgl cimport *

IF USE_SDL2:
    cdef extern from "SDL.h":
        void *SDL_GL_GetProcAddress(const char*)


cpdef is_backend_supported():
    return USE_SDL2


def init_backend():
    IF not USE_SDL2:
        raise TypeError('SDL2 is not available. Recompile with USE_SDL2=1')
    ELSE:
        # sdl2 window must have been created by now
        cgl.glActiveTexture = <GLACTIVETEXTUREPTR>SDL_GL_GetProcAddress("glActiveTexture")
        cgl.glAttachShader = <GLATTACHSHADERPTR>SDL_GL_GetProcAddress("glAttachShader")

        cgl.glBeginTransformFeedback = <GLBEGINTRANSFORMFEEDBACK>SDL_GL_GetProcAddress("glBeginTransformFeedback")
        cgl.glEndTransformFeedback = <GLENDTRANSFORMFEEDBACK>SDL_GL_GetProcAddress("glEndTransformFeedback")
        cgl.glTransformFeedbackVaryings = <GLTRANSFORMFEEDBACKVARYINGS>SDL_GL_GetProcAddress("glTransformFeedbackVaryings")
        # cgl.glGetTransformFeedbackVarying = <GLGETTRANSFORMFEEDBACKVARYING>SDL_GL_GetProcAddress("glGetTransformFeedbackVarying")
        cgl.glBindBufferBase = <GLBINDBUFFERBASE>SDL_GL_GetProcAddress("glBindBufferBase")
        cgl.glMapBufferRange = <GLMAPBUFFERRANGE>SDL_GL_GetProcAddress("glMapBufferRange")
        cgl.glUnmapBuffer = <GLUNMAPBUFFER>SDL_GL_GetProcAddress("glUnmapBuffer")

        cgl.glBindAttribLocation = <GLBINDATTRIBLOCATIONPTR>SDL_GL_GetProcAddress("glBindAttribLocation")
        cgl.glBindBuffer = <GLBINDBUFFERPTR>SDL_GL_GetProcAddress("glBindBuffer")
        cgl.glBindFramebuffer = <GLBINDFRAMEBUFFERPTR>SDL_GL_GetProcAddress("glBindFramebuffer")
        cgl.glBindRenderbuffer = <GLBINDRENDERBUFFERPTR>SDL_GL_GetProcAddress("glBindRenderbuffer")
        cgl.glBindTexture = <GLBINDTEXTUREPTR>SDL_GL_GetProcAddress("glBindTexture")
        cgl.glBlendColor = <GLBLENDCOLORPTR>SDL_GL_GetProcAddress("glBlendColor")
        cgl.glBlendEquation = <GLBLENDEQUATIONPTR>SDL_GL_GetProcAddress("glBlendEquation")
        cgl.glBlendEquationSeparate = <GLBLENDEQUATIONSEPARATEPTR>SDL_GL_GetProcAddress("glBlendEquationSeparate")
        cgl.glBlendFunc = <GLBLENDFUNCPTR>SDL_GL_GetProcAddress("glBlendFunc")
        cgl.glBlendFuncSeparate = <GLBLENDFUNCSEPARATEPTR>SDL_GL_GetProcAddress("glBlendFuncSeparate")
        cgl.glBufferData = <GLBUFFERDATAPTR>SDL_GL_GetProcAddress("glBufferData")
        cgl.glBufferSubData = <GLBUFFERSUBDATAPTR>SDL_GL_GetProcAddress("glBufferSubData")
        cgl.glCheckFramebufferStatus = <GLCHECKFRAMEBUFFERSTATUSPTR>SDL_GL_GetProcAddress("glCheckFramebufferStatus")
        cgl.glClear = <GLCLEARPTR>SDL_GL_GetProcAddress("glClear")
        cgl.glClearColor = <GLCLEARCOLORPTR>SDL_GL_GetProcAddress("glClearColor")
        cgl.glClearStencil = <GLCLEARSTENCILPTR>SDL_GL_GetProcAddress("glClearStencil")
        cgl.glColorMask = <GLCOLORMASKPTR>SDL_GL_GetProcAddress("glColorMask")
        cgl.glCompileShader = <GLCOMPILESHADERPTR>SDL_GL_GetProcAddress("glCompileShader")
        cgl.glCompressedTexImage2D = <GLCOMPRESSEDTEXIMAGE2DPTR>SDL_GL_GetProcAddress("glCompressedTexImage2D")
        cgl.glCompressedTexSubImage2D = <GLCOMPRESSEDTEXSUBIMAGE2DPTR>SDL_GL_GetProcAddress("glCompressedTexSubImage2D")
        cgl.glCopyTexImage2D = <GLCOPYTEXIMAGE2DPTR>SDL_GL_GetProcAddress("glCopyTexImage2D")
        cgl.glCopyTexSubImage2D = <GLCOPYTEXSUBIMAGE2DPTR>SDL_GL_GetProcAddress("glCopyTexSubImage2D")
        cgl.glCreateProgram = <GLCREATEPROGRAMPTR>SDL_GL_GetProcAddress("glCreateProgram")
        cgl.glCreateShader = <GLCREATESHADERPTR>SDL_GL_GetProcAddress("glCreateShader")
        cgl.glCullFace = <GLCULLFACEPTR>SDL_GL_GetProcAddress("glCullFace")
        cgl.glDeleteBuffers = <GLDELETEBUFFERSPTR>SDL_GL_GetProcAddress("glDeleteBuffers")
        cgl.glDeleteFramebuffers = <GLDELETEFRAMEBUFFERSPTR>SDL_GL_GetProcAddress("glDeleteFramebuffers")
        cgl.glDeleteProgram = <GLDELETEPROGRAMPTR>SDL_GL_GetProcAddress("glDeleteProgram")
        cgl.glDeleteRenderbuffers = <GLDELETERENDERBUFFERSPTR>SDL_GL_GetProcAddress("glDeleteRenderbuffers")
        cgl.glDeleteShader = <GLDELETESHADERPTR>SDL_GL_GetProcAddress("glDeleteShader")
        cgl.glDeleteTextures = <GLDELETETEXTURESPTR>SDL_GL_GetProcAddress("glDeleteTextures")
        cgl.glDepthFunc = <GLDEPTHFUNCPTR>SDL_GL_GetProcAddress("glDepthFunc")
        cgl.glDepthMask = <GLDEPTHMASKPTR>SDL_GL_GetProcAddress("glDepthMask")
        cgl.glDetachShader = <GLDETACHSHADERPTR>SDL_GL_GetProcAddress("glDetachShader")
        cgl.glDisable = <GLDISABLEPTR>SDL_GL_GetProcAddress("glDisable")
        cgl.glDisableVertexAttribArray = <GLDISABLEVERTEXATTRIBARRAYPTR>SDL_GL_GetProcAddress("glDisableVertexAttribArray")
        cgl.glDrawArrays = <GLDRAWARRAYSPTR>SDL_GL_GetProcAddress("glDrawArrays")
        cgl.glDrawElements = <GLDRAWELEMENTSPTR>SDL_GL_GetProcAddress("glDrawElements")
        cgl.glEnable = <GLENABLEPTR>SDL_GL_GetProcAddress("glEnable")
        cgl.glEnableVertexAttribArray = <GLENABLEVERTEXATTRIBARRAYPTR>SDL_GL_GetProcAddress("glEnableVertexAttribArray")
        cgl.glFinish = <GLFINISHPTR>SDL_GL_GetProcAddress("glFinish")
        cgl.glFlush = <GLFLUSHPTR>SDL_GL_GetProcAddress("glFlush")
        cgl.glFramebufferRenderbuffer = <GLFRAMEBUFFERRENDERBUFFERPTR>SDL_GL_GetProcAddress("glFramebufferRenderbuffer")
        cgl.glFramebufferTexture2D = <GLFRAMEBUFFERTEXTURE2DPTR>SDL_GL_GetProcAddress("glFramebufferTexture2D")
        cgl.glFrontFace = <GLFRONTFACEPTR>SDL_GL_GetProcAddress("glFrontFace")
        cgl.glGenBuffers = <GLGENBUFFERSPTR>SDL_GL_GetProcAddress("glGenBuffers")
        cgl.glGenerateMipmap = <GLGENERATEMIPMAPPTR>SDL_GL_GetProcAddress("glGenerateMipmap")
        cgl.glGenFramebuffers = <GLGENFRAMEBUFFERSPTR>SDL_GL_GetProcAddress("glGenFramebuffers")
        cgl.glGenRenderbuffers = <GLGENRENDERBUFFERSPTR>SDL_GL_GetProcAddress("glGenRenderbuffers")
        cgl.glGenTextures = <GLGENTEXTURESPTR>SDL_GL_GetProcAddress("glGenTextures")
        cgl.glGetActiveAttrib = <GLGETACTIVEATTRIBPTR>SDL_GL_GetProcAddress("glGetActiveAttrib")
        cgl.glGetActiveUniform = <GLGETACTIVEUNIFORMPTR>SDL_GL_GetProcAddress("glGetActiveUniform")
        cgl.glGetAttachedShaders = <GLGETATTACHEDSHADERSPTR>SDL_GL_GetProcAddress("glGetAttachedShaders")
        cgl.glGetAttribLocation = <GLGETATTRIBLOCATIONPTR>SDL_GL_GetProcAddress("glGetAttribLocation")
        cgl.glGetBooleanv = <GLGETBOOLEANVPTR>SDL_GL_GetProcAddress("glGetBooleanv")
        cgl.glGetBufferParameteriv = <GLGETBUFFERPARAMETERIVPTR>SDL_GL_GetProcAddress("glGetBufferParameteriv")
        cgl.glGetError = <GLGETERRORPTR>SDL_GL_GetProcAddress("glGetError")
        cgl.glGetFloatv = <GLGETFLOATVPTR>SDL_GL_GetProcAddress("glGetFloatv")
        cgl.glGetFramebufferAttachmentParameteriv = <GLGETFRAMEBUFFERATTACHMENTPARAMETERIVPTR>SDL_GL_GetProcAddress("glGetFramebufferAttachmentParameteriv")
        cgl.glGetIntegerv = <GLGETINTEGERVPTR>SDL_GL_GetProcAddress("glGetIntegerv")
        cgl.glGetProgramInfoLog = <GLGETPROGRAMINFOLOGPTR>SDL_GL_GetProcAddress("glGetProgramInfoLog")
        cgl.glGetProgramiv = <GLGETPROGRAMIVPTR>SDL_GL_GetProcAddress("glGetProgramiv")
        cgl.glGetRenderbufferParameteriv = <GLGETRENDERBUFFERPARAMETERIVPTR>SDL_GL_GetProcAddress("glGetRenderbufferParameteriv")
        cgl.glGetShaderInfoLog = <GLGETSHADERINFOLOGPTR>SDL_GL_GetProcAddress("glGetShaderInfoLog")
        cgl.glGetShaderiv = <GLGETSHADERIVPTR>SDL_GL_GetProcAddress("glGetShaderiv")
        cgl.glGetShaderSource = <GLGETSHADERSOURCEPTR>SDL_GL_GetProcAddress("glGetShaderSource")
        cgl.glGetString = <GLGETSTRINGPTR>SDL_GL_GetProcAddress("glGetString")
        cgl.glGetTexParameterfv = <GLGETTEXPARAMETERFVPTR>SDL_GL_GetProcAddress("glGetTexParameterfv")
        cgl.glGetTexParameteriv = <GLGETTEXPARAMETERIVPTR>SDL_GL_GetProcAddress("glGetTexParameteriv")
        cgl.glGetUniformfv = <GLGETUNIFORMFVPTR>SDL_GL_GetProcAddress("glGetUniformfv")
        cgl.glGetUniformiv = <GLGETUNIFORMIVPTR>SDL_GL_GetProcAddress("glGetUniformiv")
        cgl.glGetUniformLocation = <GLGETUNIFORMLOCATIONPTR>SDL_GL_GetProcAddress("glGetUniformLocation")
        cgl.glGetVertexAttribfv = <GLGETVERTEXATTRIBFVPTR>SDL_GL_GetProcAddress("glGetVertexAttribfv")
        cgl.glGetVertexAttribiv = <GLGETVERTEXATTRIBIVPTR>SDL_GL_GetProcAddress("glGetVertexAttribiv")
        cgl.glHint = <GLHINTPTR>SDL_GL_GetProcAddress("glHint")
        cgl.glIsBuffer = <GLISBUFFERPTR>SDL_GL_GetProcAddress("glIsBuffer")
        cgl.glIsEnabled = <GLISENABLEDPTR>SDL_GL_GetProcAddress("glIsEnabled")
        cgl.glIsFramebuffer = <GLISFRAMEBUFFERPTR>SDL_GL_GetProcAddress("glIsFramebuffer")
        cgl.glIsProgram = <GLISPROGRAMPTR>SDL_GL_GetProcAddress("glIsProgram")
        cgl.glIsRenderbuffer = <GLISRENDERBUFFERPTR>SDL_GL_GetProcAddress("glIsRenderbuffer")
        cgl.glIsShader = <GLISSHADERPTR>SDL_GL_GetProcAddress("glIsShader")
        cgl.glIsTexture = <GLISTEXTUREPTR>SDL_GL_GetProcAddress("glIsTexture")
        cgl.glLineWidth = <GLLINEWIDTHPTR>SDL_GL_GetProcAddress("glLineWidth")
        cgl.glLinkProgram = <GLLINKPROGRAMPTR>SDL_GL_GetProcAddress("glLinkProgram")
        cgl.glPixelStorei = <GLPIXELSTOREIPTR>SDL_GL_GetProcAddress("glPixelStorei")
        cgl.glPolygonOffset = <GLPOLYGONOFFSETPTR>SDL_GL_GetProcAddress("glPolygonOffset")
        cgl.glReadPixels = <GLREADPIXELSPTR>SDL_GL_GetProcAddress("glReadPixels")
        cgl.glRenderbufferStorage = <GLRENDERBUFFERSTORAGEPTR>SDL_GL_GetProcAddress("glRenderbufferStorage")
        cgl.glSampleCoverage = <GLSAMPLECOVERAGEPTR>SDL_GL_GetProcAddress("glSampleCoverage")
        cgl.glScissor = <GLSCISSORPTR>SDL_GL_GetProcAddress("glScissor")
        cgl.glShaderBinary = <GLSHADERBINARYPTR>SDL_GL_GetProcAddress("glShaderBinary")
        cgl.glShaderSource = <GLSHADERSOURCEPTR>SDL_GL_GetProcAddress("glShaderSource")
        cgl.glStencilFunc = <GLSTENCILFUNCPTR>SDL_GL_GetProcAddress("glStencilFunc")
        cgl.glStencilFuncSeparate = <GLSTENCILFUNCSEPARATEPTR>SDL_GL_GetProcAddress("glStencilFuncSeparate")
        cgl.glStencilMask = <GLSTENCILMASKPTR>SDL_GL_GetProcAddress("glStencilMask")
        cgl.glStencilMaskSeparate = <GLSTENCILMASKSEPARATEPTR>SDL_GL_GetProcAddress("glStencilMaskSeparate")
        cgl.glStencilOp = <GLSTENCILOPPTR>SDL_GL_GetProcAddress("glStencilOp")
        cgl.glStencilOpSeparate = <GLSTENCILOPSEPARATEPTR>SDL_GL_GetProcAddress("glStencilOpSeparate")
        cgl.glTexImage2D = <GLTEXIMAGE2DPTR>SDL_GL_GetProcAddress("glTexImage2D")
        cgl.glTexParameterf = <GLTEXPARAMETERFPTR>SDL_GL_GetProcAddress("glTexParameterf")
        cgl.glTexParameteri = <GLTEXPARAMETERIPTR>SDL_GL_GetProcAddress("glTexParameteri")
        cgl.glTexSubImage2D = <GLTEXSUBIMAGE2DPTR>SDL_GL_GetProcAddress("glTexSubImage2D")
        cgl.glUniform1f = <GLUNIFORM1FPTR>SDL_GL_GetProcAddress("glUniform1f")
        cgl.glUniform1fv = <GLUNIFORM1FVPTR>SDL_GL_GetProcAddress("glUniform1fv")
        cgl.glUniform1i = <GLUNIFORM1IPTR>SDL_GL_GetProcAddress("glUniform1i")
        cgl.glUniform1iv = <GLUNIFORM1IVPTR>SDL_GL_GetProcAddress("glUniform1iv")
        cgl.glUniform2f = <GLUNIFORM2FPTR>SDL_GL_GetProcAddress("glUniform2f")
        cgl.glUniform2fv = <GLUNIFORM2FVPTR>SDL_GL_GetProcAddress("glUniform2fv")
        cgl.glUniform2i = <GLUNIFORM2IPTR>SDL_GL_GetProcAddress("glUniform2i")
        cgl.glUniform2iv = <GLUNIFORM2IVPTR>SDL_GL_GetProcAddress("glUniform2iv")
        cgl.glUniform3f = <GLUNIFORM3FPTR>SDL_GL_GetProcAddress("glUniform3f")
        cgl.glUniform3fv = <GLUNIFORM3FVPTR>SDL_GL_GetProcAddress("glUniform3fv")
        cgl.glUniform3i = <GLUNIFORM3IPTR>SDL_GL_GetProcAddress("glUniform3i")
        cgl.glUniform3iv = <GLUNIFORM3IVPTR>SDL_GL_GetProcAddress("glUniform3iv")
        cgl.glUniform4f = <GLUNIFORM4FPTR>SDL_GL_GetProcAddress("glUniform4f")
        cgl.glUniform4fv = <GLUNIFORM4FVPTR>SDL_GL_GetProcAddress("glUniform4fv")
        cgl.glUniform4i = <GLUNIFORM4IPTR>SDL_GL_GetProcAddress("glUniform4i")
        cgl.glUniform4iv = <GLUNIFORM4IVPTR>SDL_GL_GetProcAddress("glUniform4iv")
        cgl.glUniformMatrix4fv = <GLUNIFORMMATRIX4FVPTR>SDL_GL_GetProcAddress("glUniformMatrix4fv")
        cgl.glUseProgram = <GLUSEPROGRAMPTR>SDL_GL_GetProcAddress("glUseProgram")
        cgl.glValidateProgram = <GLVALIDATEPROGRAMPTR>SDL_GL_GetProcAddress("glValidateProgram")
        cgl.glVertexAttrib1f = <GLVERTEXATTRIB1FPTR>SDL_GL_GetProcAddress("glVertexAttrib1f")
        cgl.glVertexAttrib2f = <GLVERTEXATTRIB2FPTR>SDL_GL_GetProcAddress("glVertexAttrib2f")
        cgl.glVertexAttrib3f = <GLVERTEXATTRIB3FPTR>SDL_GL_GetProcAddress("glVertexAttrib3f")
        cgl.glVertexAttrib4f = <GLVERTEXATTRIB4FPTR>SDL_GL_GetProcAddress("glVertexAttrib4f")
        cgl.glVertexAttribPointer = <GLVERTEXATTRIBPOINTERPTR>SDL_GL_GetProcAddress("glVertexAttribPointer")
        cgl.glViewport = <GLVIEWPORTPTR>SDL_GL_GetProcAddress("glViewport")
