'''
Canvas
======

The :class:`Canvas` is the root object used for drawing by a
:class:`~kivy.uix.widget.Widget`. Check the class documentation for more
information about the usage of Canvas.
'''

__all__ = ('Instruction', 'InstructionGroup',
           'ContextInstruction', 'VertexInstruction',
           'Canvas', 'CanvasBase',
           'RenderContext', 'Callback',
           'TransformFeedback')

include "../include/config.pxi"
include "opcodes.pxi"
include "gl_debug_logger.pxi"

from kivy.graphics.cgl cimport *
from kivy.compat import PY2
from kivy.logger import Logger
from kivy.graphics.context cimport get_context, Context
from weakref import proxy
from threading import get_ident

from os.path import join
from kivy import kivy_shader_dir

cdef str default_transform_vs = ''
cdef str default_transform_gs = ''
cdef str default_transform_fs = ''

with open(join(kivy_shader_dir, 'default_transform.vs')) as fin:
    default_transform_vs = fin.read()
with open(join(kivy_shader_dir, 'default_transform.gs')) as fin:
    default_transform_gs = fin.read()
with open(join(kivy_shader_dir, 'default_transform.fs')) as fin:
    default_transform_fs = fin.read()


cdef int _need_reset_gl = 1
cdef int _active_texture = -1
cdef list canvas_list = []

cdef void reset_gl_context():
    global _need_reset_gl, _active_texture
    _need_reset_gl = 0
    _active_texture = 0
    cgl.glEnable(GL_BLEND)
    cgl.glDisable(GL_DEPTH_TEST)
    cgl.glEnable(GL_STENCIL_TEST)
    cgl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    cgl.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)
    cgl.glActiveTexture(GL_TEXTURE0)
    cgl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)


cdef class Instruction(ObjectWithUid):
    '''Represents the smallest instruction available. This class is for internal
    usage only, don't use it directly.
    '''
    def __cinit__(self):
        self.__proxy_ref = None
        self.flags = 0
        self.parent = None

    def __init__(self, **kwargs):
        self.group = kwargs.get('group', None)
        if kwargs.get('noadd'):
            self.flags |= GI_NO_REMOVE
            return

        if verify_gl_main_thread and initialized_tid \
                and get_ident() != initialized_tid:
            raise TypeError("Cannot create graphics instruction outside "
                            "the main Kivy thread")

        self.parent = getActiveCanvas()
        if self.parent:
            self.parent.add(self)

    cdef int apply(self) except -1:
        return 0

    IF DEBUG:
        cpdef flag_update(self, int do_parent=1, list _instrs=None):
            cdef list instrs = _instrs if _instrs else []
            if _instrs and self in _instrs:
                raise RuntimeError('Encountered instruction group render loop: %r in %r' % (self, _instrs,))
            if do_parent == 1 and self.parent is not None:
                instrs.append(self)
                self.parent.flag_update(do_parent=1, _instrs=instrs)
            self.flags |= GI_NEEDS_UPDATE
    ELSE:
        cpdef flag_update(self, int do_parent=1):
            if do_parent == 1 and self.parent is not None:
                self.parent.flag_update()
            self.flags |= GI_NEEDS_UPDATE

    cpdef flag_data_update(self):
        if verify_gl_main_thread and initialized_tid \
                and get_ident() != initialized_tid:
            raise TypeError("Cannot change graphics instruction outside "
                            "the main Kivy thread")
        self.flag_update()

    cdef void flag_update_done(self):
        self.flags &= ~GI_NEEDS_UPDATE

    cdef void radd(self, InstructionGroup ig):
        ig.children.append(self)
        self.set_parent(ig)

    cdef void rremove(self, InstructionGroup ig):
        if self.parent is None:
            return
        ig.children.remove(self)
        self.set_parent(None)

    cdef void rinsert(self, InstructionGroup ig, int index):
        ig.children.insert(index, self)
        self.set_parent(ig)

    cdef void set_parent(self, Instruction parent):
        self.parent = parent

    cdef void reload(self) except *:
        self.flags |= GI_NEEDS_UPDATE
        self.flags &= ~GI_NO_APPLY_ONCE
        self.flags &= ~GI_IGNORE

    @property
    def needs_redraw(self):
        if (self.flags & GI_NEEDS_UPDATE) > 0:
            return True
        return False

    @property
    def proxy_ref(self):
        '''Return a proxy reference to the Instruction i.e. without creating a
        reference of the widget. See `weakref.proxy
        <http://docs.python.org/2/library/weakref.html?highlight=proxy#weakref.proxy>`_
        for more information.

        .. versionadded:: 1.7.2
        '''
        if self.__proxy_ref is None:
            self.__proxy_ref = proxy(self)
        return self.__proxy_ref


cdef class InstructionGroup(Instruction):
    """
    Group of :class:`Instruction`. Allows for the adding and removing
    of graphics instructions. It can be used directly as follows::

        blue = InstructionGroup()
        blue.add(Color(0, 0, 1, 0.2))
        blue.add(Rectangle(pos=self.pos, size=(100, 100)))

        green = InstructionGroup()
        green.add(Color(0, 1, 0, 0.4))
        green.add(Rectangle(pos=(100, 100), size=(100, 100)))

        # Here, self should be a Widget or subclass
        [self.canvas.add(group) for group in [blue, green]]
    """
    def __init__(self, **kwargs):
        Instruction.__init__(self, **kwargs)
        self.children = list()
        self.compiled_children = None
        if 'nocompiler' in kwargs:
            self.compiler = None
        else:
            self.compiler = GraphicsCompiler()

    cdef int apply(self) except -1:
        cdef Instruction c
        cdef list children
        if self.compiler is not None:
            if self.flags & GI_NEEDS_UPDATE:
                self.build()
            if self.compiled_children is not None and not (self.flags & GI_NO_APPLY_ONCE):
                children = self.compiled_children.children
                for c in children:
                    if c.flags & GI_IGNORE:
                        continue
                    c.apply()
            self.flags &= ~GI_NO_APPLY_ONCE
        else:
            for c in self.children:
                c.apply()
        return 0

    cdef void build(self):
        self.compiled_children = self.compiler.compile(self)
        self.flag_update_done()

    cpdef add(self, Instruction c):
        '''Add a new :class:`Instruction` to our list.
        '''
        c.radd(self)
        self.flag_data_update()
        return

    cpdef insert(self, int index, Instruction c):
        '''Insert a new :class:`Instruction` into our list at index.
        '''
        c.rinsert(self, index)
        self.flag_data_update()

    cpdef remove(self, Instruction c):
        '''Remove an existing :class:`Instruction` from our list.
        '''
        c.rremove(self)
        self.flag_data_update()

    def indexof(self, Instruction c):
        cdef int i
        for i in xrange(len(self.children)):
            if self.children[i] is c:
                return i
        return -1

    def length(self):
        return len(self.children)

    cpdef clear(self):
        '''Remove all the :class:`Instructions <Instruction>`.
        '''
        cdef Instruction c
        for c in self.children[:]:
            if c.flags & GI_NO_REMOVE:
                continue
            self.remove(c)

    cpdef remove_group(self, str groupname):
        '''Remove all :class:`Instructions <Instruction>` with a specific group
        name.
        '''
        cdef Instruction c
        for c in self.children[:]:
            if c.flags & GI_NO_REMOVE:
                continue
            if c.group == groupname:
                self.remove(c)

    cpdef get_group(self, str groupname):
        '''Return an iterable for all the :class:`Instructions <Instruction>`
        with a specific group name.
        '''
        cdef Instruction c
        return [c for c in self.children if c.group == groupname]

    cdef void reload(self) except *:
        Instruction.reload(self)
        cdef Instruction c
        for c in self.children:
            c.reload()


cdef class ContextInstruction(Instruction):
    '''The ContextInstruction class is the base for the creation of instructions
    that don't have a direct visual representation, but instead modify the
    current Canvas' state, e.g. texture binding, setting color parameters,
    matrix manipulation and so on.
    '''
    def __init__(self, **kwargs):
        Instruction.__init__(self, **kwargs)
        self.flags |= GI_CONTEXT_MOD
        self.context_state = dict()
        self.context_push = list()
        self.context_pop = list()

    cdef RenderContext get_context(self):
        cdef RenderContext context = getActiveContext()
        return context

    cdef int apply(self) except -1:
        cdef RenderContext context = self.get_context()
        if self.context_push:
            context.push_states(self.context_push)
        if self.context_state:
            context.set_states(self.context_state)
        if self.context_pop:
            context.pop_states(self.context_pop)
        return 0

    cdef int set_state(self, str name, value) except -1:
        self.context_state[name] = value
        self.flag_update()

    cdef int push_state(self, str name) except -1:
        self.context_push.append(name)
        self.flag_update()

    cdef int pop_state(self, str name) except -1:
        self.context_pop.append(name)
        self.flag_update()

cdef class VertexInstruction(Instruction):
    '''The VertexInstruction class is the base for all graphics instructions
    that have a direct visual representation on the canvas, such as Rectangles,
    Triangles, Lines, Ellipse and so on.
    '''
    def __init__(self, **kwargs):
        # Set a BindTexture instruction to bind the texture used for
        # this instruction before the actual vertex instruction
        self.texture_binding = BindTexture(noadd=True, **kwargs)
        self.texture = self.texture_binding.texture #auto compute tex coords
        tex_coords = kwargs.get('tex_coords')
        if tex_coords:
            self.tex_coords = tex_coords

        Instruction.__init__(self, **kwargs)
        self.flags = GI_VERTEX_DATA & GI_NEEDS_UPDATE
        self.batch = VertexBatch()

    cdef void radd(self, InstructionGroup ig):
        cdef Instruction instr = self.texture_binding
        ig.children.append(self.texture_binding)
        ig.children.append(self)
        instr.set_parent(ig)
        self.set_parent(ig)

    cdef void rinsert(self, InstructionGroup ig, int index):
        cdef Instruction instr = self.texture_binding
        ig.children.insert(index, self.texture_binding)
        ig.children.insert(index, self)
        instr.set_parent(ig)
        self.set_parent(ig)

    cdef void rremove(self, InstructionGroup ig):
        cdef Instruction instr = self.texture_binding
        ig.children.remove(self.texture_binding)
        ig.children.remove(self)
        instr.set_parent(None)
        self.set_parent(None)

    @property
    def texture(self):
        '''Property that represents the texture used for drawing this
        Instruction. You can set a new texture like this::

            from kivy.core.image import Image

            texture = Image('logo.png').texture
            with self.canvas:
                Rectangle(texture=texture, pos=self.pos, size=self.size)

        Usually, you will use the :attr:`source` attribute instead of the
        texture.
        '''
        return self.texture_binding.texture

    @texture.setter
    def texture(self, _tex):
        cdef Texture tex = _tex
        self.texture_binding.texture = tex
        if tex:
            self.tex_coords = tex.tex_coords
        else:
            self.tex_coords = [0.0,0.0, 1.0,0.0, 1.0,1.0, 0.0,1.0]
        self.flag_data_update()

    @property
    def source(self):
        '''This property represents the filename to load the texture from.
        If you want to use an image as source, do it like this::

            with self.canvas:
                Rectangle(source='mylogo.png', pos=self.pos, size=self.size)

        Here's the equivalent in Kivy language:

        .. code-block:: kv

            <MyWidget>:
                canvas:
                    Rectangle:
                        source: 'mylogo.png'
                        pos: self.pos
                        size: self.size

        .. note::

            The filename will be searched for using the
            :func:`kivy.resources.resource_find` function.

        '''
        return self.texture_binding.source

    @source.setter
    def source(self, source):
        self.texture_binding.source = source
        self.texture = self.texture_binding._texture

    @property
    def tex_coords(self):
        '''This property represents the texture coordinates used for drawing the
        vertex instruction. The value must be a list of 8 values.

        A texture coordinate has a position (u, v), and a size (w, h). The size
        can be negative, and would represent the 'flipped' texture. By default,
        the tex_coords are::

            [u, v, u + w, v, u + w, v + h, u, v + h]

        You can pass your own texture coordinates if you want to achieve fancy
        effects.

        .. warning::

            The default values just mentioned can be negative. Depending
            on the image and label providers, the coordinates are flipped
            vertically because of the order in which the image is internally
            stored. Instead of flipping the image data, we are just flipping
            the texture coordinates to be faster.

        '''
        return (
            self._tex_coords[0],
            self._tex_coords[1],
            self._tex_coords[2],
            self._tex_coords[3],
            self._tex_coords[4],
            self._tex_coords[5],
            self._tex_coords[6],
            self._tex_coords[7])

    @tex_coords.setter
    def tex_coords(self, tc):
        cdef int index
        for index in xrange(8):
            self._tex_coords[index] = tc[index]
        self.flag_data_update()

    cdef void build(self):
        pass

    cdef int apply(self) except -1:
        if self.flags & GI_NEEDS_UPDATE:
            self.build()
            self.flag_update_done()
        self.batch.draw()
        return 0

cdef convert_to_python(double *ptr, int n):
    cdef int i
    lst=[]
    for i in range(n):
        lst.append(ptr[i])
    return lst

cdef class Callback(Instruction):
    '''.. versionadded:: 1.0.4

    A Callback is an instruction that will be called when the drawing
    operation is performed. When adding instructions to a canvas, you can do
    this::

        with self.canvas:
            Color(1, 1, 1)
            Rectangle(pos=self.pos, size=self.size)
            Callback(self.my_callback)

    The definition of the callback must be::

        def my_callback(self, instr):
            print('I have been called!')

    .. warning::

        Note that if you perform many and/or costly calls to callbacks, you
        might potentially slow down the rendering performance significantly.

    The updating of your canvas does not occur until something new happens.
    From your callback, you can ask for an update::

        with self.canvas:
            self.cb = Callback(self.my_callback)
        # then later in the code
        self.cb.ask_update()

    If you use the Callback class to call rendering methods of another
    toolkit, you will have issues with the OpenGL context. The OpenGL state may
    have been manipulated by the other toolkit, and as soon as program flow
    returns to Kivy, it will just break. You can have glitches, crashes, black
    holes might occur, etc.
    To avoid that, you can activate the :attr:`reset_context` option. It will
    reset the OpenGL context state to make Kivy's rendering correct after the
    call to your callback.

    .. warning::

        The :attr:`reset_context` is not a full OpenGL reset. If you have issues
        regarding that, please contact us.

    '''
    def __init__(self, callback=None, **kwargs):
        Instruction.__init__(self, **kwargs)
        self.func = callback
        self._reset_context = int(kwargs.get('reset_context', False))

    def ask_update(self):
        '''Inform the parent canvas that we'd like it to update on the next
        frame. This is useful when you need to trigger a redraw due to some
        value having changed for example.

        .. versionadded:: 1.0.4
        '''
        self.flag_data_update()

    cdef int apply(self) except -1:
        cdef RenderContext rcx
        cdef Context ctx
        cdef Shader shader
        cdef int i

        cgl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        cgl.glBindBuffer(GL_ARRAY_BUFFER, 0)

        func = self.func
        if func is None or func(self):
            self.flag_update_done()

        if func is not None and self._reset_context:
            # FIXME do that in a proper way
            cgl.glDisable(GL_DEPTH_TEST)
            cgl.glDisable(GL_CULL_FACE)
            cgl.glDisable(GL_SCISSOR_TEST)
            cgl.glEnable(GL_BLEND)
            cgl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            cgl.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE)
            cgl.glUseProgram(0)

            # FIXME don't use 10. use max texture available from gl conf
            for i in xrange(10):
                cgl.glActiveTexture(GL_TEXTURE0 + i)
                cgl.glBindTexture(GL_TEXTURE_2D, 0)
                cgl.glDisableVertexAttribArray(i)
                cgl.glBindBuffer(GL_ARRAY_BUFFER, 0)
                cgl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

            # reset all the vertexformat in all shaders
            ctx = get_context()
            for obj in ctx.lr_shader:
                shader = obj()
                if not shader:
                    continue
                shader.bind_vertex_format(None)

            # force binding again all our textures.
            rcx = getActiveContext()
            shader = rcx._shader
            rcx.enter()
            for index, texture in rcx.bind_texture.iteritems():
                rcx.set_texture(index, texture)

            reset_gl_context()
        return 0

    cdef int enter(self) except -1:
        self._shader.use()
        return 0

    @property
    def reset_context(self):
        '''Set this to True if you want to reset the OpenGL context for Kivy
        after the callback has been called.
        '''
        return self._reset_context

    @reset_context.setter
    def reset_context(self, value):
        cdef int ivalue = int(value)
        if self._reset_context == ivalue:
            return
        self._reset_context = ivalue
        self.flag_data_update()

    @property
    def callback(self):
        '''Property for getting/setting func.
        '''
        return self.func

    @callback.setter
    def callback(self, object func):
        if self.func == func:
            return
        self.func = func
        self.flag_data_update()


cdef class CanvasBase(InstructionGroup):
    '''CanvasBase provides the context manager methods for the
    :class:`Canvas`.'''
    def __enter__(self):
        pushActiveCanvas(self)

    def __exit__(self, *largs):
        popActiveCanvas()


cdef class Canvas(CanvasBase):
    '''The important Canvas class. Use this class to add graphics or context
    instructions that you want to be used for drawing.

    .. note::

        The Canvas supports Python's ``with`` statement and its enter & exit
        semantics.

    Usage of a canvas without the ``with`` statement::

        self.canvas.add(Color(1., 1., 0))
        self.canvas.add(Rectangle(size=(50, 50)))

    Usage of a canvas with Python's ``with`` statement::

        with self.canvas:
            Color(1., 1., 0)
            Rectangle(size=(50, 50))
    '''

    def __init__(self, **kwargs):
        get_context().register_canvas(self)
        CanvasBase.__init__(self, **kwargs)
        self._opacity = kwargs.get('opacity', 1.0)
        self._before = None
        self._after = None

    cdef void reload(self) except *:
        return
        '''
        # XXX ensure it's not needed anymore.
        cdef Canvas c
        if self._before is not None:
            c = self._before
            c.reload()
        CanvasBase.reload(self)
        if self._after is not None:
            c = self._after
            c.reload()
        '''

    cpdef clear(self):
        '''Clears every :class:`Instruction` in the canvas, leaving it clean.'''
        cdef Instruction c
        for c in self.children[:]:
            if c is self._before or c is self._after:
                continue
            if c.flags & GI_NO_REMOVE:
                continue
            self.remove(c)

    cpdef draw(self):
        '''Apply the instruction to our window.
        '''
        self.apply()

    cdef int apply(self) except -1:
        cdef float opacity = self._opacity
        cdef float rc_opacity
        cdef RenderContext rc
        if opacity != 1.0:
            rc = getActiveContext()
            rc_opacity = rc['opacity']
            rc.push_state('opacity')
            rc['opacity'] = rc_opacity * opacity
        InstructionGroup.apply(self)
        if opacity != 1.0:
            rc.pop_state('opacity')
        return 0

    cpdef add(self, Instruction c):
        # the after group must remain the last one.
        if self._after is None:
            c.radd(self)
        else:
            c.rinsert(self, -1)
        self.flag_data_update()

    cpdef remove(self, Instruction c):
        c.rremove(self)
        self.flag_data_update()

    def ask_update(self):
        '''Inform the canvas that we'd like it to update on the next frame.
        This is useful when you need to trigger a redraw due to some value
        having changed for example.
        '''
        self.flag_data_update()

    @property
    def before(self):
        '''Property for getting the 'before' group.
        '''
        if self._before is None:
            self._before = CanvasBase()
            self.insert(0, self._before)
        return self._before

    @property
    def after(self):
        '''Property for getting the 'after' group.
        '''
        cdef CanvasBase c
        if self._after is None:
            c = CanvasBase()
            self.add(c)
            self._after = c
        return self._after

    @property
    def has_before(self):
        '''Property to see if the :attr:`before` group has already been created.

        .. versionadded:: 1.7.0
        '''
        return self._before is not None

    @property
    def has_after(self):
        '''Property to see if the :attr:`after` group has already been created.

        .. versionadded:: 1.7.0
        '''
        return self._after is not None


    @property
    def opacity(self):
        '''Property to get/set the opacity value of the canvas.

        .. versionadded:: 1.4.1

        The opacity attribute controls the opacity of the canvas and its
        children.  Be careful, it's a cumulative attribute: the value is
        multiplied to the current global opacity and the result is applied to
        the current context color.

        For example: if your parent has an opacity of 0.5 and a child has an
        opacity of 0.2, the real opacity of the child will be 0.5 * 0.2 = 0.1.

        Then, the opacity is applied on the shader as::

            frag_color = color * vec4(1.0, 1.0, 1.0, opacity);

        '''
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = value
        self.flag_data_update()

# Active Canvas and getActiveCanvas function is used
# by instructions, so they know which canvas to add
# themselves to
cdef CanvasBase ACTIVE_CANVAS = None

cdef CanvasBase getActiveCanvas():
    global ACTIVE_CANVAS
    return ACTIVE_CANVAS

# Canvas Stack, for internal use so canvas can be bound
# inside other canvas, and restored when other canvas is done
cdef list CANVAS_STACK = list()

cdef pushActiveCanvas(CanvasBase c):
    global ACTIVE_CANVAS, CANVAS_STACK
    CANVAS_STACK.append(ACTIVE_CANVAS)
    ACTIVE_CANVAS = c

cdef popActiveCanvas():
    global ACTIVE_CANVAS, CANVAS_STACK
    ACTIVE_CANVAS = CANVAS_STACK.pop()


#TODO: same as canvas, move back to context.pyx..fix circular import
#on actual import from python problem
include "common.pxi"
from .vertex cimport *
#from .texture cimport *

from os.path import join
from kivy import kivy_shader_dir
from kivy.cache import Cache
from kivy.core.image import Image
from kivy.graphics.transformation cimport Matrix

cdef class RenderContext(Canvas):
    '''The render context stores all the necessary information for drawing, i.e.:

    - The vertex shader
    - The geometry shader
    - The fragment shader
    - The default texture
    - The state stack (color, texture, matrix...)
    '''
    def __cinit__(self, *args, **kwargs):
        self._use_parent_projection = 0
        self._use_parent_modelview = 0
        self._use_parent_frag_modelview = 0
        self.bind_texture = dict()

    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, **kwargs)
        vs_src = kwargs.get('vs', None)
        gs_src = kwargs.get('gs', None)
        fs_src = kwargs.get('fs', None)
        self._shader = Shader(vs_src, gs_src, fs_src)

        # load default texture image
        filename = join(kivy_shader_dir, 'default.png')
        tex = Cache.get('kv.texture', filename)
        if not tex:
            tex = Image(filename).texture
            Cache.append('kv.texture', filename, tex)
        self.default_texture = tex

        self.state_stacks = {
            'opacity': [1.0],
            'texture0' : [0],
            'color'    : [[1.0,1.0,1.0,1.0]],
            'projection_mat': [Matrix()],
            'modelview_mat' : [Matrix()],
            'frag_modelview_mat' : [Matrix()],
        }

        cdef str key
        self._shader.use()
        for key, stack in self.state_stacks.iteritems():
            self.set_state(key, stack[0])

        if 'use_parent_projection' in kwargs:
            self._use_parent_projection = bool(int(kwargs['use_parent_projection']))
        if 'use_parent_modelview' in kwargs:
            self._use_parent_modelview = bool(int(kwargs['use_parent_modelview']))
        if 'use_parent_frag_modelview' in kwargs:
            self._use_parent_frag_modelview = bool(int(kwargs['use_parent_frag_modelview']))

    cdef void set_state(self, str name, value, int apply_now=0):
        # Upload the uniform value to the shader
        cdef list d
        if name not in self.state_stacks:
            self.state_stacks[name] = [value]
            self.flag_update()
        else:
            d = self.state_stacks[name]
            if value != d[-1]:
                d[-1] = value
                self.flag_update()
        self._shader.set_uniform(name, value)

    cdef get_state(self, str name):
        return self.state_stacks[name][-1]

    cdef int set_states(self, dict states) except -1:
        cdef str name
        for name, value in states.iteritems():
            self.set_state(name, value)

    cdef int push_state(self, str name) except -1:
        stack = self.state_stacks[name]
        stack.append(stack[-1])
        self.flag_update()

    cdef int push_states(self, list names) except -1:
        cdef str name
        for name in names:
            self.push_state(name)

    cdef int pop_state(self, str name) except -1:
        stack = self.state_stacks[name]
        oldvalue = stack.pop()
        if oldvalue != stack[-1]:
            self.set_state(name, stack[-1])
            self.flag_update()

    cdef int pop_states(self, list names) except -1:
        cdef str name
        for name in names:
            self.pop_state(name)

    cdef void set_texture(self, int index, Texture texture):
        # TODO this code is actually broken,
        # the binded texture can be already set, but we may changed if we came
        # from another render context.
        #if index in self.bind_texture and \
        #   self.bind_texture[index] is texture:
        #    return
        global _active_texture
        self.bind_texture[index] = texture
        if _active_texture != index:
            _active_texture = index
            cgl.glActiveTexture(GL_TEXTURE0 + index)
        texture.bind()
        self.flag_update()

    cdef int enter(self) except -1:
        self._shader.use()
        return 0

    cdef int leave(self) except -1:
        self._shader.stop()
        return 0

    cdef int apply(self) except -1:
        cdef list keys
        if PY2:
            keys = self.state_stacks.keys()
        else:
            keys = list(self.state_stacks.keys())

        cdef RenderContext active_context = getActiveContext()
        if self._use_parent_projection:
            self.set_state('projection_mat',
                    active_context.get_state('projection_mat'), 0)
        if self._use_parent_modelview:
            self.set_state('modelview_mat',
                    active_context.get_state('modelview_mat'), 0)
        if self._use_parent_frag_modelview:
            self.set_state('frag_modelview_mat',
                    active_context.get_state('frag_modelview_mat'), 0)
        pushActiveContext(self)
        if _need_reset_gl:
            reset_gl_context()
        self.push_states(keys)
        Canvas.apply(self)
        self.pop_states(keys)
        popActiveContext()
        self.flag_update_done()

        return 0

    cdef void reload(self) except *:
        pushActiveContext(self)
        reset_gl_context()
        Canvas.reload(self)
        popActiveContext()

    def __setitem__(self, key, val):
        self.set_state(key, val)

    def __getitem__(self, key):
        return self._shader.uniform_values[key]

    @property
    def shader(self):
        '''Return the shader attached to the render context.
        '''
        return self._shader

    @property
    def use_parent_projection(self):
        '''If True, the parent projection matrix will be used.

        .. versionadded:: 1.7.0

        Before::

            rc['projection_mat'] = Window.render_context['projection_mat']

        Now::

            rc = RenderContext(use_parent_projection=True)
        '''
        return bool(self._use_parent_projection)

    @use_parent_projection.setter
    def use_parent_projection(self, value):
        cdef cvalue = int(bool(value))
        if self._use_parent_projection != cvalue:
            self._use_parent_projection = cvalue
            self.flag_data_update()

    @property
    def use_parent_modelview(self):
        '''If True, the parent modelview matrix will be used.

        .. versionadded:: 1.7.0

        Before::

            rc['modelview_mat'] = Window.render_context['modelview_mat']

        Now::

            rc = RenderContext(use_parent_modelview=True)
        '''
        return bool(self._use_parent_modelview)

    @use_parent_modelview.setter
    def use_parent_modelview(self, value):
        cdef cvalue = int(bool(value))
        if self._use_parent_modelview != cvalue:
            self._use_parent_modelview = cvalue
            self.flag_data_update()

    @property
    def use_parent_frag_modelview(self):
        '''If True, the parent fragment modelview matrix will be used.

        .. versionadded:: 1.10.1

            rc = RenderContext(use_parent_frag_modelview=True)
        '''
        return bool(self._use_parent_frag_modelview)

    @use_parent_frag_modelview.setter
    def use_parent_frag_modelview(self, value):
        cdef cvalue = int(bool(value))
        if self._use_parent_frag_modelview != cvalue:
            self._use_parent_frag_modelview = cvalue
            self.flag_data_update()


cdef RenderContext ACTIVE_CONTEXT = None
cdef list CONTEXT_STACK  = list()


cdef RenderContext getActiveContext():
    global ACTIVE_CONTEXT
    return ACTIVE_CONTEXT


cdef pushActiveContext(RenderContext c):
    global CONTEXT_STACK, ACTIVE_CONTEXT
    CONTEXT_STACK.append(ACTIVE_CONTEXT)
    ACTIVE_CONTEXT = c
    c.enter()


cdef popActiveContext():
    global CONTEXT_STACK, ACTIVE_CONTEXT
    if ACTIVE_CONTEXT:
        ACTIVE_CONTEXT.leave()
    ACTIVE_CONTEXT = CONTEXT_STACK.pop()
    if ACTIVE_CONTEXT:
        ACTIVE_CONTEXT.enter()


cdef class TransformFeedback(ObjectWithUid):
    '''The render context stores all the necessary information for drawing, i.e.:

    - The vertex shader
    - The geometry shader
    - The fragment shader
    - The default texture
    - The state stack (color, texture, matrix...)
    '''
    # def __cinit__(self, *args, **kwargs):
    #     self._use_parent_projection = 0
    #     self._use_parent_modelview = 0
    #     self._use_parent_frag_modelview = 0
    #     self.bind_texture = dict()

    def __init__(self, *args, **kwargs):

        vs_src = kwargs.get('vs', None)
        gs_src = kwargs.get('gs', None)
        fs_src = kwargs.get('fs', None)

        self.max_primitives = kwargs.get('max_primitives', 3)
        self.in_format = kwargs.get('in_format', [(b'inValue', 1, 'float'),])
        self.out_varyings = kwargs.get('out_varyings ', [ "outValue"])

        if vs_src is None:
            vs_src = default_transform_vs
        if gs_src is None:
            gs_src = default_transform_gs
        if fs_src is None:
            fs_src = default_transform_fs
        
        self._shader = Shader(vs_src, gs_src, fs_src, is_transform_feedback=1)

        self._shader.set_varyings( self.out_varyings )

        self.state_stacks = {
            'opacity': [1.0],
        }
        cdef str key
        self._shader.use()
        for key, stack in self.state_stacks.iteritems():
            self.set_state(key, stack[0])

    cdef void set_state(self, str name, value, int apply_now=0):
        # Upload the uniform value to the shader
        cdef list d
        if name not in self.state_stacks:
            self.state_stacks[name] = [value]
            self.flag_update()
        else:
            d = self.state_stacks[name]
            if value != d[-1]:
                d[-1] = value
                self.flag_update()
        self._shader.set_uniform(name, value)

    cdef get_state(self, str name):
        return self.state_stacks[name][-1]

    cdef int set_states(self, dict states) except -1:
        cdef str name
        for name, value in states.iteritems():
            self.set_state(name, value)

    cdef int push_state(self, str name) except -1:
        stack = self.state_stacks[name]
        stack.append(stack[-1])
        self.flag_update()

    cdef int push_states(self, list names) except -1:
        cdef str name
        for name in names:
            self.push_state(name)

    cdef int pop_state(self, str name) except -1:
        stack = self.state_stacks[name]
        oldvalue = stack.pop()
        if oldvalue != stack[-1]:
            self.set_state(name, stack[-1])
            self.flag_update()

    cdef int pop_states(self, list names) except -1:
        cdef str name
        for name in names:
            self.pop_state(name)

    def __setitem__(self, key, val):
        self.set_state(key, val)

    def __getitem__(self, key):
        return self._shader.uniform_values[key]

    @property
    def shader(self):
        '''Return the shader attached to the render context.
        '''
        return self._shader

    def print_debug(self, debug, *args):
        if debug:
            print(*args)

    def transform(self, vi_from, vi_to, input_count, debug=False):
        # vertex_format = VertexFormat( (b'inValue', 1, 'float'),)
        vertex_format = VertexFormat( *self.in_format )
        
        self.print_debug(debug, "RECEIVING DATA", vi_to.gbatch.gvbo.gsize, vi_to.gbatch.gvbo.gdsize)
        log_gl_error("Starting transform fedback")
        self.print_debug(debug, cgl.glGetError(), "Starting transform feedback!")
        reset_gl_context()
        self.print_debug(debug, cgl.glGetError(), "Context reset!")
        self._shader.use()
        self.print_debug(debug, cgl.glGetError(), "Program in use!")

        self.print_debug(debug, cgl.glGetError(), "Binding VBO")
        cgl.glBindBuffer(GL_ARRAY_BUFFER, vi_from.gbatch.gvbo.gid)

        BUFCOUNT = input_count
        BUFSIZE = BUFCOUNT * self.max_primitives * 4 * vertex_format.vsize

        vi_to.gbatch.gvbo.set_transform_feedback(1)

        self.print_debug(debug, "BUFSIZE:", BUFSIZE)
        self._shader.bind_vertex_format(vertex_format)


        self.print_debug(debug, cgl.glGetError(), "Build receiving VBO")

        cgl.glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, vi_to.gbatch.gvbo.gid)
        cgl.glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, BUFSIZE, NULL, GL_DYNAMIC_DRAW)

        self.print_debug(debug, cgl.glGetError(), "Binding receiving buffer")
        cgl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vi_to.gbatch.gvbo.gid)


        self.print_debug(debug, cgl.glGetError(), "Disabling rasterizer")
        cgl.glEnable(GL_RASTERIZER_DISCARD)

        self.print_debug(debug, cgl.glGetError(), "Building query")
        cdef GLuint query
        cgl.glGenQueries(1, &query)
        cgl.glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query)

        self.print_debug(debug, cgl.glGetError(), "Going to activate Transform Feedback.")
        cgl.glBeginTransformFeedback(GL_POINTS)
        self.print_debug(debug, cgl.glGetError(), "Transform Feedback activated.")

        self.print_debug(debug, cgl.glGetError(), "Draw arrays")
        cgl.glDrawArrays(GL_POINTS, 0, BUFCOUNT)
        self.print_debug(debug, "GCOUNT:", BUFSIZE)

        self.print_debug(debug, "SENDING BUFFER INFO:", vi_from.gbatch.gvbo)
        self.print_debug(debug, "RECEIVING BUFFER INFO:", vi_to.gbatch.gvbo)
        self.print_debug(debug, cgl.glGetError(), "Ending Transform Feedback")
        cgl.glEndTransformFeedback()
        self.print_debug(debug, cgl.glGetError(), "Transform Feedback ended.")
        
        self.print_debug(debug, cgl.glGetError(), "Ending query")
        cgl.glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)

        self.print_debug(debug, cgl.glGetError(), "Receiving query data")
        cdef GLuint* primitives = [0]
        cgl.glGetQueryObjectuiv(query, GL_QUERY_RESULT, primitives)
        self.print_debug(debug, cgl.glGetError(), "Primitives:", primitives[0])
        ACTUAL_BUFSIZE = BUFCOUNT * primitives[0] * 4 * vertex_format.vsize

        self.print_debug(debug, cgl.glGetError(), "Actual BUFSIZE:", ACTUAL_BUFSIZE)


        self.print_debug(debug, cgl.glGetError(), "Disabling Rasterizer")
        cgl.glDisable(GL_RASTERIZER_DISCARD)

        self.print_debug(debug, cgl.glGetError(), "Flushing")
        cgl.glFlush()

        if debug:
            self.print_debug(debug, cgl.glGetError(), "Copying data back to host")

            transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, ACTUAL_BUFSIZE, GL_MAP_READ_BIT)

            self.print_debug(debug, cgl.glGetError(), "Map Buffer Initialised")

            self.print_debug(debug, cgl.glGetError(), "Unmapping copy")
            cgl.glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER)

            for i in range(ACTUAL_BUFSIZE // 4):
                self.print_debug(debug, cgl.glGetError(), f"{i}:", transform_result[i])
            self.print_debug(debug, cgl.glGetError(), "Mapbufferrange! 2")

        self.print_debug(debug, cgl.glGetError(), "Resetting context")
        self._shader.stop()
        self.print_debug(debug, cgl.glGetError(), "Shader stopped")
        reset_gl_context()
        self.print_debug(debug, cgl.glGetError(), "Context reset")

        vi_to.indices = list(range(primitives[0]))

        return primitives[0]

    # def transform_dynamic_example(self, vi_from, vi_to, input_count):
    #     cgl.glBindBuffer(GL_ARRAY_BUFFER, 0)
    #     reset_gl_context()
    #     self._shader.use()
    #     print(cgl.glGetError(), "Program in use!")

    #     print(cgl.glGetError(), "Binding VBO")
    #     cgl.glBindBuffer(GL_ARRAY_BUFFER, vi_from.gbatch.gvbo.gid)

    #     BUFCOUNT = input_count
    #     BUFSIZE = BUFCOUNT * 4

    #     vi_to.vertices = [0]*(BUFCOUNT * self.max_primitives)
    #     vi_to.indices = list(range(BUFCOUNT * self.max_primitives))

    #     print("BUFSIZE:", BUFSIZE)
    #     self._shader.bind_vertex_format(vertex_format)


    #     print(cgl.glGetError(), "Build receiving VBO")

    #     cgl.glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, vi_to.gbatch.gvbo.gid)
    #     cgl.glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, BUFSIZE * self.max_primitives, NULL, GL_DYNAMIC_DRAW)

    #     print(cgl.glGetError(), "Binding receiving buffer")
    #     cgl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vi_to.gbatch.gvbo.gid)


    #     print(cgl.glGetError(), "Disabling rasterizer")
    #     cgl.glEnable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Building query")
    #     cdef GLuint query
    #     cgl.glGenQueries(1, &query)
    #     cgl.glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query)

    #     print(cgl.glGetError(), "Going to activate Transform Feedback.")
    #     cgl.glBeginTransformFeedback(GL_TRIANGLES)
    #     print(cgl.glGetError(), "Transform Feedback activated.")

    #     print(cgl.glGetError(), "Draw arrays")
    #     cgl.glDrawArrays(GL_POINTS, 0, BUFCOUNT)
    #     print("GCOUNT:", BUFSIZE)

    #     print("RECEIVING BUFFER INFO:", vi_to.gbatch.gelements)
    #     print(cgl.glGetError(), "Ending Transform Feedback")
    #     cgl.glEndTransformFeedback()
    #     print(cgl.glGetError(), "Transform Feedback ended.")
        
    #     print(cgl.glGetError(), "Ending query")
    #     cgl.glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)

    #     print(cgl.glGetError(), "Receiving query data")
    #     cdef GLuint* primitives = [0]
    #     cgl.glGetQueryObjectuiv(query, GL_QUERY_RESULT, primitives)
    #     print(cgl.glGetError(), "Primitives:", primitives[0])


    #     print(cgl.glGetError(), "Disabling Rasterizer")
    #     cgl.glDisable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Flushing")
    #     cgl.glFlush()

    #     print(cgl.glGetError(), "Copying data back to host")

    #     transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, primitives[0] * self.max_primitives, GL_MAP_READ_BIT)

    #     print(cgl.glGetError(), "Map Buffer Initialised")

    #     print(cgl.glGetError(), "Unmapping copy")
    #     cgl.glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER)

    #     for i in range(BUFCOUNT * self.max_primitives):
    #         print(cgl.glGetError(), f"{i}:", transform_result[i])
    #     print(cgl.glGetError(), "Mapbufferrange! 2")

    #     print(cgl.glGetError(), "Resetting context")
    #     self._shader.stop()
    #     reset_gl_context()

    #     vi_to.indices = list(range(primitives[0] * self.max_primitives))

    #     return primitives[0]


    # def transform_via_shader_example(self):
        
    #     self._shader.use()
    #     print(cgl.glGetError(), "Program in use!")

    #     print(cgl.glGetError(), "Generating data!")
    #     cdef GLfloat* transform_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    #     print(cgl.glGetError(), transform_data[0])
    #     print(cgl.glGetError(), transform_data[1])
    #     print(cgl.glGetError(), transform_data[2])
    #     print(cgl.glGetError(), transform_data[3])
    #     print(cgl.glGetError(), transform_data[4])

    #     print(cgl.glGetError(), "Building VBO")
    #     cdef GLuint transform_vbo
    #     cgl.glGenBuffers(1, &transform_vbo)
    	
    #     print(cgl.glGetError(), "Binding VBO")
    #     cgl.glBindBuffer(GL_ARRAY_BUFFER, transform_vbo)

    #     BUFSIZE = 5 * 4
    #     print(cgl.glGetError(), "Upload data to buffer")
    #     cgl.glBufferData(GL_ARRAY_BUFFER, BUFSIZE, transform_data, GL_STATIC_READ) # GL_STATIC_DRAW


    #     cdef VertexFormat default_vertex = VertexFormat( (b'inValue', 1, 'float'),)
    #     self._shader.bind_vertex_format(default_vertex)


    #     print(cgl.glGetError(), "Build receiving VBO")
    #     cdef GLuint tbo
    #     cgl.glGenBuffers(1, &tbo)
    #     cgl.glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo)
    #     cgl.glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, BUFSIZE*3, NULL, GL_STATIC_READ)

    #     print(cgl.glGetError(), "Binding receiving buffer")
    #     cgl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo)


    #     print(cgl.glGetError(), "Disabling rasterizer")
    #     cgl.glEnable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Going to activate Transform Feedback.")
    #     cgl.glBeginTransformFeedback(GL_TRIANGLES)
    #     print(cgl.glGetError(), "Transform Feedback activated.")

    #     print(cgl.glGetError(), "Draw arrays")
    #     cgl.glDrawArrays(GL_POINTS, 0, 5)

    #     print(cgl.glGetError(), "Ending Transform Feedback")
    #     cgl.glEndTransformFeedback()
    #     print(cgl.glGetError(), "Transform Feedback ended.")
        
    #     print(cgl.glGetError(), "Disabling Rasterizer")
    #     cgl.glDisable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Flushing")
    #     cgl.glFlush()

    #     print(cgl.glGetError(), "Fetching data")
    #     cdef GLfloat feedback[5*3]

    #     print(cgl.glGetError(), "Copying data back to host")

    #     transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, BUFSIZE*3, GL_MAP_READ_BIT)

    #     print(cgl.glGetError(), "Map Buffer Initialised")

    #     print(cgl.glGetError(), "Unmapping copy")
    #     cgl.glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER)

    #     for i in range(15):
    #         print(cgl.glGetError(), f"{i}:", transform_result[i])
    #     print(cgl.glGetError(), "Mapbufferrange! 2")

    #     print(cgl.glGetError(), "Resetting context")
    #     self._shader.stop()
    #     reset_gl_context()
    #     return 0


    # def transform_static_example(self):
    #     cdef RenderContext rcx
    #     cdef Context ctx
    #     # cdef Shader shader
    #     cdef GLuint transform_shader
    #     cdef GLuint frag_shader
    #     cdef int i
        
    #     # Actual transform vertex shader
    #     cdef char* source = """#version 320 es

    #         #ifdef GL_ES
    #         precision mediump float;
    #         #endif

    #         in float inValue;
    #         out float outValue;

    #         void main()
    #         {
    #             outValue = sqrt(inValue);
    #         }"""

    #     # Required for Android
    #     cdef char* fragsource = """#version 320 es

    #         #ifdef GL_ES
    #         precision mediump float;
    #         #endif

    #         out vec4 fragColor;

    #         void main() {
    #           fragColor = vec4(1.0,1.0,1.0,1.0);
    #         }"""

    #     print(cgl.glGetError(), source)
    #     print(cgl.glGetError(), "Building the Transform Feedback Vertex shader.")
    #     transform_shader = cgl.glCreateShader(GL_VERTEX_SHADER)
    #     cgl.glShaderSource(transform_shader, 1, <const_char_ptr*> &source, NULL)
    #     cgl.glCompileShader(transform_shader)
    #     print(cgl.glGetError(), "Transform Feedback Vertex shader has been built and compiled.")

    #     print(cgl.glGetError(), "Building the Transform Feedback Fragment shader.")
    #     f_transform_shader = cgl.glCreateShader(GL_FRAGMENT_SHADER)
    #     cgl.glShaderSource(f_transform_shader, 1, <const_char_ptr*> &fragsource, NULL)
    #     cgl.glCompileShader(f_transform_shader)
    #     print(cgl.glGetError(), "Transform Feedback Fragment shader has been built and compiled.")

    #     print(cgl.glGetError(), "Building program.")
    #     cdef GLuint transform_program = cgl.glCreateProgram()
    #     print(cgl.glGetError(), f"Attaching shader. Program: {transform_program}")
    #     cgl.glAttachShader(transform_program, transform_shader)
    #     print(cgl.glGetError(), "Vertex attachment succesful.")
    #     cgl.glAttachShader(transform_program, f_transform_shader)
    #     print(cgl.glGetError(), "Fragment attachment succesful.")
        
    #     print(cgl.glGetError(), "Defining varyings")
    #     cdef GLchar** feedbackVaryings = [ "outValue" ]
    #     print(cgl.glGetError(), "Building varyings")
    #     cgl.glTransformFeedbackVaryings(transform_program, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS)
    #     print(cgl.glGetError(), "Varyings built!")

    #     print(cgl.glGetError(), "Linking program")
    #     cgl.glLinkProgram(transform_program)
    #     print(cgl.glGetError(), "Program linked!")

    #     cdef char msg[2048]
    #     cdef GLsizei length
    #     msg[0] = b'\0'
    #     cgl.glGetProgramInfoLog(transform_program, 2048, &length, msg)
    #     cdef bytes ret = msg[:length]
    #     print(cgl.glGetError(), ret.split(b'\0')[0].decode('utf-8'))

    #     cgl.glUseProgram(transform_program)
    #     print(cgl.glGetError(), "Program in use!")

    #     print(cgl.glGetError(), "Generating data!")
    #     cdef GLfloat* transform_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    #     print(cgl.glGetError(), transform_data[0])
    #     print(cgl.glGetError(), transform_data[1])
    #     print(cgl.glGetError(), transform_data[2])
    #     print(cgl.glGetError(), transform_data[3])
    #     print(cgl.glGetError(), transform_data[4])

    #     print(cgl.glGetError(), "Building VBO")
    #     cdef GLuint transform_vbo
    #     cgl.glGenBuffers(1, &transform_vbo)
    	
    #     print(cgl.glGetError(), "Binding VBO")
    #     cgl.glBindBuffer(GL_ARRAY_BUFFER, transform_vbo)

    #     BUFSIZE = 5 * 4
    #     print(cgl.glGetError(), "Upload data to buffer")
    #     cgl.glBufferData(GL_ARRAY_BUFFER, BUFSIZE, transform_data, GL_STATIC_READ) # GL_STATIC_DRAW

    #     # print(cgl.glGetError(), "Checking buffer")
    #     # cdef GLfloat* transform_result = <GLfloat*>cgl.glMapBufferRange(GL_ARRAY_BUFFER, 0, BUFSIZE, GL_MAP_READ_BIT)
    #     # cgl.glUnmapBuffer(GL_ARRAY_BUFFER)
        
    #     # transform_data[1] = 10
    #     # print(cgl.glGetError(), transform_result[0])
    #     # print(cgl.glGetError(), transform_result[1])
    #     # print(cgl.glGetError(), transform_result[2])
    #     # print(cgl.glGetError(), transform_result[3])
    #     # print(cgl.glGetError(), transform_result[4])
        
    #     # print(cgl.glGetError(), transform_data[0])
    #     # print(cgl.glGetError(), transform_data[1])
    #     # print(cgl.glGetError(), transform_data[2])
    #     # print(cgl.glGetError(), transform_data[3])
    #     # print(cgl.glGetError(), transform_data[4])

    #     print(cgl.glGetError(), "Building vertex pointer")
    #     cdef GLint inputAttrib = cgl.glGetAttribLocation(transform_program, "inValue")

    #     print(cgl.glGetError(), "Enabling vertex pointer")
    #     cgl.glEnableVertexAttribArray(inputAttrib)

    #     print(cgl.glGetError(), "Set vertex pointer")
    #     cgl.glVertexAttribPointer(inputAttrib, 1, GL_FLOAT, GL_FALSE,  <GLsizei>0, <GLvoid*><unsigned int>0) # 0

    #     print(cgl.glGetError(), "Build receiving VBO")
    #     cdef GLuint tbo
    #     cgl.glGenBuffers(1, &tbo)
    #     cgl.glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo)
    #     cgl.glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, BUFSIZE, NULL, GL_STATIC_READ)

    #     print(cgl.glGetError(), "Binding receiving buffer")
    #     cgl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo)


    #     print(cgl.glGetError(), "Disabling rasterizer")
    #     cgl.glEnable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Going to activate Transform Feedback.")
    #     cgl.glBeginTransformFeedback(GL_POINTS)
    #     print(cgl.glGetError(), "Transform Feedback activated.")

    #     print(cgl.glGetError(), "Draw arrays")
    #     cgl.glDrawArrays(GL_POINTS, 0, 5)

    #     print(cgl.glGetError(), "Ending Transform Feedback")
    #     cgl.glEndTransformFeedback()
    #     print(cgl.glGetError(), "Transform Feedback ended.")
        
    #     print(cgl.glGetError(), "Disabling Rasterizer")
    #     cgl.glDisable(GL_RASTERIZER_DISCARD)

    #     # cgl.glDisable(GL_RASTERIZER_DISCARD)
    #     print(cgl.glGetError(), "Flushing")
    #     cgl.glFlush()

    #     print(cgl.glGetError(), "Fetching data")
    #     cdef GLfloat feedback[5]

    #     print(cgl.glGetError(), "Copying data back to host")

    #     # cdef GLfloat* buffercontent = <GLfloat*>(cgl.glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 40, GL_MAP_READ_BIT))
    #     # cdef GLfloat* transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 80, GL_MAP_READ_BIT)
    #     transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, BUFSIZE, GL_MAP_READ_BIT)
    #     # cdef double* transform_result = <double*>malloc(8*16)
    #     print(cgl.glGetError(), "Map Buffer Initialised")
    #     # l = convert_to_python(a, 5)

    #     print(cgl.glGetError(), "Unmapping copy")
    #     cgl.glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER)

    #     print(cgl.glGetError(), "Sqrt of 1 is:", transform_result[0])
    #     print(cgl.glGetError(), "Sqrt of 2 is:", transform_result[1])
    #     print(cgl.glGetError(), "Sqrt of 3 is:", transform_result[2])
    #     print(cgl.glGetError(), "Sqrt of 4 is:", transform_result[3])
    #     print(cgl.glGetError(), "Sqrt of 5 is:", transform_result[4])
    #     print(cgl.glGetError(), "Mapbufferrange! 2")
    #     # print(cgl.glGetError(), l)
    #     # for i in range(5):
    #     #     printf("%X", *(buffercontent+i));
    #     # print(cgl.glGetError(), buffercontent)

        
    #     # void glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
    #     # GLboolean glUnmapBuffer(GLenum target);

    #     # ctypedef void (__stdcall *GLMAPBUFFERRANGE)(GLenum, GLintptr, GLsizeiptr, GLbitfield)
    #     # ctypedef GLboolean (__stdcall *GLUNMAPBUFFER)(GLenum)
    #     print(cgl.glGetError(), "Resetting context")
    #     reset_gl_context()
    #     return 0


    # def transform_geometry_static_example(self):
    #     cdef RenderContext rcx
    #     cdef Context ctx
    #     # cdef Shader shader
    #     cdef GLuint transform_shader
    #     cdef GLuint frag_shader
    #     cdef int i
        
    #     # # Actual transform vertex shader
    #     # cdef char* vertsource = """#version 320 es

    #     #     #ifdef GL_ES
    #     #     precision mediump float;
    #     #     #endif

    #     #     in float inValue;

    #     #     out VS_OUT{
    #     #         float geoValue;
    #     #     }vs_out;

    #     #     void main()
    #     #     {
    #     #         vs_out.geoValue = sqrt(inValue);
    #     #     }"""

    #     # # Actual transform geometry shader
    #     # cdef char* geosource = """#version 320 es

    #     #     #ifdef GL_ES
    #     #     precision mediump float;
    #     #     #endif

    #     #     layout(points)in;
    #     #     layout(triangle_strip,max_primitives=3)out;


    #     #     in VS_OUT{
    #     #         float geoValue;
    #     #     }gs_in[];

    #     #     out float outValue;

    #     #     void main()
    #     #     {
    #     #         for (int i = 0; i < 3; i++) {
    #     #             outValue = gs_in[0].geoValue + float(i);
    #     #             EmitVertex();
    #     #         }

    #     #         EndPrimitive();
    #     #     }
    #     #     """
    #     # # Required for Android
    #     # cdef char* fragsource = """#version 320 es

    #     #     #ifdef GL_ES
    #     #     precision mediump float;
    #     #     #endif

    #     #     out vec4 fragColor;

    #     #     void main() {
    #     #       fragColor = vec4(1.0,1.0,1.0,1.0);
    #     #     }"""

    #     cdef bytes vertbytes = default_transform_vs.encode('utf-8')
    #     cdef bytes geobytes = default_transform_gs.encode('utf-8')
    #     cdef bytes fragbytes = default_transform_fs.encode('utf-8')

    #     cdef char* vertsource = <char*> vertbytes
    #     cdef char* geosource = <char*> geobytes
    #     cdef char* fragsource = <char*> fragbytes

    #     print("Using file sources for static geo!")

    #     print(cgl.glGetError(), vertsource)
    #     print(cgl.glGetError(), "Building the Transform Feedback Vertex shader.")
    #     transform_shader = cgl.glCreateShader(GL_VERTEX_SHADER)
    #     cgl.glShaderSource(transform_shader, 1, <const_char_ptr*> &vertsource, NULL)
    #     cgl.glCompileShader(transform_shader)
    #     print(cgl.glGetError(), "Transform Feedback Vertex shader has been built and compiled.")

    #     print(cgl.glGetError(), "Building the Transform Feedback Geometry shader.")
    #     g_transform_shader = cgl.glCreateShader(GL_GEOMETRY_SHADER)
    #     cgl.glShaderSource(g_transform_shader, 1, <const_char_ptr*> &geosource, NULL)
    #     cgl.glCompileShader(g_transform_shader)
    #     print(cgl.glGetError(), "Transform Feedback Geometry shader has been built and compiled.")

    #     print(cgl.glGetError(), "Building the Transform Feedback Fragment shader.")
    #     f_transform_shader = cgl.glCreateShader(GL_FRAGMENT_SHADER)
    #     cgl.glShaderSource(f_transform_shader, 1, <const_char_ptr*> &fragsource, NULL)
    #     cgl.glCompileShader(f_transform_shader)
    #     print(cgl.glGetError(), "Transform Feedback Fragment shader has been built and compiled.")

    #     print(cgl.glGetError(), "Building program.")
    #     cdef GLuint transform_program = cgl.glCreateProgram()
    #     print(cgl.glGetError(), f"Attaching shader. Program: {transform_program}")
    #     cgl.glAttachShader(transform_program, transform_shader)
    #     print(cgl.glGetError(), "Vertex attachment succesful.")
    #     cgl.glAttachShader(transform_program, g_transform_shader)
    #     print(cgl.glGetError(), "Geometry attachment succesful.")
    #     cgl.glAttachShader(transform_program, f_transform_shader)
    #     print(cgl.glGetError(), "Fragment attachment succesful.")
        
    #     print(cgl.glGetError(), "Defining varyings")
    #     cdef GLchar** feedbackVaryings = [ "outValue" ]
    #     print(cgl.glGetError(), "Building varyings")
    #     cgl.glTransformFeedbackVaryings(transform_program, 4, feedbackVaryings, GL_INTERLEAVED_ATTRIBS)
    #     print(cgl.glGetError(), "Varyings built!")

    #     print(cgl.glGetError(), "Linking program")
    #     cgl.glLinkProgram(transform_program)
    #     print(cgl.glGetError(), "Program linked!")

    #     cdef char msg[2048]
    #     cdef GLsizei length
    #     msg[0] = b'\0'
    #     cgl.glGetProgramInfoLog(transform_program, 2048, &length, msg)
    #     cdef bytes ret = msg[:length]
    #     print(cgl.glGetError(), ret.split(b'\0')[0].decode('utf-8'))

    #     cgl.glUseProgram(transform_program)
    #     print(cgl.glGetError(), "Program in use!")

    #     print(cgl.glGetError(), "Generating data!")
    #     cdef GLfloat* transform_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    #     print(cgl.glGetError(), transform_data[0])
    #     print(cgl.glGetError(), transform_data[1])
    #     print(cgl.glGetError(), transform_data[2])
    #     print(cgl.glGetError(), transform_data[3])
    #     print(cgl.glGetError(), transform_data[4])

    #     print(cgl.glGetError(), "Building VBO")
    #     cdef GLuint transform_vbo
    #     cgl.glGenBuffers(1, &transform_vbo)
    	
    #     print(cgl.glGetError(), "Binding VBO")
    #     cgl.glBindBuffer(GL_ARRAY_BUFFER, transform_vbo)

    #     BUFSIZE = 5 * 4
    #     print(cgl.glGetError(), "Upload data to buffer")
    #     cgl.glBufferData(GL_ARRAY_BUFFER, BUFSIZE, transform_data, GL_STATIC_READ) # GL_STATIC_DRAW

    #     # print(cgl.glGetError(), "Checking buffer")
    #     # cdef GLfloat* transform_result = <GLfloat*>cgl.glMapBufferRange(GL_ARRAY_BUFFER, 0, BUFSIZE, GL_MAP_READ_BIT)
    #     # cgl.glUnmapBuffer(GL_ARRAY_BUFFER)
        
    #     # transform_data[1] = 10
    #     # print(cgl.glGetError(), transform_result[0])
    #     # print(cgl.glGetError(), transform_result[1])
    #     # print(cgl.glGetError(), transform_result[2])
    #     # print(cgl.glGetError(), transform_result[3])
    #     # print(cgl.glGetError(), transform_result[4])
        
    #     # print(cgl.glGetError(), transform_data[0])
    #     # print(cgl.glGetError(), transform_data[1])
    #     # print(cgl.glGetError(), transform_data[2])
    #     # print(cgl.glGetError(), transform_data[3])
    #     # print(cgl.glGetError(), transform_data[4])

    #     print(cgl.glGetError(), "Building vertex pointer")
    #     cdef GLint inputAttrib = cgl.glGetAttribLocation(transform_program, "inValue")

    #     print(cgl.glGetError(), "Enabling vertex pointer")
    #     cgl.glEnableVertexAttribArray(inputAttrib)

    #     print(cgl.glGetError(), "Set vertex pointer")
    #     cgl.glVertexAttribPointer(inputAttrib, 1, GL_FLOAT, GL_FALSE,  <GLsizei>0, <GLvoid*><unsigned int>0) # 0

    #     print(cgl.glGetError(), "Build receiving VBO")
    #     cdef GLuint tbo
    #     cgl.glGenBuffers(1, &tbo)
    #     cgl.glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo)
    #     cgl.glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, BUFSIZE*3, NULL, GL_STATIC_READ)

    #     print(cgl.glGetError(), "Binding receiving buffer")
    #     cgl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo)


    #     print(cgl.glGetError(), "Disabling rasterizer")
    #     cgl.glEnable(GL_RASTERIZER_DISCARD)

    #     print(cgl.glGetError(), "Going to activate Transform Feedback.")
    #     cgl.glBeginTransformFeedback(GL_TRIANGLES)
    #     print(cgl.glGetError(), "Transform Feedback activated.")

    #     print(cgl.glGetError(), "Draw arrays")
    #     cgl.glDrawArrays(GL_POINTS, 0, 5)

    #     print(cgl.glGetError(), "Ending Transform Feedback")
    #     cgl.glEndTransformFeedback()
    #     print(cgl.glGetError(), "Transform Feedback ended.")
        
    #     print(cgl.glGetError(), "Disabling Rasterizer")
    #     cgl.glDisable(GL_RASTERIZER_DISCARD)

    #     # cgl.glDisable(GL_RASTERIZER_DISCARD)
    #     print(cgl.glGetError(), "Flushing")
    #     cgl.glFlush()

    #     print(cgl.glGetError(), "Fetching data")
    #     cdef GLfloat feedback[5*3]

    #     print(cgl.glGetError(), "Copying data back to host")

    #     # cdef GLfloat* buffercontent = <GLfloat*>(cgl.glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 40, GL_MAP_READ_BIT))
    #     # cdef GLfloat* transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 80, GL_MAP_READ_BIT)
    #     transform_result = <GLfloat*>cgl.glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, BUFSIZE*3, GL_MAP_READ_BIT)
    #     # cdef double* transform_result = <double*>malloc(8*16)
    #     print(cgl.glGetError(), "Map Buffer Initialised")
    #     # l = convert_to_python(a, 5)

    #     print(cgl.glGetError(), "Unmapping copy")
    #     cgl.glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER)

    #     for i in range(15):
    #         print(cgl.glGetError(), f"{i}:", transform_result[i])
    #     print(cgl.glGetError(), "Mapbufferrange! 2")
    #     # print(cgl.glGetError(), l)
    #     # for i in range(5):
    #     #     printf("%X", *(buffercontent+i));
    #     # print(cgl.glGetError(), buffercontent)

        
    #     # void glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
    #     # GLboolean glUnmapBuffer(GLenum target);

    #     # ctypedef void (__stdcall *GLMAPBUFFERRANGE)(GLenum, GLintptr, GLsizeiptr, GLbitfield)
    #     # ctypedef GLboolean (__stdcall *GLUNMAPBUFFER)(GLenum)
    #     print(cgl.glGetError(), "Resetting context")
    #     reset_gl_context()
    #     return 0
