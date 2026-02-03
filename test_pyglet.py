import pyglet
print("Pyglet version:", pyglet.version)
try:
    from pyglet.gl import *
    print("glPushMatrix from import *:", 'glPushMatrix' in locals())
    print("glVertex3f from import *:", 'glVertex3f' in locals())
except Exception as e:
    print("pyglet.gl import * failed:", e)

try:
    import pyglet.gl
    print("pyglet.gl.glPushMatrix:", hasattr(pyglet.gl, 'glPushMatrix'))
except Exception as e:
    print("pyglet.gl access failed:", e)
